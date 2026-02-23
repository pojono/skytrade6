#!/usr/bin/env python3
"""
Precision3 Strategy — Walk-Forward Backtest (Zero Lookahead)

Architecture:
  5 purpose-built models, NO meta-model stacking:
    1. DIRECTION: LightGBM regressor on tgt_ret_1 → predict 1-bar forward return
    2. SIZING:    LightGBM regressor on tgt_realized_vol_5 → inverse-vol position sizing
    3. STOP:      LightGBM regressor on tgt_max_drawdown_long_3 → adaptive stop-loss
    4. FILTER 1:  LightGBM classifier on tgt_consolidation_3 → skip low-opportunity bars
    5. FILTER 2:  LightGBM classifier on tgt_crash_10 → reduce exposure before crashes

  Trade Logic:
    - Direction model predicts next-bar return
    - If |pred_ret| > calibrated threshold AND consolidation_prob < 0.5:
        → LONG if pred_ret > 0, SHORT if pred_ret < 0
    - Position size = base_size / max(predicted_vol / median_vol, 0.5)
    - If crash_prob > calibrated crash threshold: halve position size
    - Stop-loss = entry ± predicted_drawdown * 1.5 (buffer)
    - Hold up to 3 bars, early exit on signal reversal or stop hit
    - Entry at next-bar open (no lookahead)

  Anti-Lookahead Measures:
    - Expanding-window WFO: train on all data up to purge boundary
    - 3-day purge gap between training and trade windows
    - Feature selection from predictable_targets.json (pre-validated per target)
    - Direction threshold calibrated via inner 3-fold CV on training data only
    - All models retrained each period on full training window
    - Entry at next-bar open after signal
    - Stop-loss evaluated at bar close (conservative — real would be intrabar)

Usage:
  python microstructure_research/strategy_precision3.py [--periods N] [--symbol SYMBOL]
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import json
import time
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


# ============================================================
# PARAMETERS
# ============================================================
TF = "4h"
FEATURES_DIR = Path("./features")
CONSTRAINTS_PATH = Path("./microstructure_research/predictable_targets.json")
RESULTS_DIR = Path("./microstructure_research/results")

SELECTION_DAYS = 360       # training window
PURGE_DAYS = 3             # gap between train and trade
TRADE_DAYS = 30            # each trade period

FEE_BPS = 4.0              # round-trip fee (2 bps each way)
FEE_FRAC = FEE_BPS / 10000.0
INITIAL_CAPITAL = 10000.0

HOLD_BARS = 3              # max hold period
MAX_POSITION_FRAC = 1.0    # max fraction of capital per trade
EARLY_EXIT = True          # close on signal reversal

# Model definitions — each model has ONE job
MODELS = {
    "direction": {
        "target": "tgt_ret_1",
        "type": "regressor",
        "features": "direction",   # will resolve to target-specific or core features
    },
    "vol_sizing": {
        "target": "tgt_realized_vol_5",
        "type": "regressor",
        "features": "volatility",
    },
    "stop_loss": {
        "target": "tgt_max_drawdown_long_3",
        "type": "regressor",
        "features": "volatility",
    },
    "filter_consolidation": {
        "target": "tgt_consolidation_3",
        "type": "classifier",
        "features": "volatility",
    },
    "filter_crash": {
        "target": "tgt_crash_10",
        "type": "classifier",
        "features": "volatility",
    },
}

# LightGBM hyperparams — conservative to avoid overfit
LGB_CLF_PARAMS = dict(
    objective="binary", metric="auc", verbosity=-1,
    n_estimators=200, max_depth=5, learning_rate=0.05,
    num_leaves=20, min_child_samples=30,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=0.5, random_state=42,
)
LGB_REG_PARAMS = dict(
    objective="regression", metric="rmse", verbosity=-1,
    n_estimators=200, max_depth=5, learning_rate=0.05,
    num_leaves=20, min_child_samples=30,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=0.5, random_state=42,
)


# ============================================================
# DATA LOADING
# ============================================================
def load_features(features_dir: Path, symbol: str, tf: str) -> pd.DataFrame:
    tf_dir = features_dir / symbol / tf
    files = sorted(tf_dir.glob("*.parquet"))
    if not files:
        print(f"ERROR: No parquet files in {tf_dir}", flush=True)
        sys.exit(1)
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def get_candles_per_day(tf):
    return {"5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}[tf]


# ============================================================
# FEATURE ENGINEERING (no lookahead)
# ============================================================
def add_lag_features(df, feat_cols, lags=(1, 2, 3)):
    """Add lagged features. Only uses past data — no lookahead."""
    new_cols = {}
    for col in feat_cols:
        if col in df.columns:
            for lag in lags:
                new_cols[f"{col}_lag{lag}"] = df[col].shift(lag)
    lag_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, lag_df], axis=1)


def get_feature_set(df, feat_list, add_lags=True):
    """Get feature columns from df, optionally with lags."""
    available = [f for f in feat_list if f in df.columns]
    if not available:
        return df, []
    if add_lags:
        df_out = add_lag_features(df, available, lags=(1, 2, 3))
        lag_cols = [c for c in df_out.columns if c not in df.columns and not c.startswith("tgt_")]
        return df_out, available + lag_cols
    return df, available


def get_Xy(df_feat, feat_cols, target_series):
    """Extract aligned X, y arrays with NaN handling."""
    actual_feats = [f for f in feat_cols if f in df_feat.columns]
    if not actual_feats:
        return None, None, None, []
    X = df_feat[actual_feats].values
    y = target_series.values

    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X_clean = np.nan_to_num(X[valid], nan=0, posinf=0, neginf=0)
    y_clean = y[valid]
    return X_clean, y_clean, valid, actual_feats


# ============================================================
# MODEL TRAINING
# ============================================================
def train_model(X_train, y_train, model_type):
    """Train a single LightGBM model. Returns (model, scaler)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    if model_type == "classifier":
        y_int = y_train.astype(int)
        if len(np.unique(y_int)) < 2:
            return None, None
        model = lgb.LGBMClassifier(**LGB_CLF_PARAMS)
        model.fit(X_scaled, y_int)
    else:
        model = lgb.LGBMRegressor(**LGB_REG_PARAMS)
        model.fit(X_scaled, y_train)

    return model, scaler


def predict_model(model, scaler, X, model_type):
    """Generate predictions from a trained model."""
    X_clean = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    X_scaled = scaler.transform(X_clean)

    if model_type == "classifier":
        return model.predict_proba(X_scaled)[:, 1]
    else:
        return model.predict(X_scaled)


# ============================================================
# FEATURE RESOLUTION
# ============================================================
def resolve_features(model_name, model_cfg, constraints):
    """Resolve which features to use for a given model."""
    target = model_cfg["target"]
    feat_type = model_cfg["features"]
    target_features_map = constraints.get("target_features", {})
    core_features = constraints.get("core_features", [])

    # Direction features: use target-specific features if available, else core
    if feat_type == "direction":
        # For tgt_ret_1, we don't have pre-validated features in the JSON
        # Use features from profitable_long_1 + profitable_short_1 (closest proxy)
        feats = set()
        for proxy in ["tgt_profitable_long_1", "tgt_profitable_short_1",
                       "tgt_alpha_1", "tgt_relative_ret_1"]:
            if proxy in target_features_map:
                feats.update(target_features_map[proxy])
        if not feats:
            feats = set(core_features[:30])
        return list(feats)

    elif feat_type == "volatility":
        # Volatility/regime features
        feats = set()
        for proxy in ["tgt_vol_expansion_5", "tgt_vol_expansion_10",
                       "tgt_consolidation_3", "tgt_crash_10",
                       "tgt_tail_event_3"]:
            if proxy in target_features_map:
                feats.update(target_features_map[proxy])
        if not feats:
            feats = set(core_features[:30])
        return list(feats)

    else:
        # Fallback to target-specific or core
        if target in target_features_map:
            return target_features_map[target]
        return core_features[:30]


# ============================================================
# SINGLE PERIOD EXECUTION
# ============================================================
def run_period(df, period_idx, sel_start, sel_end, trade_start, trade_end,
               constraints, cpd):
    """
    Run one WFO period:
      1. Train 5 models on selection window
      2. Calibrate direction threshold via inner 3-fold CV
      3. Predict on trade window
      4. Execute trades with filters, sizing, and stops
    """
    t0 = time.time()

    df_sel = df.iloc[sel_start:sel_end].copy()
    df_trade = df.iloc[trade_start:trade_end].copy()

    print(f"\n  Period {period_idx+1}:")
    print(f"    Selection: {len(df_sel)} bars [{df_sel.index[0]} -> {df_sel.index[-1]}]")
    print(f"    Trade:     {len(df_trade)} bars [{df_trade.index[0]} -> {df_trade.index[-1]}]")

    # ---- Step 1: Train all 5 models ----
    trained = {}  # model_name -> (model, scaler, feat_cols)

    for model_name, cfg in MODELS.items():
        target = cfg["target"]
        model_type = cfg["type"]

        if target not in df_sel.columns:
            print(f"    WARN: target {target} not in data, skipping {model_name}")
            continue

        feat_list = resolve_features(model_name, cfg, constraints)
        df_feat, feat_cols = get_feature_set(df_sel, feat_list, add_lags=True)

        if len(feat_cols) < 3:
            print(f"    WARN: only {len(feat_cols)} features for {model_name}, skipping")
            continue

        X_tr, y_tr, valid_mask, actual_feats = get_Xy(df_feat, feat_cols, df_sel[target])
        if X_tr is None or len(X_tr) < 200:
            print(f"    WARN: insufficient training data for {model_name} ({0 if X_tr is None else len(X_tr)} rows)")
            continue

        model, scaler = train_model(X_tr, y_tr, model_type)
        if model is None:
            print(f"    WARN: {model_name} training failed (single class?)")
            continue

        trained[model_name] = (model, scaler, actual_feats, feat_list, model_type)

    n_ok = len(trained)
    print(f"    Models trained: {n_ok}/5 "
          f"[{', '.join(k for k in trained)}]")

    if "direction" not in trained:
        print(f"    SKIP: direction model failed — cannot trade")
        return None

    # ---- Step 2: Calibrate direction threshold via inner 3-fold CV ----
    dir_model_info = trained["direction"]
    _, _, dir_feats, dir_feat_list, _ = dir_model_info

    # Inner CV on training data
    n_sel = len(df_sel)
    fold_size = n_sel // 4  # use last 3/4 for 3-fold CV, first 1/4 always in train
    min_train = n_sel // 4

    cv_preds = np.full(n_sel, np.nan)
    cv_actual_ret = df_sel["tgt_ret_1"].values if "tgt_ret_1" in df_sel.columns else None

    if cv_actual_ret is not None:
        for fold in range(3):
            cv_val_start = min_train + fold * fold_size
            cv_val_end = min(cv_val_start + fold_size, n_sel)
            if cv_val_end <= cv_val_start:
                continue

            df_cv_train = df_sel.iloc[:cv_val_start]
            df_cv_val = df_sel.iloc[cv_val_start:cv_val_end]

            df_feat_tr, feat_cols_tr = get_feature_set(df_cv_train, dir_feat_list, add_lags=True)
            df_feat_val, feat_cols_val = get_feature_set(df_cv_val, dir_feat_list, add_lags=True)
            common = [f for f in feat_cols_tr if f in feat_cols_val]

            X_cv_tr, y_cv_tr, _, _ = get_Xy(df_feat_tr, common, df_cv_train["tgt_ret_1"])
            if X_cv_tr is None or len(X_cv_tr) < 100:
                continue

            try:
                cv_model, cv_scaler = train_model(X_cv_tr, y_cv_tr, "regressor")
                if cv_model is None:
                    continue
                X_cv_val = df_feat_val[common].values
                cv_preds[cv_val_start:cv_val_end] = predict_model(
                    cv_model, cv_scaler, X_cv_val, "regressor"
                )
            except Exception as e:
                print(f"    WARN: CV fold {fold} failed: {e}")

    # Find best threshold on CV predictions
    best_thresh = 0.001  # default: ~10 bps
    best_avg_ret = -999
    cv_valid = np.isfinite(cv_preds) & np.isfinite(cv_actual_ret)

    for thresh in [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005]:
        rets = []
        for j in range(n_sel):
            if not cv_valid[j]:
                continue
            pred = cv_preds[j]
            actual = cv_actual_ret[j]
            if abs(pred) > thresh:
                direction = 1.0 if pred > 0 else -1.0
                trade_ret = direction * actual - FEE_FRAC
                rets.append(trade_ret)
        if len(rets) >= 30:
            avg = np.mean(rets)
            if avg > best_avg_ret:
                best_avg_ret = avg
                best_thresh = thresh

    # Also calibrate crash threshold
    crash_thresh = 0.25  # default
    if "filter_crash" in trained:
        crash_info = trained["filter_crash"]
        # Simple: use 0.25 as default, could CV this too but keep it simple
        crash_thresh = 0.25

    # Also calibrate consolidation threshold
    consol_thresh = 0.50  # default

    print(f"    Direction threshold (CV): {best_thresh*10000:.1f} bps "
          f"(CV avg ret: {best_avg_ret*100:+.3f}%)")
    print(f"    Crash filter: >{crash_thresh:.0%}, Consolidation filter: >{consol_thresh:.0%}")

    # ---- Step 3: Generate predictions on trade window ----
    trade_preds = {}

    for model_name, (model, scaler, actual_feats, feat_list, model_type) in trained.items():
        df_feat_trade, feat_cols_trade = get_feature_set(df_trade, feat_list, add_lags=True)
        common = [f for f in actual_feats if f in feat_cols_trade]
        if len(common) < 3:
            continue
        X_trade = df_feat_trade[common].values
        trade_preds[model_name] = predict_model(model, scaler, X_trade, model_type)

    if "direction" not in trade_preds:
        print(f"    SKIP: direction prediction failed")
        return None

    # ---- Step 4: Generate signals ----
    n_trade = len(df_trade)
    pred_ret = trade_preds["direction"]
    pred_vol = trade_preds.get("vol_sizing", None)
    pred_dd = trade_preds.get("stop_loss", None)
    pred_consol = trade_preds.get("filter_consolidation", None)
    pred_crash = trade_preds.get("filter_crash", None)

    # Compute median vol from training data for normalization
    if pred_vol is not None and "tgt_realized_vol_5" in df_sel.columns:
        train_vol = df_sel["tgt_realized_vol_5"].dropna()
        median_vol = train_vol.median() if len(train_vol) > 0 else 0.01
    else:
        median_vol = 0.01

    close_prices = df_trade["close"].values
    high_prices = df_trade["high"].values
    low_prices = df_trade["low"].values
    open_prices = df_trade["open"].values if "open" in df_trade.columns else close_prices

    signals = np.zeros(n_trade)
    sizes = np.zeros(n_trade)
    stop_levels = np.full(n_trade, np.nan)  # stop-loss price level

    n_filtered_consol = 0
    n_filtered_crash = 0
    n_filtered_thresh = 0

    for bar in range(n_trade):
        # Direction signal
        ret_pred = pred_ret[bar]
        if not np.isfinite(ret_pred):
            continue

        # Filter 1: direction threshold
        if abs(ret_pred) < best_thresh:
            n_filtered_thresh += 1
            continue

        # Filter 2: consolidation — skip if high consolidation probability
        if pred_consol is not None and np.isfinite(pred_consol[bar]):
            if pred_consol[bar] > consol_thresh:
                n_filtered_consol += 1
                continue

        # Direction
        direction = 1.0 if ret_pred > 0 else -1.0

        # Position sizing: inverse predicted volatility
        size = MAX_POSITION_FRAC
        if pred_vol is not None and np.isfinite(pred_vol[bar]) and pred_vol[bar] > 0:
            vol_ratio = pred_vol[bar] / max(median_vol, 1e-6)
            size = MAX_POSITION_FRAC / max(vol_ratio, 0.5)
            size = min(size, MAX_POSITION_FRAC)  # cap at max
            size = max(size, 0.2)  # floor at 20%

        # Crash filter: halve size if crash probability is high
        if pred_crash is not None and np.isfinite(pred_crash[bar]):
            if pred_crash[bar] > crash_thresh:
                size *= 0.5
                n_filtered_crash += 1

        signals[bar] = direction
        sizes[bar] = size

        # Stop-loss level from predicted max drawdown
        if pred_dd is not None and np.isfinite(pred_dd[bar]):
            # pred_dd is predicted max_drawdown_long_3 (negative for longs)
            # Use 1.5x buffer to avoid premature stops
            dd_est = abs(pred_dd[bar]) * 1.5
            dd_est = max(dd_est, 0.002)  # minimum 20 bps stop
            dd_est = min(dd_est, 0.05)   # maximum 5% stop
            if direction > 0:
                stop_levels[bar] = close_prices[bar] * (1.0 - dd_est)
            else:
                stop_levels[bar] = close_prices[bar] * (1.0 + dd_est)

    n_signals = (signals != 0).sum()
    print(f"    Signals: {n_signals}/{n_trade} "
          f"(filtered: {n_filtered_thresh} thresh, {n_filtered_consol} consol, "
          f"{n_filtered_crash} crash-reduced)")

    # ---- Step 5: Simulate trades ----
    bar_pnl = np.zeros(n_trade)
    bar_gross = np.zeros(n_trade)
    bar_fees = np.zeros(n_trade)
    bar_position = np.zeros(n_trade)
    n_trades = 0
    n_wins = 0
    n_stopped = 0
    trade_returns = []
    active_trades = []
    current_exit_bar = -1

    for bar in range(n_trade):
        # Early exit: close current position if signal reverses
        if EARLY_EXIT and bar <= current_exit_bar and len(active_trades) > 0:
            last_trade = active_trades[-1]
            if (signals[bar] != 0 and
                signals[bar] != last_trade["direction"] and
                bar > last_trade["entry_bar"]):
                early_exit_bar = bar + 1
                if early_exit_bar < n_trade:
                    last_trade["exit_bar"] = early_exit_bar
                    last_trade["exit_reason"] = "reversal"
                    current_exit_bar = early_exit_bar

        # Stop-loss check: evaluate at bar close for active position
        if bar <= current_exit_bar and len(active_trades) > 0:
            last_trade = active_trades[-1]
            if bar > last_trade["entry_bar"] and last_trade.get("exit_reason", "") != "reversal":
                stop = last_trade.get("stop_price", None)
                if stop is not None:
                    if last_trade["direction"] > 0 and low_prices[bar] <= stop:
                        # Long stop hit — exit at next bar open
                        stop_exit = bar + 1
                        if stop_exit < n_trade:
                            last_trade["exit_bar"] = stop_exit
                            last_trade["exit_reason"] = "stop"
                            current_exit_bar = stop_exit
                            n_stopped += 1
                    elif last_trade["direction"] < 0 and high_prices[bar] >= stop:
                        # Short stop hit — exit at next bar open
                        stop_exit = bar + 1
                        if stop_exit < n_trade:
                            last_trade["exit_bar"] = stop_exit
                            last_trade["exit_reason"] = "stop"
                            current_exit_bar = stop_exit
                            n_stopped += 1

        # Only enter if no active position and signal exists
        # Signal at bar (after bar closes) -> enter at bar+1 open
        if bar > current_exit_bar and signals[bar] != 0 and bar + HOLD_BARS + 1 < n_trade:
            direction = signals[bar]
            size = sizes[bar]
            entry_bar = bar + 1
            entry_price = open_prices[entry_bar]
            exit_bar = entry_bar + HOLD_BARS
            if exit_bar >= n_trade:
                continue
            current_exit_bar = exit_bar

            # Compute stop price from entry price
            stop_price = None
            if not np.isnan(stop_levels[bar]):
                # Recompute stop relative to actual entry price
                if pred_dd is not None and np.isfinite(pred_dd[bar]):
                    dd_est = abs(pred_dd[bar]) * 1.5
                    dd_est = max(dd_est, 0.002)
                    dd_est = min(dd_est, 0.05)
                    if direction > 0:
                        stop_price = entry_price * (1.0 - dd_est)
                    else:
                        stop_price = entry_price * (1.0 + dd_est)

            active_trades.append({
                "entry_bar": entry_bar,
                "exit_bar": exit_bar,
                "direction": direction,
                "size": size,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "exit_reason": "hold",
            })

    # Calculate PnL: entry at open[entry_bar], exit at open[exit_bar]
    for trade in active_trades:
        eb = trade["entry_bar"]
        xb = trade["exit_bar"]
        d = trade["direction"]
        sz = trade["size"]
        ep = trade["entry_price"]

        if xb >= n_trade or ep <= 0:
            continue

        exit_price = open_prices[xb]
        gross_ret = d * (exit_price / ep - 1.0)
        fee = FEE_FRAC
        net_ret = gross_ret - fee

        trade_returns.append(net_ret)
        n_trades += 1
        if net_ret > 0:
            n_wins += 1

        # Distribute PnL across holding bars
        actual_hold = max(xb - eb, 1)
        for b in range(eb, min(xb + 1, n_trade)):
            bar_pnl[b] += net_ret * sz / actual_hold
            bar_gross[b] += gross_ret * sz / actual_hold
            bar_fees[b] += fee * sz / actual_hold
            bar_position[b] = d * sz

    # ---- Compute period metrics ----
    total_return = bar_pnl.sum()
    total_gross = bar_gross.sum()
    total_fees = bar_fees.sum()
    win_rate = n_wins / n_trades if n_trades > 0 else 0
    avg_trade_ret = np.mean(trade_returns) if trade_returns else 0
    signal_rate = (signals != 0).mean()

    # Buy-and-hold comparison
    bnh_return = close_prices[-1] / close_prices[0] - 1.0 if close_prices[0] > 0 else 0

    elapsed = time.time() - t0

    long_trades = [r for r, t in zip(trade_returns, active_trades) if t["direction"] > 0]
    short_trades = [r for r, t in zip(trade_returns, active_trades) if t["direction"] < 0]
    stop_trades = sum(1 for t in active_trades if t.get("exit_reason") == "stop")
    reversal_trades = sum(1 for t in active_trades if t.get("exit_reason") == "reversal")

    print(f"    Trades: {n_trades} ({len(long_trades)}L/{len(short_trades)}S), "
          f"Stops: {stop_trades}, Reversals: {reversal_trades}")
    print(f"    Net return: {total_return*100:+.2f}%, "
          f"Gross: {total_gross*100:+.2f}%, Fees: {total_fees*100:.2f}%")
    print(f"    Win rate: {win_rate:.1%}, Avg trade: {avg_trade_ret*100:+.3f}%")
    print(f"    Avg size: {sizes[signals != 0].mean():.2f}" if n_signals > 0 else "    Avg size: N/A")
    print(f"    Buy&Hold: {bnh_return*100:+.2f}%")
    print(f"    [{elapsed:.1f}s]")

    return {
        "period": period_idx + 1,
        "trade_start": str(df_trade.index[0]),
        "trade_end": str(df_trade.index[-1]),
        "n_trades": n_trades,
        "n_long": len(long_trades),
        "n_short": len(short_trades),
        "n_stopped": stop_trades,
        "n_reversals": reversal_trades,
        "net_return": total_return,
        "gross_return": total_gross,
        "total_fees": total_fees,
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_ret,
        "signal_rate": signal_rate,
        "bnh_return": bnh_return,
        "bar_pnl": bar_pnl,
        "bar_position": bar_position,
        "trade_returns": trade_returns,
        "close_prices": close_prices,
        "trade_dates": df_trade.index,
        "direction_thresh": best_thresh,
    }


# ============================================================
# VISUALIZATION
# ============================================================
def plot_results(all_results, symbol, tf):
    """Generate equity curve and summary plots."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    fig.suptitle(f"Precision3 Strategy — {symbol} {tf}\n"
                 f"5 models (direction + vol sizing + stop + 2 filters), "
                 f"hold {HOLD_BARS}, early exit",
                 fontsize=13, fontweight="bold")

    # 1. Equity curve
    ax = axes[0]
    cumulative_pnl = []
    cumulative_bnh = []
    dates = []
    running_pnl = 0
    running_bnh = 0

    for r in all_results:
        for i, pnl in enumerate(r["bar_pnl"]):
            running_pnl += pnl
            cumulative_pnl.append(running_pnl)
            dates.append(r["trade_dates"][i])
        running_bnh += r["bnh_return"]
        cumulative_bnh.append(running_bnh)

    ax.plot(dates, [x * 100 for x in cumulative_pnl], "b-", linewidth=1.5,
            label="Precision3")
    bnh_dates = [r["trade_dates"][-1] for r in all_results]
    ax.plot(bnh_dates, [x * 100 for x in cumulative_bnh], "r--", linewidth=1,
            label="Buy & Hold", alpha=0.7)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Equity Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Per-period returns
    ax = axes[1]
    periods = [r["period"] for r in all_results]
    net_rets = [r["net_return"] * 100 for r in all_results]
    bnh_rets = [r["bnh_return"] * 100 for r in all_results]
    x = np.arange(len(periods))
    width = 0.35
    ax.bar(x - width/2, net_rets, width, label="Precision3", color="steelblue")
    ax.bar(x + width/2, bnh_rets, width, label="Buy & Hold", color="salmon", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}" for p in periods])
    ax.set_ylabel("Return (%)")
    ax.set_title("Per-Period Returns")
    ax.legend()
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Trade distribution
    ax = axes[2]
    all_trade_rets = []
    for r in all_results:
        all_trade_rets.extend([x * 100 for x in r["trade_returns"]])
    if all_trade_rets:
        ax.hist(all_trade_rets, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(np.mean(all_trade_rets), color="red", linestyle="--",
                   label=f"Mean: {np.mean(all_trade_rets):.3f}%")
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Trade Return (%)")
        ax.set_ylabel("Count")
        ax.set_title(f"Trade Return Distribution (n={len(all_trade_rets)})")
        ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Position sizing over time
    ax = axes[3]
    all_sizes = []
    size_dates = []
    for r in all_results:
        for i in range(len(r["bar_position"])):
            all_sizes.append(abs(r["bar_position"][i]))
            size_dates.append(r["trade_dates"][i])
    if all_sizes:
        ax.plot(size_dates, all_sizes, "g-", alpha=0.5, linewidth=0.5)
        ax.set_ylabel("Position Size (fraction)")
        ax.set_title("Dynamic Position Sizing")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_DIR / f"precision3_{symbol}_{tf}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", type=int, default=12)
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL)
    parser.add_argument("--symbol", type=str, default="SOLUSDT")
    args = parser.parse_args()

    symbol = args.symbol
    t_start = time.time()

    print("=" * 80)
    print("  PRECISION3 STRATEGY — Walk-Forward Backtest")
    print(f"  Symbol: {symbol} {TF}")
    print(f"  Models: 5 (direction + vol_sizing + stop_loss + consol_filter + crash_filter)")
    print(f"  WFO: {SELECTION_DAYS}d train, {PURGE_DAYS}d purge, {TRADE_DAYS}d trade")
    print(f"  Periods: {args.periods}, Fees: {FEE_BPS} bps round-trip, Hold: {HOLD_BARS} bars")
    print(f"  Sizing: inverse predicted vol | Stops: predicted drawdown * 1.5x")
    print("=" * 80)

    # Load data
    print("\n  Loading data...", flush=True)
    df = load_features(FEATURES_DIR, symbol, TF)
    print(f"  Loaded {len(df)} candles, range: {df.index[0]} -> {df.index[-1]}")

    # Load constraints
    with open(CONSTRAINTS_PATH) as f:
        constraints = json.load(f)

    cpd = get_candles_per_day(TF)
    sel_candles = int(SELECTION_DAYS * cpd)
    purge_candles = int(PURGE_DAYS * cpd)
    trade_candles = int(TRADE_DAYS * cpd)

    total_needed = sel_candles + purge_candles + trade_candles
    n_candles = len(df)

    if n_candles < total_needed:
        print(f"  ERROR: Need {total_needed} candles, have {n_candles}")
        sys.exit(1)

    max_periods = (n_candles - sel_candles - purge_candles) // trade_candles
    n_periods = min(args.periods, max_periods)

    print(f"  Max possible periods: {max_periods}, running: {n_periods}")
    print(f"  Candles/day: {cpd}, Selection: {sel_candles}, "
          f"Purge: {purge_candles}, Trade: {trade_candles}")

    # Verify required targets exist
    required = [cfg["target"] for cfg in MODELS.values()]
    missing = [t for t in required if t not in df.columns]
    if missing:
        print(f"  WARNING: Missing targets: {missing}")
        print(f"  Will skip models for missing targets")

    # ---- Run periods ----
    all_results = []
    for p in range(n_periods):
        sel_start = p * trade_candles
        sel_end = sel_start + sel_candles
        trade_start = sel_end + purge_candles
        trade_end = trade_start + trade_candles

        if trade_end > n_candles:
            print(f"\n  Period {p+1}: not enough data, stopping")
            break

        result = run_period(df, p, sel_start, sel_end, trade_start, trade_end,
                           constraints, cpd)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("\n  ERROR: No periods completed successfully")
        sys.exit(1)

    # ---- Aggregate Results ----
    print(f"\n\n{'='*80}")
    print(f"  PRECISION3 — AGGREGATE RESULTS ({len(all_results)} periods)")
    print(f"{'='*80}")

    total_net = sum(r["net_return"] for r in all_results)
    total_gross = sum(r["gross_return"] for r in all_results)
    total_fees = sum(r["total_fees"] for r in all_results)
    total_bnh = sum(r["bnh_return"] for r in all_results)
    total_trades = sum(r["n_trades"] for r in all_results)
    total_longs = sum(r["n_long"] for r in all_results)
    total_shorts = sum(r["n_short"] for r in all_results)
    total_stopped = sum(r["n_stopped"] for r in all_results)
    total_reversals = sum(r["n_reversals"] for r in all_results)

    all_trade_rets = []
    for r in all_results:
        all_trade_rets.extend(r["trade_returns"])

    avg_win_rate = np.mean([r["win_rate"] for r in all_results])
    avg_signal_rate = np.mean([r["signal_rate"] for r in all_results])

    # Sharpe ratio (annualized, trade-level)
    all_bar_pnl = np.concatenate([r["bar_pnl"] for r in all_results])
    if len(all_trade_rets) > 1 and np.std(all_trade_rets) > 0:
        trade_years = len(all_results) * TRADE_DAYS / 365.0
        trades_per_year = total_trades / trade_years if trade_years > 0 else total_trades
        sharpe = (np.mean(all_trade_rets) / np.std(all_trade_rets)
                  * np.sqrt(trades_per_year))
    else:
        sharpe = 0

    # Max drawdown
    cum_pnl = np.cumsum(all_bar_pnl)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max
    max_dd = drawdown.min()

    # Profit factor
    wins = sum(r for r in all_trade_rets if r > 0)
    losses = abs(sum(r for r in all_trade_rets if r < 0))
    profit_factor = wins / losses if losses > 0 else float("inf")

    # Periods positive
    n_positive = sum(1 for r in all_results if r["net_return"] > 0)

    # Compounding equity
    equity = INITIAL_CAPITAL
    for r in all_trade_rets:
        equity *= (1.0 + r)
    compound_return = equity / INITIAL_CAPITAL - 1.0

    print(f"\n  Total net return:    {total_net*100:+.2f}% (simple)")
    print(f"  Compounding return:  {compound_return*100:+.2f}%")
    print(f"  Total gross return:  {total_gross*100:+.2f}%")
    print(f"  Total fees paid:     {total_fees*100:.2f}%")
    print(f"  Buy & Hold return:   {total_bnh*100:+.2f}%")
    print(f"  Excess vs B&H:       {(total_net - total_bnh)*100:+.2f}%")
    print(f"\n  Total trades:        {total_trades} ({total_longs}L / {total_shorts}S)")
    print(f"  Stopped out:         {total_stopped}")
    print(f"  Early reversals:     {total_reversals}")
    print(f"  Avg trade return:    {np.mean(all_trade_rets)*100:+.3f}%")
    print(f"  Median trade return: {np.median(all_trade_rets)*100:+.3f}%")
    print(f"  Win rate:            {avg_win_rate:.1%}")
    print(f"  Profit factor:       {profit_factor:.2f}")
    print(f"  Signal rate:         {avg_signal_rate:.1%}")
    print(f"\n  Sharpe ratio (ann):  {sharpe:.2f}")
    print(f"  Max drawdown:        {max_dd*100:.2f}%")
    print(f"  Periods positive:    {n_positive}/{len(all_results)}")

    # Per-period table
    print(f"\n  {'Period':<8s} {'Dates':<45s} {'Net%':>7s} {'B&H%':>7s} "
          f"{'Trades':>7s} {'WinR':>6s} {'Stops':>6s} {'Thresh':>8s}")
    print(f"  {'-'*105}")
    for r in all_results:
        print(f"  P{r['period']:<6d} {r['trade_start'][:10]} -> {r['trade_end'][:10]}"
              f"    {r['net_return']*100:>+6.2f}% {r['bnh_return']*100:>+6.2f}% "
              f"{r['n_trades']:>6d}  {r['win_rate']:>5.1%} {r['n_stopped']:>5d} "
              f"{r['direction_thresh']*10000:>6.1f}bp")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "strategy": "Precision3",
        "symbol": symbol, "tf": TF,
        "n_periods": len(all_results),
        "total_net_return": total_net,
        "compound_return": compound_return,
        "total_gross_return": total_gross,
        "total_fees": total_fees,
        "bnh_return": total_bnh,
        "total_trades": total_trades,
        "total_longs": total_longs,
        "total_shorts": total_shorts,
        "total_stopped": total_stopped,
        "total_reversals": total_reversals,
        "avg_win_rate": avg_win_rate,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
        "periods_positive": n_positive,
    }
    with open(RESULTS_DIR / f"precision3_{symbol}_{TF}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save per-period CSV
    period_df = pd.DataFrame([{
        "period": r["period"],
        "trade_start": r["trade_start"],
        "trade_end": r["trade_end"],
        "net_return": r["net_return"],
        "gross_return": r["gross_return"],
        "fees": r["total_fees"],
        "bnh_return": r["bnh_return"],
        "n_trades": r["n_trades"],
        "n_long": r["n_long"],
        "n_short": r["n_short"],
        "n_stopped": r["n_stopped"],
        "n_reversals": r["n_reversals"],
        "win_rate": r["win_rate"],
        "signal_rate": r["signal_rate"],
        "direction_thresh": r["direction_thresh"],
    } for r in all_results])
    period_df.to_csv(RESULTS_DIR / f"precision3_{symbol}_{TF}_periods.csv",
                     index=False)

    # Plot
    plot_results(all_results, symbol, TF)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
