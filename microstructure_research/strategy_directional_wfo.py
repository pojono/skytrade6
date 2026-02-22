#!/usr/bin/env python3
"""
Strategy 1: Directional Momentum — Walk-Forward Backtest (Zero Lookahead)

Architecture:
  For each 30-day trade period:
    1. Train 6 base models on training window (360d):
       - P(breakout_up_3)     : LightGBM  + core+lags
       - P(breakout_down_3)   : LightGBM  + core+lags  (mirror)
       - P(profitable_long_1) : Logistic   + raw+lags
       - P(profitable_short_1): Logistic   + raw+lags   (mirror)
       - P(vol_expansion_5)   : Logistic   + raw
       - alpha_1              : Ridge      + all_core    (continuous)
    2. Inner train/val split (80/20) to generate OOS base predictions
    3. Train meta-model (Logistic) on val_inner base predictions
       → predicts P(profitable trade over next 3 bars)
    4. Retrain base models on full training window
    5. Generate base predictions on trade window
    6. Meta-model outputs trade signal + position size
    7. Track PnL bar-by-bar with realistic fees

Anti-Lookahead Measures:
  - Purge gap between training and trade windows
  - Feature lags computed only from available data
  - No threshold optimization on OOS data
  - Models retrained each period
  - Entry at next-bar open
  - Meta-model trained on inner OOS predictions only

Usage:
  python strategy_directional_wfo.py [--periods N] [--capital AMOUNT]
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb


# ============================================================
# PARAMETERS
# ============================================================
SYMBOL = "SOLUSDT"
TF = "4h"
FEATURES_DIR = Path("./features")
CONSTRAINTS_PATH = Path("./microstructure_research/predictable_targets.json")
RESULTS_DIR = Path("./microstructure_research/results")

SELECTION_DAYS = 360
PURGE_DAYS = 3
TRADE_DAYS = 30
INNER_TRAIN_FRAC = 0.80   # 80% train, 20% val for meta-model

FEE_BPS = 4.0             # round-trip fee (2 bps each way, maker)
FEE_FRAC = FEE_BPS / 10000.0
INITIAL_CAPITAL = 10000.0

# Base model configurations (from ML comparison results)
BASE_MODELS = {
    "breakout_up_3":     {"model": "lightgbm", "features": "core+lags"},
    "breakout_down_3":   {"model": "lightgbm", "features": "core+lags"},
    "profitable_long_1": {"model": "logistic",  "features": "raw+lags"},
    "profitable_short_1":{"model": "logistic",  "features": "raw+lags"},
    "vol_expansion_5":   {"model": "logistic",  "features": "raw"},
    "alpha_1":           {"model": "ridge",     "features": "all_core"},
}

# Meta-model target: profitable trade over 3 bars
META_TARGET_LONG = "tgt_profitable_long_3"
META_TARGET_SHORT = "tgt_profitable_short_3"

# Gate model: regime-based confidence filter
# These targets predict market CONDITIONS, not direction
GATE_MODELS = {
    "vol_expansion_5":  {"model": "logistic", "features": "raw"},
    "consolidation_3":  {"model": "logistic", "features": "raw+lags"},
    "tail_event_3":     {"model": "logistic", "features": "raw"},
}
# Gate target: |3-bar cumulative return| > median of training data
# This gives ~50% base rate — gate learns "above-median movement" vs quiet bars
# Threshold is computed per-period from training data (no lookahead)

# Position sizing
MAX_POSITION_FRAC = 1.0    # max fraction of capital per trade
HOLD_BARS = 3              # hold for 3 bars (12h on 4h TF)


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
def add_lag_features(df, feat_cols, lags=[1, 2, 3]):
    """Add lagged features. Only uses past data — no lookahead."""
    new_cols = {}
    for col in feat_cols:
        if col in df.columns:
            for lag in lags:
                new_cols[f"{col}_lag{lag}"] = df[col].shift(lag)
    lag_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, lag_df], axis=1)


def prepare_feature_sets(df, target_features, core_features):
    """Prepare different feature sets from a dataframe slice."""
    available_raw = [f for f in target_features if f in df.columns]
    available_core = [f for f in core_features if f in df.columns]

    # Add lags
    df_lag = add_lag_features(df, available_raw, lags=[1, 2, 3])
    lag_cols = [c for c in df_lag.columns
                if c not in df.columns and not c.startswith("tgt_")]

    # Core + lags
    df_core_lag = add_lag_features(df, available_core, lags=[1, 2, 3])
    core_lag_cols = [c for c in df_core_lag.columns
                     if c not in df.columns and not c.startswith("tgt_")]

    feature_sets = {
        "raw": (df, available_raw),
        "raw+lags": (df_lag, available_raw + lag_cols),
        "all_core": (df, available_core),
        "core+lags": (df_core_lag, available_core + core_lag_cols),
    }
    return feature_sets


# ============================================================
# BASE MODEL TRAINING
# ============================================================
def train_base_model(X_train, y_train, X_test, model_type, is_binary=True):
    """Train a single base model and return predictions on X_test."""
    if model_type == "lightgbm":
        if is_binary:
            model = lgb.LGBMClassifier(
                objective="binary", metric="auc", verbosity=-1,
                n_estimators=300, max_depth=6, learning_rate=0.05,
                num_leaves=31, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1, random_state=42,
            )
            model.fit(X_train, y_train)
            return model.predict_proba(X_test)[:, 1]
        else:
            model = lgb.LGBMRegressor(
                objective="regression", metric="rmse", verbosity=-1,
                n_estimators=300, max_depth=6, learning_rate=0.05,
                num_leaves=31, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1, random_state=42,
            )
            model.fit(X_train, y_train)
            return model.predict(X_test)

    elif model_type == "logistic":
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)
        model = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        model.fit(Xtr, y_train)
        return model.predict_proba(Xte)[:, 1]

    elif model_type == "ridge":
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)
        model = Ridge(alpha=1.0)
        model.fit(Xtr, y_train)
        return model.predict(Xte)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_Xy(df_feat, feat_cols, df_target, target_name):
    """Extract aligned X, y arrays with NaN handling."""
    actual_feats = [f for f in feat_cols if f in df_feat.columns]
    X = df_feat[actual_feats].values
    y = df_target[target_name].values

    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X_clean = np.nan_to_num(X[valid], nan=0, posinf=0, neginf=0)
    y_clean = y[valid]
    return X_clean, y_clean, valid, actual_feats


# ============================================================
# SINGLE PERIOD EXECUTION
# ============================================================
def run_period(df, period_idx, sel_start, sel_end, trade_start, trade_end,
               constraints, cpd):
    """
    Run one WFO period:
      1. Inner split for meta-model + gate-model training
      2. Train 6 base models + 3 gate models on inner_train, predict on inner_val
      3. Train meta-model (direction) + gate-model (confidence) on inner_val predictions
      4. Retrain all models on full selection window
      5. Predict on trade window
      6. Gate filters signals, meta provides direction + sizing
    """
    t0 = time.time()

    df_sel = df.iloc[sel_start:sel_end].copy()
    df_trade = df.iloc[trade_start:trade_end].copy()

    print(f"\n  Period {period_idx+1}:")
    print(f"    Selection: {len(df_sel)} bars [{df_sel.index[0]} -> {df_sel.index[-1]}]")
    print(f"    Trade:     {len(df_trade)} bars [{df_trade.index[0]} -> {df_trade.index[-1]}]")

    target_features_map = constraints["target_features"]
    core_features = constraints["core_features"]

    # ---- Step 1: Inner train/val split ----
    inner_split = int(len(df_sel) * INNER_TRAIN_FRAC)
    df_inner_train = df_sel.iloc[:inner_split].copy()
    df_inner_val = df_sel.iloc[inner_split:].copy()

    print(f"    Inner train: {len(df_inner_train)}, Inner val: {len(df_inner_val)}")

    # ---- Helper: train a set of models on inner split ----
    def train_model_set(model_configs, df_tr, df_val):
        """Train models from a config dict, return val predictions and valid masks."""
        preds_out = {}
        masks_out = {}
        for tgt_name, cfg in model_configs.items():
            full_tgt = f"tgt_{tgt_name}"
            model_type = cfg["model"]
            feat_set_name = cfg["features"]
            is_binary = tgt_name != "alpha_1"

            tgt_feats = target_features_map.get(full_tgt, core_features)
            fs_train = prepare_feature_sets(df_tr, tgt_feats, core_features)
            fs_val = prepare_feature_sets(df_val, tgt_feats, core_features)

            if feat_set_name not in fs_train:
                continue

            df_feat_tr, feat_cols_tr = fs_train[feat_set_name]
            df_feat_val, feat_cols_val = fs_val[feat_set_name]
            common_feats = [f for f in feat_cols_tr if f in feat_cols_val]
            if len(common_feats) < 3:
                continue

            X_tr, y_tr, _, _ = get_Xy(df_feat_tr, common_feats, df_tr, full_tgt)
            X_val, y_val, val_mask, _ = get_Xy(df_feat_val, common_feats, df_val, full_tgt)

            if is_binary:
                y_tr = y_tr.astype(int)
                if len(np.unique(y_tr)) < 2:
                    continue

            try:
                p = train_base_model(X_tr, y_tr, X_val, model_type, is_binary)
                preds_out[tgt_name] = p
                masks_out[tgt_name] = val_mask
            except Exception as e:
                print(f"    WARN: {tgt_name} inner train failed: {e}")
        return preds_out, masks_out

    # ---- Helper: retrain on full selection, predict on trade window ----
    def retrain_and_predict(model_configs, df_full, df_trd):
        """Retrain models on full selection window, predict on trade window."""
        preds_out = {}
        for tgt_name, cfg in model_configs.items():
            full_tgt = f"tgt_{tgt_name}"
            model_type = cfg["model"]
            feat_set_name = cfg["features"]
            is_binary = tgt_name != "alpha_1"

            tgt_feats = target_features_map.get(full_tgt, core_features)
            fs_full = prepare_feature_sets(df_full, tgt_feats, core_features)
            fs_trade = prepare_feature_sets(df_trd, tgt_feats, core_features)

            if feat_set_name not in fs_full:
                continue

            df_feat_full, feat_cols_full = fs_full[feat_set_name]
            df_feat_trade, feat_cols_trade = fs_trade[feat_set_name]
            common_feats = [f for f in feat_cols_full if f in feat_cols_trade]
            if len(common_feats) < 3:
                continue

            X_full, y_full, _, _ = get_Xy(df_feat_full, common_feats, df_full, full_tgt)
            X_trade_raw = df_feat_trade[common_feats].values
            X_trade_clean = np.nan_to_num(X_trade_raw, nan=0, posinf=0, neginf=0)

            if is_binary:
                y_full = y_full.astype(int)
                if len(np.unique(y_full)) < 2:
                    continue

            try:
                p = train_base_model(X_full, y_full, X_trade_clean, model_type, is_binary)
                preds_out[tgt_name] = p
            except Exception as e:
                print(f"    WARN: {tgt_name} full train failed: {e}")
        return preds_out

    # ---- Step 2: Train base + regime models on inner split ----
    all_models = {**BASE_MODELS, **GATE_MODELS}
    val_all_preds, val_all_masks = train_model_set(all_models, df_inner_train, df_inner_val)

    base_ok = sum(1 for k in BASE_MODELS if k in val_all_preds)
    gate_ok = sum(1 for k in GATE_MODELS if k in val_all_preds)
    if base_ok < 4:
        print(f"    SKIP: only {base_ok} base models succeeded")
        return None
    print(f"    Models OK: {base_ok} base + {gate_ok} regime")

    # ---- Step 3: Build ENRICHED meta-model (base + regime predictions) ----
    n_val = len(df_inner_val)
    n_meta_feats = len(BASE_MODELS) + len(GATE_MODELS)  # 6 + 3 = 9
    meta_features_val = np.zeros((n_val, n_meta_feats))
    meta_valid = np.ones(n_val, dtype=bool)

    all_model_names = list(BASE_MODELS.keys()) + list(GATE_MODELS.keys())
    for i, tgt_name in enumerate(all_model_names):
        if tgt_name in val_all_preds:
            mask = val_all_masks[tgt_name]
            preds = val_all_preds[tgt_name]
            col = np.full(n_val, np.nan)
            col[mask] = preds
            meta_features_val[:, i] = col
            meta_valid &= np.isfinite(col)
        else:
            # Fill with 0.5 (neutral) for missing regime models
            if tgt_name in GATE_MODELS:
                meta_features_val[:, i] = 0.5
            else:
                meta_valid[:] = False

    # Meta targets: profitable_long_3 and profitable_short_3
    meta_y_long = df_inner_val[META_TARGET_LONG].values if META_TARGET_LONG in df_inner_val.columns else None
    meta_y_short = df_inner_val[META_TARGET_SHORT].values if META_TARGET_SHORT in df_inner_val.columns else None

    if meta_y_long is None or meta_y_short is None:
        print(f"    SKIP: meta targets not available")
        return None

    meta_valid &= np.isfinite(meta_y_long) & np.isfinite(meta_y_short)

    X_meta = meta_features_val[meta_valid]
    y_meta_long = meta_y_long[meta_valid].astype(int)
    y_meta_short = meta_y_short[meta_valid].astype(int)

    if len(X_meta) < 50 or len(np.unique(y_meta_long)) < 2 or len(np.unique(y_meta_short)) < 2:
        print(f"    SKIP: insufficient meta training data ({len(X_meta)} samples)")
        return None

    # Train two meta-models (direction) — now with 9 features including regime
    scaler_meta = StandardScaler()
    X_meta_scaled = scaler_meta.fit_transform(X_meta)

    meta_long = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
    meta_long.fit(X_meta_scaled, y_meta_long)

    meta_short = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
    meta_short.fit(X_meta_scaled, y_meta_short)

    try:
        auc_long = roc_auc_score(y_meta_long, meta_long.predict_proba(X_meta_scaled)[:, 1])
        auc_short = roc_auc_score(y_meta_short, meta_short.predict_proba(X_meta_scaled)[:, 1])
        print(f"    Meta AUC (9-feat): long={auc_long:.3f}, short={auc_short:.3f}")
    except:
        pass

    # ---- Step 3b: Calibrate confidence threshold from inner_val ----
    # Find the threshold where meta-model predictions are actually profitable
    meta_pred_long_val = meta_long.predict_proba(X_meta_scaled)[:, 1]
    meta_pred_short_val = meta_short.predict_proba(X_meta_scaled)[:, 1]

    # For each bar, what would the trade return be?
    cum_ret_3_col = "tgt_cum_ret_3"
    conf_threshold = 0.5  # default
    if cum_ret_3_col in df_inner_val.columns:
        cum_ret_3_val = df_inner_val[cum_ret_3_col].values[meta_valid]
        # Test thresholds: find one that maximizes avg trade return
        best_thresh = 0.5
        best_avg_ret = -999
        for thresh in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]:
            rets = []
            for j in range(len(meta_pred_long_val)):
                pl = meta_pred_long_val[j]
                ps = meta_pred_short_val[j]
                max_p = max(pl, ps)
                if max_p > thresh and np.isfinite(cum_ret_3_val[j]):
                    if pl > ps:
                        rets.append(cum_ret_3_val[j] - FEE_FRAC)
                    else:
                        rets.append(-cum_ret_3_val[j] - FEE_FRAC)
            if len(rets) >= 20:
                avg = np.mean(rets)
                if avg > best_avg_ret:
                    best_avg_ret = avg
                    best_thresh = thresh
        conf_threshold = best_thresh
        print(f"    Calibrated confidence threshold: {conf_threshold:.2f} "
              f"(val avg ret: {best_avg_ret*100:+.3f}%)")

    # ---- Step 4: Retrain ALL models on full selection window ----
    trade_all_preds = retrain_and_predict(all_models, df_sel, df_trade)

    base_trade_ok = sum(1 for k in BASE_MODELS if k in trade_all_preds)
    if base_trade_ok < 4:
        print(f"    SKIP: only {base_trade_ok} base models for trade window")
        return None

    # ---- Step 5: Generate meta predictions on trade window ----
    n_trade = len(df_trade)

    meta_features_trade = np.zeros((n_trade, n_meta_feats))
    trade_meta_valid = np.ones(n_trade, dtype=bool)
    for i, tgt_name in enumerate(all_model_names):
        if tgt_name in trade_all_preds:
            meta_features_trade[:, i] = trade_all_preds[tgt_name]
        elif tgt_name in GATE_MODELS:
            meta_features_trade[:, i] = 0.5  # neutral fallback
        else:
            trade_meta_valid[:] = False

    meta_features_trade_scaled = scaler_meta.transform(
        np.nan_to_num(meta_features_trade, nan=0, posinf=0, neginf=0)
    )
    p_long = meta_long.predict_proba(meta_features_trade_scaled)[:, 1]
    p_short = meta_short.predict_proba(meta_features_trade_scaled)[:, 1]
    p_gate = np.maximum(p_long, p_short)  # confidence = max of the two

    # ---- Step 6: Generate signals with calibrated confidence threshold ----
    close_prices = df_trade["close"].values
    open_prices = df_trade["open"].values if "open" in df_trade.columns else close_prices

    signals = np.zeros(n_trade)
    sizes = np.zeros(n_trade)
    n_gated_out = 0

    for bar in range(n_trade):
        if not trade_meta_valid[bar]:
            continue

        pl = p_long[bar]
        ps = p_short[bar]
        max_p = max(pl, ps)

        # Confidence gate: only trade when meta-model is confident enough
        if max_p < conf_threshold:
            n_gated_out += 1
            continue

        # Direction from meta-model
        if pl > ps:
            signals[bar] = 1.0
            sizes[bar] = min((pl - 0.5) * 2.0, MAX_POSITION_FRAC)
        else:
            signals[bar] = -1.0
            sizes[bar] = min((ps - 0.5) * 2.0, MAX_POSITION_FRAC)

    # ---- Step 7: Simulate trades — ONE position at a time (no overlap) ----
    bar_pnl = np.zeros(n_trade)
    bar_gross = np.zeros(n_trade)
    bar_fees = np.zeros(n_trade)
    bar_position = np.zeros(n_trade)  # current position direction
    n_trades = 0
    n_wins = 0
    trade_returns = []

    active_trades = []
    current_exit_bar = -1  # tracks when current position expires

    for bar in range(n_trade):
        # Only enter if no active position and signal exists
        if bar > current_exit_bar and signals[bar] != 0 and bar + HOLD_BARS < n_trade:
            direction = signals[bar]
            size = sizes[bar]
            # Entry at next bar's open (realistic execution)
            entry_bar = bar + 1
            if entry_bar < n_trade:
                entry_price = open_prices[entry_bar]
                exit_bar = min(entry_bar + HOLD_BARS, n_trade - 1)
                current_exit_bar = exit_bar  # block new entries until this expires

                active_trades.append({
                    "entry_bar": entry_bar,
                    "exit_bar": exit_bar,
                    "direction": direction,
                    "size": size,
                    "entry_price": entry_price,
                })

    # Calculate PnL for each trade
    for trade in active_trades:
        eb = trade["entry_bar"]
        xb = trade["exit_bar"]
        d = trade["direction"]
        sz = trade["size"]
        ep = trade["entry_price"]

        if xb >= n_trade or ep <= 0:
            continue

        exit_price = close_prices[xb]
        gross_ret = d * (exit_price / ep - 1.0)
        fee = FEE_FRAC
        net_ret = gross_ret - fee

        trade_returns.append(net_ret)
        n_trades += 1
        if net_ret > 0:
            n_wins += 1

        # Distribute PnL across holding bars
        for b in range(eb, xb + 1):
            if b < n_trade:
                bar_pnl[b] += net_ret * sz / HOLD_BARS
                bar_gross[b] += gross_ret * sz / HOLD_BARS
                bar_fees[b] += fee * sz / HOLD_BARS
                bar_position[b] = d * sz

    # ---- Compute period metrics ----
    total_return = bar_pnl.sum()
    total_gross = bar_gross.sum()
    total_fees = bar_fees.sum()
    win_rate = n_wins / n_trades if n_trades > 0 else 0
    avg_trade_ret = np.mean(trade_returns) if trade_returns else 0
    signal_rate = (signals != 0).mean()

    # Buy-and-hold comparison
    if close_prices[0] > 0:
        bnh_return = close_prices[-1] / close_prices[0] - 1.0
    else:
        bnh_return = 0

    elapsed = time.time() - t0

    # Long/short breakdown
    long_trades = [r for r, t in zip(trade_returns, active_trades) if t["direction"] > 0]
    short_trades = [r for r, t in zip(trade_returns, active_trades) if t["direction"] < 0]

    gate_rate = 1.0 - n_gated_out / n_trade if n_trade > 0 else 1.0
    print(f"    Trades: {n_trades} ({len(long_trades)}L/{len(short_trades)}S), "
          f"Signal rate: {signal_rate:.1%}, Gate pass: {gate_rate:.1%}")
    print(f"    Net return: {total_return*100:+.2f}%, "
          f"Gross: {total_gross*100:+.2f}%, Fees: {total_fees*100:.2f}%")
    print(f"    Win rate: {win_rate:.1%}, Avg trade: {avg_trade_ret*100:+.3f}%")
    print(f"    Buy&Hold: {bnh_return*100:+.2f}%")
    print(f"    P(long): mean={p_long.mean():.3f}, P(short): mean={p_short.mean():.3f}, "
          f"P(gate): mean={p_gate.mean():.3f}")
    print(f"    [{elapsed:.1f}s]")

    return {
        "period": period_idx + 1,
        "trade_start": str(df_trade.index[0]),
        "trade_end": str(df_trade.index[-1]),
        "n_trades": n_trades,
        "n_long": len(long_trades),
        "n_short": len(short_trades),
        "net_return": total_return,
        "gross_return": total_gross,
        "total_fees": total_fees,
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_ret,
        "signal_rate": signal_rate,
        "gate_rate": gate_rate,
        "bnh_return": bnh_return,
        "p_long_mean": p_long.mean(),
        "p_short_mean": p_short.mean(),
        "p_gate_mean": p_gate.mean(),
        "bar_pnl": bar_pnl,
        "bar_position": bar_position,
        "trade_returns": trade_returns,
        "close_prices": close_prices,
        "trade_dates": df_trade.index,
    }


# ============================================================
# VISUALIZATION
# ============================================================
def plot_results(all_results, symbol, tf):
    """Generate equity curve and summary plots."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    fig.suptitle(f"Directional Momentum Strategy — {symbol} {tf}\n"
                 f"WFO: {len(all_results)} periods, {SELECTION_DAYS}d train, "
                 f"{PURGE_DAYS}d purge, {TRADE_DAYS}d trade",
                 fontsize=14, fontweight="bold")

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
            label="Strategy")
    # Plot BnH as step function at period boundaries
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
    ax.bar(x - width/2, net_rets, width, label="Strategy", color="steelblue")
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

    # 4. Signal rate and win rate per period
    ax = axes[3]
    signal_rates = [r["signal_rate"] * 100 for r in all_results]
    win_rates = [r["win_rate"] * 100 for r in all_results]
    ax.plot(periods, signal_rates, "b-o", label="Signal Rate %", markersize=6)
    ax.plot(periods, win_rates, "g-s", label="Win Rate %", markersize=6)
    ax.axhline(50, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Period")
    ax.set_ylabel("%")
    ax.set_title("Signal Rate & Win Rate per Period")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_DIR / f"strategy_directional_{symbol}_{tf}.png"
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
    args = parser.parse_args()

    t_start = time.time()

    print("=" * 80)
    print("  DIRECTIONAL MOMENTUM STRATEGY — Walk-Forward Backtest")
    print(f"  Symbol: {SYMBOL} {TF}")
    print(f"  WFO: {SELECTION_DAYS}d train, {PURGE_DAYS}d purge, {TRADE_DAYS}d trade")
    print(f"  Periods: {args.periods}, Fees: {FEE_BPS} bps round-trip")
    print(f"  Meta-model: Logistic on 6 base model predictions")
    print("=" * 80)

    # Load data
    print("\n  Loading data...", flush=True)
    df = load_features(FEATURES_DIR, SYMBOL, TF)
    print(f"  Loaded {len(df)} candles, range: {df.index[0]} -> {df.index[-1]}")

    # Load constraints
    with open(CONSTRAINTS_PATH) as f:
        constraints = json.load(f)

    cpd = get_candles_per_day(TF)
    sel_candles = int(SELECTION_DAYS * cpd)
    purge_candles = int(PURGE_DAYS * cpd)
    trade_candles = int(TRADE_DAYS * cpd)

    # Generate period boundaries (rolling forward)
    total_needed = sel_candles + purge_candles + trade_candles
    n_candles = len(df)

    if n_candles < total_needed:
        print(f"  ERROR: Need {total_needed} candles, have {n_candles}")
        sys.exit(1)

    # Start from the end of data, work backwards to find first period
    # Then roll forward
    max_periods = (n_candles - sel_candles - purge_candles) // trade_candles
    n_periods = min(args.periods, max_periods)

    # First trade window starts after first selection + purge
    first_trade_start = sel_candles + purge_candles

    print(f"  Max possible periods: {max_periods}, running: {n_periods}")
    print(f"  Candles per day: {cpd}, Selection: {sel_candles}, "
          f"Purge: {purge_candles}, Trade: {trade_candles}")

    # Verify targets exist
    required_targets = [f"tgt_{t}" for t in BASE_MODELS.keys()]
    required_targets += [META_TARGET_LONG, META_TARGET_SHORT]
    missing = [t for t in required_targets if t not in df.columns]
    if missing:
        print(f"  ERROR: Missing targets: {missing}")
        sys.exit(1)
    print(f"  All {len(required_targets)} required targets present")

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
    print(f"  AGGREGATE RESULTS ({len(all_results)} periods)")
    print(f"{'='*80}")

    total_net = sum(r["net_return"] for r in all_results)
    total_gross = sum(r["gross_return"] for r in all_results)
    total_fees = sum(r["total_fees"] for r in all_results)
    total_bnh = sum(r["bnh_return"] for r in all_results)
    total_trades = sum(r["n_trades"] for r in all_results)
    total_longs = sum(r["n_long"] for r in all_results)
    total_shorts = sum(r["n_short"] for r in all_results)

    all_trade_rets = []
    for r in all_results:
        all_trade_rets.extend(r["trade_returns"])

    avg_win_rate = np.mean([r["win_rate"] for r in all_results])
    avg_signal_rate = np.mean([r["signal_rate"] for r in all_results])
    avg_gate_rate = np.mean([r["gate_rate"] for r in all_results])

    # Sharpe ratio (annualized from per-bar PnL)
    all_bar_pnl = np.concatenate([r["bar_pnl"] for r in all_results])
    bars_per_year = cpd * 365
    if all_bar_pnl.std() > 0:
        sharpe = all_bar_pnl.mean() / all_bar_pnl.std() * np.sqrt(bars_per_year)
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

    print(f"\n  Total net return:    {total_net*100:+.2f}%")
    print(f"  Total gross return:  {total_gross*100:+.2f}%")
    print(f"  Total fees paid:     {total_fees*100:.2f}%")
    print(f"  Buy & Hold return:   {total_bnh*100:+.2f}%")
    print(f"  Excess vs B&H:      {(total_net - total_bnh)*100:+.2f}%")
    print(f"\n  Total trades:        {total_trades} ({total_longs}L / {total_shorts}S)")
    print(f"  Avg trade return:    {np.mean(all_trade_rets)*100:+.3f}%")
    print(f"  Median trade return: {np.median(all_trade_rets)*100:+.3f}%")
    print(f"  Win rate:            {avg_win_rate:.1%}")
    print(f"  Profit factor:       {profit_factor:.2f}")
    print(f"  Signal rate:         {avg_signal_rate:.1%}")
    print(f"  Gate pass rate:      {avg_gate_rate:.1%}")
    print(f"\n  Sharpe ratio (ann):  {sharpe:.2f}")
    print(f"  Max drawdown:        {max_dd*100:.2f}%")
    print(f"  Periods positive:    {n_positive}/{len(all_results)}")

    # Per-period table
    print(f"\n  {'Period':<8s} {'Dates':<45s} {'Net%':>7s} {'B&H%':>7s} "
          f"{'Trades':>7s} {'WinR':>6s} {'SigR':>6s} {'Gate':>6s}")
    print(f"  {'-'*100}")
    for r in all_results:
        print(f"  P{r['period']:<6d} {r['trade_start'][:10]} -> {r['trade_end'][:10]}"
              f"    {r['net_return']*100:>+6.2f}% {r['bnh_return']*100:>+6.2f}% "
              f"{r['n_trades']:>6d}  {r['win_rate']:>5.1%} {r['signal_rate']:>5.1%} "
              f"{r['gate_rate']:>5.1%}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "symbol": SYMBOL, "tf": TF,
        "n_periods": len(all_results),
        "total_net_return": total_net,
        "total_gross_return": total_gross,
        "total_fees": total_fees,
        "bnh_return": total_bnh,
        "total_trades": total_trades,
        "avg_win_rate": avg_win_rate,
        "avg_gate_rate": avg_gate_rate,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
    }
    with open(RESULTS_DIR / f"strategy_directional_{SYMBOL}_{TF}_summary.json", "w") as f:
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
        "win_rate": r["win_rate"],
        "signal_rate": r["signal_rate"],
        "gate_rate": r["gate_rate"],
    } for r in all_results])
    period_df.to_csv(RESULTS_DIR / f"strategy_directional_{SYMBOL}_{TF}_periods.csv",
                     index=False)

    # Plot
    plot_results(all_results, SYMBOL, TF)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
