#!/usr/bin/env python3
"""
Strategy 4: Multi-Signal Ensemble

Combines 13 non-redundant targets across 3 roles:
  - Direction (5): breakout_up/down_3, alpha_3/5, relative_ret_1
  - Gate (5): consolidation_3, vol_expansion_10, tail_event_3, crash_10, liq_cascade_3
  - Sizing (3): max_drawup_long_3, max_drawdown_long_3, ret_magnitude_1

Architecture:
  Layer 1: 13 base models (LightGBM, top-30 auto-selected features per target)
  Layer 2: Three specialized heads
    - Direction head (LightGBM meta): long vs short from direction model outputs
    - Gate head (logistic): trade vs no-trade from gate model outputs
    - Sizing head (ridge): position size from sizing model outputs
  Layer 3: Trade execution with dynamic stops

12-period WFO, 3-day purge gap, proper train/val/trade splits.

Usage:
  python microstructure_research/strategy4_multisignal.py
  python microstructure_research/strategy4_multisignal.py --symbol XRPUSDT
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
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
RESULTS_DIR = Path("./microstructure_research/results")

# WFO params
SELECTION_DAYS = 360       # 1 year training window
PURGE_DAYS = 3             # gap between train and trade
TRADE_DAYS = 30            # 1 month trade window
INNER_TRAIN_FRAC = 0.80    # 80/20 split for meta-model training

# Execution
FEE_BPS = 4.0
FEE_FRAC = FEE_BPS / 10000.0
INITIAL_CAPITAL = 10000.0
HOLD_BARS = 3
EARLY_EXIT = True
MAX_POSITION_FRAC = 1.0

# Feature selection
TOP_FEATURES = 30

# ============================================================
# TARGET DEFINITIONS — 13 non-redundant targets
# ============================================================
DIRECTION_TARGETS = {
    "breakout_up_3":    {"type": "binary"},
    "breakout_down_3":  {"type": "binary"},
    "alpha_3":          {"type": "continuous"},
    "alpha_5":          {"type": "continuous"},
    "relative_ret_1":   {"type": "continuous"},
}

GATE_TARGETS = {
    "consolidation_3":       {"type": "binary"},
    "vol_expansion_10":      {"type": "binary"},
    "tail_event_3":          {"type": "binary"},
    "crash_10":              {"type": "binary"},
    "liquidation_cascade_3": {"type": "binary"},
}

SIZING_TARGETS = {
    "max_drawup_long_3":    {"type": "continuous"},
    "max_drawdown_long_3":  {"type": "continuous"},
    "ret_magnitude_1":      {"type": "continuous"},
}

ALL_TARGETS = {**DIRECTION_TARGETS, **GATE_TARGETS, **SIZING_TARGETS}

# Meta targets for direction head
META_TARGET_LONG = "tgt_profitable_long_3"
META_TARGET_SHORT = "tgt_profitable_short_3"


# ============================================================
# DATA LOADING
# ============================================================
def load_features(symbol, tf):
    tf_dir = FEATURES_DIR / symbol / tf
    files = sorted(tf_dir.glob("*.parquet"))
    if not files:
        print(f"ERROR: No parquet files in {tf_dir}")
        sys.exit(1)
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def get_candles_per_day(tf):
    return {"5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}[tf]


# ============================================================
# FEATURE SELECTION (per-target, training data only)
# ============================================================
def select_features(df, target_col, top_n=TOP_FEATURES):
    """Auto-select top N features by |Spearman correlation| with target."""
    feat_cols = [c for c in df.columns
                 if not c.startswith("tgt_")
                 and c not in ("open", "high", "low", "close", "volume")]

    y = df[target_col].values
    valid_y = np.isfinite(y)
    corrs = []
    for f in feat_cols:
        x = df[f].values
        mask = valid_y & np.isfinite(x)
        if mask.sum() < 100:
            corrs.append(0.0)
            continue
        try:
            c, _ = spearmanr(x[mask], y[mask])
            corrs.append(abs(c) if np.isfinite(c) else 0.0)
        except:
            corrs.append(0.0)
    idx = np.argsort(corrs)[::-1][:top_n]
    return [feat_cols[i] for i in idx if corrs[feat_cols.index(feat_cols[i])] > 0.01]


# ============================================================
# BASE MODEL TRAINING
# ============================================================
def train_base_model(X_train, y_train, X_pred, is_binary):
    """Train LightGBM and return predictions."""
    if is_binary:
        y_train = y_train.astype(int)
        if len(np.unique(y_train)) < 2:
            return None
        model = lgb.LGBMClassifier(
            objective="binary", metric="auc", verbosity=-1,
            n_estimators=200, max_depth=5, learning_rate=0.05,
            num_leaves=20, min_child_samples=30,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42,
        )
        model.fit(X_train, y_train)
        return model.predict_proba(X_pred)[:, 1]
    else:
        model = lgb.LGBMRegressor(
            objective="regression", metric="rmse", verbosity=-1,
            n_estimators=200, max_depth=5, learning_rate=0.05,
            num_leaves=20, min_child_samples=30,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42,
        )
        model.fit(X_train, y_train)
        return model.predict(X_pred)


def train_all_base_models(df_train, df_pred, target_dict):
    """Train base models for a set of targets, return predictions dict."""
    preds = {}
    for tgt_name, cfg in target_dict.items():
        full_tgt = f"tgt_{tgt_name}"
        if full_tgt not in df_train.columns:
            continue

        is_binary = cfg["type"] == "binary"

        # Select features on training data
        selected = select_features(df_train, full_tgt)
        if len(selected) < 5:
            continue

        # Prepare data
        X_tr = df_train[selected].values
        y_tr = df_train[full_tgt].values
        X_pr = df_pred[selected].values

        # Clean NaNs
        tr_valid = np.all(np.isfinite(X_tr), axis=1) & np.isfinite(y_tr)
        X_tr_c = X_tr[tr_valid]
        y_tr_c = y_tr[tr_valid]
        X_pr_c = np.nan_to_num(X_pr, nan=0, posinf=0, neginf=0)

        if len(X_tr_c) < 200:
            continue

        try:
            p = train_base_model(X_tr_c, y_tr_c, X_pr_c, is_binary)
            if p is not None:
                preds[tgt_name] = p
        except Exception as e:
            pass

    return preds


# ============================================================
# SINGLE PERIOD EXECUTION
# ============================================================
def run_period(df, period_idx, sel_start, sel_end, purge_end, trade_start, trade_end):
    t0 = time.time()

    df_sel = df.iloc[sel_start:sel_end].copy()
    df_trade = df.iloc[trade_start:trade_end].copy()

    if len(df_sel) < 500 or len(df_trade) < 10:
        print(f"  Period {period_idx+1}: SKIP (sel={len(df_sel)}, trade={len(df_trade)})")
        return None

    print(f"\n  Period {period_idx+1}:")
    print(f"    Selection: {len(df_sel)} bars [{df_sel.index[0]} -> {df_sel.index[-1]}]")
    print(f"    Purge:     {purge_end - sel_end} bars")
    print(f"    Trade:     {len(df_trade)} bars [{df_trade.index[0]} -> {df_trade.index[-1]}]")

    # ---- Step 1: Inner train/val split ----
    inner_split = int(len(df_sel) * INNER_TRAIN_FRAC)
    df_inner_train = df_sel.iloc[:inner_split].copy()
    df_inner_val = df_sel.iloc[inner_split:].copy()

    # ---- Step 2: Train all 13 base models on inner_train, predict on inner_val ----
    val_dir_preds = train_all_base_models(df_inner_train, df_inner_val, DIRECTION_TARGETS)
    val_gate_preds = train_all_base_models(df_inner_train, df_inner_val, GATE_TARGETS)
    val_size_preds = train_all_base_models(df_inner_train, df_inner_val, SIZING_TARGETS)

    n_dir = len(val_dir_preds)
    n_gate = len(val_gate_preds)
    n_size = len(val_size_preds)
    print(f"    Base models OK: {n_dir} direction + {n_gate} gate + {n_size} sizing")

    if n_dir < 2 or n_gate < 2:
        print(f"    SKIP: insufficient models")
        return None

    # ---- Step 3: Build meta-features for inner_val ----
    n_val = len(df_inner_val)

    # Direction meta-features
    dir_names = sorted(val_dir_preds.keys())
    dir_meta = np.zeros((n_val, len(dir_names)))
    for i, name in enumerate(dir_names):
        dir_meta[:, i] = val_dir_preds[name]

    # Gate meta-features
    gate_names = sorted(val_gate_preds.keys())
    gate_meta = np.zeros((n_val, len(gate_names)))
    for i, name in enumerate(gate_names):
        gate_meta[:, i] = val_gate_preds[name]

    # Sizing meta-features
    size_names = sorted(val_size_preds.keys())
    size_meta = np.zeros((n_val, len(size_names)))
    for i, name in enumerate(size_names):
        size_meta[:, i] = val_size_preds[name]

    # ---- Step 4: Train Direction Head ----
    # Meta target: profitable_long_3 and profitable_short_3
    meta_y_long = df_inner_val[META_TARGET_LONG].values if META_TARGET_LONG in df_inner_val.columns else None
    meta_y_short = df_inner_val[META_TARGET_SHORT].values if META_TARGET_SHORT in df_inner_val.columns else None

    if meta_y_long is None or meta_y_short is None:
        print(f"    SKIP: meta targets not available")
        return None

    # Combine all meta-features for direction head (direction + gate signals as context)
    all_meta = np.hstack([dir_meta, gate_meta])
    meta_valid = np.all(np.isfinite(all_meta), axis=1) & np.isfinite(meta_y_long) & np.isfinite(meta_y_short)

    X_meta = all_meta[meta_valid]
    y_long = meta_y_long[meta_valid].astype(int)
    y_short = meta_y_short[meta_valid].astype(int)

    if len(X_meta) < 50 or len(np.unique(y_long)) < 2 or len(np.unique(y_short)) < 2:
        print(f"    SKIP: insufficient meta data ({len(X_meta)} samples)")
        return None

    # Train direction meta-models
    def _train_meta_clf(X, y):
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        m = lgb.LGBMClassifier(
            objective="binary", metric="auc", verbosity=-1,
            n_estimators=100, max_depth=3, learning_rate=0.1,
            num_leaves=8, min_child_samples=30,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=1.0, random_state=42,
        )
        m.fit(Xs, y)
        return m, sc

    meta_long, sc_long = _train_meta_clf(X_meta, y_long)
    meta_short, sc_short = _train_meta_clf(X_meta, y_short)

    try:
        auc_l = roc_auc_score(y_long, meta_long.predict_proba(sc_long.transform(X_meta))[:, 1])
        auc_s = roc_auc_score(y_short, meta_short.predict_proba(sc_short.transform(X_meta))[:, 1])
        print(f"    Direction head AUC: long={auc_l:.3f}, short={auc_s:.3f}")
    except:
        pass

    # ---- Step 5: Train Gate Head ----
    # Gate target: should we trade? Use |cum_ret_3| > fee as proxy
    cum_ret_col = "tgt_cum_ret_3"
    if cum_ret_col in df_inner_val.columns:
        cum_ret = df_inner_val[cum_ret_col].values
        gate_y = (np.abs(cum_ret) > FEE_FRAC).astype(int)
        gate_valid = meta_valid & np.isfinite(cum_ret)
        X_gate = gate_meta[gate_valid]
        y_gate = gate_y[gate_valid]

        if len(X_gate) >= 50 and len(np.unique(y_gate)) >= 2:
            sc_gate = StandardScaler()
            X_gate_s = sc_gate.fit_transform(X_gate)
            gate_model = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
            gate_model.fit(X_gate_s, y_gate)
            gate_auc = roc_auc_score(y_gate, gate_model.predict_proba(X_gate_s)[:, 1])
            print(f"    Gate head AUC: {gate_auc:.3f}")
            has_gate = True
        else:
            has_gate = False
    else:
        has_gate = False

    # ---- Step 6: Calibrate threshold via 3-fold CV on inner_val ----
    conf_threshold = 0.50
    if cum_ret_col in df_inner_val.columns:
        cum_ret_3_val = cum_ret[meta_valid]
        n_cv = len(X_meta)
        fold_size = n_cv // 3
        cv_pred_long = np.full(n_cv, np.nan)
        cv_pred_short = np.full(n_cv, np.nan)

        for fold in range(3):
            vs = fold * fold_size
            ve = n_cv if fold == 2 else (fold + 1) * fold_size
            cv_mask = np.ones(n_cv, dtype=bool)
            cv_mask[vs:ve] = False
            try:
                ml, sl = _train_meta_clf(X_meta[cv_mask], y_long[cv_mask])
                ms, ss = _train_meta_clf(X_meta[cv_mask], y_short[cv_mask])
                cv_pred_long[vs:ve] = ml.predict_proba(sl.transform(X_meta[~cv_mask]))[:, 1]
                cv_pred_short[vs:ve] = ms.predict_proba(ss.transform(X_meta[~cv_mask]))[:, 1]
            except:
                pass

        best_thresh = 0.50
        best_avg = -999
        for thresh in [0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60]:
            rets = []
            for j in range(n_cv):
                pl, ps = cv_pred_long[j], cv_pred_short[j]
                if np.isnan(pl) or np.isnan(ps):
                    continue
                max_p = max(pl, ps)
                if max_p > thresh and np.isfinite(cum_ret_3_val[j]):
                    if pl > ps:
                        rets.append(cum_ret_3_val[j] - FEE_FRAC)
                    else:
                        rets.append(-cum_ret_3_val[j] - FEE_FRAC)
            if len(rets) >= 20:
                avg = np.mean(rets)
                if avg > best_avg:
                    best_avg = avg
                    best_thresh = thresh
        conf_threshold = best_thresh
        print(f"    Calibrated threshold: {conf_threshold:.2f} (OOS avg: {best_avg*100:+.3f}%)")

    # ---- Step 7: Retrain ALL base models on full selection, predict on trade ----
    trade_dir_preds = train_all_base_models(df_sel, df_trade, DIRECTION_TARGETS)
    trade_gate_preds = train_all_base_models(df_sel, df_trade, GATE_TARGETS)
    trade_size_preds = train_all_base_models(df_sel, df_trade, SIZING_TARGETS)

    n_trade = len(df_trade)

    # Build trade meta-features
    trade_dir_meta = np.zeros((n_trade, len(dir_names)))
    for i, name in enumerate(dir_names):
        if name in trade_dir_preds:
            trade_dir_meta[:, i] = trade_dir_preds[name]
        else:
            trade_dir_meta[:, i] = 0.5 if DIRECTION_TARGETS[name]["type"] == "binary" else 0.0

    trade_gate_meta = np.zeros((n_trade, len(gate_names)))
    for i, name in enumerate(gate_names):
        if name in trade_gate_preds:
            trade_gate_meta[:, i] = trade_gate_preds[name]
        else:
            trade_gate_meta[:, i] = 0.5

    trade_all_meta = np.hstack([trade_dir_meta, trade_gate_meta])

    # Direction predictions
    p_long = meta_long.predict_proba(sc_long.transform(
        np.nan_to_num(trade_all_meta, nan=0, posinf=0, neginf=0)))[:, 1]
    p_short = meta_short.predict_proba(sc_short.transform(
        np.nan_to_num(trade_all_meta, nan=0, posinf=0, neginf=0)))[:, 1]

    # Gate predictions
    if has_gate:
        p_gate = gate_model.predict_proba(sc_gate.transform(
            np.nan_to_num(trade_gate_meta, nan=0, posinf=0, neginf=0)))[:, 1]
    else:
        p_gate = np.ones(n_trade)

    # Sizing predictions (use raw predictions for stop/TP)
    pred_drawup = trade_size_preds.get("max_drawup_long_3")
    pred_drawdown = trade_size_preds.get("max_drawdown_long_3")
    pred_magnitude = trade_size_preds.get("ret_magnitude_1")

    # ---- Step 8: Generate signals ----
    close_prices = df_trade["close"].values
    open_prices = df_trade["open"].values if "open" in df_trade.columns else close_prices

    signals = np.zeros(n_trade)
    sizes = np.zeros(n_trade)
    n_gated_out = 0

    GATE_THRESHOLD = 0.50  # gate must predict >50% chance of meaningful move

    for bar in range(n_trade):
        pl = p_long[bar]
        ps = p_short[bar]
        max_p = max(pl, ps)

        # Confidence gate
        if max_p < conf_threshold:
            n_gated_out += 1
            continue

        # Gate filter: is this a good time to trade?
        if has_gate and p_gate[bar] < GATE_THRESHOLD:
            n_gated_out += 1
            continue

        # Direction
        if pl > ps:
            signals[bar] = 1.0
        else:
            signals[bar] = -1.0

        # Sizing: scale by confidence
        confidence = max_p - 0.5
        base_size = min(confidence * 2.0, MAX_POSITION_FRAC)

        # Adjust by predicted magnitude if available
        if pred_magnitude is not None:
            mag = pred_magnitude[bar]
            # Scale up when expecting big moves, down when expecting small
            if mag > 0:
                mag_factor = np.clip(mag / np.median(pred_magnitude), 0.5, 2.0)
                base_size *= mag_factor

        sizes[bar] = np.clip(base_size, 0.1, MAX_POSITION_FRAC)

    # ---- Step 9: Simulate trades ----
    bar_pnl = np.zeros(n_trade)
    n_trades = 0
    n_wins = 0
    trade_returns = []
    active_trades = []
    current_exit_bar = -1

    for bar in range(n_trade):
        # Early exit
        if EARLY_EXIT and bar <= current_exit_bar and len(active_trades) > 0:
            last_trade = active_trades[-1]
            if (signals[bar] != 0 and
                signals[bar] != last_trade["direction"] and
                bar > last_trade["entry_bar"]):
                early_exit_bar = bar + 1
                if early_exit_bar < n_trade:
                    last_trade["exit_bar"] = early_exit_bar
                    current_exit_bar = early_exit_bar

        # New entry
        if bar > current_exit_bar and signals[bar] != 0 and bar + HOLD_BARS + 1 < n_trade:
            entry_bar = bar + 1
            exit_bar = entry_bar + HOLD_BARS
            if exit_bar >= n_trade:
                continue
            current_exit_bar = exit_bar

            active_trades.append({
                "entry_bar": entry_bar,
                "exit_bar": exit_bar,
                "direction": signals[bar],
                "size": sizes[bar],
                "entry_price": open_prices[entry_bar],
            })

    # Calculate PnL
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
        net_ret = gross_ret - FEE_FRAC

        trade_returns.append(net_ret)
        n_trades += 1
        if net_ret > 0:
            n_wins += 1

        actual_hold = max(xb - eb + 1, 1)
        for b in range(eb, xb + 1):
            if b < n_trade:
                bar_pnl[b] += net_ret * sz / actual_hold

    total_return = bar_pnl.sum()
    win_rate = n_wins / n_trades if n_trades > 0 else 0
    avg_trade = np.mean(trade_returns) if trade_returns else 0
    bnh = close_prices[-1] / close_prices[0] - 1.0 if close_prices[0] > 0 else 0

    elapsed = time.time() - t0
    signal_count = (signals != 0).sum()

    print(f"    Signals: {int(signal_count)}, Gated out: {n_gated_out}, "
          f"Trades: {n_trades}, WR: {win_rate:.1%}")
    print(f"    Return: {total_return*100:+.2f}%, B&H: {bnh*100:+.2f}%, "
          f"Avg trade: {avg_trade*100:+.3f}% ({elapsed:.1f}s)")

    return {
        "period": period_idx + 1,
        "sel_start": str(df_sel.index[0].date()),
        "sel_end": str(df_sel.index[-1].date()),
        "trade_start": str(df_trade.index[0].date()),
        "trade_end": str(df_trade.index[-1].date()),
        "n_trades": n_trades,
        "n_signals": int(signal_count),
        "win_rate": win_rate,
        "total_return": total_return,
        "bnh_return": bnh,
        "avg_trade": avg_trade,
        "n_dir_models": len(trade_dir_preds),
        "n_gate_models": len(trade_gate_preds),
        "n_size_models": len(trade_size_preds),
        "threshold": conf_threshold,
        "bar_pnl": bar_pnl,
        "trade_returns": trade_returns,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--tf", default=TF)
    args = parser.parse_args()

    symbol = args.symbol
    tf = args.tf

    t0 = time.time()
    print("=" * 80)
    print("  STRATEGY 4: Multi-Signal Ensemble")
    print(f"  {symbol} {tf}, 12-period WFO, 13 targets (5 dir + 5 gate + 3 sizing)")
    print("=" * 80)

    # Load data
    print("\n  Loading data...", flush=True)
    df = load_features(symbol, tf)
    print(f"  Loaded {len(df)} candles, {df.index[0].date()} -> {df.index[-1].date()}")

    cpd = get_candles_per_day(tf)
    sel_candles = int(SELECTION_DAYS * cpd)
    purge_candles = int(PURGE_DAYS * cpd)
    trade_candles = int(TRADE_DAYS * cpd)

    # Check available targets
    available = [f"tgt_{t}" for t in ALL_TARGETS if f"tgt_{t}" in df.columns]
    print(f"  Available targets: {len(available)}/{len(ALL_TARGETS)}")

    # Generate WFO periods (working backwards from end)
    n = len(df)
    periods = []
    for p in range(20):  # max 20 periods
        trade_end = n - p * trade_candles
        trade_start = trade_end - trade_candles
        purge_end = trade_start
        sel_end = purge_end - purge_candles
        sel_start = sel_end - sel_candles

        if sel_start < 0 or trade_start < 0:
            break
        periods.append((sel_start, sel_end, purge_end, trade_start, trade_end))

    periods.reverse()
    print(f"  WFO periods: {len(periods)}")

    # Run all periods
    all_results = []
    all_bar_pnl = []

    for i, (ss, se, pe, ts, te) in enumerate(periods):
        result = run_period(df, i, ss, se, pe, ts, te)
        if result is not None:
            all_results.append(result)
            all_bar_pnl.extend(result["bar_pnl"].tolist())

    # ============================================================
    # SUMMARY
    # ============================================================
    elapsed_total = time.time() - t0

    print(f"\n{'=' * 80}")
    print(f"  RESULTS SUMMARY — {symbol} {tf}")
    print(f"{'=' * 80}")

    if not all_results:
        print("  No valid periods!")
        return

    n_periods = len(all_results)
    total_trades = sum(r["n_trades"] for r in all_results)
    all_trade_rets = []
    for r in all_results:
        all_trade_rets.extend(r["trade_returns"])

    # Compounded return
    equity = INITIAL_CAPITAL
    equity_curve = [equity]
    for r in all_results:
        for pnl in r["bar_pnl"]:
            equity *= (1 + pnl)
            equity_curve.append(equity)

    net_return = equity / INITIAL_CAPITAL - 1.0
    equity_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - peak) / peak
    max_dd = drawdown.min()

    # Per-period stats
    period_rets = [r["total_return"] for r in all_results]
    positive_periods = sum(1 for r in period_rets if r > 0)
    bnh_rets = [r["bnh_return"] for r in all_results]
    total_bnh = np.prod([1 + b for b in bnh_rets]) - 1.0

    # Sharpe (annualized from monthly periods)
    if len(period_rets) > 1 and np.std(period_rets) > 0:
        sharpe = np.mean(period_rets) / np.std(period_rets) * np.sqrt(12)
    else:
        sharpe = 0

    # Profit factor
    gross_wins = sum(r for r in all_trade_rets if r > 0)
    gross_losses = abs(sum(r for r in all_trade_rets if r < 0))
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    avg_wr = np.mean([r["win_rate"] for r in all_results])

    print(f"\n  Periods: {n_periods} ({positive_periods} positive)")
    print(f"  Trades:  {total_trades} (avg {total_trades/n_periods:.0f}/period)")
    print(f"  Win Rate: {avg_wr:.1%}")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Sharpe (ann.): {sharpe:.2f}")
    print(f"\n  Net Return:  {net_return*100:+.1f}%")
    print(f"  B&H Return:  {total_bnh*100:+.1f}%")
    print(f"  Max Drawdown: {max_dd*100:.1f}%")
    print(f"  Final Equity: ${equity:,.0f} (from ${INITIAL_CAPITAL:,.0f})")

    print(f"\n  Per-Period Returns:")
    for r in all_results:
        marker = "+" if r["total_return"] > 0 else " "
        print(f"    P{r['period']:>2}: {r['total_return']*100:>+6.2f}% "
              f"({r['n_trades']:>3} trades, WR {r['win_rate']:.0%}, "
              f"thresh={r['threshold']:.2f}) "
              f"B&H: {r['bnh_return']*100:>+6.2f}%")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_res = pd.DataFrame([{k: v for k, v in r.items()
                            if k not in ("bar_pnl", "trade_returns")}
                           for r in all_results])
    csv_path = RESULTS_DIR / f"strategy4_{symbol}_{tf}.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    print(f"  Total time: {elapsed_total:.0f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
