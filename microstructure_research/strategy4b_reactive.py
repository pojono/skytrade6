#!/usr/bin/env python3
"""
Strategy 4b: Momentum/Breakout Reactive

Core idea: Don't predict direction. Predict WHEN a big move is coming,
then ride it once it starts.

Phase 1 — READINESS (ML): Predict regime using strong gate targets
  - consolidation_3: squeeze detection (score +0.35)
  - vol_expansion_10: vol regime shift (score +0.32)
  - tail_event_3: tail event warning (score +0.15)
  - crash_10: crash warning (score +0.19)
  - ret_magnitude_1: expected move size (score +0.20)

Phase 2 — TRIGGER (price action): Confirm direction
  - Breakout: price breaks above previous high or below previous low
  - Momentum: 2 consecutive bars in same direction
  - No ML — pure reactive

Phase 3 — EXECUTION (ML-assisted sizing):
  - Size by readiness confidence × predicted magnitude
  - Hold 3 bars with early exit on reversal
  - Dynamic stop from predicted drawdown

12-period WFO, 3-day purge gap.

Usage:
  python microstructure_research/strategy4b_reactive.py
  python microstructure_research/strategy4b_reactive.py --symbol XRPUSDT
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
SELECTION_DAYS = 360
PURGE_DAYS = 3
TRADE_DAYS = 30
INNER_TRAIN_FRAC = 0.80

# Execution
FEE_BPS = 4.0
FEE_FRAC = FEE_BPS / 10000.0
INITIAL_CAPITAL = 10000.0
HOLD_BARS = 3
EARLY_EXIT = True
MAX_POSITION_FRAC = 1.0

# Feature selection
TOP_FEATURES = 30

# Readiness threshold (calibrated per period)
DEFAULT_READINESS_THRESHOLD = 0.55

# ============================================================
# TARGET DEFINITIONS
# ============================================================
# Phase 1: Readiness targets (strong gate/vol targets)
READINESS_TARGETS = {
    "consolidation_3":       {"type": "binary"},
    "vol_expansion_10":      {"type": "binary"},
    "tail_event_3":          {"type": "binary"},
    "crash_10":              {"type": "binary"},
    "liquidation_cascade_3": {"type": "binary"},
}

# Phase 3: Sizing targets
SIZING_TARGETS = {
    "ret_magnitude_1":      {"type": "continuous"},
    "max_drawup_long_3":    {"type": "continuous"},
    "max_drawdown_long_3":  {"type": "continuous"},
}

ALL_TARGETS = {**READINESS_TARGETS, **SIZING_TARGETS}


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
# FEATURE SELECTION
# ============================================================
def select_features(df, target_col, top_n=TOP_FEATURES):
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
    preds = {}
    for tgt_name, cfg in target_dict.items():
        full_tgt = f"tgt_{tgt_name}"
        if full_tgt not in df_train.columns:
            continue
        is_binary = cfg["type"] == "binary"
        selected = select_features(df_train, full_tgt)
        if len(selected) < 5:
            continue
        X_tr = df_train[selected].values
        y_tr = df_train[full_tgt].values
        X_pr = df_pred[selected].values
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
        except:
            pass
    return preds


# ============================================================
# READINESS MODEL (meta-model over gate predictions)
# ============================================================
def build_readiness_model(gate_preds, df_val):
    """
    Train a meta-model that predicts: "will the next 3 bars have a move > fee?"
    Uses gate model outputs as features.
    """
    gate_names = sorted(gate_preds.keys())
    n_val = len(df_val)

    # Build meta-features from gate predictions
    X_meta = np.zeros((n_val, len(gate_names)))
    for i, name in enumerate(gate_names):
        X_meta[:, i] = gate_preds[name]

    # Meta target: |cum_ret_3| > fee (i.e., "a meaningful move happens")
    cum_ret_col = "tgt_cum_ret_3"
    if cum_ret_col not in df_val.columns:
        return None, None, None, gate_names

    cum_ret = df_val[cum_ret_col].values
    y_meta = (np.abs(cum_ret) > FEE_FRAC * 1.5).astype(int)  # 1.5x fee = meaningful

    valid = np.all(np.isfinite(X_meta), axis=1) & np.isfinite(cum_ret)
    X_v = X_meta[valid]
    y_v = y_meta[valid]

    if len(X_v) < 50 or len(np.unique(y_v)) < 2:
        return None, None, None, gate_names

    sc = StandardScaler()
    X_vs = sc.fit_transform(X_v)

    model = lgb.LGBMClassifier(
        objective="binary", metric="auc", verbosity=-1,
        n_estimators=100, max_depth=3, learning_rate=0.1,
        num_leaves=8, min_child_samples=30,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=1.0, random_state=42,
    )
    model.fit(X_vs, y_v)

    try:
        auc = roc_auc_score(y_v, model.predict_proba(X_vs)[:, 1])
    except:
        auc = 0.5

    return model, sc, auc, gate_names


# ============================================================
# PHASE 2: TRIGGER DETECTION (pure price action)
# ============================================================
def detect_triggers(df):
    """
    Detect momentum/breakout triggers from price action.
    Returns array of signals: +1 (bullish trigger), -1 (bearish trigger), 0 (no trigger).
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    opn = df["open"].values
    n = len(df)

    triggers = np.zeros(n)

    for bar in range(2, n):
        # Trigger 1: Breakout — close above previous high or below previous low
        breakout_up = close[bar] > high[bar - 1]
        breakout_down = close[bar] < low[bar - 1]

        # Trigger 2: Momentum — 2 consecutive bullish or bearish bars
        bar_bullish = close[bar] > opn[bar]
        bar_bearish = close[bar] < opn[bar]
        prev_bullish = close[bar-1] > opn[bar-1]
        prev_bearish = close[bar-1] < opn[bar-1]

        momentum_up = bar_bullish and prev_bullish
        momentum_down = bar_bearish and prev_bearish

        # Combine: breakout OR momentum
        if breakout_up or momentum_up:
            triggers[bar] = 1.0
        elif breakout_down or momentum_down:
            triggers[bar] = -1.0
        # If both up and down trigger (rare), use breakout direction
        if breakout_up and breakout_down:
            triggers[bar] = 1.0 if (close[bar] - opn[bar]) > 0 else -1.0

    return triggers


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
    print(f"    Trade:     {len(df_trade)} bars [{df_trade.index[0]} -> {df_trade.index[-1]}]")

    # ---- Inner train/val split ----
    inner_split = int(len(df_sel) * INNER_TRAIN_FRAC)
    df_inner_train = df_sel.iloc[:inner_split].copy()
    df_inner_val = df_sel.iloc[inner_split:].copy()

    # ---- Phase 1: Train readiness models ----
    val_gate_preds = train_all_base_models(df_inner_train, df_inner_val, READINESS_TARGETS)
    val_size_preds = train_all_base_models(df_inner_train, df_inner_val, SIZING_TARGETS)

    n_gate = len(val_gate_preds)
    n_size = len(val_size_preds)
    print(f"    Base models: {n_gate} readiness + {n_size} sizing")

    if n_gate < 2:
        print(f"    SKIP: insufficient readiness models")
        return None

    # Build readiness meta-model
    readiness_model, readiness_sc, readiness_auc, gate_names = \
        build_readiness_model(val_gate_preds, df_inner_val)

    if readiness_model is None:
        print(f"    SKIP: readiness model failed")
        return None

    print(f"    Readiness meta AUC: {readiness_auc:.3f}")

    # ---- Calibrate readiness threshold via 3-fold CV ----
    n_val = len(df_inner_val)
    gate_meta_val = np.zeros((n_val, len(gate_names)))
    for i, name in enumerate(gate_names):
        gate_meta_val[:, i] = val_gate_preds[name]

    cum_ret_col = "tgt_cum_ret_3"
    cum_ret_val = df_inner_val[cum_ret_col].values if cum_ret_col in df_inner_val.columns else None

    # Detect triggers on inner_val for calibration
    val_triggers = detect_triggers(df_inner_val)

    readiness_threshold = DEFAULT_READINESS_THRESHOLD

    if cum_ret_val is not None:
        valid_mask = np.all(np.isfinite(gate_meta_val), axis=1) & np.isfinite(cum_ret_val)

        # 3-fold CV for threshold calibration
        n_cv = valid_mask.sum()
        valid_indices = np.where(valid_mask)[0]
        fold_size = n_cv // 3

        cv_readiness = np.full(n_val, np.nan)
        for fold in range(3):
            vs = fold * fold_size
            ve = n_cv if fold == 2 else (fold + 1) * fold_size
            cv_mask = np.ones(n_cv, dtype=bool)
            cv_mask[vs:ve] = False

            train_idx = valid_indices[cv_mask]
            val_idx = valid_indices[~cv_mask]

            X_cv_tr = gate_meta_val[train_idx]
            y_cv_tr = (np.abs(cum_ret_val[train_idx]) > FEE_FRAC * 1.5).astype(int)

            if len(np.unique(y_cv_tr)) < 2:
                continue

            try:
                sc_cv = StandardScaler()
                X_cv_trs = sc_cv.fit_transform(X_cv_tr)
                m_cv = lgb.LGBMClassifier(
                    objective="binary", metric="auc", verbosity=-1,
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    num_leaves=8, min_child_samples=30,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=1.0, reg_lambda=1.0, random_state=42,
                )
                m_cv.fit(X_cv_trs, y_cv_tr)
                X_cv_val = gate_meta_val[val_idx]
                X_cv_vals = sc_cv.transform(X_cv_val)
                cv_readiness[val_idx] = m_cv.predict_proba(X_cv_vals)[:, 1]
            except:
                pass

        # Calibrate: find threshold that maximizes avg trade return
        # Only count bars where trigger fires AND readiness > threshold
        best_thresh = DEFAULT_READINESS_THRESHOLD
        best_avg = -999.0
        for thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            rets = []
            for bar in range(2, n_val - HOLD_BARS - 1):
                if np.isnan(cv_readiness[bar]):
                    continue
                if cv_readiness[bar] < thresh:
                    continue
                if val_triggers[bar] == 0:
                    continue
                # Simulate: enter at next bar, hold for HOLD_BARS
                entry_bar = bar + 1
                exit_bar = entry_bar + HOLD_BARS
                if exit_bar >= n_val:
                    continue

                entry_price = df_inner_val["open"].values[entry_bar]
                exit_price = df_inner_val["open"].values[exit_bar]
                if entry_price <= 0:
                    continue

                direction = val_triggers[bar]
                gross = direction * (exit_price / entry_price - 1.0)
                net = gross - FEE_FRAC
                rets.append(net)

            if len(rets) >= 15:
                avg = np.mean(rets)
                if avg > best_avg:
                    best_avg = avg
                    best_thresh = thresh

        readiness_threshold = best_thresh
        n_cal_trades = len(rets) if 'rets' in dir() else 0
        print(f"    Calibrated readiness threshold: {readiness_threshold:.2f} "
              f"(OOS avg: {best_avg*100:+.3f}%)")

    # ---- Retrain on full selection, predict on trade window ----
    trade_gate_preds = train_all_base_models(df_sel, df_trade, READINESS_TARGETS)
    trade_size_preds = train_all_base_models(df_sel, df_trade, SIZING_TARGETS)

    n_trade = len(df_trade)

    # Build readiness predictions for trade window
    trade_gate_meta = np.zeros((n_trade, len(gate_names)))
    for i, name in enumerate(gate_names):
        if name in trade_gate_preds:
            trade_gate_meta[:, i] = trade_gate_preds[name]
        else:
            trade_gate_meta[:, i] = 0.5

    trade_gate_clean = np.nan_to_num(trade_gate_meta, nan=0, posinf=0, neginf=0)
    p_readiness = readiness_model.predict_proba(
        readiness_sc.transform(trade_gate_clean))[:, 1]

    # Phase 2: Detect triggers on trade window
    trade_triggers = detect_triggers(df_trade)

    # Sizing predictions
    pred_magnitude = trade_size_preds.get("ret_magnitude_1")
    pred_drawup = trade_size_preds.get("max_drawup_long_3")
    pred_drawdown = trade_size_preds.get("max_drawdown_long_3")

    # ---- Generate signals: readiness × trigger ----
    close_prices = df_trade["close"].values
    open_prices = df_trade["open"].values if "open" in df_trade.columns else close_prices
    high_prices = df_trade["high"].values
    low_prices = df_trade["low"].values

    signals = np.zeros(n_trade)
    sizes = np.zeros(n_trade)
    n_ready_no_trigger = 0
    n_trigger_no_ready = 0
    n_both = 0

    for bar in range(n_trade):
        is_ready = p_readiness[bar] >= readiness_threshold
        has_trigger = trade_triggers[bar] != 0

        if is_ready and has_trigger:
            signals[bar] = trade_triggers[bar]  # direction from price action
            n_both += 1

            # Size by readiness confidence
            readiness_conf = (p_readiness[bar] - 0.5) * 2.0
            base_size = np.clip(readiness_conf, 0.2, MAX_POSITION_FRAC)

            # Scale by predicted magnitude if available
            if pred_magnitude is not None:
                mag = pred_magnitude[bar]
                if mag > 0 and np.isfinite(mag):
                    med_mag = np.median(pred_magnitude[pred_magnitude > 0])
                    if med_mag > 0:
                        mag_factor = np.clip(mag / med_mag, 0.5, 2.0)
                        base_size *= mag_factor

            sizes[bar] = np.clip(base_size, 0.1, MAX_POSITION_FRAC)
        elif is_ready and not has_trigger:
            n_ready_no_trigger += 1
        elif has_trigger and not is_ready:
            n_trigger_no_ready += 1

    # ---- Simulate trades ----
    bar_pnl = np.zeros(n_trade)
    n_trades = 0
    n_wins = 0
    trade_returns = []
    active_trades = []
    current_exit_bar = -1

    for bar in range(n_trade):
        # Early exit on reversal
        if EARLY_EXIT and bar <= current_exit_bar and len(active_trades) > 0:
            last_trade = active_trades[-1]
            if (trade_triggers[bar] != 0 and
                trade_triggers[bar] != last_trade["direction"] and
                bar > last_trade["entry_bar"] and
                p_readiness[bar] >= readiness_threshold):
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
    trigger_count = int((trade_triggers != 0).sum())
    signal_count = int((signals != 0).sum())

    print(f"    Triggers: {trigger_count}, Ready+Trigger: {n_both}, "
          f"Trades: {n_trades}, WR: {win_rate:.1%}")
    print(f"    Return: {total_return*100:+.2f}%, B&H: {bnh*100:+.2f}%, "
          f"Avg trade: {avg_trade*100:+.3f}% ({elapsed:.1f}s)")

    return {
        "period": period_idx + 1,
        "sel_start": str(df_sel.index[0].date()),
        "trade_start": str(df_trade.index[0].date()),
        "trade_end": str(df_trade.index[-1].date()),
        "n_trades": n_trades,
        "n_triggers": trigger_count,
        "n_signals": signal_count,
        "n_ready_no_trigger": n_ready_no_trigger,
        "n_trigger_no_ready": n_trigger_no_ready,
        "win_rate": win_rate,
        "total_return": total_return,
        "bnh_return": bnh,
        "avg_trade": avg_trade,
        "threshold": readiness_threshold,
        "readiness_auc": readiness_auc,
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
    print("  STRATEGY 4b: Momentum/Breakout Reactive")
    print(f"  {symbol} {tf}, 12-period WFO")
    print(f"  Phase 1: ML readiness (5 gate targets)")
    print(f"  Phase 2: Price trigger (breakout + momentum)")
    print(f"  Phase 3: ML sizing (3 targets)")
    print("=" * 80)

    # Load data
    print("\n  Loading data...", flush=True)
    df = load_features(symbol, tf)
    print(f"  Loaded {len(df)} candles, {df.index[0].date()} -> {df.index[-1].date()}")

    cpd = get_candles_per_day(tf)
    sel_candles = int(SELECTION_DAYS * cpd)
    purge_candles = int(PURGE_DAYS * cpd)
    trade_candles = int(TRADE_DAYS * cpd)

    available = [f"tgt_{t}" for t in ALL_TARGETS if f"tgt_{t}" in df.columns]
    print(f"  Available targets: {len(available)}/{len(ALL_TARGETS)}")

    # Generate WFO periods
    n = len(df)
    periods = []
    for p in range(20):
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

    for i, (ss, se, pe, ts, te) in enumerate(periods):
        result = run_period(df, i, ss, se, pe, ts, te)
        if result is not None:
            all_results.append(result)

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

    period_rets = [r["total_return"] for r in all_results]
    positive_periods = sum(1 for r in period_rets if r > 0)
    bnh_rets = [r["bnh_return"] for r in all_results]
    total_bnh = np.prod([1 + b for b in bnh_rets]) - 1.0

    if len(period_rets) > 1 and np.std(period_rets) > 0:
        sharpe = np.mean(period_rets) / np.std(period_rets) * np.sqrt(12)
    else:
        sharpe = 0

    gross_wins = sum(r for r in all_trade_rets if r > 0)
    gross_losses = abs(sum(r for r in all_trade_rets if r < 0))
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    avg_wr = np.mean([r["win_rate"] for r in all_results])

    # Signal analysis
    total_triggers = sum(r["n_triggers"] for r in all_results)
    total_signals = sum(r["n_signals"] for r in all_results)
    avg_readiness_auc = np.mean([r["readiness_auc"] for r in all_results])

    print(f"\n  Periods: {n_periods} ({positive_periods} positive)")
    print(f"  Trades:  {total_trades} (avg {total_trades/n_periods:.0f}/period)")
    print(f"  Triggers: {total_triggers}, Signals (ready+trigger): {total_signals}")
    print(f"  Win Rate: {avg_wr:.1%}")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Sharpe (ann.): {sharpe:.2f}")
    print(f"  Avg Readiness AUC: {avg_readiness_auc:.3f}")

    print(f"\n  Net Return:  {net_return*100:+.1f}%")
    print(f"  B&H Return:  {total_bnh*100:+.1f}%")
    print(f"  Max Drawdown: {max_dd*100:.1f}%")
    print(f"  Final Equity: ${equity:,.0f} (from ${INITIAL_CAPITAL:,.0f})")

    print(f"\n  Per-Period Returns:")
    for r in all_results:
        print(f"    P{r['period']:>2}: {r['total_return']*100:>+6.2f}% "
              f"({r['n_trades']:>3} trades, WR {r['win_rate']:.0%}, "
              f"thresh={r['threshold']:.2f}, AUC={r['readiness_auc']:.3f}) "
              f"B&H: {r['bnh_return']*100:>+6.2f}%")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_res = pd.DataFrame([{k: v for k, v in r.items()
                            if k not in ("bar_pnl", "trade_returns")}
                           for r in all_results])
    csv_path = RESULTS_DIR / f"strategy4b_{symbol}_{tf}.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    print(f"  Total time: {elapsed_total:.0f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
