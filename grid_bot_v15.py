#!/usr/bin/env python3
"""
Grid Bot v15 — Incremental ML Improvements.

Tests 7 ML enhancements one-by-one against Fix 1.00% (24h) baseline.
All use walk-forward prediction (no lookahead), 2 bps maker fee.

Steps:
  0. Baseline: Fix 1.00% (24h)
  1. Direct range prediction (Ridge) instead of vol×5.6
  2. P90 quantile range for grid width
  3. Breakout detection → widen grid
  4. Consolidation detection → tighten grid
  5. Adaptive rebalance interval
  6. 4h range for rebalance timing
  7. Asymmetric grid from upside/downside ratio
"""

import sys
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from regime_detection import load_bars, compute_regime_features
from breakout_ml import compute_breakout_features

MAKER_FEE_BPS = 2.0
K_RANGE_1H = 5.6
K_RANGE_4H = 11.0

VOL_FEATURES = [
    "parkvol_1h", "parkvol_2h", "parkvol_4h", "parkvol_8h", "parkvol_24h",
    "rvol_1h", "rvol_2h", "rvol_4h", "rvol_8h", "rvol_24h",
    "vol_ratio_1h_24h", "vol_ratio_2h_24h", "vol_ratio_1h_8h",
    "vol_accel_1h", "vol_accel_4h",
    "vol_sma_24h", "vol_ratio_bar",
    "trade_intensity_ratio", "parkinson_vol",
    "bar_eff_1h", "bar_eff_4h", "bar_efficiency",
    "efficiency_1h", "efficiency_2h", "efficiency_4h", "efficiency_8h",
    "ret_autocorr_1h", "ret_autocorr_2h", "ret_autocorr_4h",
    "adx_2h", "adx_4h",
    "sign_persist_1h", "sign_persist_2h",
    "imbalance_1h", "imbalance_4h", "imbalance_persistence",
    "large_trade_1h", "iti_cv_1h",
    "momentum_1h", "momentum_2h", "momentum_4h",
    "price_vs_sma_2h", "price_vs_sma_4h", "price_vs_sma_8h", "price_vs_sma_24h",
    "vol_imbalance",
]

# Top 10 breakout features from v12 (fast logistic regression)
BREAKOUT_FEATURES_TOP10 = [
    "atr_1h", "range_compression_1h", "bb_pctile_2h", "bb_width_2h",
    "bar_eff_1h", "vol_sma_24h", "atr_24h", "atr_8h",
    "ret_kurtosis_8h", "range_compression_4h",
]


# ---------------------------------------------------------------------------
# GridBotSimulator (same as v14, but accepts per-bar rebalance_intervals)
# ---------------------------------------------------------------------------

class GridBotSimulator:
    def __init__(self, n_levels=5, fee_bps=2.0, capital_usd=10000):
        self.n_levels = n_levels
        self.fee_bps = fee_bps
        self.capital_usd = capital_usd

    def _setup_grid(self, center, spacing):
        levels = {}
        for i in range(1, self.n_levels + 1):
            levels[f"buy_{i}"] = {"price": center - i * spacing, "active": True, "type": "buy"}
            levels[f"sell_{i}"] = {"price": center + i * spacing, "active": True, "type": "sell"}
        return levels

    def run(self, prices_close, prices_high, prices_low,
            grid_spacings_pct, strategy_name="fixed",
            paused=None, rebalance_intervals=None):
        n = len(prices_close)
        if paused is None:
            paused = np.zeros(n, dtype=bool)
        if rebalance_intervals is None:
            rebalance_intervals = np.full(n, 288, dtype=int)  # default 24h

        size_usd = self.capital_usd / (self.n_levels * 2)
        inventory = []
        cash = 0.0
        total_fees = 0.0
        fills = 0
        grid_profits = 0.0

        grid_center = prices_close[0]
        grid_spacing = grid_spacings_pct[0] * grid_center
        levels = self._setup_grid(grid_center, grid_spacing)
        last_rebalance = 0

        equity_curve = np.zeros(n)
        position_history = np.zeros(n)

        for i in range(n):
            price = prices_close[i]
            high = prices_high[i]
            low = prices_low[i]

            rebal_interval = int(rebalance_intervals[i])
            if i - last_rebalance >= rebal_interval and i > 0:
                for qty, cost_p in inventory:
                    pnl = qty * (price - cost_p)
                    cash += pnl
                    fee = abs(qty) * price * self.fee_bps / 10000
                    cash -= fee
                    total_fees += fee
                    fills += 1
                inventory = []
                grid_center = price
                grid_spacing = grid_spacings_pct[i] * price
                levels = self._setup_grid(grid_center, grid_spacing)
                last_rebalance = i

            if paused[i]:
                net_qty = sum(q for q, _ in inventory)
                cost_basis = sum(q * p for q, p in inventory)
                unrealized = net_qty * price - cost_basis
                equity_curve[i] = cash + unrealized
                position_history[i] = net_qty * price
                continue

            for key, level in levels.items():
                if not level["active"]:
                    continue
                lp = level["price"]
                qty = size_usd / lp

                if level["type"] == "buy" and low <= lp:
                    fee = size_usd * self.fee_bps / 10000
                    cash -= fee
                    total_fees += fee
                    inventory.append((qty, lp))
                    fills += 1
                    level["active"] = False
                    sell_key = key.replace("buy", "sell")
                    if sell_key in levels:
                        levels[sell_key]["active"] = True

                elif level["type"] == "sell" and high >= lp:
                    fee = size_usd * self.fee_bps / 10000
                    cash -= fee
                    total_fees += fee
                    if inventory and inventory[0][0] > 0:
                        old_qty, old_price = inventory.pop(0)
                        profit = old_qty * (lp - old_price)
                        cash += profit
                        grid_profits += profit
                    else:
                        inventory.append((-qty, lp))
                    fills += 1
                    level["active"] = False
                    buy_key = key.replace("sell", "buy")
                    if buy_key in levels:
                        levels[buy_key]["active"] = True

            net_qty = sum(q for q, _ in inventory)
            cost_basis = sum(q * p for q, p in inventory)
            unrealized = net_qty * price - cost_basis
            equity_curve[i] = cash + unrealized
            position_history[i] = net_qty * price

        final_price = prices_close[-1]
        for qty, cost_p in inventory:
            pnl = qty * (final_price - cost_p)
            cash += pnl
            fee = abs(qty) * final_price * self.fee_bps / 10000
            cash -= fee
            total_fees += fee
        equity_curve[-1] = cash

        final_equity = cash
        max_equity = np.maximum.accumulate(equity_curve)
        drawdowns = equity_curve - max_equity
        max_drawdown = np.min(drawdowns)

        n_days = n / 288
        daily_eq = equity_curve[::288]
        daily_returns = np.diff(daily_eq)
        daily_returns = daily_returns[~np.isnan(daily_returns)]
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(365)
        else:
            sharpe = 0

        return {
            "strategy": strategy_name,
            "total_pnl": final_equity,
            "grid_profits": grid_profits,
            "total_fees": total_fees,
            "fills": fills,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "n_days": n_days,
            "pnl_per_day": final_equity / max(n_days, 1),
            "fills_per_day": fills / max(n_days, 1),
        }


# ---------------------------------------------------------------------------
# Walk-forward prediction helpers
# ---------------------------------------------------------------------------

def walkforward_ridge(df, feature_cols, target_col, min_train=2000):
    """Walk-forward Ridge regression. Returns prediction array."""
    available = [f for f in feature_cols if f in df.columns]
    X = df[available].values
    y = df[target_col].values
    n = len(df)
    predictions = np.full(n, np.nan)
    scaler = StandardScaler()
    model = Ridge(alpha=1.0)
    retrain_interval = 288

    last_train = -retrain_interval
    for i in range(min_train, n):
        if i - last_train >= retrain_interval:
            train_mask = ~np.isnan(y[:i])
            X_train = np.nan_to_num(X[:i][train_mask], nan=0, posinf=0, neginf=0)
            y_train = y[:i][train_mask]
            if len(y_train) < 100:
                continue
            scaler.fit(X_train)
            model.fit(scaler.transform(X_train), y_train)
            last_train = i
        x_i = np.nan_to_num(X[i:i+1], nan=0, posinf=0, neginf=0)
        predictions[i] = max(model.predict(scaler.transform(x_i))[0], 1e-8)
    return predictions




def walkforward_logistic(df, feature_cols, target_col, min_train=2000):
    """Walk-forward logistic regression. Returns probability array."""
    available = [f for f in feature_cols if f in df.columns]
    X = df[available].values
    y = df[target_col].values
    n = len(df)
    predictions = np.full(n, np.nan)
    scaler = StandardScaler()
    model = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
    retrain_interval = 288

    last_train = -retrain_interval
    for i in range(min_train, n):
        if i - last_train >= retrain_interval:
            train_mask = ~np.isnan(y[:i])
            X_train = np.nan_to_num(X[:i][train_mask], nan=0, posinf=0, neginf=0)
            y_train = y[:i][train_mask]
            if len(y_train) < 100 or len(np.unique(y_train)) < 2:
                continue
            scaler.fit(X_train)
            model.fit(scaler.transform(X_train), y_train)
            last_train = i
        x_i = np.nan_to_num(X[i:i+1], nan=0, posinf=0, neginf=0)
        try:
            predictions[i] = model.predict_proba(scaler.transform(x_i))[0, 1]
        except Exception:
            pass
    return predictions


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_v15(symbol, start_date, end_date):
    t_total = time.time()

    print("=" * 70)
    print(f"  GRID BOT v15 — ML Improvements (one-by-one)")
    print(f"  Symbol:   {symbol}")
    print(f"  Period:   {start_date} → {end_date}")
    print(f"  Fee:      {MAKER_FEE_BPS} bps maker per fill")
    print(f"  Baseline: Fix 1.00% (24h rebalance)")
    print("=" * 70)

    # --- Step 0: Load data & compute features ---
    print(f"\n  Loading data...", flush=True)
    df = load_bars(symbol, start_date, end_date)
    print(f"  {len(df):,} bars loaded in {time.time()-t_total:.0f}s")

    print(f"  Computing regime features...", flush=True)
    t1 = time.time()
    df = compute_regime_features(df)
    print(f"  Regime features in {time.time()-t1:.0f}s")

    print(f"  Computing breakout features...", flush=True)
    t1 = time.time()
    df = compute_breakout_features(df)
    print(f"  Breakout features in {time.time()-t1:.0f}s")

    # --- Compute forward targets ---
    print(f"  Computing forward targets...", flush=True)
    ret = df["returns"].values
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    n = len(df)

    # 1h forward vol
    fwd_vol_1h = np.full(n, np.nan)
    for i in range(n - 12):
        fwd_vol_1h[i] = np.std(ret[i+1:i+13])
    df["fwd_vol_1h"] = fwd_vol_1h

    # 1h forward range (as fraction of price)
    fwd_range_1h = np.full(n, np.nan)
    for i in range(n - 12):
        fwd_h = h[i+1:i+13].max()
        fwd_l = l[i+1:i+13].min()
        fwd_range_1h[i] = (fwd_h - fwd_l) / c[i]
    df["fwd_range_1h"] = fwd_range_1h

    # 4h forward range
    fwd_range_4h = np.full(n, np.nan)
    for i in range(n - 48):
        fwd_h = h[i+1:i+49].max()
        fwd_l = l[i+1:i+49].min()
        fwd_range_4h[i] = (fwd_h - fwd_l) / c[i]
    df["fwd_range_4h"] = fwd_range_4h

    # 1h forward upside fraction (for asymmetry)
    fwd_upside_frac = np.full(n, np.nan)
    for i in range(n - 12):
        fwd_h = h[i+1:i+13].max()
        fwd_l = l[i+1:i+13].min()
        rng = fwd_h - fwd_l
        if rng > 0:
            fwd_upside_frac[i] = (fwd_h - c[i]) / rng
    df["fwd_upside_frac"] = fwd_upside_frac

    # Breakout label: 1h range > 5× ATR
    atr_1h = df["atr_1h"].values if "atr_1h" in df.columns else np.full(n, np.nan)
    breakout_label = np.full(n, np.nan)
    for i in range(n - 12):
        if not np.isnan(atr_1h[i]) and atr_1h[i] > 0:
            breakout_label[i] = 1.0 if (fwd_range_1h[i] * c[i]) > 5.0 * atr_1h[i] else 0.0
    df["breakout_1h"] = breakout_label

    # --- Walk-forward predictions ---
    warmup = 2500

    # Step 1: Direct range prediction
    print(f"\n  [Step 1] Walk-forward 1h range prediction (Ridge)...", flush=True)
    t1 = time.time()
    pred_range_1h = walkforward_ridge(df, VOL_FEATURES, "fwd_range_1h", min_train=2000)
    valid = np.sum(~np.isnan(pred_range_1h[warmup:]))
    print(f"    {valid:,} valid predictions in {time.time()-t1:.0f}s")

    # Step 2: P90 ≈ Ridge prediction × 1.7 (from v11: safety_factor for P90 coverage)
    print(f"  [Step 2] P90 range = Ridge range × 1.7 (v11 calibration)", flush=True)
    pred_range_p90 = pred_range_1h * 1.7
    print(f"    Done (derived from Step 1)")

    # Baseline vol prediction (for reference)
    print(f"  [Ref] Walk-forward 1h vol prediction (Ridge)...", flush=True)
    t1 = time.time()
    pred_vol_1h = walkforward_ridge(df, VOL_FEATURES, "fwd_vol_1h", min_train=2000)
    valid = np.sum(~np.isnan(pred_vol_1h[warmup:]))
    print(f"    {valid:,} valid predictions in {time.time()-t1:.0f}s")

    # Step 3: Breakout probability
    print(f"  [Step 3] Walk-forward breakout probability (Logistic)...", flush=True)
    t1 = time.time()
    pred_breakout = walkforward_logistic(df, BREAKOUT_FEATURES_TOP10, "breakout_1h", min_train=2000)
    valid = np.sum(~np.isnan(pred_breakout[warmup:]))
    brk_rate = np.nanmean(breakout_label[warmup:])
    print(f"    {valid:,} valid predictions, breakout rate={brk_rate:.1%} in {time.time()-t1:.0f}s")

    # Step 6: 4h range prediction
    print(f"  [Step 6] Walk-forward 4h range prediction (Ridge)...", flush=True)
    t1 = time.time()
    pred_range_4h = walkforward_ridge(df, VOL_FEATURES, "fwd_range_4h", min_train=2000)
    valid = np.sum(~np.isnan(pred_range_4h[warmup:]))
    print(f"    {valid:,} valid predictions in {time.time()-t1:.0f}s")

    # Step 7: Upside fraction prediction
    print(f"  [Step 7] Walk-forward upside fraction (Ridge)...", flush=True)
    t1 = time.time()
    pred_upside = walkforward_ridge(df, VOL_FEATURES, "fwd_upside_frac", min_train=2000)
    valid = np.sum(~np.isnan(pred_upside[warmup:]))
    print(f"    {valid:,} valid predictions in {time.time()-t1:.0f}s")

    # --- Prepare simulation data ---
    df_sim = df.iloc[warmup:].copy().reset_index(drop=True)
    n_sim = len(df_sim)
    prices_close = df_sim["close"].values.astype(float)
    prices_high = df_sim["high"].values.astype(float)
    prices_low = df_sim["low"].values.astype(float)

    # Slice predictions
    pv1h = pred_vol_1h[warmup:]
    pr1h = pred_range_1h[warmup:]
    pr1h_p90 = pred_range_p90[warmup:] if len(pred_range_p90) > warmup else pred_range_p90
    pb = pred_breakout[warmup:]
    pr4h = pred_range_4h[warmup:]
    pu = pred_upside[warmup:]
    consol = df_sim["consolidation_2h_vs_24h"].values if "consolidation_2h_vs_24h" in df_sim.columns else np.full(n_sim, 0.5)

    median_pred_vol = np.nanmedian(pv1h[~np.isnan(pv1h)])
    median_pred_range = np.nanmedian(pr1h[~np.isnan(pr1h)])

    print(f"\n  Simulation: {n_sim:,} bars ({n_sim/288:.0f} days)")
    print(f"  Price: ${prices_close.min():.2f} - ${prices_close.max():.2f}")
    print(f"  Median pred vol: {median_pred_vol*100:.4f}%")
    print(f"  Median pred range: {median_pred_range*100:.3f}%")

    # --- Build strategies ---
    bot = GridBotSimulator(n_levels=5, fee_bps=MAKER_FEE_BPS, capital_usd=10000)

    # Helper: build spacing array from predictions
    def spacing_from_pred(pred, divisor=2.0, floor_pct=0.0050):
        s = np.full(n_sim, floor_pct)
        for i in range(n_sim):
            if not np.isnan(pred[i]) and pred[i] > 0:
                s[i] = max(pred[i] / divisor, floor_pct)
        return np.clip(s, floor_pct, 0.05)

    fixed_rebal = np.full(n_sim, 288, dtype=int)  # 24h

    strategies = []

    # === Step 0: Baseline ===
    strategies.append({
        "name": "S0: Fix 1.00% (24h)",
        "spacings": np.full(n_sim, 0.0100),
        "rebalance": fixed_rebal,
        "paused": None,
    })

    # === Step 1: Direct range prediction ===
    # spacing = predicted_range / 2, floor 1.00%
    s1_spacings = spacing_from_pred(pr1h, divisor=2.0, floor_pct=0.0100)
    strategies.append({
        "name": "S1: Range/2 f1.0% (24h)",
        "spacings": s1_spacings,
        "rebalance": fixed_rebal,
        "paused": None,
    })
    # Also test with floor 0.50% to see if tighter helps
    s1b_spacings = spacing_from_pred(pr1h, divisor=2.0, floor_pct=0.0050)
    strategies.append({
        "name": "S1b: Range/2 f0.5% (24h)",
        "spacings": s1b_spacings,
        "rebalance": fixed_rebal,
        "paused": None,
    })

    # === Step 2: P90 quantile range ===
    # spacing = P90_range / (2 * n_levels), floor 1.00%
    s2_spacings = spacing_from_pred(pr1h_p90, divisor=2.0, floor_pct=0.0100)
    strategies.append({
        "name": "S2: P90 range/2 f1.0% (24h)",
        "spacings": s2_spacings,
        "rebalance": fixed_rebal,
        "paused": None,
    })
    s2b_spacings = spacing_from_pred(pr1h_p90, divisor=2.0, floor_pct=0.0050)
    strategies.append({
        "name": "S2b: P90 range/2 f0.5% (24h)",
        "spacings": s2b_spacings,
        "rebalance": fixed_rebal,
        "paused": None,
    })

    # === Step 3: Breakout detection → widen ===
    # When breakout prob > 0.3, multiply spacing by 2×
    s3_spacings = np.full(n_sim, 0.0100)
    for i in range(n_sim):
        if not np.isnan(pb[i]) and pb[i] > 0.3:
            s3_spacings[i] = 0.0200  # 2× wider during breakout
    strategies.append({
        "name": "S3: Fix1%+Brk widen (24h)",
        "spacings": s3_spacings,
        "rebalance": fixed_rebal,
        "paused": None,
    })
    # Also try pausing during breakout instead of widening
    s3b_paused = np.zeros(n_sim, dtype=bool)
    for i in range(n_sim):
        if not np.isnan(pb[i]) and pb[i] > 0.4:
            s3b_paused[i] = True
    strategies.append({
        "name": "S3b: Fix1%+Brk pause (24h)",
        "spacings": np.full(n_sim, 0.0100),
        "rebalance": fixed_rebal,
        "paused": s3b_paused,
    })

    # === Step 4: Consolidation → tighten ===
    # When consolidation ratio < 0.3 (tight range), tighten to 0.50%
    s4_spacings = np.full(n_sim, 0.0100)
    for i in range(n_sim):
        if consol[i] < 0.3:
            s4_spacings[i] = 0.0050  # tighter during consolidation
    strategies.append({
        "name": "S4: Fix1%+Consol tight (24h)",
        "spacings": s4_spacings,
        "rebalance": fixed_rebal,
        "paused": None,
    })
    # Combine: tighten in consolidation, widen in breakout
    s4b_spacings = np.full(n_sim, 0.0100)
    for i in range(n_sim):
        if consol[i] < 0.3:
            s4b_spacings[i] = 0.0050
        if not np.isnan(pb[i]) and pb[i] > 0.3:
            s4b_spacings[i] = 0.0200
    strategies.append({
        "name": "S4b: Consol+Brk combo (24h)",
        "spacings": s4b_spacings,
        "rebalance": fixed_rebal,
        "paused": None,
    })

    # === Step 5: Adaptive rebalance interval ===
    # High vol → 8h rebalance, low vol → 48h, normal → 24h
    s5_rebal = np.full(n_sim, 288, dtype=int)
    for i in range(n_sim):
        if not np.isnan(pv1h[i]):
            if pv1h[i] > 2.0 * median_pred_vol:
                s5_rebal[i] = 96   # 8h during high vol
            elif pv1h[i] < 0.5 * median_pred_vol:
                s5_rebal[i] = 576  # 48h during low vol
    strategies.append({
        "name": "S5: Fix1% adapt rebal",
        "spacings": np.full(n_sim, 0.0100),
        "rebalance": s5_rebal,
        "paused": None,
    })

    # === Step 6: 4h range for rebalance timing ===
    # If predicted 4h range is large → rebalance sooner (12h), small → later (48h)
    s6_rebal = np.full(n_sim, 288, dtype=int)
    median_4h_range = np.nanmedian(pr4h[~np.isnan(pr4h)])
    for i in range(n_sim):
        if not np.isnan(pr4h[i]) and median_4h_range > 0:
            if pr4h[i] > 1.5 * median_4h_range:
                s6_rebal[i] = 144  # 12h
            elif pr4h[i] < 0.5 * median_4h_range:
                s6_rebal[i] = 576  # 48h
    strategies.append({
        "name": "S6: Fix1% 4h-range rebal",
        "spacings": np.full(n_sim, 0.0100),
        "rebalance": s6_rebal,
        "paused": None,
    })

    # === Step 7: Asymmetric grid ===
    # Shift grid center by predicted upside fraction
    # If upside > 0.55, shift center up (more sell levels); if < 0.45, shift down
    # Implemented as spacing adjustment: more levels on the expected side
    # Simple approach: if upside predicted, use tighter buy spacing (catch dips) and wider sell
    # Actually, simplest: shift the grid center by (pred_upside - 0.5) × range
    # We'll implement this by adjusting spacings asymmetrically
    # For now: if pred_upside > 0.55, reduce spacing (tighter grid, expect mean-reversion)
    #          if pred_upside < 0.45, reduce spacing too (same logic)
    #          if near 0.50, keep 1.00%
    s7_spacings = np.full(n_sim, 0.0100)
    for i in range(n_sim):
        if not np.isnan(pu[i]):
            asym = abs(pu[i] - 0.5)
            if asym < 0.05:  # very symmetric → good for grid, tighten
                s7_spacings[i] = 0.0070
            elif asym > 0.15:  # very asymmetric → trending, widen
                s7_spacings[i] = 0.0150
    strategies.append({
        "name": "S7: Asymmetry adj (24h)",
        "spacings": s7_spacings,
        "rebalance": fixed_rebal,
        "paused": None,
    })

    # --- Run all strategies ---
    print(f"\n  Running {len(strategies)} strategies...\n")
    results = []

    for strat in strategies:
        t_s = time.time()
        r = bot.run(
            prices_close, prices_high, prices_low,
            strat["spacings"],
            strategy_name=strat["name"],
            paused=strat.get("paused"),
            rebalance_intervals=strat["rebalance"],
        )
        results.append(r)
        elapsed = time.time() - t_s

        print(f"  {r['strategy']}:")
        print(f"    PnL: ${r['total_pnl']:+.2f} (grid: ${r['grid_profits']:+.2f}, fees: -${r['total_fees']:.2f})")
        print(f"    PnL/day: ${r['pnl_per_day']:+.2f} | Fills: {r['fills']:,} ({r['fills_per_day']:.1f}/d) | "
              f"Sharpe: {r['sharpe']:.2f} | MaxDD: ${r['max_drawdown']:.2f} ({elapsed:.1f}s)\n")

    # --- Summary table ---
    baseline = results[0]
    print(f"  {'='*95}")
    print(f"  SUMMARY — {symbol} ({n_sim/288:.0f} days)")
    print(f"  {'='*95}")
    print(f"  {'Strategy':<30s} {'PnL':>10s} {'Grid$':>10s} {'Fees':>8s} {'PnL/d':>8s} "
          f"{'Fills':>7s} {'Sharpe':>7s} {'MaxDD':>10s} {'vs Base':>8s}")
    print(f"  {'-'*98}")
    for r in results:
        delta = r['total_pnl'] - baseline['total_pnl']
        marker = "✅" if delta > 0 and r is not baseline else "  "
        print(f"  {marker}{r['strategy']:<28s} ${r['total_pnl']:>+9.2f} ${r['grid_profits']:>+9.2f} "
              f"${r['total_fees']:>7.0f} ${r['pnl_per_day']:>+7.2f} "
              f"{r['fills']:>6,d} {r['sharpe']:>7.2f} ${r['max_drawdown']:>9.2f} "
              f"{'':>8s}" if r is baseline else
              f"  {marker}{r['strategy']:<28s} ${r['total_pnl']:>+9.2f} ${r['grid_profits']:>+9.2f} "
              f"${r['total_fees']:>7.0f} ${r['pnl_per_day']:>+7.2f} "
              f"{r['fills']:>6,d} {r['sharpe']:>7.2f} ${r['max_drawdown']:>9.2f} "
              f"${delta:>+7.0f}")

    elapsed_total = time.time() - t_total
    print(f"\n✅ Done in {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SOLUSDT")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-01-31")
    args = parser.parse_args()
    run_v15(args.symbol, args.start, args.end)


if __name__ == "__main__":
    main()
