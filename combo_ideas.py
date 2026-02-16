#!/usr/bin/env python3
"""
Combination Ideas: v3 Directional Signals + v9-12 ML Predictions.

Tests ideas #1-#6 from IDEAS_v3_plus_ml.md.
Each idea is tested on a small dataset first (1 symbol, 30 days).
If promising, expand to more symbols and longer periods.

Data pipeline: merges v3 tick-based features with v9 regime features on timestamp_us.
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

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from experiments import (
    load_features, add_derived_features, backtest_signal,
    zscore_signal, rank_composite, ROUND_TRIP_FEE_BPS
)
from regime_detection import load_bars, compute_regime_features

PARQUET_DIR = Path("./parquet")

# v9 features for vol prediction (Ridge model)
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


# ---------------------------------------------------------------------------
# Data loading: merge v3 + v9 pipelines
# ---------------------------------------------------------------------------

def load_merged_data(symbol, start_date, end_date):
    """Load and merge v3 tick features with v9 regime features."""
    t0 = time.time()

    # v3 features (tick-based microstructure)
    print(f"  Loading v3 tick features...", flush=True)
    df_v3 = load_features(symbol, start_date, end_date)
    if df_v3.empty:
        print(f"  ERROR: No v3 data")
        return pd.DataFrame()
    print(f"  v3: {len(df_v3):,} bars in {time.time()-t0:.0f}s")

    # v9 features (OHLCV regime features)
    t1 = time.time()
    print(f"  Loading v9 regime features...", flush=True)
    df_v9 = load_bars(symbol, start_date, end_date)
    print(f"  v9: {len(df_v9):,} bars in {time.time()-t1:.0f}s")

    # Compute regime features
    t2 = time.time()
    print(f"  Computing regime features...", flush=True)
    df_v9 = compute_regime_features(df_v9)
    print(f"  Regime features in {time.time()-t2:.0f}s")

    # Merge on timestamp_us
    # v9 has regime features we need; v3 has microstructure features
    # Keep all v3 columns, add v9 regime features that don't exist in v3
    v9_regime_cols = [c for c in df_v9.columns if c in VOL_FEATURES and c not in df_v3.columns]
    v9_merge = df_v9[["timestamp_us"] + v9_regime_cols].copy()

    df = df_v3.merge(v9_merge, on="timestamp_us", how="inner")
    print(f"  Merged: {len(df):,} bars ({len(df_v3.columns)} v3 + {len(v9_regime_cols)} v9 regime cols)")

    # Add v3 derived features
    df = add_derived_features(df)

    # Compute forward vol target for Ridge model training
    ret = df["returns"].values
    n = len(df)
    fwd_vol = np.full(n, np.nan)
    for i in range(n - 48):
        fwd_vol[i] = np.std(ret[i+1:i+49])
    df["fwd_vol_4h"] = fwd_vol

    # Forward range for TP calculations
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    fwd_range = np.full(n, np.nan)
    for i in range(n - 48):
        future_h = h[i+1:i+49]
        future_l = l[i+1:i+49]
        fwd_range[i] = (np.max(future_h) - np.min(future_l)) / max(c[i], 1e-10)
    df["fwd_range_4h"] = fwd_range

    print(f"  Total load time: {time.time()-t0:.0f}s")
    return df


# ---------------------------------------------------------------------------
# Vol prediction helper (walk-forward Ridge)
# ---------------------------------------------------------------------------

def predict_vol_walkforward(df, feature_cols, target_col="fwd_vol_4h", min_train=2000):
    """
    Walk-forward vol prediction using Ridge regression.
    Returns array of predicted vol (NaN where no prediction available).
    """
    available = [f for f in feature_cols if f in df.columns]
    X = df[available].values
    y = df[target_col].values
    n = len(df)

    predictions = np.full(n, np.nan)
    scaler = StandardScaler()
    model = Ridge(alpha=1.0)

    # Expanding window: retrain every 288 bars (1 day)
    retrain_interval = 288
    last_train = 0

    for i in range(min_train, n):
        if i - last_train >= retrain_interval or i == min_train:
            # Train on all data up to i
            train_mask = ~np.isnan(y[:i])
            X_train = X[:i][train_mask]
            y_train = y[:i][train_mask]
            if len(y_train) < 100:
                continue
            X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
            scaler.fit(X_train)
            model.fit(scaler.transform(X_train), y_train)
            last_train = i

        x_i = np.nan_to_num(X[i:i+1], nan=0, posinf=0, neginf=0)
        predictions[i] = model.predict(scaler.transform(x_i))[0]

    return predictions


# ---------------------------------------------------------------------------
# Enhanced backtest with position sizing and dynamic TP
# ---------------------------------------------------------------------------

def backtest_enhanced(df, signal_col, entry_threshold, holding_bars, fee_bps,
                      direction, predicted_vol=None, predicted_range=None,
                      vol_sizing=False, dynamic_tp=None, vol_filter=None):
    """
    Enhanced backtest supporting:
    - vol_sizing: scale position by target_vol / predicted_vol
    - dynamic_tp: exit early if price hits predicted_range * tp_mult
    - vol_filter: only trade when vol regime matches ('low', 'high', 'rising', None)
    """
    data = df.dropna(subset=[signal_col]).copy()
    signals = data[signal_col].values
    closes = data["close"].values
    highs = data["high"].values
    lows = data["low"].values
    n = len(data)

    # Predicted vol and range aligned with data
    if predicted_vol is not None:
        pred_vol = data["predicted_vol"].values if "predicted_vol" in data.columns else predicted_vol
    else:
        pred_vol = np.ones(n)

    if predicted_range is not None:
        pred_range = data["predicted_range"].values if "predicted_range" in data.columns else predicted_range
    else:
        pred_range = np.ones(n) * 0.01  # default 1%

    # Vol filter
    if vol_filter is not None and "predicted_vol" in data.columns:
        median_vol = np.nanmedian(pred_vol[pred_vol > 0])
        if vol_filter == "low":
            vol_ok = pred_vol < median_vol
        elif vol_filter == "high":
            vol_ok = pred_vol >= median_vol
        elif vol_filter == "rising":
            vol_change = np.diff(pred_vol, prepend=pred_vol[0])
            vol_ok = vol_change > 0
        else:
            vol_ok = np.ones(n, dtype=bool)
    else:
        vol_ok = np.ones(n, dtype=bool)

    # Target vol for sizing
    if vol_sizing:
        target_vol = np.nanmedian(pred_vol[pred_vol > 0])
    else:
        target_vol = 1.0

    pnls = []
    sizes = []
    hold_durations = []
    in_trade = False
    entry_idx = 0
    trade_dir = 0
    trade_size = 1.0

    for i in range(n - holding_bars):
        # Check exit conditions
        if in_trade:
            bars_held = i - entry_idx

            # Dynamic TP check (bar by bar)
            tp_hit = False
            if dynamic_tp is not None and pred_range[entry_idx] > 0:
                tp_dist = pred_range[entry_idx] * dynamic_tp
                if trade_dir == 1:
                    if (highs[i] - closes[entry_idx]) / max(closes[entry_idx], 1e-10) >= tp_dist:
                        tp_hit = True
                        exit_ret = tp_dist  # assume fill at TP
                elif trade_dir == -1:
                    if (closes[entry_idx] - lows[i]) / max(closes[entry_idx], 1e-10) >= tp_dist:
                        tp_hit = True
                        exit_ret = tp_dist

            if tp_hit:
                raw = exit_ret * 10000 * trade_dir * trade_size
                pnls.append(raw - fee_bps * trade_size)
                sizes.append(trade_size)
                hold_durations.append(bars_held)
                in_trade = False
            elif bars_held >= holding_bars:
                raw = (closes[i] / closes[entry_idx] - 1) * 10000 * trade_dir * trade_size
                pnls.append(raw - fee_bps * trade_size)
                sizes.append(trade_size)
                hold_durations.append(bars_held)
                in_trade = False

        # Entry
        if not in_trade and vol_ok[i]:
            enter = False
            if direction == "contrarian":
                if signals[i] > entry_threshold:
                    enter = True; trade_dir = -1
                elif signals[i] < -entry_threshold:
                    enter = True; trade_dir = 1
            else:  # momentum
                if signals[i] > entry_threshold:
                    enter = True; trade_dir = 1
                elif signals[i] < -entry_threshold:
                    enter = True; trade_dir = -1

            if enter:
                in_trade = True
                entry_idx = i
                if vol_sizing and pred_vol[i] > 0:
                    trade_size = target_vol / pred_vol[i]
                    trade_size = np.clip(trade_size, 0.2, 5.0)  # cap at 0.2x-5x
                else:
                    trade_size = 1.0

    return (np.array(pnls) if pnls else np.array([]),
            np.array(sizes) if sizes else np.array([]),
            np.array(hold_durations) if hold_durations else np.array([]))


# ---------------------------------------------------------------------------
# Signal generators (from v3)
# ---------------------------------------------------------------------------

def generate_signals(df):
    """Generate all v3 signals on the merged dataframe."""
    signals = {}

    # E01: Contrarian imbalance
    cols = ["vol_imbalance", "dollar_imbalance", "large_imbalance",
            "count_imbalance", "close_vs_vwap"]
    available = [c for c in cols if c in df.columns]
    if len(available) >= 3:
        df["sig_E01"] = rank_composite(df, available)
        signals["E01"] = {"col": "sig_E01", "direction": "contrarian", "name": "Contrarian imbalance"}

    # E03: Vol breakout
    if "range_zscore" in df.columns:
        df["range_dir"] = df["range_zscore"] * np.sign(df["returns"])
        df["sig_E03"] = zscore_signal(df, "range_dir")
        signals["E03"] = {"col": "sig_E03", "direction": "momentum", "name": "Vol breakout"}

    # E09: Cumulative imbalance momentum
    if "cum_imbalance_12" in df.columns:
        df["sig_E09"] = zscore_signal(df, "cum_imbalance_12")
        signals["E09"] = {"col": "sig_E09", "direction": "momentum", "name": "Cum imbalance momentum"}

    # E06: Volume surge + direction
    if "vol_zscore" in df.columns:
        df["vol_dir"] = df["vol_zscore"] * np.sign(df["returns"])
        df["sig_E06"] = zscore_signal(df, "vol_dir")
        signals["E06"] = {"col": "sig_E06", "direction": "momentum", "name": "Volume surge"}

    # E11: Kyle's lambda momentum
    if "kyle_lambda" in df.columns:
        df["kyle_dir"] = df["kyle_lambda"] * np.sign(df["returns"])
        df["sig_E11"] = zscore_signal(df, "kyle_dir")
        signals["E11"] = {"col": "sig_E11", "direction": "momentum", "name": "Kyle lambda momentum"}

    return signals


# ---------------------------------------------------------------------------
# Idea #1: Vol-Adaptive Position Sizing
# ---------------------------------------------------------------------------

def idea1_vol_sizing(df, signals, predicted_vol):
    """Test vol-adaptive position sizing vs fixed sizing."""
    print("\n" + "#" * 70)
    print("  IDEA #1: Vol-Adaptive Position Sizing")
    print("#" * 70)

    df["predicted_vol"] = predicted_vol

    results = []
    for sig_id, sig_info in signals.items():
        for thresh in [1.0, 1.5]:
            t0 = time.time()

            # Baseline: fixed sizing
            pnls_fixed, _, _ = backtest_enhanced(
                df, sig_info["col"], thresh, 48, ROUND_TRIP_FEE_BPS,
                sig_info["direction"], vol_sizing=False)

            # Vol-adaptive sizing
            pnls_vol, sizes_vol, _ = backtest_enhanced(
                df, sig_info["col"], thresh, 48, ROUND_TRIP_FEE_BPS,
                sig_info["direction"], predicted_vol=predicted_vol, vol_sizing=True)

            if len(pnls_fixed) < 10 or len(pnls_vol) < 10:
                continue

            # Metrics
            avg_fixed = pnls_fixed.mean()
            avg_vol = pnls_vol.mean()
            sharpe_fixed = avg_fixed / max(pnls_fixed.std(), 1e-10)
            sharpe_vol = avg_vol / max(pnls_vol.std(), 1e-10)
            total_fixed = pnls_fixed.sum()
            total_vol = pnls_vol.sum()
            wr_fixed = (pnls_fixed > 0).mean()
            wr_vol = (pnls_vol > 0).mean()
            avg_size = sizes_vol.mean() if len(sizes_vol) > 0 else 1.0

            improvement = avg_vol - avg_fixed
            sharpe_imp = sharpe_vol - sharpe_fixed

            marker = "✅" if improvement > 0 else "  "
            print(f"\n  {marker} {sig_id} ({sig_info['name']}) thresh={thresh} hold=4h:")
            print(f"    Fixed:    trades={len(pnls_fixed):3d} avg={avg_fixed:+7.2f} bps "
                  f"total={total_fixed:+8.1f} WR={wr_fixed:.0%} sharpe={sharpe_fixed:.3f}")
            print(f"    VolAdapt: trades={len(pnls_vol):3d} avg={avg_vol:+7.2f} bps "
                  f"total={total_vol:+8.1f} WR={wr_vol:.0%} sharpe={sharpe_vol:.3f} "
                  f"avg_size={avg_size:.2f}x")
            print(f"    Δ avg={improvement:+.2f} bps  Δ sharpe={sharpe_imp:+.3f}")

            results.append({
                "idea": 1, "signal": sig_id, "thresh": thresh,
                "avg_fixed": avg_fixed, "avg_vol": avg_vol,
                "sharpe_fixed": sharpe_fixed, "sharpe_vol": sharpe_vol,
                "total_fixed": total_fixed, "total_vol": total_vol,
                "n_trades": len(pnls_vol), "improvement": improvement,
            })

    return results


# ---------------------------------------------------------------------------
# Idea #2: Vol-Regime Filtered Signals
# ---------------------------------------------------------------------------

def idea2_vol_filter(df, signals, predicted_vol):
    """Test signals filtered by vol regime."""
    print("\n" + "#" * 70)
    print("  IDEA #2: Vol-Regime Filtered Signals")
    print("#" * 70)

    df["predicted_vol"] = predicted_vol

    results = []
    for sig_id, sig_info in signals.items():
        thresh = 1.0 if sig_id in ["E01", "E09"] else 1.5
        print(f"\n  {sig_id} ({sig_info['name']}) thresh={thresh} hold=4h:")

        for vol_filter, filter_name in [(None, "All"), ("low", "Low-vol"), ("high", "High-vol"), ("rising", "Rising-vol")]:
            pnls, _, _ = backtest_enhanced(
                df, sig_info["col"], thresh, 48, ROUND_TRIP_FEE_BPS,
                sig_info["direction"], predicted_vol=predicted_vol,
                vol_filter=vol_filter)

            if len(pnls) < 5:
                print(f"    {filter_name:12s}: trades={len(pnls):3d} (too few)")
                continue

            avg = pnls.mean()
            sharpe = avg / max(pnls.std(), 1e-10)
            total = pnls.sum()
            wr = (pnls > 0).mean()

            marker = "✅" if avg > 0 else "  "
            print(f"    {marker} {filter_name:12s}: trades={len(pnls):3d} avg={avg:+7.2f} bps "
                  f"total={total:+8.1f} WR={wr:.0%} sharpe={sharpe:.3f}")

            results.append({
                "idea": 2, "signal": sig_id, "filter": filter_name,
                "n_trades": len(pnls), "avg_pnl": avg, "sharpe": sharpe,
                "total_pnl": total, "win_rate": wr,
            })

    return results


# ---------------------------------------------------------------------------
# Idea #3: Dynamic TP from Predicted Range
# ---------------------------------------------------------------------------

def idea3_dynamic_tp(df, signals, predicted_vol, predicted_range):
    """Test dynamic take-profit based on predicted range."""
    print("\n" + "#" * 70)
    print("  IDEA #3: Dynamic Hold/TP from Predicted Range")
    print("#" * 70)

    df["predicted_vol"] = predicted_vol
    df["predicted_range"] = predicted_range

    results = []
    for sig_id, sig_info in signals.items():
        thresh = 1.0 if sig_id in ["E01", "E09"] else 1.5
        print(f"\n  {sig_id} ({sig_info['name']}) thresh={thresh}:")

        # Baseline: fixed 4h hold
        pnls_base, _, dur_base = backtest_enhanced(
            df, sig_info["col"], thresh, 48, ROUND_TRIP_FEE_BPS,
            sig_info["direction"])

        if len(pnls_base) < 10:
            print(f"    Too few trades ({len(pnls_base)})")
            continue

        avg_base = pnls_base.mean()
        sharpe_base = avg_base / max(pnls_base.std(), 1e-10)
        print(f"    Fixed 4h:  trades={len(pnls_base):3d} avg={avg_base:+7.2f} bps "
              f"sharpe={sharpe_base:.3f}")

        # Dynamic TP at different multipliers of predicted range
        for tp_mult, tp_name in [(0.5, "TP@P50"), (0.75, "TP@P75"), (1.0, "TP@P100")]:
            pnls_tp, _, dur_tp = backtest_enhanced(
                df, sig_info["col"], thresh, 48, ROUND_TRIP_FEE_BPS,
                sig_info["direction"], predicted_range=predicted_range,
                dynamic_tp=tp_mult)

            if len(pnls_tp) < 10:
                continue

            avg_tp = pnls_tp.mean()
            sharpe_tp = avg_tp / max(pnls_tp.std(), 1e-10)
            wr_tp = (pnls_tp > 0).mean()
            avg_dur = dur_tp.mean() if len(dur_tp) > 0 else 48
            tp_hit_rate = (dur_tp < 48).mean() if len(dur_tp) > 0 else 0

            improvement = avg_tp - avg_base
            marker = "✅" if improvement > 0 else "  "
            print(f"    {marker} {tp_name:8s}: trades={len(pnls_tp):3d} avg={avg_tp:+7.2f} bps "
                  f"sharpe={sharpe_tp:.3f} WR={wr_tp:.0%} avg_hold={avg_dur:.0f}bars "
                  f"TP_hit={tp_hit_rate:.0%} Δ={improvement:+.2f}")

            results.append({
                "idea": 3, "signal": sig_id, "tp_mult": tp_mult,
                "n_trades": len(pnls_tp), "avg_pnl": avg_tp, "sharpe": sharpe_tp,
                "avg_base": avg_base, "improvement": improvement,
                "tp_hit_rate": tp_hit_rate, "avg_hold": avg_dur,
            })

        # Combo: vol sizing + dynamic TP
        pnls_combo, sizes_combo, dur_combo = backtest_enhanced(
            df, sig_info["col"], thresh, 48, ROUND_TRIP_FEE_BPS,
            sig_info["direction"], predicted_vol=predicted_vol,
            predicted_range=predicted_range, vol_sizing=True, dynamic_tp=0.75)

        if len(pnls_combo) >= 10:
            avg_combo = pnls_combo.mean()
            sharpe_combo = avg_combo / max(pnls_combo.std(), 1e-10)
            improvement = avg_combo - avg_base
            marker = "✅" if improvement > 0 else "  "
            print(f"    {marker} {'Combo':8s}: trades={len(pnls_combo):3d} avg={avg_combo:+7.2f} bps "
                  f"sharpe={sharpe_combo:.3f} Δ={improvement:+.2f} (vol_sizing + TP@P75)")

    return results


# ---------------------------------------------------------------------------
# Idea #4: Breakout Confirmation Filter
# ---------------------------------------------------------------------------

def idea4_breakout_filter(df, signals, predicted_vol):
    """Test breakout probability as confirmation filter."""
    print("\n" + "#" * 70)
    print("  IDEA #4: Breakout Confirmation Filter")
    print("#" * 70)

    # Simple breakout proxy: vol_ratio (short/long) + range compression
    # We don't have the full LGBM model here, but we can use the key features
    if "vol_ratio_1h_24h" in df.columns:
        vol_ratio = df["vol_ratio_1h_24h"].values
    elif "vol_ratio" in df.columns:
        vol_ratio = df["vol_ratio"].values
    else:
        print("  No vol_ratio feature available — skipping")
        return []

    # Breakout proxy: high vol_ratio = breakout likely
    median_vr = np.nanmedian(vol_ratio)
    df["breakout_proxy"] = vol_ratio

    results = []
    for sig_id, sig_info in signals.items():
        thresh = 1.0 if sig_id in ["E01", "E09"] else 1.5
        print(f"\n  {sig_id} ({sig_info['name']}) thresh={thresh}:")

        # Baseline
        pnls_base, _, _ = backtest_enhanced(
            df, sig_info["col"], thresh, 48, ROUND_TRIP_FEE_BPS,
            sig_info["direction"])

        if len(pnls_base) < 10:
            continue

        avg_base = pnls_base.mean()
        print(f"    Baseline:        trades={len(pnls_base):3d} avg={avg_base:+7.2f} bps")

        # Filter: only trade when breakout proxy is above/below median
        for filter_type, filter_name in [("high", "BO likely"), ("low", "BO unlikely")]:
            df_filtered = df.copy()
            if filter_type == "high":
                mask = vol_ratio >= median_vr
            else:
                mask = vol_ratio < median_vr
            # Zero out signal where filter doesn't pass
            df_filtered.loc[~mask, sig_info["col"]] = 0

            pnls_f, _, _ = backtest_enhanced(
                df_filtered, sig_info["col"], thresh, 48, ROUND_TRIP_FEE_BPS,
                sig_info["direction"])

            if len(pnls_f) < 5:
                print(f"    {filter_name:16s}: trades={len(pnls_f):3d} (too few)")
                continue

            avg_f = pnls_f.mean()
            improvement = avg_f - avg_base
            marker = "✅" if improvement > 0 else "  "
            print(f"    {marker} {filter_name:16s}: trades={len(pnls_f):3d} avg={avg_f:+7.2f} bps "
                  f"Δ={improvement:+.2f}")

            results.append({
                "idea": 4, "signal": sig_id, "filter": filter_name,
                "n_trades": len(pnls_f), "avg_pnl": avg_f,
                "avg_base": avg_base, "improvement": improvement,
            })

    return results


# ---------------------------------------------------------------------------
# Idea #6: Contrarian + Vol Compression
# ---------------------------------------------------------------------------

def idea6_contrarian_compression(df, signals, predicted_vol):
    """Test contrarian signal during vol compression periods."""
    print("\n" + "#" * 70)
    print("  IDEA #6: Contrarian + Vol Compression Setup")
    print("#" * 70)

    # Compression: short-term vol much lower than long-term
    if "vol_ratio_1h_24h" in df.columns:
        compression = df["vol_ratio_1h_24h"].values
    elif "vol_ratio" in df.columns:
        compression = df["vol_ratio"].values
    else:
        print("  No vol ratio feature — skipping")
        return []

    # Only test E01 (contrarian) — the hypothesis is specific to contrarian
    if "E01" not in signals:
        print("  E01 signal not available — skipping")
        return []

    sig_info = signals["E01"]
    results = []

    for thresh in [1.0, 1.5]:
        print(f"\n  E01 (Contrarian) thresh={thresh}:")

        # Baseline
        pnls_base, _, _ = backtest_enhanced(
            df, sig_info["col"], thresh, 48, ROUND_TRIP_FEE_BPS,
            sig_info["direction"])

        if len(pnls_base) < 10:
            continue

        avg_base = pnls_base.mean()
        print(f"    Baseline:         trades={len(pnls_base):3d} avg={avg_base:+7.2f} bps")

        # During compression (vol_ratio < 0.8 = short vol much lower than long)
        for comp_thresh, comp_name in [(0.6, "Strong compress"), (0.8, "Mild compress"), (1.2, "Expanding")]:
            df_f = df.copy()
            if comp_thresh < 1.0:
                mask = compression < comp_thresh
            else:
                mask = compression >= comp_thresh
            df_f.loc[~mask, sig_info["col"]] = 0

            pnls_f, _, _ = backtest_enhanced(
                df_f, sig_info["col"], thresh, 48, ROUND_TRIP_FEE_BPS,
                sig_info["direction"])

            if len(pnls_f) < 5:
                print(f"    {comp_name:20s}: trades={len(pnls_f):3d} (too few)")
                continue

            avg_f = pnls_f.mean()
            improvement = avg_f - avg_base
            marker = "✅" if improvement > 0 else "  "
            print(f"    {marker} {comp_name:20s}: trades={len(pnls_f):3d} avg={avg_f:+7.2f} bps "
                  f"Δ={improvement:+.2f}")

            results.append({
                "idea": 6, "signal": "E01", "filter": comp_name, "thresh": thresh,
                "n_trades": len(pnls_f), "avg_pnl": avg_f,
                "avg_base": avg_base, "improvement": improvement,
            })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Combination ideas: v3 + v9-12")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2025-12-01")
    parser.add_argument("--end", default="2025-12-30")
    parser.add_argument("--ideas", default="1,2,3,4,6", help="Comma-separated idea numbers to run")
    args = parser.parse_args()

    symbol = args.symbol
    ideas_to_run = [int(x) for x in args.ideas.split(",")]

    print("=" * 70)
    print(f"  COMBINATION IDEAS: v3 Signals + v9-12 ML")
    print(f"  Symbol:   {symbol}")
    print(f"  Period:   {args.start} → {args.end}")
    print(f"  Ideas:    {ideas_to_run}")
    print(f"  Fee:      {ROUND_TRIP_FEE_BPS} bps RT")
    print("=" * 70)

    t_total = time.time()

    # Load merged data
    df = load_merged_data(symbol, args.start, args.end)
    if df.empty:
        print("ERROR: No data loaded")
        return

    # Generate v3 signals
    print(f"\n  Generating v3 signals...", flush=True)
    signals = generate_signals(df)
    print(f"  Generated {len(signals)} signals: {list(signals.keys())}")

    # Predict vol (walk-forward)
    print(f"\n  Training walk-forward vol prediction...", flush=True)
    t_vol = time.time()
    predicted_vol = predict_vol_walkforward(df, VOL_FEATURES)
    valid_preds = np.sum(~np.isnan(predicted_vol))
    print(f"  Vol predictions: {valid_preds:,} valid ({100*valid_preds/len(df):.0f}%) in {time.time()-t_vol:.0f}s")
    df["predicted_vol"] = predicted_vol

    # Predict range (simple: vol * scaling factor from v11)
    # range/vol ratio ≈ 11x for 4h horizon
    predicted_range = predicted_vol * 11.0
    df["predicted_range"] = predicted_range

    # Run ideas
    all_results = []

    if 1 in ideas_to_run:
        r = idea1_vol_sizing(df, signals, predicted_vol)
        all_results.extend(r)

    if 2 in ideas_to_run:
        r = idea2_vol_filter(df, signals, predicted_vol)
        all_results.extend(r)

    if 3 in ideas_to_run:
        r = idea3_dynamic_tp(df, signals, predicted_vol, predicted_range)
        all_results.extend(r)

    if 4 in ideas_to_run:
        r = idea4_breakout_filter(df, signals, predicted_vol)
        all_results.extend(r)

    if 6 in ideas_to_run:
        r = idea6_contrarian_compression(df, signals, predicted_vol)
        all_results.extend(r)

    elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"  ALL DONE — {symbol} in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Total results: {len(all_results)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
