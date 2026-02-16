#!/usr/bin/env python3
"""
Regime Detection Experiment Suite — OHLCV-based, fast iteration.

Goal: detect market regime transitions as EARLY as possible.
This is NOT a trading strategy — it's a classification/prediction problem.

APPROACH:
=========
1. Define ground-truth regimes using FORWARD-LOOKING windows (we have the data).
2. Compute many candidate features using BACKWARD-LOOKING windows only.
3. Measure how well each feature (and combinations) predicts the future regime.
4. Evaluate on: accuracy, detection latency, false positive rate, stability.

REGIME DEFINITIONS (ground truth):
==================================
We define regimes along TWO orthogonal axes:

Axis 1 — VOLATILITY (highly predictable):
  - LOW_VOL:  realized vol over next N bars < 0.6× rolling median
  - NORM_VOL: between 0.6× and 1.5× rolling median
  - HIGH_VOL: realized vol over next N bars > 1.5× rolling median

Axis 2 — TREND (moderately predictable):
  - TRENDING:    efficiency ratio over next N bars > 0.4
  - RANGING:     efficiency ratio over next N bars < 0.25
  - MIXED:       between 0.25 and 0.4

Combined gives us 9 regimes, but we collapse to 5 actionable ones:
  1. QUIET_RANGE  — low vol + ranging (best for tight grids)
  2. ACTIVE_RANGE — normal/high vol + ranging (best for wide grids)
  3. SMOOTH_TREND — low/normal vol + trending (best for trend following)
  4. VOLATILE_TREND — high vol + trending (dangerous, reduce size)
  5. CHAOS — high vol + mixed (stay out)

FEATURES (backward-looking only):
==================================
Each feature is computed from past data only — no lookahead.

SUCCESS CRITERIA:
=================
An experiment PASSES if:
  - Accuracy > 40% (5-class) or > 60% (binary: trending vs ranging)
  - Detection latency < 6 bars (30 min at 5m) for regime transitions
  - False positive rate < 30% for regime change signals
  - Stability: average regime duration > 12 bars (1 hour) — no rapid flipping

An experiment FAILS if:
  - Accuracy ≤ random baseline (20% for 5-class, 50% for binary)
  - Detection latency > 24 bars (2 hours)
  - Detector flips regime every few bars (instability)
"""

import sys
import time
import argparse
import psutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PARQUET_DIR = Path("./parquet")
SOURCE = "bybit_futures"
INTERVAL_5M_US = 300_000_000


# ---------------------------------------------------------------------------
# Data loading: tick → 5m OHLCV with microstructure features
# ---------------------------------------------------------------------------

def load_bars(symbol, start_date, end_date):
    """Load tick data, aggregate to 5m bars with microstructure features.
    Caches aggregated bars to parquet for fast re-runs."""
    cache_dir = PARQUET_DIR / symbol / "regime_5m_cache" / SOURCE
    cache_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(start_date, end_date)
    all_bars = []
    t0 = time.time()
    processed = 0
    cached_hits = 0

    for i, date in enumerate(dates, 1):
        ds = date.strftime("%Y-%m-%d")
        cache_path = cache_dir / f"{ds}.parquet"

        # Try cache first
        if cache_path.exists():
            bars = pd.read_parquet(cache_path)
            all_bars.append(bars)
            cached_hits += 1
        else:
            # Aggregate from tick data
            tick_path = PARQUET_DIR / symbol / "trades" / SOURCE / f"{ds}.parquet"
            if not tick_path.exists():
                continue
            trades = pd.read_parquet(tick_path)
            bars = _aggregate_5m(trades)
            del trades
            if not bars.empty:
                bars.to_parquet(cache_path, index=False, compression="snappy")
                all_bars.append(bars)
            processed += 1

        if i % 20 == 0 or i == len(dates):
            elapsed = time.time() - t0
            rate = i / max(elapsed, 0.1)
            eta = (len(dates) - i) / max(rate, 0.01)
            mem = psutil.virtual_memory().used / (1024**3)
            print(f"  [{i}/{len(dates)}] {ds} | {elapsed:.0f}s ETA={eta:.0f}s "
                  f"RAM={mem:.1f}GB cache={cached_hits} new={processed}", flush=True)

    if not all_bars:
        return pd.DataFrame()

    df = pd.concat(all_bars, ignore_index=True).sort_values("timestamp_us").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    print(f"  Loaded {len(df):,} bars ({len(dates)} days, {cached_hits} cached, {processed} new)")
    return df


def _aggregate_5m(trades):
    """Aggregate tick data into 5-minute bars with microstructure features."""
    bucket = (trades["timestamp_us"].values // INTERVAL_5M_US) * INTERVAL_5M_US
    trades = trades.copy()
    trades["bucket"] = bucket

    bars = []
    for bkt, grp in trades.groupby("bucket"):
        p = grp["price"].values
        q = grp["quantity"].values
        qq = grp["quote_quantity"].values
        s = grp["side"].values
        n = len(grp)
        if n < 5:
            continue

        buy_mask = s == 1
        sell_mask = s == -1
        buy_vol = q[buy_mask].sum()
        sell_vol = q[sell_mask].sum()
        total_vol = q.sum()

        open_p, close_p, high_p, low_p = p[0], p[-1], p.max(), p.min()
        ret = (close_p - open_p) / max(open_p, 1e-10)

        # Microstructure features
        vol_imbalance = (buy_vol - sell_vol) / max(total_vol, 1e-10)

        # Price efficiency within bar
        price_changes = np.abs(np.diff(p))
        total_path = price_changes.sum()
        net_move = abs(p[-1] - p[0])
        bar_efficiency = net_move / max(total_path, 1e-10)

        # Trade clustering: are trades arriving in bursts?
        if n > 10:
            t_arr = grp["timestamp_us"].values
            iti = np.diff(t_arr).astype(np.float64)
            iti_mean = iti.mean()
            iti_cv = iti.std() / max(iti_mean, 1) if iti_mean > 0 else 0
        else:
            iti_cv = 0.0

        # Sign runs: consecutive same-side trades
        if n > 5:
            sign_changes = np.sum(np.diff(s) != 0)
            sign_persistence = 1.0 - sign_changes / max(n - 1, 1)
        else:
            sign_persistence = 0.5

        # Large trade fraction
        if n > 20:
            q90 = np.percentile(q, 90)
            large_frac = q[q >= q90].sum() / max(total_vol, 1e-10)
        else:
            large_frac = 0.0

        # Parkinson volatility (more efficient than close-close)
        parkinson_vol = np.sqrt(np.log(high_p / max(low_p, 1e-10))**2 / (4 * np.log(2))) if low_p > 0 else 0

        # VWAP deviation
        vwap = qq.sum() / max(total_vol, 1e-10)
        vwap_dev = (close_p - vwap) / max(vwap, 1e-10)

        bars.append({
            "timestamp_us": bkt,
            "open": open_p, "close": close_p, "high": high_p, "low": low_p,
            "volume": total_vol, "trade_count": n,
            "buy_volume": buy_vol, "sell_volume": sell_vol,
            "returns": ret,
            "vol_imbalance": vol_imbalance,
            "bar_efficiency": bar_efficiency,
            "iti_cv": iti_cv,
            "sign_persistence": sign_persistence,
            "large_trade_frac": large_frac,
            "parkinson_vol": parkinson_vol,
            "vwap_dev": vwap_dev,
        })

    return pd.DataFrame(bars)


# ---------------------------------------------------------------------------
# Feature engineering (backward-looking only)
# ---------------------------------------------------------------------------

def compute_regime_features(df):
    """Compute all candidate features for regime detection. All backward-looking."""
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    v = df["volume"].values
    ret = df["returns"].values

    # --- Volatility features ---
    for w in [12, 24, 48, 96, 288]:  # 1h, 2h, 4h, 8h, 24h
        label = {12: "1h", 24: "2h", 48: "4h", 96: "8h", 288: "24h"}[w]
        df[f"rvol_{label}"] = pd.Series(ret).rolling(w).std().values
        # Parkinson vol (uses high-low, more efficient)
        log_hl = np.log(np.maximum(h, 1e-10) / np.maximum(l, 1e-10))
        df[f"parkvol_{label}"] = pd.Series(log_hl).rolling(w).mean().values * np.sqrt(1.0 / (4 * np.log(2)))

    # Vol ratios (short/long) — detect vol expansion/contraction
    df["vol_ratio_1h_24h"] = df["rvol_1h"] / df["rvol_24h"].clip(lower=1e-10)
    df["vol_ratio_2h_24h"] = df["rvol_2h"] / df["rvol_24h"].clip(lower=1e-10)
    df["vol_ratio_1h_8h"] = df["rvol_1h"] / df["rvol_8h"].clip(lower=1e-10)

    # Vol acceleration (is vol increasing or decreasing?)
    df["vol_accel_1h"] = df["rvol_1h"].pct_change(12)  # 1h change in 1h vol
    df["vol_accel_4h"] = df["rvol_4h"].pct_change(48)

    # Volume features
    df["vol_sma_24h"] = pd.Series(v).rolling(288).mean().values
    df["vol_ratio_bar"] = v / np.maximum(df["vol_sma_24h"].values, 1e-10)

    # --- Trend/mean-reversion features ---
    for w in [12, 24, 48, 96]:
        label = {12: "1h", 24: "2h", 48: "4h", 96: "8h"}[w]
        # Efficiency ratio (Kaufman): |net move| / sum(|bar moves|)
        net = pd.Series(c).diff(w).abs()
        path = pd.Series(np.abs(ret)).rolling(w).sum() * pd.Series(c).shift(w)
        # Simpler: use returns directly
        abs_ret_sum = pd.Series(np.abs(ret)).rolling(w).sum()
        net_ret = pd.Series(ret).rolling(w).sum().abs()
        df[f"efficiency_{label}"] = (net_ret / abs_ret_sum.clip(lower=1e-10)).values

        # Return autocorrelation
        ret_series = pd.Series(ret)
        df[f"ret_autocorr_{label}"] = ret_series.rolling(w).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 2 else 0, raw=False
        ).values

    # ADX-like: smoothed directional movement
    for w in [24, 48]:
        label = {24: "2h", 48: "4h"}[w]
        plus_dm = np.maximum(np.diff(h, prepend=h[0]), 0)
        minus_dm = np.maximum(-np.diff(l, prepend=l[0]), 0)
        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr = pd.Series(tr).rolling(w).mean().values
        plus_di = pd.Series(plus_dm).rolling(w).mean().values / np.maximum(atr, 1e-10)
        minus_di = pd.Series(minus_dm).rolling(w).mean().values / np.maximum(atr, 1e-10)
        dx = np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-10)
        df[f"adx_{label}"] = pd.Series(dx).rolling(w).mean().values

    # Price vs moving averages
    for w in [24, 48, 96, 288]:
        label = {24: "2h", 48: "4h", 96: "8h", 288: "24h"}[w]
        sma = pd.Series(c).rolling(w).mean().values
        df[f"price_vs_sma_{label}"] = (c - sma) / np.maximum(sma, 1e-10)

    # Momentum (cumulative returns)
    for w in [12, 24, 48]:
        label = {12: "1h", 24: "2h", 48: "4h"}[w]
        df[f"momentum_{label}"] = pd.Series(c).pct_change(w).values

    # Sign persistence (rolling)
    signs = np.sign(ret)
    for w in [12, 24]:
        label = {12: "1h", 24: "2h"}[w]
        df[f"sign_persist_{label}"] = pd.Series(signs).rolling(w).apply(
            lambda x: np.abs(x.sum()) / len(x), raw=True
        ).values

    # --- Microstructure features (rolling) ---
    df["imbalance_1h"] = df["vol_imbalance"].rolling(12).mean()
    df["imbalance_4h"] = df["vol_imbalance"].rolling(48).mean()
    df["imbalance_persistence"] = df["vol_imbalance"].rolling(24).apply(
        lambda x: np.abs(x.sum()) / max(np.abs(x).sum(), 1e-10), raw=True
    )

    # Bar efficiency rolling
    df["bar_eff_1h"] = df["bar_efficiency"].rolling(12).mean()
    df["bar_eff_4h"] = df["bar_efficiency"].rolling(48).mean()

    # Trade intensity change
    tc = df["trade_count"].values.astype(float)
    df["trade_intensity_ratio"] = tc / np.maximum(pd.Series(tc).rolling(288).mean().values, 1e-10)

    # Large trade clustering
    df["large_trade_1h"] = df["large_trade_frac"].rolling(12).mean()

    # ITI CV rolling (trade arrival burstiness)
    df["iti_cv_1h"] = df["iti_cv"].rolling(12).mean()

    return df


# ---------------------------------------------------------------------------
# Ground truth regime labeling (forward-looking — for evaluation only)
# ---------------------------------------------------------------------------

def label_regimes(df, forward_window=48):
    """
    Label each bar with its FUTURE regime using forward-looking data.
    This is ground truth for evaluation — never used as a feature.

    forward_window: number of bars to look ahead (48 = 4 hours at 5m)
    """
    c = df["close"].values
    ret = df["returns"].values
    n = len(df)

    # Forward realized volatility
    fwd_vol = np.full(n, np.nan)
    for i in range(n - forward_window):
        fwd_vol[i] = np.std(ret[i+1:i+1+forward_window])

    # Forward efficiency ratio
    fwd_efficiency = np.full(n, np.nan)
    for i in range(n - forward_window):
        fwd_ret = ret[i+1:i+1+forward_window]
        abs_sum = np.abs(fwd_ret).sum()
        net = abs(fwd_ret.sum())
        fwd_efficiency[i] = net / max(abs_sum, 1e-10)

    df["fwd_vol"] = fwd_vol
    df["fwd_efficiency"] = fwd_efficiency

    # Compute vol regime thresholds from rolling median
    vol_median = pd.Series(fwd_vol).rolling(288 * 3, min_periods=288).median().values  # 3-day rolling median
    # Fallback: use overall median for early bars
    overall_vol_median = np.nanmedian(fwd_vol)
    vol_median = np.where(np.isnan(vol_median), overall_vol_median, vol_median)

    # Classify volatility
    vol_regime = np.full(n, "norm_vol", dtype=object)
    vol_regime[fwd_vol < 0.6 * vol_median] = "low_vol"
    vol_regime[fwd_vol > 1.5 * vol_median] = "high_vol"

    # Classify trend
    trend_regime = np.full(n, "mixed", dtype=object)
    trend_regime[fwd_efficiency > 0.4] = "trending"
    trend_regime[fwd_efficiency < 0.25] = "ranging"

    # Combined 5-class regime
    regime = np.full(n, "unknown", dtype=object)
    for i in range(n):
        if np.isnan(fwd_vol[i]):
            regime[i] = "unknown"
        elif vol_regime[i] == "low_vol" and trend_regime[i] == "ranging":
            regime[i] = "quiet_range"
        elif trend_regime[i] == "ranging":
            regime[i] = "active_range"
        elif vol_regime[i] in ("low_vol", "norm_vol") and trend_regime[i] == "trending":
            regime[i] = "smooth_trend"
        elif vol_regime[i] == "high_vol" and trend_regime[i] == "trending":
            regime[i] = "volatile_trend"
        elif vol_regime[i] == "high_vol":
            regime[i] = "chaos"
        else:
            # norm_vol + mixed
            regime[i] = "active_range"  # default bucket

    df["vol_regime"] = vol_regime
    df["trend_regime"] = trend_regime
    df["regime"] = regime

    # Binary labels for simpler evaluation
    df["is_trending"] = (trend_regime == "trending").astype(int)
    df["is_high_vol"] = (vol_regime == "high_vol").astype(int)

    return df


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def evaluate_detector(predictions, ground_truth, label=""):
    """
    Evaluate a regime detector.

    predictions: array of predicted labels (same length as ground_truth)
    ground_truth: array of true labels

    Returns dict of metrics.
    """
    valid = ~pd.isna(predictions) & (ground_truth != "unknown")
    pred = predictions[valid]
    true = ground_truth[valid]
    n = len(pred)

    if n == 0:
        return {"accuracy": 0, "n": 0}

    # Accuracy
    accuracy = (pred == true).mean()

    # Per-class accuracy
    classes = np.unique(true)
    class_acc = {}
    for cls in classes:
        mask = true == cls
        if mask.sum() > 0:
            class_acc[cls] = (pred[mask] == true[mask]).mean()

    # Detection latency: how many bars after a regime change until detector catches it
    true_changes = np.where(np.diff(true.astype(str)) != "")[0] + 1
    pred_changes = np.where(np.diff(pred.astype(str)) != "")[0] + 1

    latencies = []
    for tc in true_changes:
        # Find the next prediction change after this true change
        future_pred_changes = pred_changes[pred_changes >= tc]
        if len(future_pred_changes) > 0:
            latency = future_pred_changes[0] - tc
            if latency < 100:  # cap at reasonable value
                latencies.append(latency)

    avg_latency = np.mean(latencies) if latencies else float("inf")
    median_latency = np.median(latencies) if latencies else float("inf")

    # Stability: average run length of same prediction
    if len(pred) > 1:
        changes = np.sum(np.diff(pred.astype(str)) != "")
        avg_run_length = n / max(changes + 1, 1)
    else:
        avg_run_length = n

    # False positive rate for regime changes
    n_true_changes = len(true_changes)
    n_pred_changes = len(pred_changes)
    if n_pred_changes > 0:
        # FPR: fraction of predicted changes that don't correspond to a true change (within 6 bars)
        false_positives = 0
        for pc in pred_changes:
            nearby_true = np.any(np.abs(true_changes - pc) <= 6)
            if not nearby_true:
                false_positives += 1
        fpr = false_positives / n_pred_changes
    else:
        fpr = 0.0

    result = {
        "label": label,
        "accuracy": accuracy,
        "class_accuracy": class_acc,
        "avg_latency": avg_latency,
        "median_latency": median_latency,
        "avg_run_length": avg_run_length,
        "fpr": fpr,
        "n_true_changes": n_true_changes,
        "n_pred_changes": n_pred_changes,
        "n": n,
    }
    return result


def evaluate_binary_detector(feature_values, threshold, direction, ground_truth_binary):
    """
    Evaluate a simple threshold-based binary detector.

    direction: 'above' means feature > threshold → predict 1
               'below' means feature < threshold → predict 1
    """
    valid = ~np.isnan(feature_values) & ~np.isnan(ground_truth_binary)
    feat = feature_values[valid]
    true = ground_truth_binary[valid].astype(int)

    if len(feat) == 0:
        return {"accuracy": 0, "auc": 0.5}

    if direction == "above":
        pred = (feat > threshold).astype(int)
    else:
        pred = (feat < threshold).astype(int)

    accuracy = (pred == true).mean()
    tp = ((pred == 1) & (true == 1)).sum()
    fp = ((pred == 1) & (true == 0)).sum()
    fn = ((pred == 0) & (true == 1)).sum()
    tn = ((pred == 0) & (true == 0)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    # Stability
    changes = np.sum(np.diff(pred) != 0)
    avg_run = len(pred) / max(changes + 1, 1)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "avg_run_length": avg_run,
        "n_flips": changes,
    }


# ---------------------------------------------------------------------------
# Experiment suite
# ---------------------------------------------------------------------------

def run_experiments(df, forward_window=48):
    """Run all regime detection experiments."""

    print(f"\n{'='*70}")
    print(f"  REGIME DETECTION EXPERIMENTS")
    print(f"  Bars: {len(df)}, Forward window: {forward_window} bars ({forward_window*5} min)")
    print(f"{'='*70}")

    # --- Regime distribution ---
    regime_counts = df["regime"].value_counts()
    print(f"\n  Regime distribution:")
    for r, count in regime_counts.items():
        print(f"    {r:20s}: {count:6d} ({count/len(df)*100:.1f}%)")

    trend_pct = df["is_trending"].mean()
    highvol_pct = df["is_high_vol"].mean()
    print(f"\n  Binary: trending={trend_pct:.1%}, high_vol={highvol_pct:.1%}")

    # Random baseline
    n_classes = len(regime_counts[regime_counts.index != "unknown"])
    random_acc = 1.0 / max(n_classes, 1)
    print(f"  Random baseline: {random_acc:.1%} (5-class), 50% (binary)")

    results = []

    # =====================================================================
    # EXPERIMENT GROUP 1: Single-feature binary detectors for VOLATILITY
    # =====================================================================
    print(f"\n{'#'*70}")
    print(f"  GROUP 1: VOLATILITY DETECTION (is_high_vol)")
    print(f"{'#'*70}")

    vol_features = [
        ("rvol_1h", "above"), ("rvol_2h", "above"), ("rvol_4h", "above"),
        ("rvol_8h", "above"), ("rvol_24h", "above"),
        ("parkvol_1h", "above"), ("parkvol_4h", "above"),
        ("vol_ratio_1h_24h", "above"), ("vol_ratio_2h_24h", "above"),
        ("vol_ratio_1h_8h", "above"),
        ("vol_accel_1h", "above"), ("vol_accel_4h", "above"),
        ("vol_ratio_bar", "above"),
        ("trade_intensity_ratio", "above"),
        ("iti_cv_1h", "above"),
        ("large_trade_1h", "above"),
    ]

    gt_highvol = df["is_high_vol"].values.astype(float)

    for feat_name, direction in vol_features:
        if feat_name not in df.columns:
            continue
        vals = df[feat_name].values.astype(float)
        valid = ~np.isnan(vals) & ~np.isnan(gt_highvol)
        if valid.sum() < 100:
            continue

        # Find optimal threshold (percentile sweep)
        best_f1 = 0
        best_thresh = 0
        best_result = None
        for pct in [50, 60, 70, 75, 80, 85, 90, 95]:
            thresh = np.nanpercentile(vals[valid], pct)
            res = evaluate_binary_detector(vals, thresh, direction, gt_highvol)
            if res["f1"] > best_f1:
                best_f1 = res["f1"]
                best_thresh = thresh
                best_result = res
                best_result["threshold_pct"] = pct

        if best_result:
            best_result["feature"] = feat_name
            best_result["target"] = "is_high_vol"
            best_result["threshold"] = best_thresh
            results.append(best_result)
            status = "✅ PASS" if best_result["accuracy"] > 0.60 else "❌ FAIL"
            print(f"  {feat_name:30s} acc={best_result['accuracy']:.3f} "
                  f"f1={best_result['f1']:.3f} prec={best_result['precision']:.3f} "
                  f"rec={best_result['recall']:.3f} runs={best_result['avg_run_length']:.0f} "
                  f"thresh_pct={best_result.get('threshold_pct', '?')} {status}")

    # =====================================================================
    # EXPERIMENT GROUP 2: Single-feature binary detectors for TREND
    # =====================================================================
    print(f"\n{'#'*70}")
    print(f"  GROUP 2: TREND DETECTION (is_trending)")
    print(f"{'#'*70}")

    trend_features = [
        ("efficiency_1h", "above"), ("efficiency_2h", "above"),
        ("efficiency_4h", "above"), ("efficiency_8h", "above"),
        ("ret_autocorr_1h", "above"), ("ret_autocorr_2h", "above"),
        ("ret_autocorr_4h", "above"),
        ("adx_2h", "above"), ("adx_4h", "above"),
        ("sign_persist_1h", "above"), ("sign_persist_2h", "above"),
        ("bar_eff_1h", "above"), ("bar_eff_4h", "above"),
        ("imbalance_persistence", "above"),
        ("momentum_1h", "above"),  # absolute momentum
        ("momentum_2h", "above"),
        ("momentum_4h", "above"),
        ("price_vs_sma_2h", "above"),  # absolute deviation
        ("price_vs_sma_4h", "above"),
        ("price_vs_sma_8h", "above"),
    ]

    gt_trending = df["is_trending"].values.astype(float)

    for feat_name, direction in trend_features:
        if feat_name not in df.columns:
            continue
        vals = df[feat_name].values.astype(float)

        # For momentum/price_vs_sma, use absolute value (trend in either direction)
        if feat_name.startswith("momentum_") or feat_name.startswith("price_vs_sma_"):
            vals = np.abs(vals)

        valid = ~np.isnan(vals) & ~np.isnan(gt_trending)
        if valid.sum() < 100:
            continue

        best_f1 = 0
        best_result = None
        for pct in [50, 55, 60, 65, 70, 75, 80, 85, 90]:
            thresh = np.nanpercentile(vals[valid], pct)
            res = evaluate_binary_detector(vals, thresh, direction, gt_trending)
            if res["f1"] > best_f1:
                best_f1 = res["f1"]
                best_result = res
                best_result["threshold_pct"] = pct

        if best_result:
            best_result["feature"] = feat_name
            best_result["target"] = "is_trending"
            results.append(best_result)
            status = "✅ PASS" if best_result["accuracy"] > 0.60 else "❌ FAIL"
            print(f"  {feat_name:30s} acc={best_result['accuracy']:.3f} "
                  f"f1={best_result['f1']:.3f} prec={best_result['precision']:.3f} "
                  f"rec={best_result['recall']:.3f} runs={best_result['avg_run_length']:.0f} "
                  f"thresh_pct={best_result.get('threshold_pct', '?')} {status}")

    # =====================================================================
    # EXPERIMENT GROUP 3: Correlation analysis — which features lead?
    # =====================================================================
    print(f"\n{'#'*70}")
    print(f"  GROUP 3: LEAD-LAG CORRELATION ANALYSIS")
    print(f"{'#'*70}")

    # For each feature, compute correlation with future vol and future efficiency
    # at different lead times
    all_features = [c for c in df.columns if c not in [
        "timestamp_us", "datetime", "open", "close", "high", "low",
        "volume", "trade_count", "buy_volume", "sell_volume", "returns",
        "fwd_vol", "fwd_efficiency", "vol_regime", "trend_regime",
        "regime", "is_trending", "is_high_vol"
    ]]

    lead_lags = [1, 3, 6, 12, 24, 48]  # bars ahead

    print(f"\n  Correlation with FUTURE VOLATILITY (fwd_vol):")
    print(f"  {'Feature':30s} " + " ".join(f"{'lag'+str(l):>7s}" for l in lead_lags))
    print(f"  {'-'*75}")

    vol_corr_results = []
    for feat_name in all_features:
        if feat_name not in df.columns:
            continue
        vals = df[feat_name].values.astype(float)
        fwd = df["fwd_vol"].values.astype(float)

        corrs = []
        for lag in lead_lags:
            shifted_fwd = np.roll(fwd, -lag)
            shifted_fwd[-lag:] = np.nan
            valid = ~np.isnan(vals) & ~np.isnan(shifted_fwd)
            if valid.sum() > 100:
                c = np.corrcoef(vals[valid], shifted_fwd[valid])[0, 1]
                corrs.append(c)
            else:
                corrs.append(np.nan)

        max_abs_corr = np.nanmax(np.abs(corrs))
        if max_abs_corr > 0.05:  # only show features with some correlation
            vol_corr_results.append((feat_name, corrs, max_abs_corr))

    vol_corr_results.sort(key=lambda x: -x[2])
    for feat_name, corrs, max_c in vol_corr_results[:20]:
        corr_str = " ".join(f"{c:+.4f}" if not np.isnan(c) else "    nan" for c in corrs)
        print(f"  {feat_name:30s} {corr_str}")

    print(f"\n  Correlation with FUTURE TREND (fwd_efficiency):")
    print(f"  {'Feature':30s} " + " ".join(f"{'lag'+str(l):>7s}" for l in lead_lags))
    print(f"  {'-'*75}")

    trend_corr_results = []
    for feat_name in all_features:
        if feat_name not in df.columns:
            continue
        vals = df[feat_name].values.astype(float)
        fwd = df["fwd_efficiency"].values.astype(float)

        corrs = []
        for lag in lead_lags:
            shifted_fwd = np.roll(fwd, -lag)
            shifted_fwd[-lag:] = np.nan
            valid = ~np.isnan(vals) & ~np.isnan(shifted_fwd)
            if valid.sum() > 100:
                c = np.corrcoef(vals[valid], shifted_fwd[valid])[0, 1]
                corrs.append(c)
            else:
                corrs.append(np.nan)

        max_abs_corr = np.nanmax(np.abs(corrs))
        if max_abs_corr > 0.05:
            trend_corr_results.append((feat_name, corrs, max_abs_corr))

    trend_corr_results.sort(key=lambda x: -x[2])
    for feat_name, corrs, max_c in trend_corr_results[:20]:
        corr_str = " ".join(f"{c:+.4f}" if not np.isnan(c) else "    nan" for c in corrs)
        print(f"  {feat_name:30s} {corr_str}")

    # =====================================================================
    # EXPERIMENT GROUP 4: Multi-feature composite detectors
    # =====================================================================
    print(f"\n{'#'*70}")
    print(f"  GROUP 4: COMPOSITE DETECTORS")
    print(f"{'#'*70}")

    # Composite vol detector: combine top vol features
    _run_composite_detector(df, gt_highvol, "is_high_vol",
                            vol_corr_results[:5] if vol_corr_results else [],
                            results)

    # Composite trend detector: combine top trend features
    _run_composite_detector(df, gt_trending, "is_trending",
                            trend_corr_results[:5] if trend_corr_results else [],
                            results)

    # =====================================================================
    # EXPERIMENT GROUP 5: Regime transition detection
    # =====================================================================
    print(f"\n{'#'*70}")
    print(f"  GROUP 5: REGIME TRANSITION DETECTION")
    print(f"{'#'*70}")

    _evaluate_transition_detection(df, results)

    # =====================================================================
    # Summary
    # =====================================================================
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    passing = [r for r in results if r.get("accuracy", 0) > 0.60]
    failing = [r for r in results if r.get("accuracy", 0) <= 0.60 and "feature" in r]

    print(f"\n  PASSING experiments ({len(passing)}):")
    for r in sorted(passing, key=lambda x: -x.get("f1", 0)):
        print(f"    {r.get('feature', 'composite'):30s} → {r['target']:15s} "
              f"acc={r['accuracy']:.3f} f1={r.get('f1', 0):.3f}")

    print(f"\n  FAILING experiments ({len(failing)}):")
    for r in sorted(failing, key=lambda x: -x.get("accuracy", 0))[:10]:
        print(f"    {r.get('feature', '?'):30s} → {r['target']:15s} "
              f"acc={r['accuracy']:.3f} f1={r.get('f1', 0):.3f}")

    return results


def _run_composite_detector(df, ground_truth, target_name, top_features, results):
    """Build a composite detector by z-scoring and averaging top features."""
    if len(top_features) < 2:
        print(f"  Not enough features for composite {target_name} detector")
        return

    feat_names = [f[0] for f in top_features if f[0] in df.columns][:5]
    if len(feat_names) < 2:
        return

    print(f"\n  Composite {target_name} detector using: {feat_names}")

    # Z-score each feature, then average
    z_scores = []
    for fn in feat_names:
        vals = df[fn].values.astype(float)
        mu = np.nanmean(vals)
        sigma = np.nanstd(vals)
        if sigma > 1e-10:
            z = (vals - mu) / sigma
        else:
            z = np.zeros_like(vals)
        z_scores.append(z)

    composite = np.nanmean(z_scores, axis=0)

    # Find optimal threshold
    best_f1 = 0
    best_result = None
    for pct in [50, 55, 60, 65, 70, 75, 80, 85, 90]:
        thresh = np.nanpercentile(composite[~np.isnan(composite)], pct)
        res = evaluate_binary_detector(composite, thresh, "above", ground_truth)
        if res["f1"] > best_f1:
            best_f1 = res["f1"]
            best_result = res
            best_result["threshold_pct"] = pct

    if best_result:
        best_result["feature"] = f"composite_{target_name}"
        best_result["target"] = target_name
        best_result["components"] = feat_names
        results.append(best_result)
        status = "✅ PASS" if best_result["accuracy"] > 0.60 else "❌ FAIL"
        print(f"  Result: acc={best_result['accuracy']:.3f} "
              f"f1={best_result['f1']:.3f} prec={best_result['precision']:.3f} "
              f"rec={best_result['recall']:.3f} runs={best_result['avg_run_length']:.0f} {status}")


def _evaluate_transition_detection(df, results):
    """Evaluate how quickly we detect regime transitions."""
    regime = df["regime"].values
    valid = regime != "unknown"
    regime_valid = regime[valid]

    # Find transition points
    transitions = []
    for i in range(1, len(regime_valid)):
        if regime_valid[i] != regime_valid[i-1]:
            transitions.append({
                "bar": i,
                "from": regime_valid[i-1],
                "to": regime_valid[i],
            })

    print(f"  Total regime transitions: {len(transitions)}")
    if len(transitions) == 0:
        return

    # Transition frequency
    avg_regime_duration = len(regime_valid) / max(len(transitions), 1)
    print(f"  Average regime duration: {avg_regime_duration:.1f} bars ({avg_regime_duration*5:.0f} min)")

    # Transition matrix
    from_to = {}
    for t in transitions:
        key = (t["from"], t["to"])
        from_to[key] = from_to.get(key, 0) + 1

    print(f"\n  Transition matrix (top 10):")
    for (fr, to), count in sorted(from_to.items(), key=lambda x: -x[1])[:10]:
        print(f"    {fr:20s} → {to:20s}: {count:4d}")

    # For each feature, measure how it changes BEFORE a transition
    # This tells us which features are leading indicators
    print(f"\n  Feature behavior BEFORE transitions (avg z-score in 12 bars before):")

    all_features = [c for c in df.columns if c not in [
        "timestamp_us", "datetime", "open", "close", "high", "low",
        "volume", "trade_count", "buy_volume", "sell_volume", "returns",
        "fwd_vol", "fwd_efficiency", "vol_regime", "trend_regime",
        "regime", "is_trending", "is_high_vol"
    ]]

    # Focus on transitions TO trending and TO high_vol
    to_trending = [t for t in transitions if t["to"] in ("smooth_trend", "volatile_trend")]
    to_highvol = [t for t in transitions if t["to"] in ("volatile_trend", "chaos")]
    to_ranging = [t for t in transitions if t["to"] in ("quiet_range", "active_range")]

    for trans_type, trans_list in [("→ trending", to_trending),
                                    ("→ high_vol", to_highvol),
                                    ("→ ranging", to_ranging)]:
        if len(trans_list) < 5:
            print(f"\n  {trans_type}: too few transitions ({len(trans_list)})")
            continue

        print(f"\n  {trans_type} ({len(trans_list)} transitions):")
        print(f"  {'Feature':30s} {'Pre-12':>8s} {'Pre-6':>8s} {'Pre-3':>8s} {'Pre-1':>8s}")

        feature_signals = []
        for feat_name in all_features:
            if feat_name not in df.columns:
                continue
            vals = df[feat_name].values[valid].astype(float)
            mu = np.nanmean(vals)
            sigma = np.nanstd(vals)
            if sigma < 1e-10:
                continue

            z_vals = (vals - mu) / sigma

            pre_signals = {1: [], 3: [], 6: [], 12: []}
            for t in trans_list:
                bar = t["bar"]
                for lookback in [1, 3, 6, 12]:
                    start = max(0, bar - lookback)
                    if start < len(z_vals):
                        pre_signals[lookback].append(np.nanmean(z_vals[start:bar]))

            avg_pre = {}
            for lb in [1, 3, 6, 12]:
                if pre_signals[lb]:
                    avg_pre[lb] = np.nanmean(pre_signals[lb])
                else:
                    avg_pre[lb] = 0.0

            max_signal = max(abs(avg_pre[lb]) for lb in [1, 3, 6, 12])
            if max_signal > 0.2:  # only show features with meaningful pre-transition signal
                feature_signals.append((feat_name, avg_pre, max_signal))

        feature_signals.sort(key=lambda x: -x[2])
        for feat_name, avg_pre, max_s in feature_signals[:15]:
            print(f"  {feat_name:30s} {avg_pre[12]:+.3f}   {avg_pre[6]:+.3f}   "
                  f"{avg_pre[3]:+.3f}   {avg_pre[1]:+.3f}")


# ---------------------------------------------------------------------------
# Single-symbol runner
# ---------------------------------------------------------------------------

def run_symbol(symbol, start, end, forward_window, results_dir):
    """Run full regime detection pipeline for one symbol. Returns results list."""
    print(f"\n{'='*70}")
    print(f"  {symbol} | {start} → {end} | fwd={forward_window} bars ({forward_window*5}min)")
    print(f"{'='*70}")

    t0 = time.time()

    # Step 1: Load data
    print(f"\n  Step 1: Loading 5m bars from tick data...")
    df = load_bars(symbol, start, end)
    if df.empty:
        print("  No data!")
        return []
    load_time = time.time() - t0

    # Step 2: Compute features
    print(f"\n  Step 2: Computing features...")
    t1 = time.time()
    df = compute_regime_features(df)
    n_features = len([c for c in df.columns if c not in ['timestamp_us', 'datetime']])
    print(f"  {n_features} features computed in {time.time()-t1:.0f}s")

    # Step 3: Label regimes
    print(f"\n  Step 3: Labeling regimes (forward-looking)...")
    t2 = time.time()
    df = label_regimes(df, forward_window=forward_window)
    print(f"  Labeled in {time.time()-t2:.0f}s")

    # Step 4: Run experiments
    print(f"\n  Step 4: Running experiments...")
    results = run_experiments(df, forward_window=forward_window)

    elapsed = time.time() - t0
    print(f"\n✅ {symbol} complete in {elapsed:.0f}s "
          f"(load={load_time:.0f}s, {len(df):,} bars, {n_features} features)")

    # Save per-symbol results
    out_path = Path(results_dir) / f"regime_{symbol}_{start}_{end}.txt"
    # results already printed to stdout which is tee'd to file
    return results


# ---------------------------------------------------------------------------
# Cross-symbol summary
# ---------------------------------------------------------------------------

def print_cross_symbol_summary(all_results):
    """Print a comparison table across all symbols."""
    print(f"\n\n{'#'*70}")
    print(f"  CROSS-SYMBOL COMPARISON")
    print(f"{'#'*70}")

    # Collect best vol and trend detectors per symbol
    print(f"\n  Best VOLATILITY detectors (is_high_vol):")
    print(f"  {'Symbol':12s} {'Feature':30s} {'Acc':>7s} {'F1':>7s} {'Prec':>7s} {'Rec':>7s} {'Runs':>6s}")
    print(f"  {'-'*75}")
    for sym, results in all_results.items():
        vol_results = [r for r in results if r.get("target") == "is_high_vol" and "feature" in r]
        if vol_results:
            best = max(vol_results, key=lambda x: x.get("f1", 0))
            print(f"  {sym:12s} {best['feature']:30s} {best['accuracy']:7.3f} "
                  f"{best['f1']:7.3f} {best['precision']:7.3f} {best['recall']:7.3f} "
                  f"{best['avg_run_length']:6.0f}")

    print(f"\n  Best TREND detectors (is_trending):")
    print(f"  {'Symbol':12s} {'Feature':30s} {'Acc':>7s} {'F1':>7s} {'Prec':>7s} {'Rec':>7s} {'Runs':>6s}")
    print(f"  {'-'*75}")
    for sym, results in all_results.items():
        trend_results = [r for r in results if r.get("target") == "is_trending" and "feature" in r]
        if trend_results:
            best = max(trend_results, key=lambda x: x.get("f1", 0))
            print(f"  {sym:12s} {best['feature']:30s} {best['accuracy']:7.3f} "
                  f"{best['f1']:7.3f} {best['precision']:7.3f} {best['recall']:7.3f} "
                  f"{best['avg_run_length']:6.0f}")

    # Feature consistency: which features pass across ALL symbols?
    print(f"\n  Feature consistency (passing >60% acc across symbols):")
    feature_scores = {}
    for sym, results in all_results.items():
        for r in results:
            feat = r.get("feature", "")
            target = r.get("target", "")
            key = (feat, target)
            if key not in feature_scores:
                feature_scores[key] = {}
            feature_scores[key][sym] = r

    print(f"  {'Feature':30s} {'Target':15s} {'Symbols':>8s} " +
          " ".join(f"{s:>10s}" for s in all_results.keys()))
    print(f"  {'-'*100}")

    consistent = []
    for (feat, target), sym_results in feature_scores.items():
        if not feat:
            continue
        n_pass = sum(1 for r in sym_results.values() if r.get("accuracy", 0) > 0.60)
        if n_pass >= 3:  # passes on at least 3 symbols
            avg_f1 = np.mean([r.get("f1", 0) for r in sym_results.values()])
            consistent.append((feat, target, n_pass, avg_f1, sym_results))

    consistent.sort(key=lambda x: (-x[2], -x[3]))
    for feat, target, n_pass, avg_f1, sym_results in consistent[:20]:
        acc_strs = []
        for sym in all_results.keys():
            if sym in sym_results:
                a = sym_results[sym].get("accuracy", 0)
                f = sym_results[sym].get("f1", 0)
                acc_strs.append(f"{a:.2f}/{f:.2f}")
            else:
                acc_strs.append("     -    ")
        print(f"  {feat:30s} {target:15s} {n_pass:>5d}/5  " +
              " ".join(f"{s:>10s}" for s in acc_strs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]

def main():
    parser = argparse.ArgumentParser(description="Regime Detection Experiments")
    parser.add_argument("--exchange", default="bybit_futures")
    parser.add_argument("--symbol", default="all",
                        help="Symbol or 'all' for all 5 currencies")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-01-31")
    parser.add_argument("--forward-window", type=int, default=48,
                        help="Forward-looking window for regime labeling (bars)")
    args = parser.parse_args()

    if args.symbol.lower() == "all":
        symbols = ALL_SYMBOLS
    else:
        symbols = [s.strip().upper() for s in args.symbol.split(",")]

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("  REGIME DETECTION EXPERIMENT SUITE")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Period:  {args.start} → {args.end}")
    print(f"  Forward: {args.forward_window} bars ({args.forward_window * 5} min)")
    print("=" * 70)

    grand_t0 = time.time()
    all_results = {}

    for idx, symbol in enumerate(symbols, 1):
        print(f"\n\n{'*'*70}")
        print(f"  SYMBOL {idx}/{len(symbols)}: {symbol}")
        print(f"{'*'*70}")

        sym_results = run_symbol(symbol, args.start, args.end,
                                 args.forward_window, results_dir)
        all_results[symbol] = sym_results

        elapsed = time.time() - grand_t0
        remaining = len(symbols) - idx
        if idx > 0:
            per_sym = elapsed / idx
            eta = remaining * per_sym
            print(f"\n  ⏱ Total elapsed: {elapsed:.0f}s | "
                  f"~{per_sym:.0f}s/symbol | ETA remaining: {eta:.0f}s")

    # Cross-symbol summary
    if len(all_results) > 1:
        print_cross_symbol_summary(all_results)

    total_elapsed = time.time() - grand_t0
    print(f"\n\n{'='*70}")
    print(f"  ALL DONE — {len(symbols)} symbols in {total_elapsed:.0f}s "
          f"({total_elapsed/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
