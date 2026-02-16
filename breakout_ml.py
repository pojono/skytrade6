#!/usr/bin/env python3
"""
ML Breakout Prediction — Phase 1 & Phase 2.

Phase 1: Breakout Occurrence Prediction
  - Define breakout = price moves > K × ATR within next N bars after consolidation
  - Binary classification: will a breakout occur in the next 1h / 4h?
  - Evaluate: AUC, F1, precision/recall, calibration

Phase 2: Support/Resistance Level Strength
  - Detect S/R levels from local highs/lows
  - When price approaches a level, predict: break vs reject
  - Features: touches, volume profile, approach speed, vol compression

Feature engineering strategy:
  - Start with ~80 candidate features (existing vol + new breakout-specific)
  - Compute correlation matrix, drop highly correlated pairs (|r| > 0.90)
  - Keep only features with meaningful univariate signal (|corr with target| > 0.02)
  - Report final feature set used

All models are CPU-only with walk-forward time-series CV.
"""

import sys
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, r2_score, mean_absolute_error, classification_report,
    brier_score_loss
)
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from regime_detection import load_bars, compute_regime_features

PARQUET_DIR = Path("./parquet")


# ---------------------------------------------------------------------------
# Breakout-specific feature engineering
# ---------------------------------------------------------------------------

def compute_breakout_features(df):
    """
    Compute breakout-specific features on top of existing regime features.
    All backward-looking only.
    """
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    v = df["volume"].values.astype(float)
    ret = df["returns"].values.astype(float)

    # --- ATR at multiple timeframes ---
    tr = np.maximum(h - l,
         np.maximum(np.abs(h - np.roll(c, 1)),
                    np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    for w, label in [(12, "1h"), (24, "2h"), (48, "4h"), (96, "8h"), (288, "24h")]:
        df[f"atr_{label}"] = pd.Series(tr).rolling(w).mean().values
        # Normalized ATR (% of price)
        df[f"natr_{label}"] = df[f"atr_{label}"].values / np.maximum(c, 1e-10)

    # --- Bollinger Band features (vol compression/expansion) ---
    for w, label in [(24, "2h"), (48, "4h"), (96, "8h")]:
        sma = pd.Series(c).rolling(w).mean().values
        std = pd.Series(c).rolling(w).std().values
        bw = 2 * std / np.maximum(sma, 1e-10)  # Bollinger bandwidth
        df[f"bb_width_{label}"] = bw
        # Bandwidth percentile (how compressed vs recent history)
        bw_series = pd.Series(bw)
        df[f"bb_pctile_{label}"] = bw_series.rolling(288).rank(pct=True).values

    # --- Range compression (current range vs historical) ---
    bar_range = (h - l) / np.maximum(c, 1e-10)
    for w, label in [(12, "1h"), (48, "4h"), (288, "24h")]:
        rolling_range = pd.Series(bar_range).rolling(w).mean().values
        long_range = pd.Series(bar_range).rolling(288).mean().values
        df[f"range_compression_{label}"] = rolling_range / np.maximum(long_range, 1e-10)

    # --- Price position within recent range ---
    for w, label in [(48, "4h"), (96, "8h"), (288, "24h")]:
        rolling_high = pd.Series(h).rolling(w).max().values
        rolling_low = pd.Series(l).rolling(w).min().values
        range_size = rolling_high - rolling_low
        df[f"price_position_{label}"] = (c - rolling_low) / np.maximum(range_size, 1e-10)
        # Distance to range high/low (normalized)
        df[f"dist_to_high_{label}"] = (rolling_high - c) / np.maximum(c, 1e-10)
        df[f"dist_to_low_{label}"] = (c - rolling_low) / np.maximum(c, 1e-10)

    # --- Support/Resistance level features ---
    # Number of times price touched the current high/low zone
    for w, label in [(96, "8h"), (288, "24h")]:
        rolling_high = pd.Series(h).rolling(w).max().values
        rolling_low = pd.Series(l).rolling(w).min().values
        range_size = rolling_high - rolling_low
        touch_zone = 0.02  # within 2% of range boundary
        # Count touches near high
        near_high = (h >= rolling_high - touch_zone * range_size).astype(float)
        df[f"touches_high_{label}"] = pd.Series(near_high).rolling(w).sum().values
        # Count touches near low
        near_low = (l <= rolling_low + touch_zone * range_size).astype(float)
        df[f"touches_low_{label}"] = pd.Series(near_low).rolling(w).sum().values

    # --- Volume at extremes ---
    for w, label in [(48, "4h"), (96, "8h")]:
        rolling_high = pd.Series(h).rolling(w).max().values
        rolling_low = pd.Series(l).rolling(w).min().values
        range_size = rolling_high - rolling_low
        touch_zone = 0.05
        near_high = h >= rolling_high - touch_zone * range_size
        near_low = l <= rolling_low + touch_zone * range_size
        # Volume when near high vs average
        vol_at_high = np.where(near_high, v, np.nan)
        vol_avg = pd.Series(v).rolling(w).mean().values
        val = pd.Series(vol_at_high).rolling(w).mean().values / np.maximum(vol_avg, 1e-10)
        df[f"vol_at_high_{label}"] = np.nan_to_num(val, nan=1.0)  # 1.0 = neutral ratio
        vol_at_low = np.where(near_low, v, np.nan)
        val2 = pd.Series(vol_at_low).rolling(w).mean().values / np.maximum(vol_avg, 1e-10)
        df[f"vol_at_low_{label}"] = np.nan_to_num(val2, nan=1.0)

    # --- Approach speed (momentum toward boundary) ---
    for w, label in [(6, "30m"), (12, "1h"), (24, "2h")]:
        # Speed = recent return magnitude / ATR
        recent_ret = pd.Series(c).pct_change(w).values
        atr_ref = df["atr_4h"].values if "atr_4h" in df.columns else pd.Series(tr).rolling(48).mean().values
        df[f"approach_speed_{label}"] = np.abs(recent_ret) / np.maximum(atr_ref / np.maximum(c, 1e-10), 1e-10)

    # --- Consolidation detection ---
    # Ratio of recent range to longer-term range
    for short_w, long_w, label in [(12, 96, "1h_vs_8h"), (24, 288, "2h_vs_24h"), (48, 288, "4h_vs_24h")]:
        short_range = pd.Series(h).rolling(short_w).max().values - pd.Series(l).rolling(short_w).min().values
        long_range = pd.Series(h).rolling(long_w).max().values - pd.Series(l).rolling(long_w).min().values
        df[f"consolidation_{label}"] = short_range / np.maximum(long_range, 1e-10)

    # --- Candle pattern features ---
    body = np.abs(c - df["open"].values)
    wick_upper = h - np.maximum(c, df["open"].values)
    wick_lower = np.minimum(c, df["open"].values) - l
    candle_range = h - l
    # Body ratio (how much of candle is body vs wick)
    df["body_ratio_avg_1h"] = pd.Series(body / np.maximum(candle_range, 1e-10)).rolling(12).mean().values
    df["body_ratio_avg_4h"] = pd.Series(body / np.maximum(candle_range, 1e-10)).rolling(48).mean().values
    # Wick dominance (upper wick / total wick) — rejection signal
    total_wick = wick_upper + wick_lower
    df["upper_wick_ratio_1h"] = pd.Series(wick_upper / np.maximum(total_wick, 1e-10)).rolling(12).mean().values
    df["upper_wick_ratio_4h"] = pd.Series(wick_upper / np.maximum(total_wick, 1e-10)).rolling(48).mean().values

    # --- Volume trend (is volume increasing or decreasing?) ---
    for w, label in [(12, "1h"), (48, "4h")]:
        vol_short = pd.Series(v).rolling(w).mean().values
        vol_long = pd.Series(v).rolling(w * 4).mean().values
        df[f"vol_trend_{label}"] = vol_short / np.maximum(vol_long, 1e-10)

    # --- Consecutive bars in same direction ---
    signs = np.sign(ret)
    consec = np.zeros(len(signs))
    for i in range(1, len(signs)):
        if signs[i] == signs[i-1] and signs[i] != 0:
            consec[i] = consec[i-1] + 1
        else:
            consec[i] = 0
    df["consecutive_bars"] = consec
    df["consecutive_bars_avg_1h"] = pd.Series(consec).rolling(12).mean().values

    # --- Return distribution features (skew, kurtosis) ---
    for w, label in [(48, "4h"), (96, "8h")]:
        ret_series = pd.Series(ret)
        df[f"ret_skew_{label}"] = ret_series.rolling(w).skew().values
        df[f"ret_kurtosis_{label}"] = ret_series.rolling(w).kurt().values

    # --- High-low range trend ---
    hl_range = (h - l) / np.maximum(c, 1e-10)
    df["range_trend_1h"] = pd.Series(hl_range).rolling(12).mean().values / \
                           np.maximum(pd.Series(hl_range).rolling(48).mean().values, 1e-10)
    df["range_trend_4h"] = pd.Series(hl_range).rolling(48).mean().values / \
                           np.maximum(pd.Series(hl_range).rolling(288).mean().values, 1e-10)

    return df


# ---------------------------------------------------------------------------
# All candidate features (existing + new breakout-specific)
# ---------------------------------------------------------------------------

EXISTING_VOL_FEATURES = [
    "parkvol_1h", "parkvol_2h", "parkvol_4h", "parkvol_8h", "parkvol_24h",
    "rvol_1h", "rvol_2h", "rvol_4h", "rvol_8h", "rvol_24h",
    "vol_ratio_1h_24h", "vol_ratio_2h_24h", "vol_ratio_1h_8h",
    "vol_accel_1h", "vol_accel_4h",
    "vol_sma_24h", "vol_ratio_bar",
    "trade_intensity_ratio", "parkinson_vol",
]

EXISTING_TREND_FEATURES = [
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

NEW_BREAKOUT_FEATURES = [
    # ATR
    "atr_1h", "atr_2h", "atr_4h", "atr_8h", "atr_24h",
    "natr_1h", "natr_2h", "natr_4h", "natr_8h", "natr_24h",
    # Bollinger
    "bb_width_2h", "bb_width_4h", "bb_width_8h",
    "bb_pctile_2h", "bb_pctile_4h", "bb_pctile_8h",
    # Range compression
    "range_compression_1h", "range_compression_4h", "range_compression_24h",
    # Price position
    "price_position_4h", "price_position_8h", "price_position_24h",
    "dist_to_high_4h", "dist_to_high_8h", "dist_to_high_24h",
    "dist_to_low_4h", "dist_to_low_8h", "dist_to_low_24h",
    # S/R touches
    "touches_high_8h", "touches_high_24h",
    "touches_low_8h", "touches_low_24h",
    # Volume at extremes
    "vol_at_high_4h", "vol_at_high_8h",
    "vol_at_low_4h", "vol_at_low_8h",
    # Approach speed
    "approach_speed_30m", "approach_speed_1h", "approach_speed_2h",
    # Consolidation
    "consolidation_1h_vs_8h", "consolidation_2h_vs_24h", "consolidation_4h_vs_24h",
    # Candle patterns
    "body_ratio_avg_1h", "body_ratio_avg_4h",
    "upper_wick_ratio_1h", "upper_wick_ratio_4h",
    # Volume trend
    "vol_trend_1h", "vol_trend_4h",
    # Consecutive bars
    "consecutive_bars", "consecutive_bars_avg_1h",
    # Return distribution
    "ret_skew_4h", "ret_skew_8h",
    "ret_kurtosis_4h", "ret_kurtosis_8h",
    # Range trend
    "range_trend_1h", "range_trend_4h",
]

ALL_CANDIDATE_FEATURES = EXISTING_VOL_FEATURES + EXISTING_TREND_FEATURES + NEW_BREAKOUT_FEATURES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def walk_forward_splits(n_samples, n_splits=5, min_train=10000, min_test=2000):
    """Walk-forward expanding-window splits."""
    test_size = max(n_samples // (n_splits + 1), min_test)
    splits = []
    for i in range(n_splits):
        test_end = n_samples - (n_splits - 1 - i) * test_size
        test_start = test_end - test_size
        train_end = test_start
        if train_end < min_train:
            continue
        if test_start < 0 or test_end > n_samples:
            continue
        splits.append((np.arange(0, train_end), np.arange(test_start, test_end)))
    return splits


# ---------------------------------------------------------------------------
# Breakout labeling
# ---------------------------------------------------------------------------

def label_breakouts(df, forward_window=48, atr_multiplier=2.0):
    """
    Label breakout events.
    A breakout occurs when the forward price range exceeds K × ATR.

    Returns columns:
      - fwd_max_move: max(|high - current_close|, |current_close - low|) over forward window, normalized
      - fwd_range: (max_high - min_low) / close over forward window
      - is_breakout: binary, 1 if fwd_range > atr_multiplier × current ATR
      - breakout_magnitude: fwd_range / ATR (how many ATRs the breakout was)
    """
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    n = len(df)

    fwd_range = np.full(n, np.nan)
    fwd_max_up = np.full(n, np.nan)
    fwd_max_down = np.full(n, np.nan)

    for i in range(n - forward_window):
        future_h = h[i+1:i+1+forward_window]
        future_l = l[i+1:i+1+forward_window]
        max_high = np.max(future_h)
        min_low = np.min(future_l)
        fwd_range[i] = (max_high - min_low) / max(c[i], 1e-10)
        fwd_max_up[i] = (max_high - c[i]) / max(c[i], 1e-10)
        fwd_max_down[i] = (c[i] - min_low) / max(c[i], 1e-10)

    df["fwd_range"] = fwd_range
    df["fwd_max_up"] = fwd_max_up
    df["fwd_max_down"] = fwd_max_down

    # ATR reference for breakout threshold
    atr_col = f"natr_{'4h' if forward_window >= 48 else '1h'}"
    if atr_col not in df.columns:
        atr_col = "natr_4h"
    atr_vals = df[atr_col].values

    df["breakout_threshold"] = atr_multiplier * atr_vals
    df["is_breakout"] = (fwd_range > df["breakout_threshold"].values).astype(int)
    df["breakout_magnitude"] = fwd_range / np.maximum(atr_vals, 1e-10)

    return df


def label_sr_events(df, lookback=288, touch_zone_pct=0.02, forward_window=48):
    """
    Label support/resistance test events.

    For each bar, check if price is near a recent high or low (within touch_zone_pct
    of the range). If so, label whether price broke through or rejected.

    Returns columns:
      - near_resistance: 1 if close is within touch_zone of recent high
      - near_support: 1 if close is within touch_zone of recent low
      - sr_break: 1 if price broke through the level in the next forward_window bars
      - sr_reject: 1 if price rejected (moved away from level)
    """
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    n = len(df)

    near_resistance = np.zeros(n, dtype=int)
    near_support = np.zeros(n, dtype=int)
    sr_break = np.full(n, np.nan)
    sr_level = np.full(n, np.nan)
    sr_type = np.full(n, np.nan)  # 1 = resistance test, -1 = support test

    for i in range(lookback, n - forward_window):
        past_h = h[i-lookback:i]
        past_l = l[i-lookback:i]
        recent_high = np.max(past_h)
        recent_low = np.min(past_l)
        range_size = recent_high - recent_low
        if range_size < 1e-10:
            continue

        zone = touch_zone_pct * range_size

        # Near resistance?
        if c[i] >= recent_high - zone:
            near_resistance[i] = 1
            sr_level[i] = recent_high
            sr_type[i] = 1
            # Did price break above in forward window?
            future_h = h[i+1:i+1+forward_window]
            if np.max(future_h) > recent_high + zone:
                sr_break[i] = 1
            else:
                sr_break[i] = 0

        # Near support?
        if c[i] <= recent_low + zone:
            near_support[i] = 1
            sr_level[i] = recent_low
            sr_type[i] = -1
            # Did price break below in forward window?
            future_l = l[i+1:i+1+forward_window]
            if np.min(future_l) < recent_low - zone:
                sr_break[i] = 1
            else:
                sr_break[i] = 0

    df["near_resistance"] = near_resistance
    df["near_support"] = near_support
    df["sr_break"] = sr_break
    df["sr_level"] = sr_level
    df["sr_type"] = sr_type

    return df


# ---------------------------------------------------------------------------
# Feature selection: correlation filter
# ---------------------------------------------------------------------------

def select_features(X, y, max_inter_corr=0.90, min_target_corr=0.02, verbose=True):
    """
    Feature selection pipeline:
    1. Drop features with |corr with target| < min_target_corr
    2. Among remaining, drop one of each pair with |inter-corr| > max_inter_corr
       (keep the one with higher target correlation)

    Returns: list of selected feature names
    """
    n_start = X.shape[1]

    # Step 1: univariate target correlation
    target_corrs = {}
    for col in X.columns:
        valid = ~(X[col].isna() | np.isnan(y))
        if valid.sum() < 100:
            continue
        r, _ = stats.pearsonr(X[col].values[valid], y[valid])
        target_corrs[col] = abs(r)

    # Keep features with meaningful signal
    keep = [f for f, r in target_corrs.items() if r >= min_target_corr]
    dropped_weak = n_start - len(keep)

    if verbose:
        print(f"  Feature selection: {n_start} candidates")
        print(f"    Dropped {dropped_weak} with |target_corr| < {min_target_corr}")

    if len(keep) < 3:
        if verbose:
            print(f"    WARNING: only {len(keep)} features passed target corr filter, relaxing to top 20")
        # Fallback: keep top 20 by target correlation
        sorted_feats = sorted(target_corrs.items(), key=lambda x: -x[1])
        keep = [f for f, _ in sorted_feats[:20]]

    # Step 2: inter-feature correlation
    X_keep = X[keep].copy()
    corr_matrix = X_keep.corr().abs()

    # Greedy removal: for each highly correlated pair, drop the one with lower target corr
    to_drop = set()
    cols = list(X_keep.columns)
    for i in range(len(cols)):
        if cols[i] in to_drop:
            continue
        for j in range(i+1, len(cols)):
            if cols[j] in to_drop:
                continue
            if corr_matrix.loc[cols[i], cols[j]] > max_inter_corr:
                # Drop the one with lower target correlation
                if target_corrs.get(cols[i], 0) >= target_corrs.get(cols[j], 0):
                    to_drop.add(cols[j])
                else:
                    to_drop.add(cols[i])

    selected = [f for f in keep if f not in to_drop]

    if verbose:
        print(f"    Dropped {len(to_drop)} with inter-corr > {max_inter_corr}")
        print(f"    Final: {len(selected)} features")
        # Show top 10 by target correlation
        top = sorted([(f, target_corrs[f]) for f in selected], key=lambda x: -x[1])[:10]
        for f, r in top:
            print(f"      {f:35s} target_corr={r:.4f}")

    return selected, target_corrs


# ---------------------------------------------------------------------------
# Phase 1: Breakout Occurrence Prediction
# ---------------------------------------------------------------------------

def phase1_breakout_occurrence(df, horizons, all_features, verbose=True):
    """
    Binary classification: will a breakout occur in the next N bars?
    Models: Logistic Regression, LGBM
    """
    print("\n" + "#" * 70)
    print("  PHASE 1: Breakout Occurrence Prediction")
    print("#" * 70)

    for fw, label in horizons.items():
        t0 = time.time()
        print(f"\n  {'='*60}")
        print(f"  Horizon: {fw} bars ({label})")
        print(f"  {'='*60}")

        # Label breakouts for this horizon
        # ATR multiplier scales with horizon: normal range/vol ratio is ~5.6x (1h) and ~11x (4h)
        # We want breakout = significantly above normal, targeting ~20-30% breakout rate
        atr_mult = {12: 5.0, 48: 10.0}.get(fw, 5.0)
        df_h = df.copy()
        df_h = label_breakouts(df_h, forward_window=fw, atr_multiplier=atr_mult)
        print(f"  ATR multiplier: {atr_mult}x")

        # Get valid rows: drop warmup period, require target
        available = [f for f in all_features if f in df_h.columns]
        warmup = 576  # 2 days of 5m bars for longest rolling windows
        df_h = df_h.iloc[warmup:].copy()
        valid = df_h["is_breakout"].notna()
        df_valid = df_h[valid].reset_index(drop=True)
        # Fill residual NaN in features with 0 (after warmup, very few remain)
        df_valid[available] = df_valid[available].fillna(0)

        y = df_valid["is_breakout"].values
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if len(y) == 0:
            print(f"  No valid samples — skipping")
            continue
        print(f"  Samples: {len(y):,} | Breakouts: {n_pos:,} ({100*n_pos/len(y):.1f}%) | Non-breakouts: {n_neg:,}")

        # Get feature matrix
        X_raw = df_valid[available].copy()

        # Feature selection
        print()
        selected, target_corrs = select_features(X_raw, y, max_inter_corr=0.90, min_target_corr=0.02)
        X = df_valid[selected].values
        feature_names = selected

        print(f"\n  Walk-forward splits:")
        splits = walk_forward_splits(len(y), n_splits=5)
        print(f"  {len(splits)} splits, {len(y):,} samples")

        # --- Logistic Regression ---
        print(f"\n  A) Logistic Regression:")
        lr_metrics = {"auc": [], "f1": [], "prec": [], "rec": [], "brier": []}
        for fold_i, (train_idx, test_idx) in enumerate(splits):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            # Handle NaN/inf
            X_tr_s = np.nan_to_num(X_tr_s, nan=0, posinf=0, neginf=0)
            X_te_s = np.nan_to_num(X_te_s, nan=0, posinf=0, neginf=0)

            clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
            clf.fit(X_tr_s, y_tr)
            y_prob = clf.predict_proba(X_te_s)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            auc = roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else 0.5
            f1 = f1_score(y_te, y_pred, zero_division=0)
            prec = precision_score(y_te, y_pred, zero_division=0)
            rec = recall_score(y_te, y_pred, zero_division=0)
            brier = brier_score_loss(y_te, y_prob)

            lr_metrics["auc"].append(auc)
            lr_metrics["f1"].append(f1)
            lr_metrics["prec"].append(prec)
            lr_metrics["rec"].append(rec)
            lr_metrics["brier"].append(brier)

            print(f"    LR fold {fold_i+1}/{len(splits)}: AUC={auc:.3f} F1={f1:.3f} P={prec:.3f} R={rec:.3f} "
                  f"Brier={brier:.4f} [train={len(train_idx):,} test={len(test_idx):,}]")

        print(f"  → LR: AUC={np.mean(lr_metrics['auc']):.3f}±{np.std(lr_metrics['auc']):.3f} "
              f"F1={np.mean(lr_metrics['f1']):.3f}±{np.std(lr_metrics['f1']):.3f} "
              f"P={np.mean(lr_metrics['prec']):.3f} R={np.mean(lr_metrics['rec']):.3f} "
              f"Brier={np.mean(lr_metrics['brier']):.4f} ({time.time()-t0:.1f}s)")

        # --- LGBM ---
        print(f"\n  B) LightGBM:")
        t1 = time.time()
        lgb_metrics = {"auc": [], "f1": [], "prec": [], "rec": [], "brier": []}
        importances = np.zeros(len(feature_names))

        for fold_i, (train_idx, test_idx) in enumerate(splits):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # Handle NaN/inf
            X_tr = np.nan_to_num(X_tr, nan=0, posinf=0, neginf=0)
            X_te = np.nan_to_num(X_te, nan=0, posinf=0, neginf=0)

            scale_pos = n_neg / max(n_pos, 1)
            model = lgb.LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                num_leaves=31, min_child_samples=50,
                scale_pos_weight=scale_pos,
                verbose=-1, n_jobs=-1
            )
            model.fit(X_tr, y_tr)
            y_prob = model.predict_proba(X_te)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            auc = roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else 0.5
            f1 = f1_score(y_te, y_pred, zero_division=0)
            prec = precision_score(y_te, y_pred, zero_division=0)
            rec = recall_score(y_te, y_pred, zero_division=0)
            brier = brier_score_loss(y_te, y_prob)

            lgb_metrics["auc"].append(auc)
            lgb_metrics["f1"].append(f1)
            lgb_metrics["prec"].append(prec)
            lgb_metrics["rec"].append(rec)
            lgb_metrics["brier"].append(brier)
            importances += model.feature_importances_

            print(f"    LGBM fold {fold_i+1}/{len(splits)}: AUC={auc:.3f} F1={f1:.3f} P={prec:.3f} R={rec:.3f} "
                  f"Brier={brier:.4f} [train={len(train_idx):,} test={len(test_idx):,}]")

        importances /= len(splits)
        print(f"  → LGBM: AUC={np.mean(lgb_metrics['auc']):.3f}±{np.std(lgb_metrics['auc']):.3f} "
              f"F1={np.mean(lgb_metrics['f1']):.3f}±{np.std(lgb_metrics['f1']):.3f} "
              f"P={np.mean(lgb_metrics['prec']):.3f} R={np.mean(lgb_metrics['rec']):.3f} "
              f"Brier={np.mean(lgb_metrics['brier']):.4f} ({time.time()-t1:.1f}s)")

        # --- Feature importance ---
        print(f"\n  C) Feature importance (LGBM):")
        imp_sorted = sorted(zip(feature_names, importances), key=lambda x: -x[1])
        total_imp = sum(importances)
        cum = 0
        print(f"  Top 15 features for breakout prediction:")
        for rank, (fname, imp) in enumerate(imp_sorted[:15], 1):
            pct = 100 * imp / max(total_imp, 1e-10)
            cum += pct
            print(f"    {rank:3d}. {fname:35s} {pct:5.1f}% (cum {cum:5.1f}%)")

        # --- Breakout magnitude regression (for breakout bars only) ---
        print(f"\n  D) Breakout magnitude prediction (Ridge, breakout bars only):")
        bo_mask = y == 1
        if bo_mask.sum() > 1000:
            X_bo = X[bo_mask]
            y_mag = df_valid.loc[bo_mask, "breakout_magnitude"].values
            splits_bo = walk_forward_splits(len(y_mag), n_splits=5, min_train=500, min_test=100)
            mag_r2s, mag_corrs = [], []
            for fold_i, (tr, te) in enumerate(splits_bo):
                scaler = StandardScaler()
                X_tr_s = np.nan_to_num(scaler.fit_transform(X_bo[tr]), nan=0, posinf=0, neginf=0)
                X_te_s = np.nan_to_num(scaler.transform(X_bo[te]), nan=0, posinf=0, neginf=0)
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_tr_s, y_mag[tr])
                pred = ridge.predict(X_te_s)
                r2 = r2_score(y_mag[te], pred)
                corr = np.corrcoef(y_mag[te], pred)[0, 1] if np.std(pred) > 1e-10 else 0
                mag_r2s.append(r2)
                mag_corrs.append(corr)
                print(f"    fold {fold_i+1}/{len(splits_bo)}: r2={r2:.3f} corr={corr:.3f} "
                      f"[train={len(tr):,} test={len(te):,}]")
            print(f"  → Magnitude: r2={np.mean(mag_r2s):.3f}±{np.std(mag_r2s):.3f} "
                  f"corr={np.mean(mag_corrs):.3f}±{np.std(mag_corrs):.3f}")
        else:
            print(f"    Not enough breakout bars ({bo_mask.sum()}) for magnitude regression")

        # --- Summary ---
        print(f"\n  {'='*60}")
        print(f"  PHASE 1 SUMMARY — {label} horizon")
        print(f"  {'='*60}")
        print(f"  {'Model':<20s} {'AUC':>8s} {'F1':>8s} {'Prec':>8s} {'Rec':>8s} {'Brier':>8s}")
        print(f"  {'-'*60}")
        print(f"  {'Logistic Reg':<20s} {np.mean(lr_metrics['auc']):8.3f} {np.mean(lr_metrics['f1']):8.3f} "
              f"{np.mean(lr_metrics['prec']):8.3f} {np.mean(lr_metrics['rec']):8.3f} {np.mean(lr_metrics['brier']):8.4f}")
        print(f"  {'LGBM':<20s} {np.mean(lgb_metrics['auc']):8.3f} {np.mean(lgb_metrics['f1']):8.3f} "
              f"{np.mean(lgb_metrics['prec']):8.3f} {np.mean(lgb_metrics['rec']):8.3f} {np.mean(lgb_metrics['brier']):8.4f}")
        print(f"  Breakout rate: {100*n_pos/len(y):.1f}%")
        print(f"  Features used: {len(feature_names)}")
        print(f"  Time: {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Phase 2: S/R Level Strength Prediction
# ---------------------------------------------------------------------------

def phase2_sr_prediction(df, all_features, verbose=True):
    """
    When price is near a S/R level, predict break vs reject.
    Binary classification on the subset of bars near S/R levels.
    """
    print("\n" + "#" * 70)
    print("  PHASE 2: Support/Resistance Level Strength")
    print("#" * 70)

    for lookback, lb_label in [(96, "8h"), (288, "24h")]:
        for fw, fw_label in [(12, "1h"), (48, "4h")]:
            t0 = time.time()
            print(f"\n  {'='*60}")
            print(f"  S/R lookback: {lb_label} | Forward: {fw_label}")
            print(f"  {'='*60}")

            df_sr = df.copy()
            df_sr = label_sr_events(df_sr, lookback=lookback, touch_zone_pct=0.05, forward_window=fw)

            # Filter to bars near S/R levels
            near_sr = (df_sr["near_resistance"] == 1) | (df_sr["near_support"] == 1)
            valid = near_sr & df_sr["sr_break"].notna()

            df_events = df_sr[valid].reset_index(drop=True)

            # Fill residual NaN in features
            avail_feats = [f for f in all_features if f in df_events.columns]
            df_events[avail_feats] = df_events[avail_feats].fillna(0)

            if len(df_events) < 500:
                print(f"  Only {len(df_events)} S/R events — skipping")
                continue

            y = df_events["sr_break"].values.astype(int)
            n_break = int(y.sum())
            n_reject = len(y) - n_break
            print(f"  S/R events: {len(y):,} | Breaks: {n_break:,} ({100*n_break/len(y):.1f}%) | Rejects: {n_reject:,}")

            # Resistance vs support breakdown
            n_res = int(df_events["near_resistance"].sum())
            n_sup = int(df_events["near_support"].sum())
            print(f"  Near resistance: {n_res:,} | Near support: {n_sup:,}")

            # Feature selection
            available = [f for f in all_features if f in df_events.columns]
            # Add S/R-specific features
            sr_extra = ["near_resistance", "near_support"]
            for f in sr_extra:
                if f in df_events.columns and f not in available:
                    available.append(f)

            X_raw = df_events[available].copy()
            print()
            selected, target_corrs = select_features(X_raw, y, max_inter_corr=0.90, min_target_corr=0.02)

            # Ensure we have at least some features
            if len(selected) < 3:
                print(f"  Only {len(selected)} features selected — using top 15 by target corr")
                sorted_feats = sorted(target_corrs.items(), key=lambda x: -x[1])
                selected = [f for f, _ in sorted_feats[:15]]

            X = df_events[selected].values
            feature_names = selected

            splits = walk_forward_splits(len(y), n_splits=5, min_train=500, min_test=100)
            if len(splits) < 2:
                print(f"  Not enough data for walk-forward CV — skipping")
                continue
            print(f"  Walk-forward: {len(splits)} splits")

            # --- Logistic Regression ---
            print(f"\n  A) Logistic Regression:")
            lr_metrics = {"auc": [], "f1": [], "prec": [], "rec": []}
            for fold_i, (train_idx, test_idx) in enumerate(splits):
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]

                scaler = StandardScaler()
                X_tr_s = np.nan_to_num(scaler.fit_transform(X_tr), nan=0, posinf=0, neginf=0)
                X_te_s = np.nan_to_num(scaler.transform(X_te), nan=0, posinf=0, neginf=0)

                clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
                clf.fit(X_tr_s, y_tr)
                y_prob = clf.predict_proba(X_te_s)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)

                auc = roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else 0.5
                f1 = f1_score(y_te, y_pred, zero_division=0)
                prec = precision_score(y_te, y_pred, zero_division=0)
                rec = recall_score(y_te, y_pred, zero_division=0)

                lr_metrics["auc"].append(auc)
                lr_metrics["f1"].append(f1)
                lr_metrics["prec"].append(prec)
                lr_metrics["rec"].append(rec)

                print(f"    LR fold {fold_i+1}/{len(splits)}: AUC={auc:.3f} F1={f1:.3f} P={prec:.3f} R={rec:.3f} "
                      f"[train={len(train_idx):,} test={len(test_idx):,}]")

            print(f"  → LR: AUC={np.mean(lr_metrics['auc']):.3f}±{np.std(lr_metrics['auc']):.3f} "
                  f"F1={np.mean(lr_metrics['f1']):.3f} P={np.mean(lr_metrics['prec']):.3f} R={np.mean(lr_metrics['rec']):.3f}")

            # --- LGBM ---
            print(f"\n  B) LightGBM:")
            t1 = time.time()
            lgb_metrics = {"auc": [], "f1": [], "prec": [], "rec": []}
            importances = np.zeros(len(feature_names))

            for fold_i, (train_idx, test_idx) in enumerate(splits):
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]

                X_tr = np.nan_to_num(X_tr, nan=0, posinf=0, neginf=0)
                X_te = np.nan_to_num(X_te, nan=0, posinf=0, neginf=0)

                scale_pos = n_reject / max(n_break, 1)
                model = lgb.LGBMClassifier(
                    n_estimators=300, max_depth=6, learning_rate=0.05,
                    num_leaves=31, min_child_samples=20,
                    scale_pos_weight=scale_pos,
                    verbose=-1, n_jobs=-1
                )
                model.fit(X_tr, y_tr)
                y_prob = model.predict_proba(X_te)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)

                auc = roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else 0.5
                f1 = f1_score(y_te, y_pred, zero_division=0)
                prec = precision_score(y_te, y_pred, zero_division=0)
                rec = recall_score(y_te, y_pred, zero_division=0)

                lgb_metrics["auc"].append(auc)
                lgb_metrics["f1"].append(f1)
                lgb_metrics["prec"].append(prec)
                lgb_metrics["rec"].append(rec)
                importances += model.feature_importances_

                print(f"    LGBM fold {fold_i+1}/{len(splits)}: AUC={auc:.3f} F1={f1:.3f} P={prec:.3f} R={rec:.3f} "
                      f"[train={len(train_idx):,} test={len(test_idx):,}]")

            importances /= len(splits)
            print(f"  → LGBM: AUC={np.mean(lgb_metrics['auc']):.3f}±{np.std(lgb_metrics['auc']):.3f} "
                  f"F1={np.mean(lgb_metrics['f1']):.3f} P={np.mean(lgb_metrics['prec']):.3f} R={np.mean(lgb_metrics['rec']):.3f} "
                  f"({time.time()-t1:.1f}s)")

            # Feature importance
            print(f"\n  C) Feature importance (LGBM):")
            imp_sorted = sorted(zip(feature_names, importances), key=lambda x: -x[1])
            total_imp = sum(importances)
            cum = 0
            print(f"  Top 10 features for S/R break prediction:")
            for rank, (fname, imp) in enumerate(imp_sorted[:10], 1):
                pct = 100 * imp / max(total_imp, 1e-10)
                cum += pct
                print(f"    {rank:3d}. {fname:35s} {pct:5.1f}% (cum {cum:5.1f}%)")

            # --- Baseline: always predict majority class ---
            majority_rate = max(n_break, n_reject) / len(y)
            print(f"\n  Baseline (majority class): accuracy={majority_rate:.3f}")

            # --- Summary ---
            print(f"\n  {'='*60}")
            print(f"  PHASE 2 SUMMARY — S/R {lb_label} lookback, {fw_label} forward")
            print(f"  {'='*60}")
            print(f"  {'Model':<20s} {'AUC':>8s} {'F1':>8s} {'Prec':>8s} {'Rec':>8s}")
            print(f"  {'-'*52}")
            print(f"  {'Logistic Reg':<20s} {np.mean(lr_metrics['auc']):8.3f} {np.mean(lr_metrics['f1']):8.3f} "
                  f"{np.mean(lr_metrics['prec']):8.3f} {np.mean(lr_metrics['rec']):8.3f}")
            print(f"  {'LGBM':<20s} {np.mean(lgb_metrics['auc']):8.3f} {np.mean(lgb_metrics['f1']):8.3f} "
                  f"{np.mean(lgb_metrics['prec']):8.3f} {np.mean(lgb_metrics['rec']):8.3f}")
            print(f"  Break rate: {100*n_break/len(y):.1f}% | Events: {len(y):,}")
            print(f"  Features: {len(feature_names)} | Time: {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Breakout ML experiments")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-01-31")
    args = parser.parse_args()

    symbol = args.symbol
    start_date = args.start
    end_date = args.end

    horizons = {12: "1h", 48: "4h"}

    print("=" * 70)
    print(f"  ML BREAKOUT PREDICTION EXPERIMENT")
    print(f"  Symbol:   {symbol}")
    print(f"  Period:   {start_date} → {end_date}")
    print(f"  Phase 1:  Breakout occurrence (binary classification)")
    print(f"  Phase 2:  S/R level break vs reject")
    print(f"  Candidate features: {len(ALL_CANDIDATE_FEATURES)}")
    print("=" * 70)

    t_total = time.time()

    # --- Load data ---
    print(f"\n  Step 1: Loading 5m bars...")
    df = load_bars(symbol, start_date, end_date)
    print(f"  Loaded {len(df):,} bars in {time.time()-t_total:.0f}s")

    # --- Compute existing regime features ---
    print(f"\n  Step 2: Computing regime features...")
    t1 = time.time()
    df = compute_regime_features(df)
    print(f"  Regime features in {time.time()-t1:.0f}s")

    # --- Compute new breakout features ---
    print(f"\n  Step 3: Computing breakout-specific features...")
    t2 = time.time()
    df = compute_breakout_features(df)
    n_available = sum(1 for f in ALL_CANDIDATE_FEATURES if f in df.columns)
    print(f"  {n_available} features available (of {len(ALL_CANDIDATE_FEATURES)} candidates) in {time.time()-t2:.0f}s")

    # --- Phase 1 ---
    phase1_breakout_occurrence(df, horizons, ALL_CANDIDATE_FEATURES)

    # --- Phase 2 ---
    phase2_sr_prediction(df, ALL_CANDIDATE_FEATURES)

    elapsed = time.time() - t_total
    print(f"\n✅ {symbol} breakout experiments complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"\n{'='*70}")
    print(f"  ALL DONE — {symbol} in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
