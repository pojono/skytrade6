#!/usr/bin/env python3
"""
ML Range Prediction — Phase 1 & Phase 2.

Phase 1: Direct Range Prediction
  - Predict fwd_range (high-low)/close over next 1h and 4h
  - Compare vs vol-derived range estimate
  - Ridge + LGBM regression

Phase 2: Quantile Regression
  - Predict P10, P50, P90 of forward range
  - Evaluate calibration: how often does actual range fall within predicted band?
  - LGBM with quantile loss + sklearn QuantileRegressor

All models are CPU-only with walk-forward time-series CV.
"""

import sys
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, QuantileRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from regime_detection import load_bars, compute_regime_features

PARQUET_DIR = Path("./parquet")

HORIZONS = {
    12:  "1h",
    48:  "4h",
}

ALL_FEATURES = [
    "parkvol_1h", "parkvol_2h", "parkvol_4h", "parkvol_8h", "parkvol_24h",
    "rvol_1h", "rvol_2h", "rvol_4h", "rvol_8h", "rvol_24h",
    "vol_ratio_1h_24h", "vol_ratio_2h_24h", "vol_ratio_1h_8h",
    "vol_accel_1h", "vol_accel_4h",
    "vol_sma_24h", "vol_ratio_bar",
    "trade_intensity_ratio",
    "parkinson_vol",
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
# Helpers
# ---------------------------------------------------------------------------

def get_feature_matrix(df, feature_list):
    """Extract feature matrix, dropping unavailable columns."""
    available = [f for f in feature_list if f in df.columns]
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        print(f"  Warning: {len(missing)} features not found: {missing[:5]}...")
    X = df[available].copy()
    return X, available


def walk_forward_splits(n_samples, n_splits=5, min_train=10000, min_test=2000):
    """Walk-forward expanding-window splits."""
    test_size = max(n_samples // (n_splits + 1), min_test)
    splits = []
    for i in range(n_splits):
        test_end = n_samples - (n_splits - 1 - i) * test_size
        test_start = test_end - test_size
        train_end = test_start
        if train_end < min_train or test_start < 0 or test_end > n_samples:
            continue
        splits.append((np.arange(0, train_end), np.arange(test_start, test_end)))
    return splits


# ---------------------------------------------------------------------------
# Labeling: compute forward range targets
# ---------------------------------------------------------------------------

def compute_range_targets(df, forward_window):
    """
    Compute forward-looking range targets for a given window.

    Targets:
      fwd_range_{N}     = (max(high) - min(low)) / close
      fwd_upside_{N}    = (max(high) - close) / close
      fwd_downside_{N}  = (close - min(low)) / close
      fwd_vol_{N}       = std(returns) over next N bars (for comparison)
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    ret = df["returns"].values
    n = len(df)

    fwd_range = np.full(n, np.nan)
    fwd_upside = np.full(n, np.nan)
    fwd_downside = np.full(n, np.nan)
    fwd_vol = np.full(n, np.nan)

    for i in range(n - forward_window):
        future_high = np.max(high[i+1:i+1+forward_window])
        future_low = np.min(low[i+1:i+1+forward_window])
        c = close[i]

        fwd_range[i] = (future_high - future_low) / c
        fwd_upside[i] = (future_high - c) / c
        fwd_downside[i] = (c - future_low) / c
        fwd_vol[i] = np.std(ret[i+1:i+1+forward_window])

    sfx = f"_{forward_window}"
    df[f"fwd_range{sfx}"] = fwd_range
    df[f"fwd_upside{sfx}"] = fwd_upside
    df[f"fwd_downside{sfx}"] = fwd_downside
    df[f"fwd_vol{sfx}"] = fwd_vol

    # Derived: asymmetry ratio (upside / range), 0.5 = symmetric
    asym = np.full(n, np.nan)
    valid = fwd_range > 1e-10
    asym[valid] = fwd_upside[valid] / fwd_range[valid]
    df[f"fwd_asymmetry{sfx}"] = asym

    return df


# ---------------------------------------------------------------------------
# Phase 1: Direct Range Prediction
# ---------------------------------------------------------------------------

def phase1_range_prediction(df, feature_list, horizons, n_splits=5):
    """
    Phase 1: Predict fwd_range directly with Ridge + LGBM.
    Compare vs vol-derived range estimate.
    """
    print(f"\n{'#'*70}")
    print(f"  PHASE 1: Direct Range Prediction")
    print(f"{'#'*70}")

    X_full, feat_names = get_feature_matrix(df, feature_list)

    all_results = {}

    for fwd, label in sorted(horizons.items()):
        print(f"\n  {'='*60}")
        print(f"  Horizon: {fwd} bars ({label})")
        print(f"  {'='*60}")

        col_range = f"fwd_range_{fwd}"
        col_vol = f"fwd_vol_{fwd}"

        y_range = df[col_range].values
        y_vol = df[col_vol].values
        valid = X_full.notna().all(axis=1).values & ~np.isnan(y_range) & ~np.isnan(y_vol)
        X = X_full.values[valid]
        yr = y_range[valid]
        yv = y_vol[valid]

        # Stats
        print(f"  Samples: {len(yr):,}")
        print(f"  fwd_range: mean={yr.mean():.6f} std={yr.std():.6f} "
              f"median={np.median(yr):.6f}")
        print(f"  fwd_vol:   mean={yv.mean():.6f} std={yv.std():.6f}")
        print(f"  range/vol ratio: {yr.mean()/yv.mean():.2f}x")

        # Correlation between range and vol
        rv_corr = np.corrcoef(yr, yv)[0, 1]
        print(f"  Correlation(range, vol): {rv_corr:.4f}")

        splits = walk_forward_splits(len(yr), n_splits=n_splits)
        print(f"  Walk-forward splits: {len(splits)}")

        regressors = {
            "Ridge": Ridge(alpha=1.0),
            "LGBM": lgb.LGBMRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=50, n_jobs=-1, random_state=42, verbose=-1
            ),
        }

        horizon_results = {}

        # --- A) Predict range directly ---
        print(f"\n  A) Direct range prediction:")
        for name, reg in regressors.items():
            t0 = time.time()
            fold_metrics = []

            for fold_i, (train_idx, test_idx) in enumerate(splits):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = yr[train_idx], yr[test_idx]

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                reg_copy = type(reg)(**reg.get_params())
                reg_copy.fit(X_train_s, y_train)
                y_pred = reg_copy.predict(X_test_s)

                r2 = r2_score(y_test, y_pred)
                corr = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 2 else 0
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                # Practical metric: if we set grid width = predicted range,
                # what % of actual ranges fall within our prediction?
                coverage = np.mean(y_test <= y_pred)  # % of bars where actual <= predicted
                # Overshoot: how much wider is our prediction vs actual on average?
                overshoot = np.mean(y_pred / np.maximum(y_test, 1e-10)) - 1.0

                fold_metrics.append({
                    "r2": r2, "corr": corr, "mae": mae, "rmse": rmse,
                    "coverage": coverage, "overshoot": overshoot,
                })

                print(f"    {name} fold {fold_i+1}/{len(splits)}: "
                      f"r2={r2:.3f} corr={corr:.3f} mae={mae:.6f} "
                      f"coverage={coverage:.3f} "
                      f"[train={len(train_idx):,} test={len(test_idx):,}]", flush=True)

            elapsed = time.time() - t0
            avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
            std = {k: np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
            horizon_results[f"range_{name}"] = {"avg": avg, "std": std, "time": elapsed}

            print(f"  → {name}: r2={avg['r2']:.3f}±{std['r2']:.3f} "
                  f"corr={avg['corr']:.3f}±{std['corr']:.3f} "
                  f"mae={avg['mae']:.6f} coverage={avg['coverage']:.3f} ({elapsed:.1f}s)", flush=True)

        # --- B) Vol-derived range estimate (baseline) ---
        print(f"\n  B) Vol-derived range estimate (baseline):")
        t0 = time.time()
        fold_metrics = []
        reg_vol = Ridge(alpha=1.0)

        for fold_i, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train_v, y_test_v = yv[train_idx], yv[test_idx]
            y_test_r = yr[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            reg_copy = Ridge(alpha=1.0)
            reg_copy.fit(X_train_s, y_train_v)
            y_pred_vol = reg_copy.predict(X_test_s)

            # Convert predicted vol to estimated range: range ≈ vol × k × √N
            # Fit k from training data
            train_range = yr[train_idx]
            train_vol = yv[train_idx]
            valid_tv = train_vol > 1e-10
            k = np.median(train_range[valid_tv] / train_vol[valid_tv])
            y_pred_range = y_pred_vol * k

            r2 = r2_score(y_test_r, y_pred_range)
            corr = np.corrcoef(y_test_r, y_pred_range)[0, 1] if len(y_test_r) > 2 else 0
            mae = mean_absolute_error(y_test_r, y_pred_range)
            coverage = np.mean(y_test_r <= y_pred_range)
            overshoot = np.mean(y_pred_range / np.maximum(y_test_r, 1e-10)) - 1.0

            fold_metrics.append({
                "r2": r2, "corr": corr, "mae": mae,
                "coverage": coverage, "overshoot": overshoot, "k": k,
            })

            print(f"    Vol→Range fold {fold_i+1}/{len(splits)}: "
                  f"r2={r2:.3f} corr={corr:.3f} mae={mae:.6f} "
                  f"coverage={coverage:.3f} k={k:.2f} "
                  f"[train={len(train_idx):,} test={len(test_idx):,}]", flush=True)

        elapsed = time.time() - t0
        avg = {k_: np.mean([m[k_] for m in fold_metrics]) for k_ in fold_metrics[0]}
        std = {k_: np.std([m[k_] for m in fold_metrics]) for k_ in fold_metrics[0]}
        horizon_results["vol_derived"] = {"avg": avg, "std": std, "time": elapsed}

        print(f"  → Vol→Range: r2={avg['r2']:.3f}±{std['r2']:.3f} "
              f"corr={avg['corr']:.3f}±{std['corr']:.3f} "
              f"mae={avg['mae']:.6f} coverage={avg['coverage']:.3f} "
              f"k={avg['k']:.2f} ({elapsed:.1f}s)", flush=True)

        # --- C) Upside/Downside prediction ---
        print(f"\n  C) Upside vs Downside prediction (Ridge):")
        for target_name, col_suffix in [("upside", f"fwd_upside_{fwd}"),
                                         ("downside", f"fwd_downside_{fwd}")]:
            yt = df[col_suffix].values[valid]
            t0 = time.time()
            fold_metrics = []

            for fold_i, (train_idx, test_idx) in enumerate(splits):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = yt[train_idx], yt[test_idx]

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                reg_copy = Ridge(alpha=1.0)
                reg_copy.fit(X_train_s, y_train)
                y_pred = reg_copy.predict(X_test_s)

                r2 = r2_score(y_test, y_pred)
                corr = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 2 else 0
                mae = mean_absolute_error(y_test, y_pred)

                fold_metrics.append({"r2": r2, "corr": corr, "mae": mae})

                print(f"    {target_name} fold {fold_i+1}/{len(splits)}: "
                      f"r2={r2:.3f} corr={corr:.3f} mae={mae:.6f} "
                      f"[train={len(train_idx):,} test={len(test_idx):,}]", flush=True)

            elapsed = time.time() - t0
            avg = {k_: np.mean([m[k_] for m in fold_metrics]) for k_ in fold_metrics[0]}
            std = {k_: np.std([m[k_] for m in fold_metrics]) for k_ in fold_metrics[0]}
            horizon_results[target_name] = {"avg": avg, "std": std, "time": elapsed}

            print(f"  → {target_name}: r2={avg['r2']:.3f}±{std['r2']:.3f} "
                  f"corr={avg['corr']:.3f}±{std['corr']:.3f} "
                  f"mae={avg['mae']:.6f} ({elapsed:.1f}s)", flush=True)

        # --- D) Asymmetry prediction ---
        print(f"\n  D) Asymmetry prediction (upside/range ratio, Ridge):")
        ya = df[f"fwd_asymmetry_{fwd}"].values[valid]
        t0 = time.time()
        fold_metrics = []

        for fold_i, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = ya[train_idx], ya[test_idx]

            # Remove NaN asymmetry
            train_valid = ~np.isnan(y_train)
            test_valid = ~np.isnan(y_test)
            if train_valid.sum() < 1000 or test_valid.sum() < 100:
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train[train_valid])
            X_test_s = scaler.transform(X_test[test_valid])

            reg_copy = Ridge(alpha=1.0)
            reg_copy.fit(X_train_s, y_train[train_valid])
            y_pred = reg_copy.predict(X_test_s)

            r2 = r2_score(y_test[test_valid], y_pred)
            corr = np.corrcoef(y_test[test_valid], y_pred)[0, 1] if len(y_pred) > 2 else 0
            mae = mean_absolute_error(y_test[test_valid], y_pred)

            # Directional accuracy: does predicted asymmetry > 0.5 match actual > 0.5?
            dir_acc = np.mean((y_pred > 0.5) == (y_test[test_valid] > 0.5))

            fold_metrics.append({"r2": r2, "corr": corr, "mae": mae, "dir_acc": dir_acc})

            print(f"    asymmetry fold {fold_i+1}/{len(splits)}: "
                  f"r2={r2:.3f} corr={corr:.3f} dir_acc={dir_acc:.3f} "
                  f"[train={train_valid.sum():,} test={test_valid.sum():,}]", flush=True)

        if fold_metrics:
            elapsed = time.time() - t0
            avg = {k_: np.mean([m[k_] for m in fold_metrics]) for k_ in fold_metrics[0]}
            std = {k_: np.std([m[k_] for m in fold_metrics]) for k_ in fold_metrics[0]}
            horizon_results["asymmetry"] = {"avg": avg, "std": std, "time": elapsed}

            print(f"  → asymmetry: r2={avg['r2']:.3f}±{std['r2']:.3f} "
                  f"corr={avg['corr']:.3f}±{std['corr']:.3f} "
                  f"dir_acc={avg['dir_acc']:.3f} ({elapsed:.1f}s)", flush=True)

        # --- E) Feature importance for range prediction ---
        print(f"\n  E) Feature importance (LGBM, range target):")
        split_idx = int(len(yr) * 0.7)
        X_train_fi, X_test_fi = X[:split_idx], X[split_idx:]
        y_train_fi = yr[:split_idx]

        scaler = StandardScaler()
        X_train_fi_s = scaler.fit_transform(X_train_fi)

        model = lgb.LGBMRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=50, n_jobs=-1, random_state=42, verbose=-1
        )
        model.fit(X_train_fi_s, y_train_fi)

        importances = model.feature_importances_
        feat_imp = sorted(zip(feat_names, importances), key=lambda x: -x[1])
        total_imp = sum(importances)
        cumulative = 0
        print(f"  Top 15 features for range prediction:")
        for rank, (feat, imp) in enumerate(feat_imp[:15], 1):
            pct = imp / total_imp * 100
            cumulative += pct
            print(f"    {rank:2d}. {feat:30s} {pct:5.1f}% (cum {cumulative:5.1f}%)")

        horizon_results["feat_importance"] = feat_imp

        # Summary
        print(f"\n  {'='*60}")
        print(f"  PHASE 1 SUMMARY — {label} horizon")
        print(f"  {'='*60}")
        print(f"  {'Method':25s} {'R²':>8s} {'Corr':>8s} {'MAE':>10s} {'Coverage':>10s}")
        print(f"  {'-'*65}")
        for method_key, method_label in [
            ("range_Ridge", "Range (Ridge)"),
            ("range_LGBM", "Range (LGBM)"),
            ("vol_derived", "Vol→Range (Ridge)"),
            ("upside", "Upside (Ridge)"),
            ("downside", "Downside (Ridge)"),
        ]:
            if method_key in horizon_results:
                a = horizon_results[method_key]["avg"]
                cov = a.get("coverage", float("nan"))
                print(f"  {method_label:25s} {a['r2']:8.3f} {a['corr']:8.3f} "
                      f"{a['mae']:10.6f} {cov:10.3f}")
        if "asymmetry" in horizon_results:
            a = horizon_results["asymmetry"]["avg"]
            print(f"  {'Asymmetry (Ridge)':25s} {a['r2']:8.3f} {a['corr']:8.3f} "
                  f"{a['mae']:10.6f} {a['dir_acc']:10.3f}")

        all_results[fwd] = horizon_results

    return all_results


# ---------------------------------------------------------------------------
# Phase 2: Quantile Regression
# ---------------------------------------------------------------------------

def phase2_quantile_regression(df, feature_list, horizons, n_splits=5):
    """
    Phase 2: Predict P10, P50, P90 of forward range.
    Evaluate calibration: how often does actual range fall within bands?
    """
    print(f"\n\n{'#'*70}")
    print(f"  PHASE 2: Quantile Regression")
    print(f"{'#'*70}")

    X_full, feat_names = get_feature_matrix(df, feature_list)

    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    all_results = {}

    for fwd, label in sorted(horizons.items()):
        print(f"\n  {'='*60}")
        print(f"  Horizon: {fwd} bars ({label})")
        print(f"  {'='*60}")

        col_range = f"fwd_range_{fwd}"
        y = df[col_range].values
        valid = X_full.notna().all(axis=1).values & ~np.isnan(y)
        X = X_full.values[valid]
        yr = y[valid]

        splits = walk_forward_splits(len(yr), n_splits=n_splits)
        print(f"  Samples: {len(yr):,} | Splits: {len(splits)}")

        # --- A) LGBM quantile regression ---
        print(f"\n  A) LGBM Quantile Regression:")
        horizon_results = {}

        for q in quantiles:
            t0 = time.time()
            fold_metrics = []

            for fold_i, (train_idx, test_idx) in enumerate(splits):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = yr[train_idx], yr[test_idx]

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                model = lgb.LGBMRegressor(
                    objective="quantile", alpha=q,
                    n_estimators=300, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    min_child_samples=50, n_jobs=-1, random_state=42, verbose=-1
                )
                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)

                # Calibration: what fraction of actual values are below the predicted quantile?
                actual_below = np.mean(y_test <= y_pred)
                # Ideal: actual_below ≈ q
                cal_error = abs(actual_below - q)
                mae = mean_absolute_error(y_test, y_pred)

                fold_metrics.append({
                    "actual_below": actual_below,
                    "cal_error": cal_error,
                    "mae": mae,
                    "pred_mean": y_pred.mean(),
                    "pred_std": y_pred.std(),
                })

                print(f"    Q{q:.2f} fold {fold_i+1}/{len(splits)}: "
                      f"actual_below={actual_below:.3f} (target={q:.2f}) "
                      f"cal_err={cal_error:.3f} mae={mae:.6f} "
                      f"[train={len(train_idx):,} test={len(test_idx):,}]", flush=True)

            elapsed = time.time() - t0
            avg = {k_: np.mean([m[k_] for m in fold_metrics]) for k_ in fold_metrics[0]}
            horizon_results[f"lgbm_q{q:.2f}"] = {"avg": avg, "folds": fold_metrics, "time": elapsed}

            print(f"  → Q{q:.2f}: actual_below={avg['actual_below']:.3f} "
                  f"cal_error={avg['cal_error']:.3f} "
                  f"pred_mean={avg['pred_mean']:.6f} ({elapsed:.1f}s)", flush=True)

        # --- B) Sklearn QuantileRegressor (linear, for comparison) ---
        print(f"\n  B) Linear Quantile Regression (sklearn):")
        for q in [0.10, 0.50, 0.90]:
            t0 = time.time()
            fold_metrics = []

            for fold_i, (train_idx, test_idx) in enumerate(splits):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = yr[train_idx], yr[test_idx]

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                # Subsample for speed (QuantileRegressor is slow)
                max_train = 20000
                if len(X_train_s) > max_train:
                    idx = np.random.RandomState(42).choice(len(X_train_s), max_train, replace=False)
                    X_train_sub = X_train_s[idx]
                    y_train_sub = y_train[idx]
                else:
                    X_train_sub = X_train_s
                    y_train_sub = y_train

                model = QuantileRegressor(quantile=q, alpha=0.01, solver="highs")
                model.fit(X_train_sub, y_train_sub)
                y_pred = model.predict(X_test_s)

                actual_below = np.mean(y_test <= y_pred)
                cal_error = abs(actual_below - q)
                mae = mean_absolute_error(y_test, y_pred)

                fold_metrics.append({
                    "actual_below": actual_below,
                    "cal_error": cal_error,
                    "mae": mae,
                })

                print(f"    LinQ{q:.2f} fold {fold_i+1}/{len(splits)}: "
                      f"actual_below={actual_below:.3f} (target={q:.2f}) "
                      f"cal_err={cal_error:.3f} "
                      f"[train={len(X_train_sub):,} test={len(test_idx):,}]", flush=True)

            elapsed = time.time() - t0
            avg = {k_: np.mean([m[k_] for m in fold_metrics]) for k_ in fold_metrics[0]}
            horizon_results[f"linear_q{q:.2f}"] = {"avg": avg, "time": elapsed}

            print(f"  → LinQ{q:.2f}: actual_below={avg['actual_below']:.3f} "
                  f"cal_error={avg['cal_error']:.3f} ({elapsed:.1f}s)", flush=True)

        # --- C) Band coverage analysis ---
        print(f"\n  C) Band Coverage Analysis (LGBM quantiles):")

        # Use last fold's predictions for analysis
        last_train, last_test = splits[-1]
        X_train, X_test = X[last_train], X[last_test]
        y_train, y_test = yr[last_train], yr[last_test]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        q_preds = {}
        for q in quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile", alpha=q,
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=50, n_jobs=-1, random_state=42, verbose=-1
            )
            model.fit(X_train_s, y_train)
            q_preds[q] = model.predict(X_test_s)

        # Band coverage
        bands = [
            ("P10-P90", 0.10, 0.90, 0.80),
            ("P25-P75", 0.25, 0.75, 0.50),
        ]
        for band_name, q_lo, q_hi, expected_coverage in bands:
            lo = q_preds[q_lo]
            hi = q_preds[q_hi]
            within = np.mean((y_test >= lo) & (y_test <= hi))
            avg_width = np.mean(hi - lo)
            avg_actual = np.mean(y_test)
            width_ratio = avg_width / avg_actual

            print(f"  {band_name}: coverage={within:.3f} (expected={expected_coverage:.2f}) "
                  f"avg_width={avg_width:.6f} ({width_ratio*100:.1f}% of mean range)")

        # Practical grid sizing
        print(f"\n  D) Practical Grid Sizing (last fold, {len(y_test):,} bars):")
        p50 = q_preds[0.50]
        p90 = q_preds[0.90]

        # If grid width = P50 prediction
        p50_coverage = np.mean(y_test <= p50)
        # If grid width = P90 prediction
        p90_coverage = np.mean(y_test <= p90)

        # Convert to dollar amounts (assume BTC ~$100k for illustration)
        price = df["close"].values[valid][-len(y_test):]
        p50_dollars = p50 * price
        p90_dollars = p90 * price
        actual_dollars = y_test * price

        print(f"  Grid width = P50 prediction:")
        print(f"    Coverage: {p50_coverage:.3f} (range stays within grid {p50_coverage*100:.1f}% of time)")
        print(f"    Avg predicted range: ${p50_dollars.mean():.0f}")
        print(f"    Avg actual range:    ${actual_dollars.mean():.0f}")

        print(f"  Grid width = P90 prediction:")
        print(f"    Coverage: {p90_coverage:.3f} (range stays within grid {p90_coverage*100:.1f}% of time)")
        print(f"    Avg predicted range: ${p90_dollars.mean():.0f}")

        # Adaptive vs fixed grid comparison
        fixed_p50 = np.median(yr)  # fixed grid at overall median range
        fixed_coverage = np.mean(y_test <= fixed_p50)
        fixed_dollars = fixed_p50 * price

        print(f"\n  Fixed grid (overall median range):")
        print(f"    Coverage: {fixed_coverage:.3f}")
        print(f"    Fixed width: ${fixed_dollars.mean():.0f}")

        # Efficiency: adaptive grid is narrower when vol is low, wider when high
        low_vol_mask = y_test < np.percentile(yr, 25)
        high_vol_mask = y_test > np.percentile(yr, 75)

        if low_vol_mask.sum() > 0 and high_vol_mask.sum() > 0:
            print(f"\n  Adaptive grid efficiency:")
            print(f"    During low-vol:  adaptive P50=${(p50*price)[low_vol_mask].mean():.0f} "
                  f"vs fixed=${fixed_dollars[low_vol_mask].mean():.0f} "
                  f"(saves ${(fixed_dollars[low_vol_mask].mean() - (p50*price)[low_vol_mask].mean()):.0f})")
            print(f"    During high-vol: adaptive P50=${(p50*price)[high_vol_mask].mean():.0f} "
                  f"vs fixed=${fixed_dollars[high_vol_mask].mean():.0f} "
                  f"(wider by ${((p50*price)[high_vol_mask].mean() - fixed_dollars[high_vol_mask].mean()):.0f})")

        # Summary
        print(f"\n  {'='*60}")
        print(f"  PHASE 2 SUMMARY — {label} horizon")
        print(f"  {'='*60}")
        print(f"  {'Quantile':>10s} {'LGBM cal':>10s} {'Linear cal':>10s} {'Target':>10s}")
        print(f"  {'-'*45}")
        for q in quantiles:
            lgbm_key = f"lgbm_q{q:.2f}"
            lin_key = f"linear_q{q:.2f}"
            lgbm_ab = horizon_results[lgbm_key]["avg"]["actual_below"] if lgbm_key in horizon_results else float("nan")
            lin_ab = horizon_results.get(lin_key, {}).get("avg", {}).get("actual_below", float("nan"))
            print(f"  P{q*100:4.0f}      {lgbm_ab:10.3f} {lin_ab:10.3f} {q:10.2f}")

        all_results[fwd] = horizon_results

    return all_results


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_range_experiments(symbol, start, end):
    """Run all range prediction experiments for one symbol."""
    print(f"\n\n{'='*70}")
    print(f"  RANGE PREDICTION EXPERIMENTS: {symbol}")
    print(f"  {start} → {end}")
    print(f"  Horizons: {', '.join(f'{v} ({k} bars)' for k, v in sorted(HORIZONS.items()))}")
    print(f"{'='*70}")

    t0 = time.time()

    # Step 1: Load data
    print(f"\n  Step 1: Loading 5m bars...")
    df = load_bars(symbol, start, end)
    if df.empty:
        print("  No data!")
        return {}
    print(f"  Loaded {len(df):,} bars in {time.time()-t0:.0f}s")

    # Step 2: Compute features
    print(f"\n  Step 2: Computing features...")
    t1 = time.time()
    df = compute_regime_features(df)
    n_features = len([c for c in df.columns if c not in ['timestamp_us', 'datetime']])
    print(f"  {n_features} features in {time.time()-t1:.0f}s")

    # Step 3: Compute range targets for all horizons
    print(f"\n  Step 3: Computing range targets...")
    t2 = time.time()
    for fwd, label in sorted(HORIZONS.items()):
        df = compute_range_targets(df, fwd)
        rng = df[f"fwd_range_{fwd}"].dropna()
        asym = df[f"fwd_asymmetry_{fwd}"].dropna()
        print(f"    {label:>4s} ({fwd:3d} bars): "
              f"range mean={rng.mean():.6f} median={rng.median():.6f} | "
              f"asymmetry mean={asym.mean():.3f} (0.5=symmetric)", flush=True)
    print(f"  Targets computed in {time.time()-t2:.0f}s")

    results = {}

    # Phase 1: Direct range prediction
    results["phase1"] = phase1_range_prediction(df, ALL_FEATURES, HORIZONS)

    # Phase 2: Quantile regression
    results["phase2"] = phase2_quantile_regression(df, ALL_FEATURES, HORIZONS)

    elapsed = time.time() - t0
    print(f"\n✅ {symbol} range prediction experiments complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    return results


def main():
    parser = argparse.ArgumentParser(description="ML Range Prediction Experiments")
    parser.add_argument("--symbol", default="BTCUSDT",
                        help="Symbol to test (default: BTCUSDT)")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-01-31")
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("  ML RANGE PREDICTION EXPERIMENT")
    print(f"  Symbol:   {args.symbol}")
    print(f"  Period:   {args.start} → {args.end}")
    print(f"  Horizons: {', '.join(f'{v} ({k} bars)' for k, v in sorted(HORIZONS.items()))}")
    print(f"  Phase 1:  Direct range prediction (Ridge, LGBM)")
    print(f"  Phase 2:  Quantile regression (LGBM, Linear)")
    print("=" * 70)

    grand_t0 = time.time()
    results = run_range_experiments(args.symbol, args.start, args.end)

    total = time.time() - grand_t0
    print(f"\n\n{'='*70}")
    print(f"  ALL DONE — {args.symbol} in {total:.0f}s ({total/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
