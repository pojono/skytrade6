#!/usr/bin/env python3
"""
ML Multi-Horizon Volatility Prediction — Focused Experiment.

Building on v9 findings (Ridge regression is the best model), this script
tests how prediction quality changes when we vary the FORWARD LABELING WINDOW:

  - 12 bars  (1h)  — scalping / tight grid
  - 48 bars  (4h)  — medium-term grid adjustment (v9 default)
  - 144 bars (12h) — swing / wider grid
  - 288 bars (24h) — daily regime decision

For each horizon we run:
  1. Ridge regression (predict fwd_vol) — our v9 winner
  2. LGBM regression (for comparison / feature importance)
  3. Binary classification via regression threshold
  4. Feature importance comparison across horizons
  5. Cross-horizon correlation (do different horizons agree?)

All models are CPU-only with walk-forward time-series CV.
"""

import sys
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, mean_squared_error, r2_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from regime_detection import load_bars, compute_regime_features

PARQUET_DIR = Path("./parquet")
ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]

# Horizons to test (bars at 5min each)
HORIZONS = {
    12:  "1h",
    48:  "4h",
    144: "12h",
    288: "24h",
}

# Features from v9
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
    """Walk-forward expanding-window splits. No lookahead."""
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


def label_regimes_multi(df, forward_window):
    """
    Label regimes for a specific forward window.
    Returns df with fwd_vol_{window} and is_high_vol_{window} columns.
    """
    ret = df["returns"].values
    n = len(df)
    col_vol = f"fwd_vol_{forward_window}"
    col_hv = f"is_high_vol_{forward_window}"

    fwd_vol = np.full(n, np.nan)
    for i in range(n - forward_window):
        fwd_vol[i] = np.std(ret[i+1:i+1+forward_window])

    df[col_vol] = fwd_vol

    # Rolling median threshold (3-day window)
    vol_median = pd.Series(fwd_vol).rolling(288 * 3, min_periods=288).median().values
    overall_median = np.nanmedian(fwd_vol)
    vol_median = np.where(np.isnan(vol_median), overall_median, vol_median)

    is_high = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(fwd_vol[i]):
            is_high[i] = 1.0 if fwd_vol[i] > 1.5 * vol_median[i] else 0.0
    df[col_hv] = is_high

    return df


# ---------------------------------------------------------------------------
# Experiment A: Ridge regression across horizons
# ---------------------------------------------------------------------------

def experiment_regression_horizons(df, feature_list, horizons, n_splits=5):
    """
    Run Ridge + LGBM regression for each forward horizon.
    Compare R², correlation, and binary F1 across horizons.
    """
    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT A: Regression Across Forward Horizons")
    print(f"{'#'*70}")

    X_full, feat_names = get_feature_matrix(df, feature_list)

    all_results = {}

    for fwd, label in horizons.items():
        print(f"\n  --- Horizon: {fwd} bars ({label}) ---")
        col_vol = f"fwd_vol_{fwd}"

        y = df[col_vol].values
        valid = X_full.notna().all(axis=1).values & ~np.isnan(y)
        X = X_full.values[valid]
        y_clean = y[valid]

        pos_rate = (df[f"is_high_vol_{fwd}"].dropna() == 1).mean()
        print(f"  Samples: {len(y_clean):,} | fwd_vol mean={y_clean.mean():.6f} "
              f"std={y_clean.std():.6f} | high_vol rate={pos_rate:.3f}")

        splits = walk_forward_splits(len(y_clean), n_splits=n_splits)
        print(f"  Walk-forward splits: {len(splits)}")

        regressors = {
            "Ridge": Ridge(alpha=1.0),
            "LGBM_reg": lgb.LGBMRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=50, n_jobs=-1, random_state=42, verbose=-1
            ),
        }

        horizon_results = {}
        for name, reg in regressors.items():
            t0 = time.time()
            fold_metrics = []

            for fold_i, (train_idx, test_idx) in enumerate(splits):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y_clean[train_idx], y_clean[test_idx]

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                reg_copy = type(reg)(**reg.get_params())
                reg_copy.fit(X_train_s, y_train)
                y_pred = reg_copy.predict(X_test_s)

                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                corr = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 2 else 0

                # Binary evaluation (above training median = high vol)
                median_vol = np.median(y_train)
                bin_true = (y_test > median_vol).astype(int)
                bin_pred = (y_pred > median_vol).astype(int)
                bin_acc = accuracy_score(bin_true, bin_pred)
                bin_f1 = f1_score(bin_true, bin_pred, zero_division=0)
                try:
                    bin_auc = roc_auc_score(bin_true, y_pred)
                except:
                    bin_auc = 0.5

                fold_metrics.append({
                    "r2": r2, "corr": corr, "rmse": rmse,
                    "bin_acc": bin_acc, "bin_f1": bin_f1, "bin_auc": bin_auc,
                })

                print(f"    {name} fold {fold_i+1}/{len(splits)}: "
                      f"r2={r2:.3f} corr={corr:.3f} rmse={rmse:.6f} "
                      f"bin_f1={bin_f1:.3f} bin_auc={bin_auc:.3f} "
                      f"[train={len(train_idx):,} test={len(test_idx):,}]", flush=True)

            elapsed = time.time() - t0
            avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
            std = {k: np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
            horizon_results[name] = {"avg": avg, "std": std, "time": elapsed}

            print(f"  → {name}: r2={avg['r2']:.3f}±{std['r2']:.3f} "
                  f"corr={avg['corr']:.3f}±{std['corr']:.3f} "
                  f"bin_f1={avg['bin_f1']:.3f}±{std['bin_f1']:.3f} "
                  f"bin_auc={avg['bin_auc']:.3f} ({elapsed:.1f}s)", flush=True)

        all_results[fwd] = horizon_results

    # Summary table
    print(f"\n  {'='*70}")
    print(f"  REGRESSION SUMMARY — Ridge (best model from v9)")
    print(f"  {'='*70}")
    print(f"  {'Horizon':>10s} {'R²':>8s} {'Corr':>8s} {'RMSE':>10s} "
          f"{'BinAcc':>8s} {'BinF1':>8s} {'BinAUC':>8s}")
    print(f"  {'-'*60}")
    for fwd in sorted(all_results.keys()):
        if "Ridge" in all_results[fwd]:
            a = all_results[fwd]["Ridge"]["avg"]
            label = HORIZONS[fwd]
            print(f"  {label:>10s} {a['r2']:8.3f} {a['corr']:8.3f} {a['rmse']:10.6f} "
                  f"{a['bin_acc']:8.3f} {a['bin_f1']:8.3f} {a['bin_auc']:8.3f}")

    print(f"\n  {'Horizon':>10s} {'R²':>8s} {'Corr':>8s} {'RMSE':>10s} "
          f"{'BinAcc':>8s} {'BinF1':>8s} {'BinAUC':>8s}")
    print(f"  {'-'*60}")
    print(f"  LGBM Regressor:")
    for fwd in sorted(all_results.keys()):
        if "LGBM_reg" in all_results[fwd]:
            a = all_results[fwd]["LGBM_reg"]["avg"]
            label = HORIZONS[fwd]
            print(f"  {label:>10s} {a['r2']:8.3f} {a['corr']:8.3f} {a['rmse']:10.6f} "
                  f"{a['bin_acc']:8.3f} {a['bin_f1']:8.3f} {a['bin_auc']:8.3f}")

    return all_results


# ---------------------------------------------------------------------------
# Experiment B: Binary classification across horizons (LogReg)
# ---------------------------------------------------------------------------

def experiment_classification_horizons(df, feature_list, horizons, n_splits=5):
    """
    Run LogReg binary classification for each forward horizon.
    """
    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT B: Binary Classification Across Horizons (LogReg)")
    print(f"{'#'*70}")

    X_full, feat_names = get_feature_matrix(df, feature_list)

    all_results = {}

    for fwd, label in horizons.items():
        print(f"\n  --- Horizon: {fwd} bars ({label}) ---")
        col_hv = f"is_high_vol_{fwd}"

        y = df[col_hv].values
        valid = X_full.notna().all(axis=1).values & ~np.isnan(y)
        X = X_full.values[valid]
        y_clean = y[valid].astype(int)

        pos_rate = y_clean.mean()
        print(f"  Samples: {len(y_clean):,} | Positive rate: {pos_rate:.3f}")

        splits = walk_forward_splits(len(y_clean), n_splits=n_splits)
        print(f"  Walk-forward splits: {len(splits)}")

        clf = LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs",
            class_weight="balanced", n_jobs=-1
        )

        t0 = time.time()
        fold_metrics = []

        for fold_i, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_clean[train_idx], y_clean[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf_copy = LogisticRegression(**clf.get_params())
            clf_copy.fit(X_train_s, y_train)

            y_pred = clf_copy.predict(X_test_s)
            y_prob = clf_copy.predict_proba(X_test_s)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_test, y_prob)
            except:
                auc = 0.5

            fold_metrics.append({
                "acc": acc, "f1": f1, "prec": prec, "rec": rec, "auc": auc,
            })

            print(f"    LogReg fold {fold_i+1}/{len(splits)}: "
                  f"acc={acc:.3f} f1={f1:.3f} auc={auc:.3f} "
                  f"[train={len(train_idx):,} test={len(test_idx):,}]", flush=True)

        elapsed = time.time() - t0
        avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        std = {k: np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        all_results[fwd] = {"avg": avg, "std": std, "time": elapsed}

        print(f"  → LogReg: acc={avg['acc']:.3f}±{std['acc']:.3f} "
              f"f1={avg['f1']:.3f}±{std['f1']:.3f} "
              f"auc={avg['auc']:.3f}±{std['auc']:.3f} "
              f"prec={avg['prec']:.3f} rec={avg['rec']:.3f} ({elapsed:.1f}s)", flush=True)

    # Summary
    print(f"\n  {'='*70}")
    print(f"  CLASSIFICATION SUMMARY — LogReg (balanced)")
    print(f"  {'='*70}")
    print(f"  {'Horizon':>10s} {'Acc':>8s} {'F1':>8s} {'AUC':>8s} {'Prec':>8s} {'Rec':>8s}")
    print(f"  {'-'*50}")
    for fwd in sorted(all_results.keys()):
        a = all_results[fwd]["avg"]
        label = HORIZONS[fwd]
        print(f"  {label:>10s} {a['acc']:8.3f} {a['f1']:8.3f} {a['auc']:8.3f} "
              f"{a['prec']:8.3f} {a['rec']:8.3f}")

    return all_results


# ---------------------------------------------------------------------------
# Experiment C: Feature importance by horizon
# ---------------------------------------------------------------------------

def experiment_feature_importance_by_horizon(df, feature_list, horizons):
    """
    Compare which features matter most at each horizon.
    Uses LGBM on 70/30 time split.
    """
    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT C: Feature Importance by Horizon")
    print(f"{'#'*70}")

    X_full, feat_names = get_feature_matrix(df, feature_list)

    all_importances = {}

    for fwd, label in horizons.items():
        print(f"\n  --- Horizon: {fwd} bars ({label}) ---")
        col_hv = f"is_high_vol_{fwd}"

        y = df[col_hv].values
        valid = X_full.notna().all(axis=1).values & ~np.isnan(y)
        X = X_full.values[valid]
        y_clean = y[valid].astype(int)

        split_idx = int(len(y_clean) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_clean[:split_idx], y_clean[split_idx:]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=50, is_unbalance=True,
            n_jobs=-1, random_state=42, verbose=-1
        )
        model.fit(X_train_s, y_train)

        importances = model.feature_importances_
        feat_imp = sorted(zip(feat_names, importances), key=lambda x: -x[1])
        all_importances[fwd] = feat_imp

        print(f"  Top 10 features:")
        total_imp = sum(importances)
        cumulative = 0
        for rank, (feat, imp) in enumerate(feat_imp[:10], 1):
            pct = imp / total_imp * 100
            cumulative += pct
            print(f"    {rank:2d}. {feat:30s} {pct:5.1f}% (cum {cumulative:5.1f}%)")

    # Cross-horizon comparison: which features shift in importance?
    print(f"\n  {'='*70}")
    print(f"  FEATURE RANK COMPARISON ACROSS HORIZONS")
    print(f"  {'='*70}")

    # Get top 15 features from each horizon
    all_top = set()
    for fwd in horizons:
        for feat, _ in all_importances[fwd][:15]:
            all_top.add(feat)

    print(f"\n  {'Feature':30s}", end="")
    for fwd, label in sorted(horizons.items()):
        print(f" {label:>6s}", end="")
    print(f" {'Shift':>6s}")
    print(f"  {'-'*70}")

    rows = []
    for feat in sorted(all_top):
        ranks = {}
        for fwd in horizons:
            rank_list = [f for f, _ in all_importances[fwd]]
            if feat in rank_list:
                ranks[fwd] = rank_list.index(feat) + 1
            else:
                ranks[fwd] = len(rank_list)
        # Shift = difference between shortest and longest horizon rank
        sorted_fwds = sorted(horizons.keys())
        shift = ranks[sorted_fwds[-1]] - ranks[sorted_fwds[0]]
        rows.append((feat, ranks, shift))

    # Sort by average rank
    rows.sort(key=lambda x: np.mean(list(x[1].values())))

    for feat, ranks, shift in rows:
        print(f"  {feat:30s}", end="")
        for fwd in sorted(horizons.keys()):
            print(f" {ranks[fwd]:6d}", end="")
        direction = "↑" if shift < -3 else "↓" if shift > 3 else "="
        print(f" {shift:+5d}{direction}")

    return all_importances


# ---------------------------------------------------------------------------
# Experiment D: Cross-horizon correlation
# ---------------------------------------------------------------------------

def experiment_cross_horizon_correlation(df, horizons):
    """
    How correlated are the fwd_vol values across different horizons?
    If 1h and 24h vol are highly correlated, one model might suffice.
    """
    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT D: Cross-Horizon Correlation")
    print(f"{'#'*70}")

    vol_cols = {fwd: f"fwd_vol_{fwd}" for fwd in horizons}
    hv_cols = {fwd: f"is_high_vol_{fwd}" for fwd in horizons}

    # Continuous vol correlation
    print(f"\n  Forward Vol Correlation Matrix:")
    sorted_fwds = sorted(horizons.keys())
    print(f"  {'':>10s}", end="")
    for fwd in sorted_fwds:
        print(f" {HORIZONS[fwd]:>8s}", end="")
    print()
    print(f"  {'-'*45}")

    for fwd_i in sorted_fwds:
        print(f"  {HORIZONS[fwd_i]:>10s}", end="")
        for fwd_j in sorted_fwds:
            vi = df[vol_cols[fwd_i]].values
            vj = df[vol_cols[fwd_j]].values
            valid = ~np.isnan(vi) & ~np.isnan(vj)
            if valid.sum() > 100:
                corr = np.corrcoef(vi[valid], vj[valid])[0, 1]
            else:
                corr = np.nan
            print(f" {corr:8.3f}", end="")
        print()

    # Binary agreement (is_high_vol)
    print(f"\n  High-Vol Agreement Matrix (% of bars where both agree):")
    print(f"  {'':>10s}", end="")
    for fwd in sorted_fwds:
        print(f" {HORIZONS[fwd]:>8s}", end="")
    print()
    print(f"  {'-'*45}")

    for fwd_i in sorted_fwds:
        print(f"  {HORIZONS[fwd_i]:>10s}", end="")
        for fwd_j in sorted_fwds:
            hi = df[hv_cols[fwd_i]].values
            hj = df[hv_cols[fwd_j]].values
            valid = ~np.isnan(hi) & ~np.isnan(hj)
            if valid.sum() > 100:
                agree = np.mean(hi[valid] == hj[valid])
            else:
                agree = np.nan
            print(f" {agree:8.3f}", end="")
        print()

    # Regime transition analysis: how often does high-vol at 1h predict high-vol at 4h/12h/24h?
    print(f"\n  Predictive Power: P(high_vol@longer | high_vol@shorter)")
    print(f"  {'Given HV at':>15s}", end="")
    for fwd in sorted_fwds:
        print(f" {HORIZONS[fwd]:>8s}", end="")
    print()
    print(f"  {'-'*50}")

    for fwd_i in sorted_fwds:
        print(f"  {HORIZONS[fwd_i]:>15s}", end="")
        hi = df[hv_cols[fwd_i]].values
        for fwd_j in sorted_fwds:
            hj = df[hv_cols[fwd_j]].values
            valid = ~np.isnan(hi) & ~np.isnan(hj) & (hi == 1)
            if valid.sum() > 50:
                p = np.mean(hj[valid] == 1)
            else:
                p = np.nan
            print(f" {p:8.3f}", end="")
        print()


# ---------------------------------------------------------------------------
# Experiment E: Optimal horizon per strategy
# ---------------------------------------------------------------------------

def experiment_strategy_horizons(df, feature_list, horizons, n_splits=5):
    """
    For each horizon, simulate a simple position-sizing strategy:
    - Predict fwd_vol with Ridge
    - Scale position: size = clip(1 - P(high_vol), 0.2, 1.0)
    - Measure: avg size reduction during actual high-vol periods
    """
    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT E: Position Sizing Effectiveness by Horizon")
    print(f"{'#'*70}")

    X_full, feat_names = get_feature_matrix(df, feature_list)

    results = {}

    for fwd, label in sorted(horizons.items()):
        print(f"\n  --- Horizon: {fwd} bars ({label}) ---")
        col_vol = f"fwd_vol_{fwd}"
        col_hv = f"is_high_vol_{fwd}"

        y_vol = df[col_vol].values
        y_hv = df[col_hv].values
        valid = X_full.notna().all(axis=1).values & ~np.isnan(y_vol) & ~np.isnan(y_hv)
        X = X_full.values[valid]
        y_v = y_vol[valid]
        y_h = y_hv[valid].astype(int)

        # Use last 20% as test
        split_idx = int(len(y_v) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train_v, y_test_v = y_v[:split_idx], y_v[split_idx:]
        y_test_h = y_h[split_idx:]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Ridge regression
        reg = Ridge(alpha=1.0)
        reg.fit(X_train_s, y_train_v)
        y_pred_v = reg.predict(X_test_s)

        # Convert predicted vol to position size
        # Higher predicted vol → smaller position
        vol_median = np.median(y_train_v)
        vol_ratio = y_pred_v / vol_median  # >1 means above-average vol
        sizes = np.clip(1.0 / vol_ratio, 0.2, 1.0)  # inverse scaling

        avg_size_hv = sizes[y_test_h == 1].mean() if (y_test_h == 1).sum() > 0 else np.nan
        avg_size_nv = sizes[y_test_h == 0].mean() if (y_test_h == 0).sum() > 0 else np.nan
        reduction = 1 - avg_size_hv / avg_size_nv if avg_size_nv > 0 else 0

        # Also measure: how many high-vol bars had size < 0.5?
        if (y_test_h == 1).sum() > 0:
            pct_reduced = (sizes[y_test_h == 1] < 0.5).mean()
        else:
            pct_reduced = 0

        results[fwd] = {
            "avg_size_hv": avg_size_hv,
            "avg_size_nv": avg_size_nv,
            "reduction": reduction,
            "pct_reduced": pct_reduced,
            "n_hv": (y_test_h == 1).sum(),
            "n_nv": (y_test_h == 0).sum(),
        }

        print(f"  Test set: {len(y_test_h):,} bars | "
              f"high_vol: {(y_test_h==1).sum():,} ({y_test_h.mean():.3f})")
        print(f"  Avg size during high_vol:  {avg_size_hv:.3f}")
        print(f"  Avg size during normal:    {avg_size_nv:.3f}")
        print(f"  Size reduction:            {reduction*100:.1f}%")
        print(f"  % of high_vol bars < 0.5x: {pct_reduced*100:.1f}%", flush=True)

    # Summary
    print(f"\n  {'='*70}")
    print(f"  POSITION SIZING SUMMARY")
    print(f"  {'='*70}")
    print(f"  {'Horizon':>10s} {'SizeHV':>8s} {'SizeNV':>8s} {'Reduction':>10s} {'%<0.5x':>8s}")
    print(f"  {'-'*50}")
    for fwd in sorted(results.keys()):
        r = results[fwd]
        label = HORIZONS[fwd]
        print(f"  {label:>10s} {r['avg_size_hv']:8.3f} {r['avg_size_nv']:8.3f} "
              f"{r['reduction']*100:9.1f}% {r['pct_reduced']*100:7.1f}%")

    return results


# ---------------------------------------------------------------------------
# Single-symbol runner
# ---------------------------------------------------------------------------

def run_symbol_horizons(symbol, start, end):
    """Run all multi-horizon experiments for one symbol."""
    print(f"\n\n{'='*70}")
    print(f"  MULTI-HORIZON VOL PREDICTION: {symbol}")
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

    # Step 3: Label regimes for ALL horizons
    print(f"\n  Step 3: Labeling regimes for {len(HORIZONS)} horizons...")
    t2 = time.time()
    for fwd, label in sorted(HORIZONS.items()):
        df = label_regimes_multi(df, fwd)
        hv_rate = (df[f"is_high_vol_{fwd}"] == 1).mean()
        print(f"    {label:>4s} ({fwd:3d} bars): high_vol rate = {hv_rate:.3f}", flush=True)
    print(f"  Labeled in {time.time()-t2:.0f}s")

    results = {}

    # Experiment A: Regression across horizons
    results["regression"] = experiment_regression_horizons(df, ALL_FEATURES, HORIZONS)

    # Experiment B: Classification across horizons
    results["classification"] = experiment_classification_horizons(df, ALL_FEATURES, HORIZONS)

    # Experiment C: Feature importance by horizon
    results["feat_importance"] = experiment_feature_importance_by_horizon(df, ALL_FEATURES, HORIZONS)

    # Experiment D: Cross-horizon correlation
    experiment_cross_horizon_correlation(df, HORIZONS)

    # Experiment E: Position sizing by horizon
    results["position_sizing"] = experiment_strategy_horizons(df, ALL_FEATURES, HORIZONS)

    elapsed = time.time() - t0
    print(f"\n✅ {symbol} multi-horizon experiments complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    return results


# ---------------------------------------------------------------------------
# Cross-symbol summary
# ---------------------------------------------------------------------------

def print_cross_symbol_summary(all_results):
    """Print cross-symbol multi-horizon comparison."""
    print(f"\n\n{'#'*70}")
    print(f"  CROSS-SYMBOL MULTI-HORIZON SUMMARY")
    print(f"{'#'*70}")

    # Ridge regression R² by symbol × horizon
    print(f"\n  Ridge Regression R² by Symbol × Horizon:")
    print(f"  {'Symbol':12s}", end="")
    for fwd in sorted(HORIZONS.keys()):
        print(f" {HORIZONS[fwd]:>8s}", end="")
    print()
    print(f"  {'-'*45}")
    for sym, results in all_results.items():
        if "regression" not in results:
            continue
        print(f"  {sym:12s}", end="")
        for fwd in sorted(HORIZONS.keys()):
            if fwd in results["regression"] and "Ridge" in results["regression"][fwd]:
                r2 = results["regression"][fwd]["Ridge"]["avg"]["r2"]
                print(f" {r2:8.3f}", end="")
            else:
                print(f" {'—':>8s}", end="")
        print()

    # Ridge correlation by symbol × horizon
    print(f"\n  Ridge Correlation by Symbol × Horizon:")
    print(f"  {'Symbol':12s}", end="")
    for fwd in sorted(HORIZONS.keys()):
        print(f" {HORIZONS[fwd]:>8s}", end="")
    print()
    print(f"  {'-'*45}")
    for sym, results in all_results.items():
        if "regression" not in results:
            continue
        print(f"  {sym:12s}", end="")
        for fwd in sorted(HORIZONS.keys()):
            if fwd in results["regression"] and "Ridge" in results["regression"][fwd]:
                corr = results["regression"][fwd]["Ridge"]["avg"]["corr"]
                print(f" {corr:8.3f}", end="")
            else:
                print(f" {'—':>8s}", end="")
        print()

    # Ridge binary F1 by symbol × horizon
    print(f"\n  Ridge Binary F1 (thresholded) by Symbol × Horizon:")
    print(f"  {'Symbol':12s}", end="")
    for fwd in sorted(HORIZONS.keys()):
        print(f" {HORIZONS[fwd]:>8s}", end="")
    print()
    print(f"  {'-'*45}")
    for sym, results in all_results.items():
        if "regression" not in results:
            continue
        print(f"  {sym:12s}", end="")
        for fwd in sorted(HORIZONS.keys()):
            if fwd in results["regression"] and "Ridge" in results["regression"][fwd]:
                bf1 = results["regression"][fwd]["Ridge"]["avg"]["bin_f1"]
                print(f" {bf1:8.3f}", end="")
            else:
                print(f" {'—':>8s}", end="")
        print()

    # LogReg classification F1 by symbol × horizon
    print(f"\n  LogReg Classification F1 by Symbol × Horizon:")
    print(f"  {'Symbol':12s}", end="")
    for fwd in sorted(HORIZONS.keys()):
        print(f" {HORIZONS[fwd]:>8s}", end="")
    print()
    print(f"  {'-'*45}")
    for sym, results in all_results.items():
        if "classification" not in results:
            continue
        print(f"  {sym:12s}", end="")
        for fwd in sorted(HORIZONS.keys()):
            if fwd in results["classification"]:
                f1 = results["classification"][fwd]["avg"]["f1"]
                print(f" {f1:8.3f}", end="")
            else:
                print(f" {'—':>8s}", end="")
        print()

    # Position sizing reduction by symbol × horizon
    print(f"\n  Position Size Reduction (%) by Symbol × Horizon:")
    print(f"  {'Symbol':12s}", end="")
    for fwd in sorted(HORIZONS.keys()):
        print(f" {HORIZONS[fwd]:>8s}", end="")
    print()
    print(f"  {'-'*45}")
    for sym, results in all_results.items():
        if "position_sizing" not in results:
            continue
        print(f"  {sym:12s}", end="")
        for fwd in sorted(HORIZONS.keys()):
            if fwd in results["position_sizing"]:
                red = results["position_sizing"][fwd]["reduction"] * 100
                print(f" {red:7.1f}%", end="")
            else:
                print(f" {'—':>8s}", end="")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-Horizon Vol Prediction")
    parser.add_argument("--symbol", default="all",
                        help="Symbol or 'all' for all 5 currencies")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-01-31")
    args = parser.parse_args()

    if args.symbol.lower() == "all":
        symbols = ALL_SYMBOLS
    else:
        symbols = [s.strip().upper() for s in args.symbol.split(",")]

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("  MULTI-HORIZON VOLATILITY PREDICTION EXPERIMENT")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Period:  {args.start} → {args.end}")
    print(f"  Horizons: {', '.join(f'{v} ({k} bars)' for k, v in sorted(HORIZONS.items()))}")
    print(f"  Models:  Ridge, LGBM (regression) + LogReg (classification)")
    print("=" * 70)

    grand_t0 = time.time()
    all_results = {}

    for idx, symbol in enumerate(symbols, 1):
        print(f"\n\n{'*'*70}")
        print(f"  SYMBOL {idx}/{len(symbols)}: {symbol}")
        print(f"{'*'*70}")

        sym_results = run_symbol_horizons(symbol, args.start, args.end)
        all_results[symbol] = sym_results

        elapsed = time.time() - grand_t0
        remaining = len(symbols) - idx
        per_sym = elapsed / idx
        eta = remaining * per_sym
        print(f"\n  ⏱ Total: {elapsed:.0f}s | ~{per_sym:.0f}s/symbol | ETA: {eta:.0f}s")

    # Cross-symbol summary
    if len(all_results) > 1:
        print_cross_symbol_summary(all_results)

    total = time.time() - grand_t0
    print(f"\n\n{'='*70}")
    print(f"  ALL DONE — {len(symbols)} symbols in {total:.0f}s ({total/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
