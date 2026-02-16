#!/usr/bin/env python3
"""
ML-based Regime Detection — Improving volatility prediction with ML.

Building on regime_detection.py findings:
- Single-feature threshold detectors: 63-70% acc, F1 0.35-0.43
- 6 features universally predictive across all 5 symbols
- Parkinson vol correlation r=0.37-0.58 with future vol

This suite tests whether ML can improve on those baselines by:
1. Combining all 60 features non-linearly
2. Proper time-series train/test splits (no lookahead)
3. Multiple targets: binary high_vol, 3-class vol regime, regression on fwd_vol
4. Feature importance analysis to find the best feature subsets
5. Calibrated probability outputs for position sizing

All models are CPU-only: RandomForest, XGBoost, LightGBM, LogisticRegression.
"""

import sys
import time
import argparse
import warnings
import psutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, mean_squared_error, r2_score, classification_report,
    log_loss
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import data loading and feature computation from regime_detection
from regime_detection import load_bars, compute_regime_features, label_regimes

PARQUET_DIR = Path("./parquet")
ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]


# ---------------------------------------------------------------------------
# Feature selection helpers
# ---------------------------------------------------------------------------

# Core features we know work from v8 experiments
CORE_VOL_FEATURES = [
    "parkvol_1h", "parkvol_2h", "parkvol_4h", "parkvol_8h", "parkvol_24h",
    "rvol_1h", "rvol_2h", "rvol_4h", "rvol_8h", "rvol_24h",
    "vol_ratio_1h_24h", "vol_ratio_2h_24h", "vol_ratio_1h_8h",
    "vol_accel_1h", "vol_accel_4h",
    "vol_sma_24h", "vol_ratio_bar",
    "trade_intensity_ratio",
    "parkinson_vol",
]

# Extended features including microstructure and trend indicators
ALL_FEATURES = CORE_VOL_FEATURES + [
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


def get_feature_matrix(df, feature_list):
    """Extract feature matrix, dropping rows with NaN."""
    available = [f for f in feature_list if f in df.columns]
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        print(f"  Warning: {len(missing)} features not found: {missing[:5]}...")
    X = df[available].copy()
    return X, available


# ---------------------------------------------------------------------------
# Time-series cross-validation (walk-forward)
# ---------------------------------------------------------------------------

def walk_forward_splits(n_samples, n_splits=5, min_train=10000, min_test=2000):
    """
    Generate walk-forward train/test splits for time series.
    Each split uses expanding window for training,
    and the next chunk for testing. No lookahead.
    """
    # Reserve last 20% as final test, split the rest into expanding folds
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

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    return splits


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_classifiers():
    """Return dict of CPU-friendly classifiers."""
    return {
        "LogReg": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", class_weight="balanced",
            n_jobs=-1
        ),
        "RF": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=50,
            class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "XGB": xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=50, tree_method="hist",
            scale_pos_weight=1.0,  # overridden per fold
            n_jobs=-1, random_state=42, verbosity=0,
            eval_metric="logloss"
        ),
        "LGBM": lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=50, is_unbalance=True,
            n_jobs=-1, random_state=42, verbose=-1
        ),
    }


def get_regressors():
    """Return dict of CPU-friendly regressors for vol prediction."""
    return {
        "Ridge": Ridge(alpha=1.0),
        "RF_reg": RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=50,
            n_jobs=-1, random_state=42
        ),
        "XGB_reg": xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=50, tree_method="hist",
            n_jobs=-1, random_state=42, verbosity=0
        ),
        "LGBM_reg": lgb.LGBMRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=50, n_jobs=-1, random_state=42,
            verbose=-1
        ),
    }


# ---------------------------------------------------------------------------
# Experiment 1: Binary high-vol classification
# ---------------------------------------------------------------------------

def experiment_binary_highvol(df, feature_list, n_splits=5):
    """
    Binary classification: is_high_vol (0/1).
    Walk-forward CV with multiple models.
    """
    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT 1: Binary High-Vol Classification")
    print(f"{'#'*70}")

    X, feat_names = get_feature_matrix(df, feature_list)
    y = df["is_high_vol"].values

    # Drop NaN rows
    valid = X.notna().all(axis=1) & ~np.isnan(y)
    X = X[valid].values
    y = y[valid].astype(int)
    print(f"  Samples: {len(y):,} | Positive rate: {y.mean():.3f}")

    splits = walk_forward_splits(len(y), n_splits=n_splits)
    print(f"  Walk-forward splits: {len(splits)}")

    classifiers = get_classifiers()
    results = {}

    for name, clf in classifiers.items():
        t0 = time.time()
        fold_metrics = []

        for fold_i, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf_copy = type(clf)(**clf.get_params())
            # Dynamic class weight for XGB
            if name == "XGB":
                neg = (y_train == 0).sum()
                pos = max((y_train == 1).sum(), 1)
                clf_copy.set_params(scale_pos_weight=neg / pos)
            clf_copy.fit(X_train_s, y_train)

            y_pred = clf_copy.predict(X_test_s)
            y_prob = clf_copy.predict_proba(X_test_s)[:, 1] if hasattr(clf_copy, "predict_proba") else y_pred.astype(float)

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
                "train_size": len(train_idx), "test_size": len(test_idx),
                "test_pos_rate": y_test.mean(),
            })

            print(f"    {name} fold {fold_i+1}/{len(splits)}: "
                  f"acc={acc:.3f} f1={f1:.3f} auc={auc:.3f} "
                  f"[train={len(train_idx):,} test={len(test_idx):,}]", flush=True)

        elapsed = time.time() - t0
        avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        std = {k: np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0] if k not in ("train_size", "test_size", "test_pos_rate")}

        results[name] = {"avg": avg, "std": std, "folds": fold_metrics, "time": elapsed}

        print(f"  → {name}: acc={avg['acc']:.3f}±{std['acc']:.3f} "
              f"f1={avg['f1']:.3f}±{std['f1']:.3f} "
              f"auc={avg['auc']:.3f}±{std['auc']:.3f} "
              f"prec={avg['prec']:.3f} rec={avg['rec']:.3f} "
              f"({elapsed:.1f}s)", flush=True)

    # Summary table
    print(f"\n  {'Model':10s} {'Acc':>8s} {'F1':>8s} {'AUC':>8s} {'Prec':>8s} {'Rec':>8s} {'Time':>6s}")
    print(f"  {'-'*55}")
    for name, r in sorted(results.items(), key=lambda x: -x[1]["avg"]["f1"]):
        a = r["avg"]
        print(f"  {name:10s} {a['acc']:8.3f} {a['f1']:8.3f} {a['auc']:8.3f} "
              f"{a['prec']:8.3f} {a['rec']:8.3f} {r['time']:6.1f}s")

    # Baseline comparison
    print(f"\n  Baseline (single-feature threshold): acc=0.63-0.70, f1=0.35-0.43")

    return results


# ---------------------------------------------------------------------------
# Experiment 2: 3-class vol regime classification
# ---------------------------------------------------------------------------

def experiment_vol_regime_3class(df, feature_list, n_splits=5):
    """
    3-class classification: low_vol / norm_vol / high_vol.
    """
    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT 2: 3-Class Vol Regime Classification")
    print(f"{'#'*70}")

    X, feat_names = get_feature_matrix(df, feature_list)
    y = df["vol_regime"].values

    valid = X.notna().all(axis=1) & (y != "unknown") & ~pd.isna(y)
    X = X[valid].values
    y_str = y[valid]

    # Encode: low_vol=0, norm_vol=1, high_vol=2
    label_map = {"low_vol": 0, "norm_vol": 1, "high_vol": 2}
    y = np.array([label_map.get(v, 1) for v in y_str])
    print(f"  Samples: {len(y):,} | Distribution: " +
          ", ".join(f"{k}={np.mean(y==v):.3f}" for k, v in label_map.items()))

    splits = walk_forward_splits(len(y), n_splits=n_splits)
    print(f"  Walk-forward splits: {len(splits)}")

    classifiers = get_classifiers()
    results = {}

    for name, clf in classifiers.items():
        t0 = time.time()
        fold_metrics = []

        for fold_i, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf_copy = type(clf)(**clf.get_params())
            # For 3-class, disable binary-specific weighting on XGB
            if name == "XGB":
                clf_copy.set_params(scale_pos_weight=1.0)
            clf_copy.fit(X_train_s, y_train)

            y_pred = clf_copy.predict(X_test_s)
            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
            f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            # Per-class F1
            f1_per = f1_score(y_test, y_pred, average=None, zero_division=0)

            fold_metrics.append({
                "acc": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted,
                "f1_low": f1_per[0] if len(f1_per) > 0 else 0,
                "f1_norm": f1_per[1] if len(f1_per) > 1 else 0,
                "f1_high": f1_per[2] if len(f1_per) > 2 else 0,
            })

            print(f"    {name} fold {fold_i+1}/{len(splits)}: "
                  f"acc={acc:.3f} f1_macro={f1_macro:.3f} "
                  f"f1_low={f1_per[0]:.3f} f1_norm={f1_per[1]:.3f} f1_high={f1_per[2]:.3f}", flush=True)

        elapsed = time.time() - t0
        avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        results[name] = {"avg": avg, "time": elapsed}

        print(f"  → {name}: acc={avg['acc']:.3f} f1_macro={avg['f1_macro']:.3f} "
              f"f1_low={avg['f1_low']:.3f} f1_norm={avg['f1_norm']:.3f} "
              f"f1_high={avg['f1_high']:.3f} ({elapsed:.1f}s)", flush=True)

    # Summary
    print(f"\n  {'Model':10s} {'Acc':>8s} {'F1mac':>8s} {'F1low':>8s} {'F1norm':>8s} {'F1high':>8s}")
    print(f"  {'-'*50}")
    for name, r in sorted(results.items(), key=lambda x: -x[1]["avg"]["f1_macro"]):
        a = r["avg"]
        print(f"  {name:10s} {a['acc']:8.3f} {a['f1_macro']:8.3f} "
              f"{a['f1_low']:8.3f} {a['f1_norm']:8.3f} {a['f1_high']:8.3f}")

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Regression — predict future volatility directly
# ---------------------------------------------------------------------------

def experiment_vol_regression(df, feature_list, n_splits=5):
    """
    Regression: predict fwd_vol (forward realized volatility).
    This is the most direct approach — predict the actual number.
    """
    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT 3: Vol Regression (predict fwd_vol)")
    print(f"{'#'*70}")

    X, feat_names = get_feature_matrix(df, feature_list)
    y = df["fwd_vol"].values

    valid = X.notna().all(axis=1) & ~np.isnan(y)
    X = X[valid].values
    y = y[valid]
    print(f"  Samples: {len(y):,} | fwd_vol mean={y.mean():.6f} std={y.std():.6f}")

    splits = walk_forward_splits(len(y), n_splits=n_splits)
    print(f"  Walk-forward splits: {len(splits)}")

    regressors = get_regressors()
    results = {}

    for name, reg in regressors.items():
        t0 = time.time()
        fold_metrics = []

        for fold_i, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            reg_copy = type(reg)(**reg.get_params())
            reg_copy.fit(X_train_s, y_train)

            y_pred = reg_copy.predict(X_test_s)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            corr = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 2 else 0

            # Also evaluate as a binary classifier (high vol = above median)
            median_vol = np.median(y_train)
            binary_true = (y_test > median_vol).astype(int)
            binary_pred = (y_pred > median_vol).astype(int)
            binary_acc = accuracy_score(binary_true, binary_pred)
            binary_f1 = f1_score(binary_true, binary_pred, zero_division=0)

            fold_metrics.append({
                "rmse": rmse, "r2": r2, "corr": corr,
                "binary_acc": binary_acc, "binary_f1": binary_f1,
            })

            print(f"    {name} fold {fold_i+1}/{len(splits)}: "
                  f"r2={r2:.3f} corr={corr:.3f} rmse={rmse:.6f} "
                  f"bin_acc={binary_acc:.3f} bin_f1={binary_f1:.3f}", flush=True)

        elapsed = time.time() - t0
        avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        results[name] = {"avg": avg, "time": elapsed}

        print(f"  → {name}: r2={avg['r2']:.3f} corr={avg['corr']:.3f} "
              f"rmse={avg['rmse']:.6f} bin_acc={avg['binary_acc']:.3f} "
              f"bin_f1={avg['binary_f1']:.3f} ({elapsed:.1f}s)", flush=True)

    # Summary
    print(f"\n  {'Model':10s} {'R²':>8s} {'Corr':>8s} {'RMSE':>10s} {'BinAcc':>8s} {'BinF1':>8s}")
    print(f"  {'-'*55}")
    for name, r in sorted(results.items(), key=lambda x: -x[1]["avg"]["corr"]):
        a = r["avg"]
        print(f"  {name:10s} {a['r2']:8.3f} {a['corr']:8.3f} {a['rmse']:10.6f} "
              f"{a['binary_acc']:8.3f} {a['binary_f1']:8.3f}")

    # Baseline
    print(f"\n  Baseline (parkvol_1h correlation): r=0.37-0.58")

    return results


# ---------------------------------------------------------------------------
# Experiment 4: Feature importance analysis
# ---------------------------------------------------------------------------

def experiment_feature_importance(df, feature_list):
    """
    Train best model on full data, extract feature importances.
    Use 70/30 time split.
    """
    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT 4: Feature Importance Analysis")
    print(f"{'#'*70}")

    X, feat_names = get_feature_matrix(df, feature_list)
    y = df["is_high_vol"].values

    valid = X.notna().all(axis=1) & ~np.isnan(y)
    X = X[valid].values
    y = y[valid].astype(int)

    # 70/30 time split
    split_idx = int(len(y) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train LGBM (usually best for feature importance)
    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_samples=50, n_jobs=-1, random_state=42, verbose=-1
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    print(f"  LGBM on 70/30 split: acc={acc:.3f} f1={f1:.3f} auc={auc:.3f}")
    print(f"  Train: {len(y_train):,} | Test: {len(y_test):,}")

    # Feature importance (gain-based)
    importances = model.feature_importances_
    feat_imp = sorted(zip(feat_names, importances), key=lambda x: -x[1])

    print(f"\n  Top 20 features by importance (gain):")
    print(f"  {'Feature':30s} {'Importance':>12s} {'Cumulative%':>12s}")
    print(f"  {'-'*55}")
    total_imp = sum(importances)
    cumulative = 0
    for feat, imp in feat_imp[:20]:
        pct = imp / total_imp * 100
        cumulative += pct
        print(f"  {feat:30s} {imp:12.0f} {cumulative:11.1f}%")

    # Test with top-N features only
    print(f"\n  Ablation: accuracy with top-N features only:")
    for n_top in [5, 10, 15, 20, len(feat_names)]:
        top_feats = [f for f, _ in feat_imp[:n_top]]
        top_idx = [feat_names.index(f) for f in top_feats]

        model_small = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=50, n_jobs=-1, random_state=42, verbose=-1
        )
        model_small.fit(X_train_s[:, top_idx], y_train)
        y_pred_s = model_small.predict(X_test_s[:, top_idx])
        y_prob_s = model_small.predict_proba(X_test_s[:, top_idx])[:, 1]

        acc_s = accuracy_score(y_test, y_pred_s)
        f1_s = f1_score(y_test, y_pred_s, zero_division=0)
        auc_s = roc_auc_score(y_test, y_prob_s)
        print(f"    Top {n_top:2d}: acc={acc_s:.3f} f1={f1_s:.3f} auc={auc_s:.3f}", flush=True)

    return feat_imp


# ---------------------------------------------------------------------------
# Experiment 5: Probability calibration for position sizing
# ---------------------------------------------------------------------------

def experiment_calibration(df, feature_list):
    """
    Test calibrated probability outputs for position sizing.
    A well-calibrated model saying P(high_vol)=0.7 should be right 70% of the time.
    """
    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT 5: Probability Calibration for Position Sizing")
    print(f"{'#'*70}")

    X, feat_names = get_feature_matrix(df, feature_list)
    y = df["is_high_vol"].values

    valid = X.notna().all(axis=1) & ~np.isnan(y)
    X = X[valid].values
    y = y[valid].astype(int)

    # 60/20/20 split: train / calibration / test
    n = len(y)
    split1 = int(n * 0.6)
    split2 = int(n * 0.8)

    X_train, X_cal, X_test = X[:split1], X[split1:split2], X[split2:]
    y_train, y_cal, y_test = y[:split1], y[split1:split2], y[split2:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_cal_s = scaler.transform(X_cal)
    X_test_s = scaler.transform(X_test)

    # Train base model
    base_model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_samples=50, n_jobs=-1, random_state=42, verbose=-1
    )
    base_model.fit(X_train_s, y_train)

    # Raw probabilities
    raw_probs = base_model.predict_proba(X_test_s)[:, 1]

    # Calibrate using isotonic regression on calibration set
    cal_model = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
    cal_model.fit(X_cal_s, y_cal)
    cal_probs = cal_model.predict_proba(X_test_s)[:, 1]

    # Evaluate calibration in probability bins
    print(f"\n  Calibration analysis (test set, n={len(y_test):,}):")
    print(f"  {'Prob Bin':15s} {'Count':>8s} {'Actual%':>8s} {'Raw%':>8s} {'Cal%':>8s} {'CalErr':>8s}")
    print(f"  {'-'*55}")

    bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
            (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]

    raw_cal_error = 0
    cal_cal_error = 0
    total_in_bins = 0

    for lo, hi in bins:
        # Use calibrated probs for binning
        mask = (cal_probs >= lo) & (cal_probs < hi)
        if mask.sum() == 0:
            continue
        actual_rate = y_test[mask].mean()
        raw_rate = raw_probs[mask].mean()
        cal_rate = cal_probs[mask].mean()
        cal_err = abs(actual_rate - cal_rate)
        raw_err = abs(actual_rate - raw_rate)

        raw_cal_error += raw_err * mask.sum()
        cal_cal_error += cal_err * mask.sum()
        total_in_bins += mask.sum()

        print(f"  [{lo:.1f}, {hi:.1f})     {mask.sum():8d} {actual_rate:8.3f} "
              f"{raw_rate:8.3f} {cal_rate:8.3f} {cal_err:8.3f}")

    if total_in_bins > 0:
        print(f"\n  Mean calibration error: raw={raw_cal_error/total_in_bins:.4f} "
              f"calibrated={cal_cal_error/total_in_bins:.4f}")

    # Position sizing simulation
    print(f"\n  Position sizing simulation:")
    print(f"  Strategy: size = 1.0 - P(high_vol), capped at [0.2, 1.0]")

    sizes = np.clip(1.0 - cal_probs, 0.2, 1.0)
    # When high_vol is true, we want small size (good)
    # When high_vol is false, we want large size (good)
    avg_size_when_highvol = sizes[y_test == 1].mean()
    avg_size_when_normal = sizes[y_test == 0].mean()
    overall_avg_size = sizes.mean()

    print(f"    Avg size when high_vol:  {avg_size_when_highvol:.3f}")
    print(f"    Avg size when normal:    {avg_size_when_normal:.3f}")
    print(f"    Overall avg size:        {overall_avg_size:.3f}")
    print(f"    Size reduction in danger: {(1 - avg_size_when_highvol/avg_size_when_normal)*100:.1f}%")

    return {
        "raw_cal_error": raw_cal_error / max(total_in_bins, 1),
        "cal_cal_error": cal_cal_error / max(total_in_bins, 1),
        "avg_size_highvol": avg_size_when_highvol,
        "avg_size_normal": avg_size_when_normal,
        "size_reduction": 1 - avg_size_when_highvol / max(avg_size_when_normal, 1e-10),
    }


# ---------------------------------------------------------------------------
# Experiment 6: Early warning — predict vol N bars ahead
# ---------------------------------------------------------------------------

def experiment_early_warning(df, feature_list):
    """
    Test how far ahead we can predict high-vol.
    Train separate models for different prediction horizons.
    """
    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT 6: Early Warning — Predict High-Vol N Bars Ahead")
    print(f"{'#'*70}")

    X_full, feat_names = get_feature_matrix(df, feature_list)
    fwd_vol = df["fwd_vol"].values
    returns = df["returns"].values

    valid_base = X_full.notna().all(axis=1).values
    X_vals = X_full.values

    horizons = [1, 6, 12, 24, 48, 96]  # 5min, 30min, 1h, 2h, 4h, 8h
    results = {}

    for horizon in horizons:
        t0 = time.time()
        # Create shifted target: is the vol HIGH at bar i+horizon?
        n = len(df)
        y_shifted = np.full(n, np.nan)

        # Use rolling median to define high-vol threshold
        vol_median = pd.Series(fwd_vol).rolling(288 * 3, min_periods=288).median().values
        overall_median = np.nanmedian(fwd_vol)
        vol_median = np.where(np.isnan(vol_median), overall_median, vol_median)

        for i in range(n - horizon):
            # Compute realized vol for the window starting at i+horizon
            end = min(i + horizon + 48, n)
            if end - (i + horizon) < 12:
                continue
            window_vol = np.std(returns[i+horizon:end])
            y_shifted[i] = 1 if window_vol > 1.5 * vol_median[i] else 0

        valid = valid_base & ~np.isnan(y_shifted)
        X = X_vals[valid]
        y = y_shifted[valid].astype(int)

        if len(y) < 5000:
            print(f"  Horizon {horizon:3d} bars: insufficient data ({len(y)})", flush=True)
            continue

        # 70/30 time split
        split_idx = int(len(y) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=50, n_jobs=-1, random_state=42, verbose=-1
        )
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = 0.5

        elapsed = time.time() - t0
        results[horizon] = {"acc": acc, "f1": f1, "auc": auc, "prec": prec, "rec": rec}

        mins = horizon * 5
        print(f"  Horizon {horizon:3d} bars ({mins:4d}min): "
              f"acc={acc:.3f} f1={f1:.3f} auc={auc:.3f} "
              f"prec={prec:.3f} rec={rec:.3f} ({elapsed:.1f}s)", flush=True)

    # Summary
    print(f"\n  {'Horizon':>10s} {'Acc':>8s} {'F1':>8s} {'AUC':>8s} {'Prec':>8s} {'Rec':>8s}")
    print(f"  {'-'*50}")
    for h in sorted(results.keys()):
        r = results[h]
        print(f"  {h*5:>7d}min {r['acc']:8.3f} {r['f1']:8.3f} {r['auc']:8.3f} "
              f"{r['prec']:8.3f} {r['rec']:8.3f}")

    return results


# ---------------------------------------------------------------------------
# Experiment 7: Core vs All features comparison
# ---------------------------------------------------------------------------

def experiment_feature_sets(df, n_splits=5):
    """
    Compare model performance with different feature sets:
    1. Core vol features only (19 features)
    2. All features (45+ features)
    """
    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT 7: Core vs All Features Comparison")
    print(f"{'#'*70}")

    feature_sets = {
        "core_vol (19)": CORE_VOL_FEATURES,
        "all (45+)": ALL_FEATURES,
    }

    y_raw = df["is_high_vol"].values

    for set_name, feat_list in feature_sets.items():
        print(f"\n  --- Feature set: {set_name} ---")
        X, feat_names = get_feature_matrix(df, feat_list)
        valid = X.notna().all(axis=1) & ~np.isnan(y_raw)
        X = X[valid].values
        y = y_raw[valid].astype(int)

        splits = walk_forward_splits(len(y), n_splits=n_splits)

        model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=50, n_jobs=-1, random_state=42, verbose=-1
        )

        fold_metrics = []
        for fold_i, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            m = lgb.LGBMClassifier(**model.get_params())
            m.fit(X_train_s, y_train)

            y_pred = m.predict(X_test_s)
            y_prob = m.predict_proba(X_test_s)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_test, y_prob)
            except:
                auc = 0.5

            fold_metrics.append({"acc": acc, "f1": f1, "auc": auc})
            print(f"    Fold {fold_i+1}: acc={acc:.3f} f1={f1:.3f} auc={auc:.3f}", flush=True)

        avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        print(f"  → {set_name}: acc={avg['acc']:.3f} f1={avg['f1']:.3f} auc={avg['auc']:.3f}")


# ---------------------------------------------------------------------------
# Single-symbol runner
# ---------------------------------------------------------------------------

def run_symbol_ml(symbol, start, end, forward_window=48):
    """Run all ML experiments for one symbol."""
    print(f"\n\n{'='*70}")
    print(f"  ML REGIME DETECTION: {symbol}")
    print(f"  {start} → {end} | fwd={forward_window} bars ({forward_window*5}min)")
    print(f"{'='*70}")

    t0 = time.time()

    # Step 1: Load data
    print(f"\n  Step 1: Loading 5m bars...")
    df = load_bars(symbol, start, end)
    if df.empty:
        print("  No data!")
        return {}
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.0f}s")

    # Step 2: Compute features
    print(f"\n  Step 2: Computing features...")
    t1 = time.time()
    df = compute_regime_features(df)
    n_features = len([c for c in df.columns if c not in ['timestamp_us', 'datetime']])
    print(f"  {n_features} features in {time.time()-t1:.0f}s")

    # Step 3: Label regimes
    print(f"\n  Step 3: Labeling regimes...")
    t2 = time.time()
    df = label_regimes(df, forward_window=forward_window)
    print(f"  Labeled in {time.time()-t2:.0f}s")

    results = {}

    # Experiment 1: Binary high-vol
    results["binary_highvol"] = experiment_binary_highvol(df, ALL_FEATURES, n_splits=5)

    # Experiment 2: 3-class vol regime
    results["vol_3class"] = experiment_vol_regime_3class(df, ALL_FEATURES, n_splits=5)

    # Experiment 3: Vol regression
    results["vol_regression"] = experiment_vol_regression(df, ALL_FEATURES, n_splits=5)

    # Experiment 4: Feature importance
    results["feature_importance"] = experiment_feature_importance(df, ALL_FEATURES)

    # Experiment 5: Calibration
    results["calibration"] = experiment_calibration(df, ALL_FEATURES)

    # Experiment 6: Early warning
    results["early_warning"] = experiment_early_warning(df, ALL_FEATURES)

    # Experiment 7: Feature sets
    experiment_feature_sets(df, n_splits=5)

    elapsed = time.time() - t0
    print(f"\n✅ {symbol} ML experiments complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    return results


# ---------------------------------------------------------------------------
# Cross-symbol ML summary
# ---------------------------------------------------------------------------

def print_ml_summary(all_results):
    """Print cross-symbol ML comparison."""
    print(f"\n\n{'#'*70}")
    print(f"  CROSS-SYMBOL ML SUMMARY")
    print(f"{'#'*70}")

    # Binary high-vol: best model per symbol
    print(f"\n  Binary High-Vol — Best model per symbol:")
    print(f"  {'Symbol':12s} {'Model':10s} {'Acc':>8s} {'F1':>8s} {'AUC':>8s}")
    print(f"  {'-'*45}")
    for sym, results in all_results.items():
        if "binary_highvol" not in results:
            continue
        best_name = max(results["binary_highvol"],
                       key=lambda k: results["binary_highvol"][k]["avg"]["f1"])
        best = results["binary_highvol"][best_name]["avg"]
        print(f"  {sym:12s} {best_name:10s} {best['acc']:8.3f} "
              f"{best['f1']:8.3f} {best['auc']:8.3f}")

    # Vol regression: best model per symbol
    print(f"\n  Vol Regression — Best model per symbol:")
    print(f"  {'Symbol':12s} {'Model':10s} {'R²':>8s} {'Corr':>8s} {'BinF1':>8s}")
    print(f"  {'-'*45}")
    for sym, results in all_results.items():
        if "vol_regression" not in results:
            continue
        best_name = max(results["vol_regression"],
                       key=lambda k: results["vol_regression"][k]["avg"]["corr"])
        best = results["vol_regression"][best_name]["avg"]
        print(f"  {sym:12s} {best_name:10s} {best['r2']:8.3f} "
              f"{best['corr']:8.3f} {best['binary_f1']:8.3f}")

    # Early warning decay
    print(f"\n  Early Warning — AUC decay by horizon:")
    header = f"  {'Symbol':12s}"
    for h in [1, 6, 12, 24, 48, 96]:
        header += f" {h*5:>5d}min"
    print(header)
    print(f"  {'-'*55}")
    for sym, results in all_results.items():
        if "early_warning" not in results:
            continue
        ew = results["early_warning"]
        row = f"  {sym:12s}"
        for h in [1, 6, 12, 24, 48, 96]:
            if h in ew:
                row += f" {ew[h]['auc']:7.3f}"
            else:
                row += f"       -"
        print(row)

    # Calibration
    print(f"\n  Calibration & Position Sizing:")
    print(f"  {'Symbol':12s} {'CalErr':>8s} {'SizeHigh':>10s} {'SizeNorm':>10s} {'Reduction':>10s}")
    print(f"  {'-'*55}")
    for sym, results in all_results.items():
        if "calibration" not in results:
            continue
        c = results["calibration"]
        print(f"  {sym:12s} {c['cal_cal_error']:8.4f} "
              f"{c['avg_size_highvol']:10.3f} {c['avg_size_normal']:10.3f} "
              f"{c['size_reduction']*100:9.1f}%")

    print(f"\n  Baseline comparison:")
    print(f"    Single-feature threshold: acc=0.63-0.70, f1=0.35-0.43")
    print(f"    Parkvol_1h correlation:   r=0.37-0.58")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ML Regime Detection Experiments")
    parser.add_argument("--symbol", default="all",
                        help="Symbol or 'all' for all 5 currencies")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-01-31")
    parser.add_argument("--forward-window", type=int, default=48)
    args = parser.parse_args()

    if args.symbol.lower() == "all":
        symbols = ALL_SYMBOLS
    else:
        symbols = [s.strip().upper() for s in args.symbol.split(",")]

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("  ML REGIME DETECTION EXPERIMENT SUITE")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Period:  {args.start} → {args.end}")
    print(f"  Forward: {args.forward_window} bars ({args.forward_window * 5} min)")
    print(f"  Models:  LogReg, RandomForest, XGBoost, LightGBM (CPU)")
    print("=" * 70)

    grand_t0 = time.time()
    all_results = {}

    for idx, symbol in enumerate(symbols, 1):
        print(f"\n\n{'*'*70}")
        print(f"  SYMBOL {idx}/{len(symbols)}: {symbol}")
        print(f"{'*'*70}")

        sym_results = run_symbol_ml(symbol, args.start, args.end, args.forward_window)
        all_results[symbol] = sym_results

        elapsed = time.time() - grand_t0
        remaining = len(symbols) - idx
        per_sym = elapsed / idx
        eta = remaining * per_sym
        print(f"\n  ⏱ Total: {elapsed:.0f}s | ~{per_sym:.0f}s/symbol | ETA: {eta:.0f}s")

    # Cross-symbol summary
    if len(all_results) > 1:
        print_ml_summary(all_results)

    total = time.time() - grand_t0
    print(f"\n\n{'='*70}")
    print(f"  ALL DONE — {len(symbols)} symbols in {total:.0f}s ({total/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
