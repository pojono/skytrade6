#!/usr/bin/env python3
"""
Settlement ML V2 — Train models with deep features
====================================================
Uses 89 features from analyse_settlement_v2.py to predict:
  - Regression: drop magnitude, time to bottom
  - Classification: profitable?, drop class, fast drop?

Handles:
  - Missing features (HistGradientBoosting handles NaN natively)
  - Small N (LOOCV, strong regularization, feature selection)
  - Multiple model types (Ridge, LGBM, RF, ElasticNet)
  - Proper comparison: old (17 feat) vs new (75+ feat)

Usage:
    python3 train_settlement_ml_v2.py settlement_features_v2.csv
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
    RandomForestRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.model_selection import (
    LeaveOneOut,
    cross_val_predict,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── Feature groups ───────────────────────────────────────────────────
# Tier 1: Available for ALL 64 settlements (no OB.1/OB.50 needed)
TIER1_FEATURES = [
    # FR
    "fr_bps", "fr_abs_bps",
    # OB.200 depth
    "total_bid_mean_usd", "total_ask_mean_usd", "total_depth_usd",
    "total_depth_imb_mean", "total_depth_trend",
    "bid_concentration", "ask_concentration",
    "depth_within_50bps", "thin_side_depth",
    # Trade microstructure
    "pre_trade_count", "pre_total_vol_usd", "trade_flow_imb",
    "pre_avg_trade_size_usd", "trade_size_median", "trade_size_p90",
    "trade_size_p99", "trade_size_max", "trade_size_skew",
    "large_trade_count", "large_trade_pct", "large_trade_imb",
    "pre_price_vol_bps",
    "trade_rate_10s", "trade_rate_2s", "trade_rate_accel", "vol_rate_accel",
    "buy_imb_last_1s", "buy_pressure_surge", "vwap_vs_mid_bps",
    # Ticker
    "oi_value_usd", "oi_change_60s", "oi_change_pct_60s",
    "basis_bps", "basis_abs_bps", "basis_trend",
    "volume_24h", "turnover_24h_usd",
    "price_change_24h_pct",
    # Liquidation
    "liq_count_pre", "liq_volume_usd", "liq_direction",
    # Interactions
    "fr_x_depth", "fr_x_spread", "fr_x_imb",
    "imb_x_vol", "spread_x_depth", "fr_squared",
]

# Tier 2: Only for 25 settlements with OB.1/OB.50
TIER2_FEATURES = [
    # Spread dynamics
    "spread_mean_bps", "spread_std_bps", "spread_max_bps", "spread_min_bps",
    "spread_trend_10s", "spread_trend_5s", "spread_trend_2s", "spread_last_vs_mean",
    # Qty imbalance
    "qty_imb_mean", "qty_imb_std", "qty_imb_trend_10s",
    "qty_imb_last_1s", "qty_imb_surge",
    # Depth imbalance (OB.50)
    "bid10_mean_usd", "ask10_mean_usd",
    "depth_imb_mean", "depth_imb_std",
    "depth_imb_trend_10s", "depth_imb_trend_5s",
    "bid_depth_trend", "ask_depth_trend",
]

ALL_FEATURES = TIER1_FEATURES + TIER2_FEATURES

# Regression targets
REG_TARGETS = {
    "drop_min_bps": "Max drop (bps)",
    "drop_500ms_bps": "Price @ T+500ms",
    "drop_1s_bps": "Price @ T+1s",
    "drop_5s_bps": "Price @ T+5s",
}

# Classification targets
CLF_TARGETS = {
    "target_profitable": "Profitable (>40bps)",
    "target_drop_class": "Drop class (0-3)",
    "target_fast_drop": "Fast drop (<500ms)",
}


def _available_features(df, feature_list):
    """Return features that exist in df and have <90% missing."""
    available = []
    for f in feature_list:
        if f in df.columns:
            missing_pct = df[f].isna().mean()
            if missing_pct < 0.90:
                available.append(f)
    return available


def train_regression(df, features, target_col, target_label):
    """Train regression models with LOOCV."""
    y = df[target_col].values
    X = df[features].values

    print(f"\n{'─'*70}")
    print(f"REGRESSION: {target_label}  ({target_col})")
    print(f"  N={len(y)}  features={len(features)}  target: mean={np.mean(y):.1f} std={np.std(y):.1f}")
    print(f"{'─'*70}")

    loo = LeaveOneOut()
    results = {}

    models = {
        "Ridge": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "ElasticNet": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000)),
        ]),
        "HGBR": HistGradientBoostingRegressor(
            max_iter=100, max_depth=4, min_samples_leaf=5,
            learning_rate=0.05, l2_regularization=1.0,
            random_state=42,
        ),
        "LGBM": lgb.LGBMRegressor(
            n_estimators=100, max_depth=4, num_leaves=15,
            min_child_samples=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=1.0,
            random_state=42, verbose=-1,
        ),
        "RF": RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=3,
            max_features=0.5, random_state=42,
        ),
    }

    for name, model in models.items():
        try:
            y_pred = cross_val_predict(model, X, y, cv=loo)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            mae_std_ratio = mae / np.std(y) if np.std(y) > 0 else float("inf")

            results[name] = {"mae": mae, "r2": r2, "ratio": mae_std_ratio}
            marker = " ⭐" if mae_std_ratio < 0.6 else ""
            print(f"  {name:12s}: MAE={mae:7.1f}  R²={r2:+.3f}  MAE/Std={mae_std_ratio:.3f}{marker}")
        except Exception as e:
            print(f"  {name:12s}: FAILED ({e})")

    # Baseline
    baseline_mae = mean_absolute_error(y, np.full_like(y, np.mean(y)))
    print(f"  {'Baseline':12s}: MAE={baseline_mae:7.1f}  (predict mean)")

    if results:
        best = min(results, key=lambda k: results[k]["mae"])
        imp = (1 - results[best]["mae"] / baseline_mae) * 100
        print(f"  → Best: {best} ({imp:+.1f}% vs baseline)")

    return results


def train_classification(df, features, target_col, target_label):
    """Train classification models with LOOCV."""
    y = df[target_col].values
    X = df[features].values
    n_classes = len(np.unique(y))

    print(f"\n{'─'*70}")
    print(f"CLASSIFICATION: {target_label}  ({target_col})")
    print(f"  N={len(y)}  features={len(features)}  classes={n_classes}  dist={dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"{'─'*70}")

    loo = LeaveOneOut()
    results = {}

    is_binary = n_classes == 2

    models = {
        "LogReg": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=0.1, max_iter=5000,
                multi_class="multinomial" if n_classes > 2 else "auto",
            )),
        ]),
        "HGBC": HistGradientBoostingClassifier(
            max_iter=100, max_depth=4, min_samples_leaf=5,
            learning_rate=0.05, l2_regularization=1.0,
            random_state=42,
        ),
        "LGBM_C": lgb.LGBMClassifier(
            n_estimators=100, max_depth=4, num_leaves=15,
            min_child_samples=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=1.0,
            random_state=42, verbose=-1,
            objective="binary" if is_binary else "multiclass",
        ),
        "RF_C": RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=3,
            max_features=0.5, random_state=42,
        ),
    }

    for name, model in models.items():
        try:
            y_pred = cross_val_predict(model, X, y, cv=loo)
            acc = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, average="macro" if n_classes > 2 else "binary")

            res = {"accuracy": acc, "f1": f1}

            if is_binary:
                try:
                    y_prob = cross_val_predict(model, X, y, cv=loo, method="predict_proba")[:, 1]
                    auc = roc_auc_score(y, y_prob)
                    res["auc"] = auc
                except:
                    auc = None
            else:
                auc = None

            results[name] = res
            auc_str = f"  AUC={auc:.3f}" if auc else ""
            marker = " ⭐" if (auc and auc > 0.85) or (not auc and acc > 0.75) else ""
            print(f"  {name:12s}: Acc={acc:.3f}  F1={f1:.3f}{auc_str}{marker}")
        except Exception as e:
            print(f"  {name:12s}: FAILED ({e})")

    # Baseline
    majority_class = np.bincount(y.astype(int)).argmax()
    baseline_acc = (y == majority_class).mean()
    print(f"  {'Baseline':12s}: Acc={baseline_acc:.3f}  (predict majority class={majority_class})")

    # Print confusion matrix for best model
    if results:
        best = max(results, key=lambda k: results[k].get("auc", results[k]["accuracy"]))
        print(f"  → Best: {best}")

        # Recompute predictions for best model
        y_pred_best = cross_val_predict(models[best], X, y, cv=loo)
        if is_binary:
            print(f"\n  Confusion Matrix ({best}):")
            cm = confusion_matrix(y, y_pred_best)
            print(f"              Predicted")
            print(f"              Skip  Trade")
            print(f"  Actual Skip  {cm[0,0]:3d}   {cm[0,1]:3d}")
            print(f"  Actual Trade {cm[1,0]:3d}   {cm[1,1]:3d}")
        else:
            print(f"\n  Classification Report ({best}):")
            print(classification_report(y, y_pred_best, digits=2))

    return results


def feature_importance_analysis(df, features, target_col, target_label):
    """Analyze feature importance with permutation-based approach."""
    y = df[target_col].values
    X = df[features].copy()

    # Use HistGradientBoosting (handles NaN)
    model = HistGradientBoostingRegressor(
        max_iter=100, max_depth=4, min_samples_leaf=5,
        learning_rate=0.05, l2_regularization=1.0,
        random_state=42,
    )
    model.fit(X, y)

    # Permutation importance via LOOCV MAE difference
    loo = LeaveOneOut()
    base_preds = cross_val_predict(model, X.values, y, cv=loo)
    base_mae = mean_absolute_error(y, base_preds)

    importances = {}
    for i, feat_name in enumerate(features):
        X_perm = X.copy()
        X_perm.iloc[:, i] = np.random.RandomState(42).permutation(X_perm.iloc[:, i].values)
        try:
            perm_preds = cross_val_predict(model, X_perm.values, y, cv=loo)
            perm_mae = mean_absolute_error(y, perm_preds)
            importances[feat_name] = perm_mae - base_mae  # Positive = feature helps
        except:
            importances[feat_name] = 0.0

    # Sort by importance
    sorted_imp = sorted(importances.items(), key=lambda x: -x[1])

    print(f"\n{'─'*70}")
    print(f"FEATURE IMPORTANCE: {target_label}")
    print(f"  (Permutation importance: MAE increase when feature shuffled)")
    print(f"{'─'*70}")
    for name, imp in sorted_imp[:20]:
        bar = "█" * int(max(0, imp) * 2)
        print(f"  {name:35s}: {imp:+7.2f} bps  {bar}")

    return sorted_imp


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 train_settlement_ml_v2.py settlement_features_v2.csv")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} settlements × {len(df.columns)} columns from {csv_path}\n")

    # ── Experiment 1: Tier 1 features on ALL 64 settlements ──────────
    print("=" * 70)
    print("EXPERIMENT 1: TIER 1 FEATURES (all 64 settlements)")
    print("  Using: FR + OB.200 + trades + ticker + liquidation + interactions")
    print("=" * 70)

    t1_feats = _available_features(df, TIER1_FEATURES)
    print(f"  Available features: {len(t1_feats)}")

    for target_col, target_label in REG_TARGETS.items():
        if target_col in df.columns:
            mask = df[target_col].notna()
            train_regression(df[mask], t1_feats, target_col, f"T1: {target_label}")

    for target_col, target_label in CLF_TARGETS.items():
        if target_col in df.columns:
            mask = df[target_col].notna()
            train_classification(df[mask], t1_feats, target_col, f"T1: {target_label}")

    # ── Experiment 2: ALL features on 25 settlements with OB.1/50 ────
    print("\n\n" + "=" * 70)
    print("EXPERIMENT 2: ALL FEATURES (25 settlements with OB.1/OB.50)")
    print("  Using: ALL 75+ features including spread dynamics, qty imbalance")
    print("=" * 70)

    df_full = df[df["has_ob1"] == True].copy()
    all_feats = _available_features(df_full, ALL_FEATURES)
    print(f"  Settlements: {len(df_full)} | Available features: {len(all_feats)}")

    for target_col, target_label in REG_TARGETS.items():
        if target_col in df_full.columns:
            mask = df_full[target_col].notna()
            train_regression(df_full[mask], all_feats, target_col, f"T2: {target_label}")

    for target_col, target_label in CLF_TARGETS.items():
        if target_col in df_full.columns:
            mask = df_full[target_col].notna()
            train_classification(df_full[mask], all_feats, target_col, f"T2: {target_label}")

    # ── Experiment 3: FR-only baseline ───────────────────────────────
    print("\n\n" + "=" * 70)
    print("EXPERIMENT 3: FR-ONLY BASELINE (how much does FR alone explain?)")
    print("=" * 70)

    fr_feats = ["fr_bps", "fr_abs_bps", "fr_squared"]
    fr_feats = _available_features(df, fr_feats)

    for target_col, target_label in [("drop_min_bps", "Max drop"), ("target_profitable", "Profitable")]:
        if target_col in df.columns:
            mask = df[target_col].notna()
            if target_col.startswith("target_"):
                train_classification(df[mask], fr_feats, target_col, f"FR-only: {target_label}")
            else:
                train_regression(df[mask], fr_feats, target_col, f"FR-only: {target_label}")

    # ── Feature importance analysis ──────────────────────────────────
    print("\n\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    feature_importance_analysis(df, t1_feats, "drop_min_bps", "Max drop (all 64)")

    if len(df_full) >= 10:
        feature_importance_analysis(df_full, all_feats, "drop_min_bps", "Max drop (25 with OB.1)")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key questions answered:
  1. How much does FR alone explain?  → Experiment 3
  2. Do OB/trade features help beyond FR?  → Compare Exp 1 vs Exp 3
  3. Do high-res OB.1/OB.50 features help?  → Compare Exp 2 vs Exp 1
  4. Which features matter most?  → Feature importance analysis
  5. Which model works best for small N?  → Model comparison in each experiment
""")


if __name__ == "__main__":
    main()
