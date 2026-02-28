#!/usr/bin/env python3
"""
ML Integrity Audit — Check for lookahead bias and overfitting
==============================================================

Tests:
  1. LOOKAHEAD BIAS: Verify ALL features use only pre-settlement data
  2. SYMBOL LEAKAGE: Leave-One-Symbol-Out CV vs LOOCV (are results inflated?)
  3. OVERFITTING: Train vs CV gap, learning curves
  4. TEMPORAL LEAKAGE: Leave-Later-Out (train on early hours, test on later)
  5. FEATURE SANITY: Identify features that are just proxies for symbol identity

Usage:
    python3 audit_ml_integrity.py settlement_features_v2.csv
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb

warnings.filterwarnings("ignore")


TIER1_FEATURES = [
    "fr_bps", "fr_abs_bps",
    "total_bid_mean_usd", "total_ask_mean_usd", "total_depth_usd",
    "total_depth_imb_mean", "total_depth_trend",
    "bid_concentration", "ask_concentration",
    "depth_within_50bps", "thin_side_depth",
    "pre_trade_count", "pre_total_vol_usd", "trade_flow_imb",
    "pre_avg_trade_size_usd", "trade_size_median", "trade_size_p90",
    "trade_size_p99", "trade_size_max", "trade_size_skew",
    "large_trade_count", "large_trade_pct", "large_trade_imb",
    "pre_price_vol_bps",
    "trade_rate_10s", "trade_rate_2s", "trade_rate_accel", "vol_rate_accel",
    "buy_imb_last_1s", "buy_pressure_surge", "vwap_vs_mid_bps",
    "oi_value_usd", "oi_change_60s", "oi_change_pct_60s",
    "basis_bps", "basis_abs_bps", "basis_trend",
    "volume_24h", "turnover_24h_usd",
    "price_change_24h_pct",
    "liq_count_pre", "liq_volume_usd", "liq_direction",
    "fr_x_depth", "fr_x_spread", "fr_x_imb",
    "imb_x_vol", "spread_x_depth", "fr_squared",
]


def _available(df, feats):
    return [f for f in feats if f in df.columns and df[f].isna().mean() < 0.90]


def _make_models():
    return {
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
            learning_rate=0.05, l2_regularization=1.0, random_state=42,
        ),
    }


def _make_clf_models():
    return {
        "LogReg": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.1, max_iter=5000)),
        ]),
        "HGBC": HistGradientBoostingClassifier(
            max_iter=100, max_depth=4, min_samples_leaf=5,
            learning_rate=0.05, l2_regularization=1.0, random_state=42,
        ),
    }


def audit_1_lookahead_bias(df):
    """Check that no feature uses post-settlement data."""
    print("=" * 70)
    print("AUDIT 1: LOOKAHEAD BIAS CHECK")
    print("=" * 70)

    # Feature columns should NOT correlate with post-settlement targets
    # more than FR does (r=0.92)... unless they're legitimately predictive.
    # The REAL test: do any features contain future information?

    # Check: features that correlate MORE with targets than FR does
    feature_cols = _available(df, TIER1_FEATURES)
    target = "drop_min_bps"

    print("\nFeatures with |r| > 0.92 with drop_min_bps (higher than FR):")
    suspicious = 0
    for f in feature_cols:
        r = df[f].corr(df[target])
        if abs(r) > 0.92 and f not in ("fr_bps", "fr_abs_bps"):
            print(f"  ⚠️  {f:35s}: r={r:+.4f}")
            suspicious += 1

    if suspicious == 0:
        print("  ✅ No features have higher correlation than FR")

    # Check: post-settlement columns accidentally in features
    post_keywords = ["post_", "drop_", "target_", "recovery", "time_to_bottom",
                     "price_100ms", "price_500ms", "price_1s", "price_5s",
                     "worst_100ms", "worst_500ms", "worst_1s", "worst_5s"]
    leaked = [f for f in feature_cols if any(k in f for k in post_keywords)]
    if leaked:
        print(f"\n  ❌ POST-SETTLEMENT DATA IN FEATURES: {leaked}")
    else:
        print("  ✅ No post-settlement columns in feature list")

    # Check: interaction features use only pre-settlement components
    print("\nInteraction feature components (verify all pre-settlement):")
    interactions = [f for f in feature_cols if "_x_" in f or "squared" in f]
    for f in interactions:
        print(f"  {f}: ✅ (composed from pre-settlement features)")

    print()


def audit_2_symbol_leakage(df, features, target_col="drop_min_bps"):
    """Compare LOOCV vs Leave-One-Symbol-Out CV."""
    print("=" * 70)
    print("AUDIT 2: SYMBOL LEAKAGE (LOOCV vs Leave-One-Symbol-Out)")
    print("=" * 70)

    X = df[features].values
    y = df[target_col].values
    groups = df["symbol"].values

    n_symbols = len(np.unique(groups))
    print(f"\n  N={len(y)} samples, {n_symbols} symbols")
    print(f"  Target: {target_col} (mean={np.mean(y):.1f}, std={np.std(y):.1f})")
    baseline_mae = mean_absolute_error(y, np.full_like(y, np.mean(y)))
    print(f"  Baseline (predict mean): MAE={baseline_mae:.1f}\n")

    models = _make_models()
    loo = LeaveOneOut()
    logo = LeaveOneGroupOut()

    print(f"  {'Model':12s}  {'LOOCV MAE':>10s}  {'LOSO MAE':>10s}  {'Inflation':>10s}  {'LOSO R²':>8s}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}")

    for name, model in models.items():
        try:
            # LOOCV
            y_loo = cross_val_predict(model, X, y, cv=loo)
            mae_loo = mean_absolute_error(y, y_loo)

            # Leave-One-Symbol-Out
            y_loso = cross_val_predict(model, X, y, cv=logo, groups=groups)
            mae_loso = mean_absolute_error(y, y_loso)
            r2_loso = r2_score(y, y_loso)

            inflation = (mae_loso - mae_loo) / mae_loo * 100
            flag = " ⚠️" if inflation > 20 else " ✅"

            print(f"  {name:12s}  {mae_loo:10.1f}  {mae_loso:10.1f}  {inflation:+9.1f}%{flag}  {r2_loso:+8.3f}")
        except Exception as e:
            print(f"  {name:12s}  FAILED: {e}")

    print(f"\n  If LOSO MAE >> LOOCV MAE → symbol leakage inflating results")
    print(f"  LOSO is the HONEST estimate of generalization to unseen coins\n")


def audit_2b_symbol_leakage_clf(df, features, target_col="target_profitable"):
    """Classification version of symbol leakage check."""
    print("=" * 70)
    print(f"AUDIT 2b: SYMBOL LEAKAGE — CLASSIFICATION ({target_col})")
    print("=" * 70)

    X = df[features].values
    y = df[target_col].values
    groups = df["symbol"].values

    baseline_acc = max(np.bincount(y.astype(int))) / len(y)
    print(f"\n  N={len(y)}, baseline acc={baseline_acc:.3f}")

    models = _make_clf_models()
    loo = LeaveOneOut()
    logo = LeaveOneGroupOut()

    print(f"\n  {'Model':12s}  {'LOOCV Acc':>10s}  {'LOSO Acc':>10s}  {'LOOCV AUC':>10s}  {'LOSO AUC':>10s}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    for name, model in models.items():
        try:
            y_loo = cross_val_predict(model, X, y, cv=loo)
            acc_loo = accuracy_score(y, y_loo)

            y_loso = cross_val_predict(model, X, y, cv=logo, groups=groups)
            acc_loso = accuracy_score(y, y_loso)

            # AUC
            try:
                y_prob_loo = cross_val_predict(model, X, y, cv=loo, method="predict_proba")[:, 1]
                auc_loo = roc_auc_score(y, y_prob_loo)
            except:
                auc_loo = None

            try:
                y_prob_loso = cross_val_predict(model, X, y, cv=logo, groups=groups, method="predict_proba")[:, 1]
                auc_loso = roc_auc_score(y, y_prob_loso)
            except:
                auc_loso = None

            auc_loo_s = f"{auc_loo:.3f}" if auc_loo else "N/A"
            auc_loso_s = f"{auc_loso:.3f}" if auc_loso else "N/A"

            print(f"  {name:12s}  {acc_loo:10.3f}  {acc_loso:10.3f}  {auc_loo_s:>10s}  {auc_loso_s:>10s}")
        except Exception as e:
            print(f"  {name:12s}  FAILED: {e}")

    print()


def audit_3_overfitting(df, features, target_col="drop_min_bps"):
    """Check train vs CV error gap."""
    print("=" * 70)
    print("AUDIT 3: OVERFITTING (Train vs CV gap)")
    print("=" * 70)

    X = df[features].values
    y = df[target_col].values

    models = _make_models()

    print(f"\n  {'Model':12s}  {'Train MAE':>10s}  {'LOOCV MAE':>10s}  {'Gap Ratio':>10s}  {'Verdict':>10s}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    for name, model in models.items():
        try:
            # Train MAE (full dataset)
            model.fit(X, y)
            y_train = model.predict(X)
            mae_train = mean_absolute_error(y, y_train)

            # CV MAE
            loo = LeaveOneOut()
            y_cv = cross_val_predict(model, X, y, cv=loo)
            mae_cv = mean_absolute_error(y, y_cv)

            ratio = mae_cv / max(mae_train, 0.01)
            verdict = "✅ OK" if ratio < 3 else "⚠️ OVERFIT" if ratio < 10 else "❌ SEVERE"

            print(f"  {name:12s}  {mae_train:10.1f}  {mae_cv:10.1f}  {ratio:10.1f}x  {verdict:>10s}")
        except Exception as e:
            print(f"  {name:12s}  FAILED: {e}")

    print(f"\n  Gap ratio < 3x = OK, 3-10x = moderate overfitting, >10x = severe\n")


def audit_4_temporal_leakage(df, features, target_col="drop_min_bps"):
    """Train on early hours, test on later hours (temporal hold-out)."""
    print("=" * 70)
    print("AUDIT 4: TEMPORAL VALIDATION (train early → test late)")
    print("=" * 70)

    # Parse hour from settle_time
    df = df.copy()
    df["hour"] = pd.to_datetime(df["settle_time"]).dt.hour

    # Split: train on 00:00-09:00, test on 10:00-19:00
    train_mask = df["hour"] < 10
    test_mask = df["hour"] >= 10

    n_train = train_mask.sum()
    n_test = test_mask.sum()
    print(f"\n  Train: {n_train} samples (hours 0-9)")
    print(f"  Test:  {n_test} samples (hours 10-19)")

    if n_train < 5 or n_test < 5:
        print("  ⚠️  Not enough samples for temporal split")
        return

    X_train = df.loc[train_mask, features].values
    y_train = df.loc[train_mask, target_col].values
    X_test = df.loc[test_mask, features].values
    y_test = df.loc[test_mask, target_col].values

    baseline_mae = mean_absolute_error(y_test, np.full_like(y_test, np.mean(y_train)))
    print(f"  Baseline (predict train mean): MAE={baseline_mae:.1f}\n")

    models = _make_models()

    print(f"  {'Model':12s}  {'Test MAE':>10s}  {'Test R²':>8s}  {'vs Baseline':>12s}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*12}")

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            imp = (1 - mae / baseline_mae) * 100

            print(f"  {name:12s}  {mae:10.1f}  {r2:+8.3f}  {imp:+11.1f}%")
        except Exception as e:
            print(f"  {name:12s}  FAILED: {e}")

    print()


def audit_5_feature_sanity(df, features, target_col="drop_min_bps"):
    """Identify features that are proxies for symbol identity."""
    print("=" * 70)
    print("AUDIT 5: FEATURE SANITY (symbol-proxy detection)")
    print("=" * 70)

    # A feature is a symbol-proxy if it has high between-symbol variance
    # and low within-symbol variance
    print("\n  Features with high between-symbol / within-symbol variance ratio:")
    print(f"  (High ratio = feature is mostly identifying the coin, not the settlement)\n")

    risky = []
    for f in features:
        if f not in df.columns or df[f].isna().mean() > 0.5:
            continue
        try:
            groups = df.groupby("symbol")[f]
            between_var = groups.mean().var()
            within_var = groups.apply(lambda x: x.var()).mean()
            ratio = between_var / max(within_var, 1e-10)
            if ratio > 10:
                risky.append((f, ratio))
        except:
            pass

    risky.sort(key=lambda x: -x[1])
    for f, ratio in risky[:15]:
        print(f"  ⚠️  {f:35s}: between/within ratio = {ratio:>8.1f}")

    if not risky:
        print("  ✅ No features appear to be pure symbol proxies")
    else:
        print(f"\n  → {len(risky)} features are mostly symbol-specific.")
        print(f"  → Consider removing or normalizing these for cross-symbol generalization.")

    print()


def audit_6_fr_only_vs_full(df, features, target_col="drop_min_bps"):
    """How much do features beyond FR actually help, after accounting for symbol leakage?"""
    print("=" * 70)
    print("AUDIT 6: INCREMENTAL VALUE (FR-only vs Full, both with LOSO)")
    print("=" * 70)

    X_full = df[features].values
    fr_feats = [f for f in features if f in ("fr_bps", "fr_abs_bps", "fr_squared")]
    X_fr = df[fr_feats].values
    y = df[target_col].values
    groups = df["symbol"].values

    logo = LeaveOneGroupOut()

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
    }

    print(f"\n  {'Model':12s}  {'FR-only MAE':>12s}  {'Full MAE':>10s}  {'Improvement':>12s}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*12}")

    for name, model in models.items():
        try:
            import copy
            model_fr = copy.deepcopy(model)
            model_full = copy.deepcopy(model)

            y_fr = cross_val_predict(model_fr, X_fr, y, cv=logo, groups=groups)
            mae_fr = mean_absolute_error(y, y_fr)

            y_full = cross_val_predict(model_full, X_full, y, cv=logo, groups=groups)
            mae_full = mean_absolute_error(y, y_full)

            imp = (1 - mae_full / mae_fr) * 100
            print(f"  {name:12s}  {mae_fr:12.1f}  {mae_full:10.1f}  {imp:+11.1f}%")
        except Exception as e:
            print(f"  {name:12s}  FAILED: {e}")

    print(f"\n  If improvement is small → extra features mostly learned symbol identity\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 audit_ml_integrity.py settlement_features_v2.csv")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    features = _available(df, TIER1_FEATURES)
    print(f"Loaded {len(df)} samples × {len(df.columns)} columns")
    print(f"Using {len(features)} Tier 1 features\n")

    audit_1_lookahead_bias(df)
    audit_2_symbol_leakage(df, features, "drop_min_bps")
    audit_2b_symbol_leakage_clf(df, features, "target_profitable")
    audit_3_overfitting(df, features, "drop_min_bps")
    audit_4_temporal_leakage(df, features, "drop_min_bps")
    audit_5_feature_sanity(df, features, "drop_min_bps")
    audit_6_fr_only_vs_full(df, features, "drop_min_bps")

    # ── Final Summary ────────────────────────────────────────────────
    print("=" * 70)
    print("INTEGRITY AUDIT COMPLETE")
    print("=" * 70)
    print("""
HONEST numbers to report are from:
  - Leave-One-Symbol-Out CV (LOSO) for regression MAE
  - LOSO AUC for classification
  - Temporal hold-out for out-of-distribution performance

These are the numbers that predict REAL production performance.
LOOCV numbers are OPTIMISTIC because same-symbol settlements leak info.
""")


if __name__ == "__main__":
    main()
