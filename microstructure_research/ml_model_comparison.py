#!/usr/bin/env python3
"""
ML Model Comparison — Single Target Deep Dive

Compares multiple ML models on tgt_breakout_any_5 using the same
WFO temporal split (12mo train, 3d purge, 30d test).

Models tested:
  1. Logistic Regression (current baseline)
  2. Ridge Classifier (linear, regularized)
  3. Random Forest
  4. LightGBM (default params)
  5. LightGBM (tuned via Optuna)
  6. XGBoost (default params)
  7. Ensemble (avg of top 3)

Also tests feature engineering:
  - Raw features only
  - Raw + lag features (1, 2, 3 bar lags)
  - Raw + lag + interaction features

Metrics: AUC, Precision@10%, Precision@20%, Brier score, Log loss
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss,
    precision_score, recall_score, f1_score,
    precision_recall_curve, r2_score, mean_squared_error
)
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

# ============================================================
# PARAMETERS
# ============================================================
TARGETS = [
    # Well-balanced targets (base rate 10-90%, balance score > 0.5)
    # Regime / volatility
    "tgt_vol_expansion_10",   # 49.3% — regime signal
    "tgt_vol_expansion_5",    # 39.4% — shorter horizon vol
    # Directional breakout
    "tgt_breakout_up_3",      # 66.2% — directional up
    "tgt_breakout_down_3",    # 64.1% — directional down
    "tgt_breakout_up_5",      # 73.1% — directional up (longer)
    "tgt_breakout_down_5",    # 71.7% — directional down (longer)
    # Risk management
    "tgt_tail_event_5",       # 35.5% — tail risk
    "tgt_tail_event_3",       # 24.9% — tail risk (shorter)
    # Profitability
    "tgt_profitable_long_1",  # 49.4% — directly tradeable
    "tgt_profitable_short_1", # 48.2% — directly tradeable
    "tgt_profitable_long_3",  # 49.9% — 3-bar hold
    "tgt_profitable_short_3", # 48.7% — 3-bar hold
    "tgt_profitable_long_5",  # 49.3% — 5-bar hold
    "tgt_profitable_short_5", # 49.8% — 5-bar hold
    "tgt_profitable_long_10", # 50.5% — 10-bar hold
    "tgt_profitable_short_10",# 48.8% — 10-bar hold
    # Alpha
    "tgt_alpha_1",            # continuous — 1-bar alpha
    "tgt_relative_ret_1",     # continuous — 1-bar relative return
    # Other
    "tgt_adverse_selection_1",# 51.0% — adverse selection
]
SYMBOL = "SOLUSDT"
TF = "4h"
FEATURES_DIR = Path("./features")
CONSTRAINTS_PATH = Path("./microstructure_research/predictable_targets.json")

SELECTION_DAYS = 360
PURGE_DAYS = 3
TRADE_DAYS = 30

# ============================================================
# DATA LOADING
# ============================================================
def load_features(features_dir, symbol, tf):
    tf_dir = features_dir / symbol / tf
    files = sorted(tf_dir.glob("*.parquet"))
    if not files:
        print(f"ERROR: No parquet files in {tf_dir}")
        sys.exit(1)
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def classify_columns(df):
    tgt_cols = [c for c in df.columns if c.startswith("tgt_")]
    feat_cols = [c for c in df.columns if not c.startswith("tgt_")]
    binary_tgts, continuous_tgts = [], []
    for c in tgt_cols:
        if df[c].dropna().nunique() <= 3:
            binary_tgts.append(c)
        else:
            continuous_tgts.append(c)
    return feat_cols, continuous_tgts, binary_tgts


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def add_lag_features(df, feat_cols, lags=[1, 2, 3]):
    """Add lagged versions of features."""
    new_cols = {}
    for lag in lags:
        for col in feat_cols:
            new_cols[f"{col}_lag{lag}"] = df[col].shift(lag)
    lag_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, lag_df], axis=1)


def add_interaction_features(df, feat_cols):
    """Add key interaction features based on domain knowledge."""
    new_cols = {}

    # Volatility × activity interactions
    vol_feats = [c for c in feat_cols if any(v in c for v in
                 ['realized_vol', 'atr_change', 'max_drawdown', 'max_drawup'])]
    activity_feats = [c for c in feat_cols if any(v in c for v in
                      ['trade_rate', 'volume_per_second', 'burstiness'])]

    for vf in vol_feats[:3]:
        for af in activity_feats[:3]:
            if vf in df.columns and af in df.columns:
                new_cols[f"ix_{vf}_x_{af}"] = df[vf] * df[af]

    # Rolling stats on key features
    key_feats = ['realized_vol', 'burstiness', 'atr_change_pct_14',
                 'trade_rate_std', 'volume_per_second_std']
    for col in key_feats:
        if col in df.columns:
            new_cols[f"{col}_roll3_mean"] = df[col].rolling(3, min_periods=1).mean()
            new_cols[f"{col}_roll3_std"] = df[col].rolling(3, min_periods=1).std()
            new_cols[f"{col}_diff1"] = df[col].diff(1)

    int_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, int_df], axis=1)


# ============================================================
# BINARY MODEL DEFINITIONS
# ============================================================
def train_logistic(X_train, y_train, X_test):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    model = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
    model.fit(Xtr, y_train)
    return model.predict_proba(Xte)[:, 1]


def train_ridge_clf(X_train, y_train, X_test):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    model = RidgeClassifier(alpha=1.0)
    model.fit(Xtr, y_train)
    scores = model.decision_function(Xte)
    return 1 / (1 + np.exp(-scores))


def train_random_forest_clf(X_train, y_train, X_test):
    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=20,
        max_features="sqrt", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def train_lightgbm_clf(X_train, y_train, X_test):
    model = lgb.LGBMClassifier(
        objective="binary", metric="auc", verbosity=-1,
        n_estimators=300, max_depth=6, learning_rate=0.05,
        num_leaves=31, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def train_xgboost_clf(X_train, y_train, X_test):
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        eval_metric="auc", random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


# ============================================================
# REGRESSION MODEL DEFINITIONS
# ============================================================
def train_ridge_reg(X_train, y_train, X_test):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    model = Ridge(alpha=1.0)
    model.fit(Xtr, y_train)
    return model.predict(Xte)


def train_random_forest_reg(X_train, y_train, X_test):
    model = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=20,
        max_features="sqrt", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def train_lightgbm_reg(X_train, y_train, X_test):
    model = lgb.LGBMRegressor(
        objective="regression", metric="rmse", verbosity=-1,
        n_estimators=300, max_depth=6, learning_rate=0.05,
        num_leaves=31, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42,
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def train_xgboost_reg(X_train, y_train, X_test):
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


# ============================================================
# EVALUATION
# ============================================================
def evaluate_binary(y_true, y_prob, model_name):
    """Compute metrics for binary classification."""
    auc = roc_auc_score(y_true, y_prob)
    metrics = {"model": model_name, "score": auc, "score_type": "AUC",
               "score_dev": auc - 0.5}

    for pct in [10, 20]:
        threshold = np.percentile(y_prob, 100 - pct)
        pred_pos = y_prob >= threshold
        if pred_pos.sum() > 0:
            prec = y_true[pred_pos].mean()
            base_rate = y_true.mean()
            lift = prec / base_rate if base_rate > 0 else 0
            metrics[f"prec_top{pct}"] = prec
            metrics[f"lift_top{pct}"] = lift
        else:
            metrics[f"prec_top{pct}"] = np.nan
            metrics[f"lift_top{pct}"] = np.nan

    metrics["brier"] = brier_score_loss(y_true, y_prob)
    return metrics


def evaluate_regression(y_true, y_pred, model_name):
    """Compute metrics for regression."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics = {"model": model_name, "score": r2, "score_type": "R2",
               "score_dev": r2}
    metrics["rmse"] = rmse
    # Directional accuracy: does prediction sign match actual sign?
    dir_acc = np.mean(np.sign(y_pred) == np.sign(y_true))
    metrics["dir_accuracy"] = dir_acc
    return metrics


# ============================================================
# MAIN EXPERIMENT
# ============================================================
def main():
    t_start = time.time()

    print("=" * 80)
    print(f"  ML MODEL COMPARISON")
    print(f"  Symbol: {SYMBOL} {TF}")
    print(f"  Targets: {TARGETS}")
    print("=" * 80)

    # Load data
    print("\n  Loading data...", flush=True)
    df = load_features(FEATURES_DIR, SYMBOL, TF)
    print(f"  Loaded {len(df)} candles, range: {df.index[0]} -> {df.index[-1]}")
    print(f"  Targets: {TARGETS}")

    # Load constraints
    with open(CONSTRAINTS_PATH) as f:
        constraints = json.load(f)

    # Run for each target
    all_target_results = []
    for TARGET in TARGETS:
        all_target_results.extend(
            run_single_target(df, TARGET, constraints, TF)
        )

    # Final cross-target summary
    results_df = pd.DataFrame(all_target_results)
    results_df = results_df.sort_values(["target", "score"], ascending=[True, False])

    print(f"\n\n{'#'*80}")
    print(f"  CROSS-TARGET SUMMARY")
    print(f"{'#'*80}")
    for tgt in TARGETS:
        tgt_df = results_df[results_df["target"] == tgt].head(3)
        if len(tgt_df) == 0:
            continue
        stype = tgt_df["score_type"].iloc[0]
        print(f"\n  {tgt} ({stype}):")
        for _, row in tgt_df.iterrows():
            print(f"    {row['model']:<20s} {row['feature_set']:<20s} "
                  f"{stype}={row['score']:.4f}")

    # Save
    out_path = Path("microstructure_research/results")
    out_path.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path / f"ml_comparison_{SYMBOL}_multi_target.csv", index=False)
    print(f"\n  Results saved.")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'#'*80}")


def run_single_target(df, TARGET, constraints, TF):
    """Run model comparison for a single target."""
    print(f"\n\n{'='*80}")
    print(f"  TARGET: {TARGET}")
    print(f"{'='*80}")

    target_features = constraints["target_features"].get(TARGET, [])
    print(f"  Features from research: {len(target_features)}")

    available_feats = [f for f in target_features if f in df.columns]
    missing_feats = [f for f in target_features if f not in df.columns]
    print(f"  Available: {len(available_feats)}, Missing: {len(missing_feats)}")
    if missing_feats:
        print(f"  Missing: {missing_feats[:5]}...")

    if TARGET not in df.columns:
        print(f"  ERROR: Target {TARGET} not in data!")
        return []

    # ---- Temporal split ----
    candles_per_day = {"5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}[TF]
    sel_candles = SELECTION_DAYS * candles_per_day
    purge_candles = PURGE_DAYS * candles_per_day
    trade_candles = TRADE_DAYS * candles_per_day

    total_needed = sel_candles + purge_candles + trade_candles
    if len(df) < total_needed:
        print(f"  ERROR: Need {total_needed} candles, have {len(df)}")
        sys.exit(1)

    # Use first period: train on first 360d, purge 3d, test on next 30d
    sel_end = sel_candles
    trade_start = sel_end + purge_candles
    trade_end = trade_start + trade_candles

    df_sel = df.iloc[:sel_end].copy()
    df_trade = df.iloc[trade_start:trade_end].copy()

    print(f"\n  Selection: {len(df_sel)} candles [{df_sel.index[0]} -> {df_sel.index[-1]}]")
    print(f"  Trade:     {len(df_trade)} candles [{df_trade.index[0]} -> {df_trade.index[-1]}]")

    # Target stats
    y_sel = df_sel[TARGET].values
    y_trade = df_trade[TARGET].values
    sel_valid = np.isfinite(y_sel)
    trade_valid = np.isfinite(y_trade)
    sel_pos_rate = y_sel[sel_valid].mean()
    trade_pos_rate = y_trade[trade_valid].mean()
    print(f"\n  Selection target: {sel_valid.sum()} valid, "
          f"positive rate = {sel_pos_rate:.4f}")
    print(f"  Trade target:     {trade_valid.sum()} valid, "
          f"positive rate = {trade_pos_rate:.4f}")

    if sel_pos_rate > 0.95 or sel_pos_rate < 0.05:
        print(f"  WARNING: Extreme base rate ({sel_pos_rate:.4f}) — target may be trivial")

    # ============================================================
    # FEATURE SETS
    # ============================================================
    all_results = []
    feature_sets = {}

    # Set 1: Raw features only
    feature_sets["raw"] = available_feats

    # Set 2: Raw + lags
    df_sel_lag = add_lag_features(df_sel, available_feats, lags=[1, 2, 3])
    df_trade_lag = add_lag_features(df_trade, available_feats, lags=[1, 2, 3])
    lag_cols = [c for c in df_sel_lag.columns
                if c not in df_sel.columns and not c.startswith("tgt_")]
    feature_sets["raw+lags"] = available_feats + lag_cols

    # Also add all core features as a feature set
    core_features = constraints["core_features"]
    available_core = [f for f in core_features if f in df.columns]
    feature_sets["all_67_core"] = available_core

    # Add core + lags
    df_sel_core_lag = add_lag_features(df_sel, available_core, lags=[1, 2, 3])
    df_trade_core_lag = add_lag_features(df_trade, available_core, lags=[1, 2, 3])
    core_lag_cols = [c for c in df_sel_core_lag.columns
                     if c not in df_sel.columns and not c.startswith("tgt_")]
    feature_sets["core+lags"] = available_core + core_lag_cols

    print(f"\n  Feature sets:")
    for name, feats in feature_sets.items():
        print(f"    {name}: {len(feats)} features")

    # Detect if binary or continuous
    is_binary = df[TARGET].dropna().nunique() <= 3
    print(f"  Target type: {'binary' if is_binary else 'continuous'}")

    # ============================================================
    # RUN ALL MODEL × FEATURE SET COMBINATIONS
    # ============================================================
    if is_binary:
        model_configs = [
            ("Logistic", train_logistic),
            ("RidgeClf", train_ridge_clf),
            ("RandomForest", train_random_forest_clf),
            ("LightGBM", train_lightgbm_clf),
            ("XGBoost", train_xgboost_clf),
        ]
    else:
        model_configs = [
            ("Ridge", train_ridge_reg),
            ("RandomForest", train_random_forest_reg),
            ("LightGBM", train_lightgbm_reg),
            ("XGBoost", train_xgboost_reg),
        ]

    for feat_name, feat_cols in feature_sets.items():
        print(f"\n{'='*70}")
        print(f"  FEATURE SET: {feat_name} ({len(feat_cols)} features)")
        print(f"{'='*70}")

        # Use the enriched dataframes for lag/interaction features
        if feat_name == "core+lags":
            df_s, df_t = df_sel_core_lag, df_trade_core_lag
        elif "lags" in feat_name:
            df_s, df_t = df_sel_lag, df_trade_lag
        else:
            df_s, df_t = df_sel, df_trade

        # Filter to available columns
        actual_feats = [f for f in feat_cols if f in df_s.columns and f in df_t.columns]
        if len(actual_feats) < 3:
            print(f"  SKIP: only {len(actual_feats)} features available")
            continue

        # Prepare arrays
        X_sel = df_s[actual_feats].values
        y_sel_arr = df_s[TARGET].values
        X_trade = df_t[actual_feats].values
        y_trade_arr = df_t[TARGET].values

        # Clean: keep only rows where all features and target are finite
        sel_mask = np.all(np.isfinite(X_sel), axis=1) & np.isfinite(y_sel_arr)
        trade_mask = np.all(np.isfinite(X_trade), axis=1) & np.isfinite(y_trade_arr)

        X_tr = np.nan_to_num(X_sel[sel_mask], nan=0, posinf=0, neginf=0)
        X_te = np.nan_to_num(X_trade[trade_mask], nan=0, posinf=0, neginf=0)
        if is_binary:
            y_tr = y_sel_arr[sel_mask].astype(int)
            y_te = y_trade_arr[trade_mask].astype(int)
        else:
            y_tr = y_sel_arr[sel_mask]
            y_te = y_trade_arr[trade_mask]

        print(f"  Train: {len(X_tr)} samples, Test: {len(X_te)} samples")
        if is_binary:
            print(f"  Train pos rate: {y_tr.mean():.4f}, Test pos rate: {y_te.mean():.4f}")
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                print(f"  SKIP: not enough class diversity")
                continue
        else:
            print(f"  Train mean: {y_tr.mean():.6f}, Test mean: {y_te.mean():.6f}")

        # Run each model
        eval_fn = evaluate_binary if is_binary else evaluate_regression
        predictions = {}
        for model_name, train_fn in model_configs:
            t0 = time.time()
            print(f"\n    {model_name}...", end="", flush=True)
            try:
                y_pred = train_fn(X_tr, y_tr, X_te)
                elapsed = time.time() - t0
                metrics = eval_fn(y_te, y_pred, model_name)
                metrics["feature_set"] = feat_name
                metrics["n_features"] = len(actual_feats)
                metrics["train_time_s"] = elapsed
                metrics["target"] = TARGET
                metrics["is_binary"] = is_binary
                all_results.append(metrics)
                predictions[model_name] = y_pred
                print(f" {metrics['score_type']}={metrics['score']:.4f} "
                      f"(dev={metrics['score_dev']:+.4f}) [{elapsed:.1f}s]")
            except Exception as e:
                print(f" ERROR: {e}")

        # Ensemble: average of tree models
        tree_models = ["LightGBM", "XGBoost", "RandomForest"]
        available_preds = {k: v for k, v in predictions.items() if k in tree_models}
        if len(available_preds) >= 2:
            ensemble_pred = np.mean(list(available_preds.values()), axis=0)
            metrics = eval_fn(y_te, ensemble_pred, "Ensemble_Trees")
            metrics["feature_set"] = feat_name
            metrics["n_features"] = len(actual_feats)
            metrics["train_time_s"] = 0
            metrics["target"] = TARGET
            metrics["is_binary"] = is_binary
            all_results.append(metrics)
            print(f"\n    Ensemble_Trees: {metrics['score_type']}={metrics['score']:.4f} "
                  f"(dev={metrics['score_dev']:+.4f})")

        # Ensemble: all models
        if len(predictions) >= 3:
            all_pred = np.mean(list(predictions.values()), axis=0)
            metrics = eval_fn(y_te, all_pred, "Ensemble_All")
            metrics["feature_set"] = feat_name
            metrics["n_features"] = len(actual_feats)
            metrics["train_time_s"] = 0
            metrics["target"] = TARGET
            metrics["is_binary"] = is_binary
            all_results.append(metrics)
            print(f"    Ensemble_All:   {metrics['score_type']}={metrics['score']:.4f} "
                  f"(dev={metrics['score_dev']:+.4f})")

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("score", ascending=False)

    stype = results_df["score_type"].iloc[0] if len(results_df) > 0 else "Score"
    print(f"\n  {'Model':<20s} {'Features':<20s} {stype:>6s} {'Dev':>8s} {'Time':>5s}")
    print(f"  {'-'*65}")

    for _, row in results_df.iterrows():
        extra = ""
        if is_binary:
            extra = f" P@10={row.get('prec_top10', 0):.3f} Lift={row.get('lift_top10', 0):.2f}"
        else:
            extra = f" DirAcc={row.get('dir_accuracy', 0):.3f}"
        print(f"  {row['model']:<20s} {row['feature_set']:<20s} "
              f"{row['score']:>6.4f} {row['score_dev']:>+8.4f} "
              f"{row['train_time_s']:>5.1f}s{extra}")

    # Best per model type
    baseline_model = "Logistic" if is_binary else "Ridge"
    baseline_df = results_df[results_df["model"] == baseline_model]
    baseline_score = baseline_df["score"].max() if len(baseline_df) > 0 else 0.5
    baseline_ref = 0.5 if is_binary else 0.0
    print(f"\n  --- Best {stype} per Model (baseline {baseline_model}={baseline_score:.4f}) ---")
    best_per_model = results_df.groupby("model")["score"].max().sort_values(ascending=False)
    for model, score in best_per_model.items():
        if (baseline_score - baseline_ref) > 0.001:
            pct_improve = ((score - baseline_ref) / (baseline_score - baseline_ref) - 1) * 100
            print(f"    {model:<20s}: {stype}={score:.4f} ({pct_improve:+.0f}% vs baseline)")
        else:
            print(f"    {model:<20s}: {stype}={score:.4f}")

    return all_results


if __name__ == "__main__":
    main()
