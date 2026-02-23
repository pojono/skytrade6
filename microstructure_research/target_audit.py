#!/usr/bin/env python3
"""
Target Audit: Quick ML validation of all 120 targets.

For each target, runs a fast 3-fold expanding-window WFO with LightGBM:
  - Train on expanding window, predict on 90-day test window
  - Binary targets: AUC ROC (score = AUC - 0.5, so >0 = better than random)
  - Continuous targets: Spearman correlation
  - Filters out targets with extreme base rates (<5% or >95%)

Output: CSV + summary with usability classification for each target.

Usage:
  python microstructure_research/target_audit.py
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


# ============================================================
# PARAMETERS
# ============================================================
SYMBOL = "SOLUSDT"
TF = "4h"
FEATURES_DIR = Path("./features")
RESULTS_DIR = Path("./microstructure_research/results")

# WFO params
MIN_TRAIN_CANDLES = 1440   # ~240 days minimum training
TEST_CANDLES = 540         # ~90 days per test fold
N_FOLDS = 3
TOP_FEATURES = 30          # use top 30 features per target (by abs correlation)

# LightGBM fast params
LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "n_jobs": -1,
}

LGB_REG_PARAMS = {
    "objective": "regression",
    "metric": "mse",
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "n_jobs": -1,
}


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


def classify_target(series):
    """Classify target as binary or continuous, return base rate info."""
    vals = series.dropna()
    n = len(vals)
    unique = vals.nunique()

    if unique <= 2:
        p1 = vals.mean()
        return {
            "type": "binary",
            "n": n,
            "base_rate": p1,
            "usable": 0.05 <= p1 <= 0.95,
            "good_rate": 0.15 <= p1 <= 0.85,
        }
    else:
        return {
            "type": "continuous",
            "n": n,
            "base_rate": None,
            "usable": True,
            "good_rate": True,
        }


def select_features(df, target_col, feat_cols, top_n=30):
    """Select top N features by absolute correlation with target."""
    y = df[target_col].values
    valid = np.isfinite(y)
    corrs = []
    for f in feat_cols:
        x = df[f].values
        mask = valid & np.isfinite(x)
        if mask.sum() < 100:
            corrs.append(0.0)
            continue
        try:
            c, _ = spearmanr(x[mask], y[mask])
            corrs.append(abs(c) if np.isfinite(c) else 0.0)
        except:
            corrs.append(0.0)
    idx = np.argsort(corrs)[::-1][:top_n]
    return [feat_cols[i] for i in idx]


def run_wfo(df, target_col, feat_cols, info):
    """Run expanding-window WFO for a single target."""
    n = len(df)
    is_binary = info["type"] == "binary"

    # Calculate fold boundaries (from end)
    fold_results = []
    for fold in range(N_FOLDS):
        test_end = n - fold * TEST_CANDLES
        test_start = test_end - TEST_CANDLES
        train_end = test_start

        if train_end < MIN_TRAIN_CANDLES:
            continue

        df_train = df.iloc[:train_end]
        df_test = df.iloc[test_start:test_end]

        # Select features on training data only
        selected = select_features(df_train, target_col, feat_cols, TOP_FEATURES)
        if len(selected) < 5:
            continue

        X_train = df_train[selected].values
        y_train = df_train[target_col].values
        X_test = df_test[selected].values
        y_test = df_test[target_col].values

        # Remove NaN rows
        train_valid = np.all(np.isfinite(X_train), axis=1) & np.isfinite(y_train)
        test_valid = np.all(np.isfinite(X_test), axis=1) & np.isfinite(y_test)

        X_tr, y_tr = X_train[train_valid], y_train[train_valid]
        X_te, y_te = X_test[test_valid], y_test[test_valid]

        if len(X_tr) < 200 or len(X_te) < 50:
            continue

        # Check test set has variance
        if is_binary and (y_te.mean() < 0.01 or y_te.mean() > 0.99):
            continue
        if not is_binary and y_te.std() < 1e-10:
            continue

        try:
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            if is_binary:
                model = lgb.LGBMClassifier(**LGB_PARAMS)
                model.fit(X_tr_s, y_tr)
                preds = model.predict_proba(X_te_s)[:, 1]
                score = roc_auc_score(y_te, preds) - 0.5  # center at 0
            else:
                model = lgb.LGBMRegressor(**LGB_REG_PARAMS)
                model.fit(X_tr_s, y_tr)
                preds = model.predict(X_te_s)
                score, _ = spearmanr(y_te, preds)
                if not np.isfinite(score):
                    score = 0.0

            fold_results.append({
                "fold": fold,
                "score": score,
                "n_train": len(X_tr),
                "n_test": len(X_te),
            })
        except Exception as e:
            continue

    if not fold_results:
        return None

    scores = [r["score"] for r in fold_results]
    return {
        "mean_score": np.mean(scores),
        "min_score": np.min(scores),
        "max_score": np.max(scores),
        "n_folds": len(fold_results),
        "all_positive": all(s > 0 for s in scores),
    }


def main():
    t0 = time.time()
    print("=" * 80)
    print("  TARGET AUDIT — Quick ML Validation of All Targets")
    print(f"  {SYMBOL} {TF}, {N_FOLDS}-fold expanding WFO, LightGBM")
    print("=" * 80)

    # Load data
    print("\n  Loading data...", flush=True)
    df = load_features(FEATURES_DIR, SYMBOL, TF)
    print(f"  Loaded {len(df)} candles, {df.index[0].date()} -> {df.index[-1].date()}")

    # Identify targets and features
    tgt_cols = sorted([c for c in df.columns if c.startswith("tgt_")])
    feat_cols = [c for c in df.columns
                 if not c.startswith("tgt_")
                 and c not in ("open", "high", "low", "close", "volume")]
    print(f"  Targets: {len(tgt_cols)}, Features: {len(feat_cols)}")

    # Classify all targets
    print("\n  Classifying targets...", flush=True)
    target_info = {}
    for t in tgt_cols:
        target_info[t] = classify_target(df[t])

    binary_count = sum(1 for v in target_info.values() if v["type"] == "binary")
    cont_count = sum(1 for v in target_info.values() if v["type"] == "continuous")
    usable_count = sum(1 for v in target_info.values() if v["usable"])
    print(f"  Binary: {binary_count}, Continuous: {cont_count}")
    print(f"  Usable base rate: {usable_count}/{len(tgt_cols)}")

    # Run WFO for each usable target
    results = []
    skipped = []
    n_usable = sum(1 for v in target_info.values() if v["usable"])

    print(f"\n  Running WFO on {n_usable} usable targets...\n", flush=True)
    done = 0
    for t in tgt_cols:
        info = target_info[t]
        if not info["usable"]:
            skipped.append({
                "target": t,
                "type": info["type"],
                "base_rate": info["base_rate"],
                "reason": f"extreme base rate ({info['base_rate']:.1%})" if info["base_rate"] is not None else "unusable",
                "mean_score": None,
                "verdict": "SKIP_BASE_RATE",
            })
            continue

        done += 1
        t1 = time.time()
        wfo = run_wfo(df, t, feat_cols, info)
        elapsed = time.time() - t1

        if wfo is None:
            verdict = "FAILED"
            mean_score = None
            print(f"  [{done:>3}/{n_usable}] {t:<45} FAILED ({elapsed:.1f}s)")
        else:
            mean_score = wfo["mean_score"]
            if wfo["all_positive"] and mean_score > 0.05:
                verdict = "STRONG"
            elif wfo["all_positive"] and mean_score > 0.02:
                verdict = "GOOD"
            elif mean_score > 0.02:
                verdict = "WEAK"
            elif mean_score > 0:
                verdict = "MARGINAL"
            else:
                verdict = "UNPREDICTABLE"

            print(f"  [{done:>3}/{n_usable}] {t:<45} {verdict:<14} "
                  f"mean={mean_score:+.3f} [{wfo['min_score']:+.3f}, {wfo['max_score']:+.3f}] "
                  f"folds={wfo['n_folds']} ({elapsed:.1f}s)")

        results.append({
            "target": t,
            "type": info["type"],
            "base_rate": info["base_rate"],
            "mean_score": mean_score,
            "min_score": wfo["min_score"] if wfo else None,
            "max_score": wfo["max_score"] if wfo else None,
            "n_folds": wfo["n_folds"] if wfo else 0,
            "all_positive": wfo["all_positive"] if wfo else False,
            "verdict": verdict,
        })

    # Combine results
    all_results = results + skipped
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values("mean_score", ascending=False, na_position="last")

    # Save CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"target_audit_{SYMBOL}_{TF}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # Print summary
    elapsed_total = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY ({elapsed_total:.0f}s)")
    print(f"{'=' * 80}")

    for verdict in ["STRONG", "GOOD", "WEAK", "MARGINAL", "UNPREDICTABLE", "FAILED", "SKIP_BASE_RATE"]:
        subset = df_results[df_results["verdict"] == verdict]
        if len(subset) > 0:
            print(f"\n  {verdict} ({len(subset)}):")
            for _, row in subset.iterrows():
                br = f" base={row['base_rate']:.1%}" if pd.notna(row['base_rate']) else ""
                sc = f" score={row['mean_score']:+.3f}" if pd.notna(row['mean_score']) else ""
                print(f"    {row['target']:<45} {row['type']:<12}{br}{sc}")

    # Cross-reference with predictable_targets.json
    import json
    pt_path = Path("./microstructure_research/predictable_targets.json")
    if pt_path.exists():
        with open(pt_path) as f:
            pt = json.load(f)
        pt_list = pt.get("predictable_targets", [])
        print(f"\n  Cross-reference with predictable_targets.json ({len(pt_list)} targets):")
        for t in pt_list:
            row = df_results[df_results["target"] == t]
            if len(row) > 0:
                r = row.iloc[0]
                sc = f"score={r['mean_score']:+.3f}" if pd.notna(r['mean_score']) else "N/A"
                print(f"    {t:<45} {r['verdict']:<14} {sc}")

    print(f"\n  Total time: {elapsed_total:.0f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
