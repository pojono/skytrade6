#!/usr/bin/env python3
"""
Target Audit: 1h vs 4h comparison for 54 deduplicated targets.
Re-runs 4h with clean (non-contaminated) features as baseline,
then runs 1h for SOL + DOGE.

Usage:
  python microstructure_research/target_audit_1h_vs_4h.py
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import json
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
FEATURES_DIR = Path("./features")
RESULTS_DIR = Path("./microstructure_research/results")
PT_PATH = Path("./microstructure_research/predictable_targets.json")

N_FOLDS = 3
TOP_FEATURES = 30

# Per-timeframe settings
TF_CONFIG = {
    "4h": {
        "min_train_candles": 1440,   # ~240 days
        "test_candles": 540,         # ~90 days
        "coins": ["SOLUSDT", "XRPUSDT", "DOGEUSDT"],
    },
    "1h": {
        "min_train_candles": 5760,   # ~240 days
        "test_candles": 2160,        # ~90 days
        "coins": ["SOLUSDT", "DOGEUSDT"],  # no XRP 1h data
    },
}

LGB_PARAMS = {
    "objective": "binary", "metric": "auc",
    "n_estimators": 100, "max_depth": 4, "learning_rate": 0.1,
    "num_leaves": 15, "min_child_samples": 50,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "verbose": -1, "n_jobs": -1,
}

LGB_REG_PARAMS = {
    "objective": "regression", "metric": "mse",
    "n_estimators": 100, "max_depth": 4, "learning_rate": 0.1,
    "num_leaves": 15, "min_child_samples": 50,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "verbose": -1, "n_jobs": -1,
}


def load_features(symbol, tf):
    tf_dir = FEATURES_DIR / symbol / tf
    files = sorted(tf_dir.glob("*.parquet"))
    if not files:
        return None
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def classify_target(series):
    vals = series.dropna()
    unique = vals.nunique()
    if unique <= 2:
        p1 = vals.mean()
        return {"type": "binary", "base_rate": p1, "usable": 0.05 <= p1 <= 0.95}
    else:
        return {"type": "continuous", "base_rate": None, "usable": True}


def select_features(df, target_col, feat_cols, top_n=30):
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


def run_wfo(df, target_col, feat_cols, info, min_train, test_candles):
    n = len(df)
    is_binary = info["type"] == "binary"
    fold_results = []

    for fold in range(N_FOLDS):
        test_end = n - fold * test_candles
        test_start = test_end - test_candles
        train_end = test_start

        if train_end < min_train:
            continue

        df_train = df.iloc[:train_end]
        df_test = df.iloc[test_start:test_end]

        selected = select_features(df_train, target_col, feat_cols, TOP_FEATURES)
        if len(selected) < 5:
            continue

        X_train = df_train[selected].values
        y_train = df_train[target_col].values
        X_test = df_test[selected].values
        y_test = df_test[target_col].values

        train_valid = np.all(np.isfinite(X_train), axis=1) & np.isfinite(y_train)
        test_valid = np.all(np.isfinite(X_test), axis=1) & np.isfinite(y_test)

        X_tr, y_tr = X_train[train_valid], y_train[train_valid]
        X_te, y_te = X_test[test_valid], y_test[test_valid]

        if len(X_tr) < 200 or len(X_te) < 50:
            continue
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
                score = roc_auc_score(y_te, preds) - 0.5
            else:
                model = lgb.LGBMRegressor(**LGB_REG_PARAMS)
                model.fit(X_tr_s, y_tr)
                preds = model.predict(X_te_s)
                score, _ = spearmanr(y_te, preds)
                if not np.isfinite(score):
                    score = 0.0

            fold_results.append(score)
        except:
            continue

    if not fold_results:
        return None

    return {
        "mean_score": np.mean(fold_results),
        "min_score": np.min(fold_results),
        "max_score": np.max(fold_results),
        "n_folds": len(fold_results),
        "all_positive": all(s > 0 for s in fold_results),
    }


def audit_coin_tf(symbol, tf, targets, target_details):
    """Run audit for one coin/timeframe combination."""
    cfg = TF_CONFIG[tf]
    df = load_features(symbol, tf)
    if df is None:
        print(f"    No data for {symbol} {tf}, skipping")
        return {}

    print(f"    {symbol} {tf}: {len(df)} candles, "
          f"{df.index[0].date()} -> {df.index[-1].date()}")

    feat_cols = [c for c in df.columns
                 if not c.startswith("tgt_")
                 and c not in ("open", "high", "low", "close", "volume")]

    results = {}
    done = 0
    total = len(targets)

    for tgt in targets:
        if tgt not in df.columns:
            continue

        # Get type from target_details
        detail = next((d for d in target_details if d["target"] == tgt), None)
        if detail is None:
            continue

        info = {"type": detail["type"], "usable": True}
        done += 1
        t1 = time.time()
        wfo = run_wfo(df, tgt, feat_cols, info,
                      cfg["min_train_candles"], cfg["test_candles"])
        elapsed = time.time() - t1

        if wfo is None:
            print(f"      [{done:>2}/{total}] {tgt:<40} FAILED ({elapsed:.1f}s)")
            results[tgt] = None
        else:
            ms = wfo["mean_score"]
            print(f"      [{done:>2}/{total}] {tgt:<40} "
                  f"mean={ms:+.4f} [{wfo['min_score']:+.4f}, {wfo['max_score']:+.4f}] "
                  f"({elapsed:.1f}s)")
            results[tgt] = wfo["mean_score"]

    return results


def main():
    t0 = time.time()
    print("=" * 80)
    print("  TARGET AUDIT: 1h vs 4h Comparison (Clean Features)")
    print("  54 deduplicated targets, 3-fold expanding WFO, LightGBM")
    print("=" * 80)

    # Load target list
    with open(PT_PATH) as f:
        pt = json.load(f)
    targets = pt["predictable_targets"]
    target_details = pt["target_details"]
    print(f"\n  Targets: {len(targets)} from {PT_PATH.name}")

    # Run all coin/tf combinations
    all_scores = {}  # key: (coin, tf) -> {target: score}

    for tf in ["4h", "1h"]:
        cfg = TF_CONFIG[tf]
        print(f"\n{'=' * 60}")
        print(f"  TIMEFRAME: {tf}")
        print(f"  Coins: {cfg['coins']}")
        print(f"  Min train: {cfg['min_train_candles']}, Test: {cfg['test_candles']}")
        print(f"{'=' * 60}")

        for coin in cfg["coins"]:
            print(f"\n  --- {coin} {tf} ---")
            scores = audit_coin_tf(coin, tf, targets, target_details)
            all_scores[(coin, tf)] = scores

    # Build comparison table
    print(f"\n\n{'=' * 80}")
    print(f"  BUILDING COMPARISON TABLE")
    print(f"{'=' * 80}")

    rows = []
    for tgt in targets:
        detail = next((d for d in target_details if d["target"] == tgt), None)
        if detail is None:
            continue

        row = {
            "target": tgt,
            "type": detail["type"],
        }

        # 4h scores
        for coin in TF_CONFIG["4h"]["coins"]:
            key = (coin, "4h")
            score = all_scores.get(key, {}).get(tgt)
            row[f"{coin}_4h"] = score

        # 1h scores
        for coin in TF_CONFIG["1h"]["coins"]:
            key = (coin, "1h")
            score = all_scores.get(key, {}).get(tgt)
            row[f"{coin}_1h"] = score

        # Averages
        scores_4h = [row.get(f"{c}_4h") for c in TF_CONFIG["4h"]["coins"]
                     if row.get(f"{c}_4h") is not None]
        scores_1h = [row.get(f"{c}_1h") for c in TF_CONFIG["1h"]["coins"]
                     if row.get(f"{c}_1h") is not None]

        row["avg_4h"] = np.mean(scores_4h) if scores_4h else None
        row["avg_1h"] = np.mean(scores_1h) if scores_1h else None

        if row["avg_4h"] is not None and row["avg_1h"] is not None:
            row["delta_1h_vs_4h"] = row["avg_1h"] - row["avg_4h"]
            row["ratio_1h_vs_4h"] = row["avg_1h"] / row["avg_4h"] if row["avg_4h"] != 0 else None
        else:
            row["delta_1h_vs_4h"] = None
            row["ratio_1h_vs_4h"] = None

        rows.append(row)

    df_cmp = pd.DataFrame(rows)
    df_cmp = df_cmp.sort_values("avg_4h", ascending=False, na_position="last")

    # Save CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "target_audit_1h_vs_4h.csv"
    df_cmp.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # Print summary
    elapsed_total = time.time() - t0

    print(f"\n{'=' * 80}")
    print(f"  SUMMARY ({elapsed_total:.0f}s)")
    print(f"{'=' * 80}")

    valid = df_cmp.dropna(subset=["avg_4h", "avg_1h"])
    print(f"\n  Targets with both 4h and 1h scores: {len(valid)}/{len(targets)}")

    if len(valid) > 0:
        print(f"\n  Average scores:")
        print(f"    4h mean: {valid['avg_4h'].mean():.4f}")
        print(f"    1h mean: {valid['avg_1h'].mean():.4f}")
        print(f"    Delta (1h - 4h): {valid['delta_1h_vs_4h'].mean():+.4f}")

        better_1h = (valid["delta_1h_vs_4h"] > 0).sum()
        worse_1h = (valid["delta_1h_vs_4h"] < 0).sum()
        print(f"\n  1h better: {better_1h}, 1h worse: {worse_1h}")

        print(f"\n  Top 10 targets (by 4h score):")
        print(f"  {'Target':<40} {'4h':>8} {'1h':>8} {'Delta':>8}")
        print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8}")
        for _, row in valid.head(10).iterrows():
            print(f"  {row['target']:<40} {row['avg_4h']:>+.4f} {row['avg_1h']:>+.4f} "
                  f"{row['delta_1h_vs_4h']:>+.4f}")

        print(f"\n  Biggest 1h improvements:")
        top_improve = valid.nlargest(5, "delta_1h_vs_4h")
        for _, row in top_improve.iterrows():
            print(f"    {row['target']:<40} 4h={row['avg_4h']:+.4f} -> 1h={row['avg_1h']:+.4f} "
                  f"({row['delta_1h_vs_4h']:+.4f})")

        print(f"\n  Biggest 1h degradations:")
        top_degrade = valid.nsmallest(5, "delta_1h_vs_4h")
        for _, row in top_degrade.iterrows():
            print(f"    {row['target']:<40} 4h={row['avg_4h']:+.4f} -> 1h={row['avg_1h']:+.4f} "
                  f"({row['delta_1h_vs_4h']:+.4f})")

    print(f"\n  Total time: {elapsed_total:.0f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
