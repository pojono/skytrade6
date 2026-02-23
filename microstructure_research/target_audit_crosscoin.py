#!/usr/bin/env python3
"""
Cross-coin target audit: validate SOL's STRONG+GOOD targets on XRP and DOGE.

Only tests the 86 targets that passed on SOL 4h. If they also pass on XRP and DOGE,
these are genuine market inefficiencies, not coin-specific artifacts.

Usage:
  python microstructure_research/target_audit_crosscoin.py
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
TF = "4h"
FEATURES_DIR = Path("./features")
RESULTS_DIR = Path("./microstructure_research/results")
SOL_AUDIT = RESULTS_DIR / "target_audit_SOLUSDT_4h.csv"

ALL_COINS = ["XRPUSDT", "DOGEUSDT"]

MIN_TRAIN_CANDLES = 1440
TEST_CANDLES = 540
N_FOLDS = 3
TOP_FEATURES = 30

LGB_PARAMS = {
    "objective": "binary", "metric": "auc", "n_estimators": 100,
    "max_depth": 4, "learning_rate": 0.1, "num_leaves": 15,
    "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8,
    "verbose": -1, "n_jobs": -1,
}
LGB_REG_PARAMS = {
    "objective": "regression", "metric": "mse", "n_estimators": 100,
    "max_depth": 4, "learning_rate": 0.1, "num_leaves": 15,
    "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8,
    "verbose": -1, "n_jobs": -1,
}


def load_features(features_dir, symbol, tf):
    tf_dir = features_dir / symbol / tf
    files = sorted(tf_dir.glob("*.parquet"))
    if not files:
        return None
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def classify_target(series):
    vals = series.dropna()
    if vals.nunique() <= 2:
        return "binary", vals.mean()
    return "continuous", None


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


def run_wfo(df, target_col, feat_cols, tgt_type):
    n = len(df)
    is_binary = tgt_type == "binary"
    fold_results = []

    for fold in range(N_FOLDS):
        test_end = n - fold * TEST_CANDLES
        test_start = test_end - TEST_CANDLES
        train_end = test_start
        if train_end < MIN_TRAIN_CANDLES:
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, default=None, help="Single coin to run")
    args = parser.parse_args()

    coins = [args.coin] if args.coin else ALL_COINS

    t0 = time.time()

    # Load deduplicated predictable targets
    import json
    pt_path = Path("./microstructure_research/predictable_targets.json")
    with open(pt_path) as f:
        pt = json.load(f)
    target_list = pt["predictable_targets"]

    # Get scores and types from SOL audit
    sol_df = pd.read_csv(SOL_AUDIT)
    sol_scores = dict(zip(sol_df["target"], sol_df["mean_score"]))
    sol_types = dict(zip(sol_df["target"], sol_df["type"]))

    print("=" * 90)
    print(f"  CROSS-COIN TARGET AUDIT — {len(target_list)} STRONG+GOOD targets from SOL")
    print(f"  Coins: {', '.join(coins)} | TF: {TF} | {N_FOLDS}-fold WFO")
    print("=" * 90)

    all_results = {}

    for coin in coins:
        print(f"\n  Loading {coin}...", flush=True)
        df = load_features(FEATURES_DIR, coin, TF)
        if df is None:
            print(f"  ERROR: No data for {coin} {TF}")
            continue
        print(f"  Loaded {len(df)} candles, {df.index[0].date()} -> {df.index[-1].date()}")

        feat_cols = [c for c in df.columns
                     if not c.startswith("tgt_")
                     and c not in ("open", "high", "low", "close", "volume")]

        coin_results = {}
        n_total = len(target_list)
        for i, tgt in enumerate(target_list):
            if tgt not in df.columns:
                print(f"  [{i+1:>3}/{n_total}] {tgt:<45} MISSING")
                continue

            tgt_type = sol_types[tgt]
            t1 = time.time()
            wfo = run_wfo(df, tgt, feat_cols, tgt_type)
            elapsed = time.time() - t1

            if wfo is None:
                print(f"  [{i+1:>3}/{n_total}] {tgt:<45} FAILED ({elapsed:.1f}s)")
                coin_results[tgt] = None
            else:
                score = wfo["mean_score"]
                coin_results[tgt] = score
                tag = "✓" if score > 0.02 else "✗"
                print(f"  [{i+1:>3}/{n_total}] {tgt:<45} {tag} score={score:+.3f} "
                      f"[{wfo['min_score']:+.3f}, {wfo['max_score']:+.3f}] ({elapsed:.1f}s)")

        all_results[coin] = coin_results

    # Build comparison table
    print(f"\n{'=' * 90}")
    print(f"  CROSS-COIN COMPARISON")
    print(f"{'=' * 90}")
    print(f"\n  {'TARGET':<40} {'SOL':>8} {'XRP':>8} {'DOGE':>8} {'Avg':>8} {'Status':>12}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")

    rows = []
    for tgt in target_list:
        sol_s = sol_scores.get(tgt, None)
        xrp_s = all_results.get("XRPUSDT", {}).get(tgt, None)
        doge_s = all_results.get("DOGEUSDT", {}).get(tgt, None)

        scores = [s for s in [sol_s, xrp_s, doge_s] if s is not None]
        avg = np.mean(scores) if scores else None
        n_positive = sum(1 for s in scores if s is not None and s > 0.02)
        n_coins = len(scores)

        if n_positive == n_coins and n_coins >= 2:
            status = "UNIVERSAL"
        elif n_positive >= 2:
            status = "CROSS-COIN"
        elif n_positive == 1:
            status = "coin-specific"
        else:
            status = "FAILED"

        sol_str = f"{sol_s:+.3f}" if sol_s is not None else "N/A"
        xrp_str = f"{xrp_s:+.3f}" if xrp_s is not None else "N/A"
        doge_str = f"{doge_s:+.3f}" if doge_s is not None else "N/A"
        avg_str = f"{avg:+.3f}" if avg is not None else "N/A"

        print(f"  {tgt:<40} {sol_str:>8} {xrp_str:>8} {doge_str:>8} {avg_str:>8} {status:>12}")

        rows.append({
            "target": tgt,
            "type": sol_types.get(tgt, ""),
            "sol_score": sol_s,
            "xrp_score": xrp_s,
            "doge_score": doge_s,
            "avg_score": avg,
            "n_positive": n_positive,
            "n_coins": n_coins,
            "status": status,
        })

    # Save CSV
    df_out = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / f"target_audit_crosscoin_{TF}.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # Summary
    universal = sum(1 for r in rows if r["status"] == "UNIVERSAL")
    cross = sum(1 for r in rows if r["status"] == "CROSS-COIN")
    specific = sum(1 for r in rows if r["status"] == "coin-specific")
    failed = sum(1 for r in rows if r["status"] == "FAILED")

    elapsed_total = time.time() - t0
    print(f"\n  UNIVERSAL (all coins >0.02): {universal}")
    print(f"  CROSS-COIN (2+ coins >0.02): {cross}")
    print(f"  Coin-specific (1 coin only): {specific}")
    print(f"  Failed (none >0.02):          {failed}")
    print(f"\n  Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
    print("=" * 90)


if __name__ == "__main__":
    main()
