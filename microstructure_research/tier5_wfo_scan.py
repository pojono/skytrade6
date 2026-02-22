#!/usr/bin/env python3
"""
Tier 5: Walk-Forward Single-Feature Predictive Power

For each (feature, target) pair from Tier 4 representatives:
  - Fit univariate Ridge (continuous) or Logistic (binary) in walk-forward fashion
  - Expanding window: min 120d train, 45d test, 2d purge
  - Each fold trains on ALL data from start; later folds have more data
  - Measure OOS R² (regression) or OOS AUC (classification)

Usage:
  python tier5_wfo_scan.py DOGEUSDT 4h
  python tier5_wfo_scan.py SOLUSDT 4h
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, r2_score

# WFO parameters
MIN_TRAIN_DAYS = 120  # minimum training window (expanding from here)
TEST_DAYS = 45        # OOS test window
PURGE_DAYS = 2        # gap between train and test to avoid lookahead
MIN_TRAIN_CANDLES = 500


def load_features(features_dir: Path, symbol: str, tf: str) -> pd.DataFrame:
    tf_dir = features_dir / symbol / tf
    files = sorted(tf_dir.glob("*.parquet"))
    if not files:
        print(f"ERROR: No parquet files in {tf_dir}", flush=True)
        sys.exit(1)
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def get_tier4_representatives(results_dir: Path, tf: str):
    """Load Tier 4 cluster representatives."""
    path = results_dir / f"tier4_{tf}_representatives.csv"
    if not path.exists():
        print(f"ERROR: {path} not found. Run tier4_clustering.py first.", flush=True)
        sys.exit(1)
    return pd.read_csv(path)


def classify_targets(df):
    """Split targets into binary and continuous."""
    tgt_cols = [c for c in df.columns if c.startswith("tgt_")]
    binary_tgts = []
    continuous_tgts = []
    for c in tgt_cols:
        unique_vals = df[c].dropna().unique()
        if len(unique_vals) <= 3:
            binary_tgts.append(c)
        else:
            continuous_tgts.append(c)
    return continuous_tgts, binary_tgts


def make_wfo_folds(n_candles, candles_per_day):
    """Create expanding-window walk-forward folds.

    Train always starts at index 0 and expands forward.
    Test windows are non-overlapping and slide forward by test_size.
    """
    min_train = int(MIN_TRAIN_DAYS * candles_per_day)
    test_size = int(TEST_DAYS * candles_per_day)
    purge_size = int(PURGE_DAYS * candles_per_day)

    folds = []
    # First test block starts after min_train + purge
    test_start = min_train + purge_size
    while test_start + test_size <= n_candles:
        train_end = test_start - purge_size  # train up to purge gap
        test_end = test_start + test_size
        folds.append((0, train_end, test_start, test_end))
        test_start += test_size  # slide test forward (non-overlapping)

    return folds


def wfo_single_feature(feat_vals, tgt_vals, folds, is_binary):
    """Run WFO for a single feature-target pair."""
    oos_scores = []

    for (tr_start, tr_end, te_start, te_end) in folds:
        X_train = feat_vals[tr_start:tr_end].reshape(-1, 1)
        y_train = tgt_vals[tr_start:tr_end]
        X_test = feat_vals[te_start:te_end].reshape(-1, 1)
        y_test = tgt_vals[te_start:te_end]

        # Remove NaN/inf
        train_mask = np.isfinite(X_train.ravel()) & np.isfinite(y_train)
        test_mask = np.isfinite(X_test.ravel()) & np.isfinite(y_test)

        if train_mask.sum() < 30 or test_mask.sum() < 10:
            oos_scores.append(np.nan)
            continue

        X_tr, y_tr = X_train[train_mask], y_train[train_mask]
        X_te, y_te = X_test[test_mask], y_test[test_mask]

        # Scale
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        if is_binary:
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                oos_scores.append(np.nan)
                continue
            try:
                model = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
                model.fit(X_tr, y_tr.astype(int))
                proba = model.predict_proba(X_te)[:, 1]
                score = roc_auc_score(y_te.astype(int), proba)
                oos_scores.append(score - 0.5)  # deviation from random
            except Exception:
                oos_scores.append(np.nan)
        else:
            try:
                model = Ridge(alpha=1.0)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)
                score = r2_score(y_te, pred)
                oos_scores.append(score)
            except Exception:
                oos_scores.append(np.nan)

    return np.array(oos_scores)


def run_tier5(df, symbol, tf, rep_feats, output_dir):
    """Run Tier 5 WFO scan."""
    continuous_tgts, binary_tgts = classify_targets(df)
    all_tgts = continuous_tgts + binary_tgts

    tf_hours = {"15m": 0.25, "30m": 0.5, "1h": 1, "2h": 2, "4h": 4}
    candles_per_day = 24 / tf_hours.get(tf, 1)
    folds = make_wfo_folds(len(df), candles_per_day)

    # Filter features present in data
    feats = [f for f in rep_feats if f in df.columns]
    n_feats = len(feats)
    n_tgts = len(all_tgts)
    total_pairs = n_feats * n_tgts

    print(f"\n{'='*70}", flush=True)
    print(f"Tier 5 WFO Scan: {symbol} {tf}", flush=True)
    print(f"  Candles:    {len(df):,}", flush=True)
    print(f"  Features:   {n_feats} (Tier 4 representatives)", flush=True)
    print(f"  Targets:    {n_tgts} ({len(continuous_tgts)} continuous + {len(binary_tgts)} binary)", flush=True)
    # Print fold details
    for i, (ts, te, tes, tee) in enumerate(folds):
        tr_days = (te - ts) / candles_per_day
        te_days = (tee - tes) / candles_per_day
        print(f"    Fold {i+1}: train {te-ts} candles ({tr_days:.0f}d), "
              f"test {tee-tes} candles ({te_days:.0f}d)", flush=True)
    print(f"  WFO folds:  {len(folds)} (expanding, min train {MIN_TRAIN_DAYS}d, "
          f"test {TEST_DAYS}d, purge {PURGE_DAYS}d)", flush=True)
    print(f"  Total pairs: {total_pairs:,}", flush=True)
    print(f"{'='*70}", flush=True)

    results = []
    t0 = time.time()
    done = 0

    for ti, tgt in enumerate(all_tgts):
        is_binary = tgt in binary_tgts
        tgt_vals = df[tgt].values.astype(float)

        for fi, feat in enumerate(feats):
            feat_vals = df[feat].values.astype(float)
            oos = wfo_single_feature(feat_vals, tgt_vals, folds, is_binary)
            valid = oos[~np.isnan(oos)]

            if len(valid) == 0:
                done += 1
                continue

            mean_oos = np.mean(valid)
            std_oos = np.std(valid)
            pct_positive = np.mean(valid > 0) if not is_binary else np.mean(valid > 0)
            snr = abs(mean_oos) / std_oos if std_oos > 0 else 0

            results.append({
                "feature": feat,
                "target": tgt,
                "is_binary": is_binary,
                "mean_oos": mean_oos,
                "std_oos": std_oos,
                "snr": snr,
                "pct_positive": pct_positive,
                "n_folds": len(valid),
                "metric": "AUC_dev" if is_binary else "R2",
            })
            done += 1

        elapsed = time.time() - t0
        speed = done / elapsed if elapsed > 0 else 0
        remaining = total_pairs - done
        eta = remaining / speed if speed > 0 else 0
        print(f"  [{done:>5}/{total_pairs}] {tgt:<35} "
              f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s ETA, {speed:.1f} pairs/s]", flush=True)

    res_df = pd.DataFrame(results)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"tier5_{symbol}_{tf}"
    csv_path = output_dir / f"{prefix}_wfo.csv"
    res_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}", flush=True)

    # --- Report ---
    print(f"\n{'='*70}", flush=True)
    print(f"TIER 5 RESULTS: {symbol} {tf}", flush=True)
    print(f"{'='*70}", flush=True)

    if len(res_df) == 0:
        print("  No results!", flush=True)
        return res_df

    # Top feature-target pairs by mean OOS score
    # Continuous: top by R²
    cont_res = res_df[~res_df["is_binary"]].copy()
    if len(cont_res) > 0:
        top_cont = cont_res.sort_values("mean_oos", ascending=False).head(30)
        print(f"\n--- Top 30 continuous pairs (by OOS R²) ---", flush=True)
        print(f"  {'Feature':<35} {'Target':<30} {'R²':>8} {'std':>8} {'SNR':>6} {'%pos':>6} {'folds':>5}", flush=True)
        print(f"  {'-'*100}", flush=True)
        for _, r in top_cont.iterrows():
            print(f"  {r['feature']:<35} {r['target']:<30} {r['mean_oos']:>+8.5f} "
                  f"{r['std_oos']:>8.5f} {r['snr']:>6.2f} {r['pct_positive']:>5.0%} "
                  f"{r['n_folds']:>5.0f}", flush=True)

    # Binary: top by AUC deviation
    bin_res = res_df[res_df["is_binary"]].copy()
    if len(bin_res) > 0:
        top_bin = bin_res.sort_values("mean_oos", key=abs, ascending=False).head(30)
        print(f"\n--- Top 30 binary pairs (by |OOS AUC - 0.5|) ---", flush=True)
        print(f"  {'Feature':<35} {'Target':<30} {'AUC_dev':>8} {'std':>8} {'SNR':>6} {'%pos':>6} {'folds':>5}", flush=True)
        print(f"  {'-'*100}", flush=True)
        for _, r in top_bin.iterrows():
            print(f"  {r['feature']:<35} {r['target']:<30} {r['mean_oos']:>+8.5f} "
                  f"{r['std_oos']:>8.5f} {r['snr']:>6.2f} {r['pct_positive']:>5.0%} "
                  f"{r['n_folds']:>5.0f}", flush=True)

    # Per-feature summary: how many targets does each feature predict OOS?
    print(f"\n--- Feature Summary: OOS predictive power ---", flush=True)
    feat_summary = []
    for feat in feats:
        feat_res = res_df[res_df["feature"] == feat]
        if len(feat_res) == 0:
            continue
        cont_f = feat_res[~feat_res["is_binary"]]
        bin_f = feat_res[feat_res["is_binary"]]
        n_cont_pos = (cont_f["mean_oos"] > 0).sum() if len(cont_f) > 0 else 0
        n_bin_pos = (bin_f["mean_oos"].abs() > 0.005).sum() if len(bin_f) > 0 else 0
        avg_r2 = cont_f["mean_oos"].mean() if len(cont_f) > 0 else 0
        avg_auc = bin_f["mean_oos"].abs().mean() if len(bin_f) > 0 else 0
        feat_summary.append({
            "feature": feat,
            "n_cont_positive_r2": n_cont_pos,
            "n_bin_significant": n_bin_pos,
            "avg_r2": avg_r2,
            "avg_auc_dev": avg_auc,
            "total_useful": n_cont_pos + n_bin_pos,
        })

    fs_df = pd.DataFrame(feat_summary).sort_values("total_useful", ascending=False)
    print(f"  {'Feature':<35} {'R²>0':>6} {'AUC>0.005':>10} {'avg_R²':>8} {'avg|AUC|':>9} {'total':>6}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for _, r in fs_df.iterrows():
        print(f"  {r['feature']:<35} {r['n_cont_positive_r2']:>6.0f} {r['n_bin_significant']:>10.0f} "
              f"{r['avg_r2']:>+8.5f} {r['avg_auc_dev']:>9.5f} {r['total_useful']:>6.0f}", flush=True)

    # Save feature summary
    fs_path = output_dir / f"{prefix}_feature_summary.csv"
    fs_df.to_csv(fs_path, index=False)
    print(f"\nSaved: {fs_path}", flush=True)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s", flush=True)
    print(f"Done! Results in {output_dir}/", flush=True)
    return res_df


def main():
    parser = argparse.ArgumentParser(description="Tier 5: Walk-Forward Single-Feature Scan")
    parser.add_argument("symbol", help="e.g. DOGEUSDT")
    parser.add_argument("timeframe", help="e.g. 4h")
    parser.add_argument("--features-dir", default="./features")
    parser.add_argument("--results-dir", default="./microstructure_research/results")
    parser.add_argument("--output-dir", default="./microstructure_research/results")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    # Load data
    df = load_features(features_dir, args.symbol, args.timeframe)
    print(f"Loaded {len(df)} candles for {args.symbol} {args.timeframe} "
          f"({df.index[0]} -> {df.index[-1]})", flush=True)

    # Get Tier 4 representatives
    rep_df = get_tier4_representatives(results_dir, args.timeframe)
    rep_feats = rep_df["feature"].tolist()
    print(f"Tier 4 representatives: {len(rep_feats)} features", flush=True)

    run_tier5(df, args.symbol, args.timeframe, rep_feats, output_dir)


if __name__ == "__main__":
    main()
