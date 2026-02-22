#!/usr/bin/env python3
"""
No-Lookahead Walk-Forward Feature Selection & Validation Pipeline

Design:
  For each temporal split:
    1. Truncate data at cutoff (selection period only)
    2. Run Tiers 1→5 on truncated data → get per-target feature lists
    3. Train multi-feature model on selection period
    4. Predict on holdout test period (never seen during feature selection)

  Final: average holdout scores across splits. A target is "predictable"
  only if it works across multiple independent test windows.

Splits for 2-year data (SOL/XRP):
  Split 1: select Jan2024-Jun2025, purge 30d, test Jul2025-Sep2025
  Split 2: select Jan2024-Sep2025, purge 30d, test Oct2025-Dec2025

Usage:
  python pipeline_nolookahead.py SOLUSDT 4h
  python pipeline_nolookahead.py XRPUSDT 4h
  python pipeline_nolookahead.py DOGEUSDT 4h
"""

import argparse
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*constant.*")
warnings.filterwarnings("ignore", message=".*Only one class.*")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, r2_score
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# ============================================================
# PIPELINE PARAMETERS
# ============================================================
MIN_SELECTION_DAYS = 360   # minimum selection period
TEST_DAYS = 90             # each test window
PURGE_DAYS = 30            # gap between selection and test

# Tier 2 parameters
T2_WINDOW_DAYS = 90
T2_STEP_DAYS = 30
T2_MIN_CANDLES = 200
T2_MIN_EFFECT_SIZE = 0.03

# Tier 4 parameters
T4_CORR_THRESHOLD = 0.7

# Tier 5 parameters (inner WFO within selection period)
T5_MIN_TRAIN_DAYS = 120
T5_TEST_DAYS = 45
T5_PURGE_DAYS = 2
T5_MIN_OOS = 0.02
T5_MIN_PCT_POS = 0.60

# Tier 6 multi-feature parameters
MAX_FEATURES = 20
MIN_FEATURES = 3
RIDGE_ALPHA = 10.0
LOGISTIC_C = 0.1


# ============================================================
# DATA LOADING
# ============================================================
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


def classify_columns(df):
    tgt_cols = [c for c in df.columns if c.startswith("tgt_")]
    feat_cols = [c for c in df.columns if not c.startswith("tgt_")]
    binary_tgts = []
    continuous_tgts = []
    for c in tgt_cols:
        vals = df[c].dropna().unique()
        if len(vals) <= 3:
            binary_tgts.append(c)
        else:
            continuous_tgts.append(c)
    return feat_cols, continuous_tgts, binary_tgts


def get_candles_per_day(tf):
    tf_hours = {"15m": 0.25, "30m": 0.5, "1h": 1, "2h": 2, "4h": 4}
    return 24 / tf_hours.get(tf, 1)


# ============================================================
# TIER 2: Temporal Stability (simplified, in-memory)
# ============================================================
def run_tier2_inmemory(df, tf):
    """Run Tier 2 stability scan on a DataFrame. Returns survivors DataFrame."""
    feat_cols, continuous_tgts, binary_tgts = classify_columns(df)
    cpd = get_candles_per_day(tf)
    all_tgts = continuous_tgts + binary_tgts

    # Make windows
    win_size = int(T2_WINDOW_DAYS * cpd)
    step_size = int(T2_STEP_DAYS * cpd)
    windows = []
    start = 0
    while start + win_size <= len(df):
        windows.append((start, start + win_size))
        start += step_size

    if len(windows) < 3:
        print(f"  WARNING: Only {len(windows)} windows, need ≥3", flush=True)
        return pd.DataFrame()

    results = []
    for ti, tgt in enumerate(all_tgts):
        is_binary = tgt in binary_tgts
        tgt_vals_full = df[tgt].values

        for feat in feat_cols:
            feat_vals_full = df[feat].values

            # Full-period correlation
            mask = np.isfinite(feat_vals_full) & np.isfinite(tgt_vals_full)
            if mask.sum() < 30:
                continue

            if is_binary:
                unique = np.unique(tgt_vals_full[mask])
                if len(unique) < 2:
                    continue
                try:
                    auc = roc_auc_score(tgt_vals_full[mask].astype(int),
                                        feat_vals_full[mask])
                    full_r = auc - 0.5
                except:
                    continue
            else:
                r, _ = stats.spearmanr(feat_vals_full[mask], tgt_vals_full[mask])
                if not np.isfinite(r):
                    continue
                full_r = r

            # Window correlations
            win_rs = []
            for ws, we in windows:
                fv = feat_vals_full[ws:we]
                tv = tgt_vals_full[ws:we]
                m = np.isfinite(fv) & np.isfinite(tv)
                if m.sum() < T2_MIN_CANDLES:
                    win_rs.append(np.nan)
                    continue
                if is_binary:
                    u = np.unique(tv[m])
                    if len(u) < 2:
                        win_rs.append(np.nan)
                        continue
                    try:
                        a = roc_auc_score(tv[m].astype(int), fv[m])
                        win_rs.append(a - 0.5)
                    except:
                        win_rs.append(np.nan)
                else:
                    rr, _ = stats.spearmanr(fv[m], tv[m])
                    win_rs.append(rr if np.isfinite(rr) else np.nan)

            valid_rs = [r for r in win_rs if np.isfinite(r)]
            if len(valid_rs) < 3:
                continue

            full_sign = np.sign(full_r)
            signs = [np.sign(r) for r in valid_rs]
            sign_pct = sum(1 for s in signs if s == full_sign) / len(signs)
            mean_r = np.mean(valid_rs)
            std_r = np.std(valid_rs)
            snr = abs(mean_r) / std_r if std_r > 0 else 0

            # Max wrong streak
            wrong = [1 if s != full_sign else 0 for s in signs]
            max_wrong = 0
            cur = 0
            for w in wrong:
                if w:
                    cur += 1
                    max_wrong = max(max_wrong, cur)
                else:
                    cur = 0

            # Regime check (simplified: first half vs second half)
            mid = len(df) // 2
            fv1, tv1 = feat_vals_full[:mid], tgt_vals_full[:mid]
            fv2, tv2 = feat_vals_full[mid:], tgt_vals_full[mid:]
            m1 = np.isfinite(fv1) & np.isfinite(tv1)
            m2 = np.isfinite(fv2) & np.isfinite(tv2)
            regime_ok = True
            if m1.sum() >= 30 and m2.sum() >= 30:
                if is_binary:
                    try:
                        r1 = roc_auc_score(tv1[m1].astype(int), fv1[m1]) - 0.5
                        r2 = roc_auc_score(tv2[m2].astype(int), fv2[m2]) - 0.5
                        regime_ok = (r1 * r2) > 0
                    except:
                        pass
                else:
                    r1, _ = stats.spearmanr(fv1[m1], tv1[m1])
                    r2, _ = stats.spearmanr(fv2[m2], tv2[m2])
                    if np.isfinite(r1) and np.isfinite(r2):
                        regime_ok = (r1 * r2) > 0

            tier2_pass = (
                abs(full_r) >= T2_MIN_EFFECT_SIZE and
                sign_pct >= 0.70 and
                snr >= 0.5 and
                max_wrong <= 3 and
                regime_ok
            )

            if tier2_pass:
                results.append({
                    "feature": feat,
                    "target": tgt,
                    "full_r": full_r,
                    "sign_pct": sign_pct,
                    "snr": snr,
                    "max_wrong_streak": max_wrong,
                    "regime_consistent": regime_ok,
                })

        if (ti + 1) % 5 == 0 or ti == len(all_tgts) - 1:
            print(f"    Tier 2: [{ti+1}/{len(all_tgts)}] targets done, "
                  f"{len(results)} pairs pass so far", flush=True)

    return pd.DataFrame(results)


# ============================================================
# TIER 4: Clustering (simplified, in-memory)
# ============================================================
def run_tier4_inmemory(df, tier2_df):
    """Cluster Tier 2 survivors, return representative features."""
    if len(tier2_df) == 0:
        return []

    # Aggregate per feature
    survivors = {}
    for _, row in tier2_df.iterrows():
        f = row["feature"]
        if f not in survivors:
            survivors[f] = {"avg_r": [], "n_targets": 0}
        survivors[f]["avg_r"].append(abs(row["full_r"]))
        survivors[f]["n_targets"] += 1

    surv_list = []
    for f, info in survivors.items():
        surv_list.append({
            "feature": f,
            "mean_r": np.mean(info["avg_r"]),
            "n_targets": info["n_targets"],
            "composite": np.mean(info["avg_r"]) * np.log1p(info["n_targets"]),
        })
    surv_df = pd.DataFrame(surv_list).sort_values("composite", ascending=False)
    feat_list = surv_df["feature"].tolist()

    # Filter to features in data
    feat_list = [f for f in feat_list if f in df.columns]
    if len(feat_list) <= 1:
        return feat_list

    # Correlation matrix
    feat_data = df[feat_list].replace([np.inf, -np.inf], np.nan).fillna(0)
    corr = feat_data.corr(method="spearman")

    # Hierarchical clustering
    dist = 1 - corr.abs()
    np.fill_diagonal(dist.values, 0)
    dist = (dist + dist.T) / 2
    dist = dist.clip(lower=0)

    try:
        condensed = squareform(dist.values)
        Z = linkage(condensed, method="average")
        clusters = fcluster(Z, t=1 - T4_CORR_THRESHOLD, criterion="distance")
    except Exception as e:
        print(f"    Tier 4: clustering failed ({e}), using all features", flush=True)
        return feat_list

    # Pick best per cluster
    cluster_df = pd.DataFrame({"feature": feat_list, "cluster": clusters})
    cluster_df = cluster_df.merge(surv_df[["feature", "composite"]], on="feature")

    reps = []
    for cid in sorted(set(clusters)):
        members = cluster_df[cluster_df["cluster"] == cid].sort_values(
            "composite", ascending=False)
        reps.append(members.iloc[0]["feature"])

    return reps


# ============================================================
# TIER 5: Walk-Forward Single-Feature OOS (simplified, in-memory)
# ============================================================
def run_tier5_inmemory(df, tf, rep_feats):
    """Run single-feature WFO within selection period. Returns survivors."""
    _, continuous_tgts, binary_tgts = classify_columns(df)
    all_tgts = continuous_tgts + binary_tgts
    cpd = get_candles_per_day(tf)

    # Make expanding folds
    min_train = int(T5_MIN_TRAIN_DAYS * cpd)
    test_size = int(T5_TEST_DAYS * cpd)
    purge_size = int(T5_PURGE_DAYS * cpd)

    folds = []
    test_start = min_train + purge_size
    while test_start + test_size <= len(df):
        train_end = test_start - purge_size
        folds.append((0, train_end, test_start, test_start + test_size))
        test_start += test_size

    if len(folds) < 2:
        print(f"    Tier 5: Only {len(folds)} folds, need ≥2", flush=True)
        return pd.DataFrame()

    feats = [f for f in rep_feats if f in df.columns]
    survivors = []
    total = len(feats) * len(all_tgts)
    done = 0

    for tgt in all_tgts:
        is_binary = tgt in binary_tgts
        tgt_vals = df[tgt].values

        for feat in feats:
            feat_vals = df[feat].values
            oos_scores = []

            for (tr_s, tr_e, te_s, te_e) in folds:
                X_tr = feat_vals[tr_s:tr_e].reshape(-1, 1)
                y_tr = tgt_vals[tr_s:tr_e]
                X_te = feat_vals[te_s:te_e].reshape(-1, 1)
                y_te = tgt_vals[te_s:te_e]

                tr_mask = np.isfinite(X_tr.ravel()) & np.isfinite(y_tr)
                te_mask = np.isfinite(X_te.ravel()) & np.isfinite(y_te)

                if tr_mask.sum() < 30 or te_mask.sum() < 10:
                    oos_scores.append(np.nan)
                    continue

                Xtr, ytr = X_tr[tr_mask], y_tr[tr_mask]
                Xte, yte = X_te[te_mask], y_te[te_mask]

                scaler = StandardScaler()
                Xtr = scaler.fit_transform(Xtr)
                Xte = scaler.transform(Xte)

                if is_binary:
                    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                        oos_scores.append(np.nan)
                        continue
                    try:
                        m = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
                        m.fit(Xtr, ytr.astype(int))
                        p = m.predict_proba(Xte)[:, 1]
                        oos_scores.append(roc_auc_score(yte.astype(int), p) - 0.5)
                    except:
                        oos_scores.append(np.nan)
                else:
                    try:
                        m = Ridge(alpha=1.0)
                        m.fit(Xtr, ytr)
                        pred = m.predict(Xte)
                        oos_scores.append(r2_score(yte, pred))
                    except:
                        oos_scores.append(np.nan)

            valid = [s for s in oos_scores if np.isfinite(s)]
            if len(valid) < 2:
                done += 1
                continue

            mean_oos = np.mean(valid)
            pct_pos = (np.array(valid) > 0).mean()

            # Hard OOS filter
            if is_binary:
                passes = abs(mean_oos) >= T5_MIN_OOS and pct_pos >= T5_MIN_PCT_POS
            else:
                passes = mean_oos >= T5_MIN_OOS and pct_pos >= T5_MIN_PCT_POS

            if passes:
                survivors.append({
                    "feature": feat,
                    "target": tgt,
                    "mean_oos": mean_oos,
                    "pct_positive": pct_pos,
                })

            done += 1

        if done % (len(feats) * 3) < len(feats) or done == total:
            print(f"    Tier 5: [{done}/{total}] pairs evaluated, "
                  f"{len(survivors)} survive", flush=True)

    return pd.DataFrame(survivors)


# ============================================================
# TIER 6: Multi-Feature Holdout Test
# ============================================================
def run_holdout_test(df_selection, df_test, survivors_df, tf):
    """Train multi-feature models on selection, test on holdout.

    Returns per-target holdout scores.
    """
    if len(survivors_df) == 0:
        return pd.DataFrame()

    targets = sorted(survivors_df["target"].unique())
    _, continuous_tgts, binary_tgts = classify_columns(df_selection)

    results = []
    for tgt in targets:
        tgt_surv = survivors_df[survivors_df["target"] == tgt].copy()
        tgt_surv = tgt_surv.sort_values("mean_oos", key=abs, ascending=False)

        feat_cols = [f for f in tgt_surv["feature"].tolist()
                     if f in df_selection.columns and f in df_test.columns]

        if len(feat_cols) < MIN_FEATURES:
            continue

        feat_cols = feat_cols[:MAX_FEATURES]
        is_binary = tgt in binary_tgts

        # Train on full selection period
        X_train = df_selection[feat_cols].values
        y_train = df_selection[tgt].values
        X_test = df_test[feat_cols].values
        y_test = df_test[tgt].values

        tr_mask = np.all(np.isfinite(X_train), axis=1) & np.isfinite(y_train)
        te_mask = np.all(np.isfinite(X_test), axis=1) & np.isfinite(y_test)

        if tr_mask.sum() < 50 or te_mask.sum() < 20:
            continue

        Xtr, ytr = X_train[tr_mask], y_train[tr_mask]
        Xte, yte = X_test[te_mask], y_test[te_mask]

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        score = np.nan
        if is_binary:
            if len(np.unique(ytr)) >= 2 and len(np.unique(yte)) >= 2:
                try:
                    model = LogisticRegression(C=LOGISTIC_C, max_iter=500, solver="lbfgs")
                    model.fit(Xtr, ytr.astype(int))
                    proba = model.predict_proba(Xte)[:, 1]
                    score = roc_auc_score(yte.astype(int), proba) - 0.5
                except:
                    pass
        else:
            try:
                model = Ridge(alpha=RIDGE_ALPHA)
                model.fit(Xtr, ytr)
                pred = model.predict(Xte)
                score = r2_score(yte, pred)
            except:
                pass

        results.append({
            "target": tgt,
            "is_binary": is_binary,
            "n_features": len(feat_cols),
            "features": ", ".join(feat_cols),
            "holdout_score": score,
            "n_train": int(tr_mask.sum()),
            "n_test": int(te_mask.sum()),
            "metric": "AUC_dev" if is_binary else "R2",
        })

    return pd.DataFrame(results)


# ============================================================
# SPLIT GENERATION
# ============================================================
def generate_splits(df, tf):
    """Generate temporal splits: selection → purge → test."""
    cpd = get_candles_per_day(tf)
    n = len(df)
    total_days = n / cpd

    test_candles = int(TEST_DAYS * cpd)
    purge_candles = int(PURGE_DAYS * cpd)
    min_select = int(MIN_SELECTION_DAYS * cpd)

    splits = []
    # Work backwards from end of data
    test_end = n
    while True:
        test_start = test_end - test_candles
        purge_end = test_start
        purge_start = purge_end - purge_candles
        select_end = purge_start

        if select_end < min_select:
            break

        splits.append({
            "select_end": select_end,
            "purge_start": purge_start,
            "purge_end": purge_end,
            "test_start": test_start,
            "test_end": test_end,
        })

        test_end = test_start  # next test window ends where this one starts

    splits.reverse()  # chronological order
    return splits


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_pipeline(symbol, tf, features_dir, output_dir):
    t0_total = time.time()

    # Load full data
    df = load_features(features_dir, symbol, tf)
    cpd = get_candles_per_day(tf)
    print(f"\nLoaded {len(df)} candles for {symbol} {tf}", flush=True)
    print(f"  Range: {df.index[0]} -> {df.index[-1]}", flush=True)
    print(f"  Days: {len(df)/cpd:.0f}", flush=True)

    # Generate splits
    splits = generate_splits(df, tf)
    if not splits:
        print("ERROR: Not enough data for even 1 split!", flush=True)
        sys.exit(1)

    print(f"\n{'='*80}", flush=True)
    print(f"NO-LOOKAHEAD PIPELINE: {symbol} {tf}", flush=True)
    print(f"  Splits: {len(splits)}", flush=True)
    print(f"  Test window: {TEST_DAYS}d, Purge: {PURGE_DAYS}d", flush=True)
    for i, sp in enumerate(splits):
        sel_days = sp["select_end"] / cpd
        test_s = df.index[sp["test_start"]]
        test_e = df.index[min(sp["test_end"] - 1, len(df) - 1)]
        print(f"  Split {i+1}: select {sel_days:.0f}d "
              f"({df.index[0].date()} -> {df.index[sp['select_end']-1].date()}), "
              f"test {test_s.date()} -> {test_e.date()}", flush=True)
    print(f"{'='*80}", flush=True)

    all_split_results = []

    for si, sp in enumerate(splits):
        t0_split = time.time()
        print(f"\n{'#'*80}", flush=True)
        print(f"# SPLIT {si+1}/{len(splits)}", flush=True)
        print(f"{'#'*80}", flush=True)

        df_select = df.iloc[:sp["select_end"]].copy()
        df_test = df.iloc[sp["test_start"]:sp["test_end"]].copy()

        sel_days = len(df_select) / cpd
        test_days = len(df_test) / cpd
        print(f"  Selection: {len(df_select)} candles ({sel_days:.0f}d)", flush=True)
        print(f"  Test:      {len(df_test)} candles ({test_days:.0f}d)", flush=True)

        # --- Tier 2 ---
        print(f"\n  --- Tier 2: Stability Scan ---", flush=True)
        t2_start = time.time()
        tier2_df = run_tier2_inmemory(df_select, tf)
        n_t2_feats = tier2_df["feature"].nunique() if len(tier2_df) > 0 else 0
        n_t2_pairs = len(tier2_df)
        print(f"  Tier 2 done: {n_t2_feats} features, {n_t2_pairs} (feat,tgt) pairs "
              f"[{time.time()-t2_start:.0f}s]", flush=True)

        if n_t2_feats == 0:
            print(f"  SKIP: No Tier 2 survivors", flush=True)
            continue

        # --- Tier 4 ---
        print(f"\n  --- Tier 4: Clustering ---", flush=True)
        t4_start = time.time()
        rep_feats = run_tier4_inmemory(df_select, tier2_df)
        print(f"  Tier 4 done: {n_t2_feats} -> {len(rep_feats)} cluster reps "
              f"[{time.time()-t4_start:.0f}s]", flush=True)

        if len(rep_feats) == 0:
            print(f"  SKIP: No Tier 4 representatives", flush=True)
            continue

        # --- Tier 5 ---
        print(f"\n  --- Tier 5: Single-Feature WFO ---", flush=True)
        t5_start = time.time()
        survivors_df = run_tier5_inmemory(df_select, tf, rep_feats)
        n_t5_feats = survivors_df["feature"].nunique() if len(survivors_df) > 0 else 0
        n_t5_tgts = survivors_df["target"].nunique() if len(survivors_df) > 0 else 0
        print(f"  Tier 5 done: {len(rep_feats)} reps -> {n_t5_feats} features "
              f"across {n_t5_tgts} targets ({len(survivors_df)} pairs) "
              f"[{time.time()-t5_start:.0f}s]", flush=True)

        if len(survivors_df) == 0:
            print(f"  SKIP: No Tier 5 survivors", flush=True)
            continue

        # --- Tier 6: Holdout Test ---
        print(f"\n  --- Tier 6: Holdout Test ---", flush=True)
        t6_start = time.time()
        holdout_df = run_holdout_test(df_select, df_test, survivors_df, tf)
        print(f"  Tier 6 done: {len(holdout_df)} targets tested "
              f"[{time.time()-t6_start:.0f}s]", flush=True)

        if len(holdout_df) > 0:
            holdout_df["split"] = si + 1
            all_split_results.append(holdout_df)

            # Quick summary
            valid = holdout_df.dropna(subset=["holdout_score"])
            n_pos = (valid["holdout_score"] > 0).sum()
            print(f"\n  Split {si+1} summary: {n_pos}/{len(valid)} targets with "
                  f"holdout > 0", flush=True)

            # Top 10
            top = valid.sort_values("holdout_score", ascending=False).head(10)
            for _, r in top.iterrows():
                print(f"    {r['target']:<35} holdout={r['holdout_score']:>+.4f} "
                      f"n_feat={r['n_features']}", flush=True)

        split_time = time.time() - t0_split
        print(f"\n  Split {si+1} total time: {split_time:.0f}s", flush=True)

    # ============================================================
    # AGGREGATE RESULTS ACROSS SPLITS
    # ============================================================
    print(f"\n{'='*80}", flush=True)
    print(f"AGGREGATED RESULTS: {symbol} {tf}", flush=True)
    print(f"{'='*80}", flush=True)

    if not all_split_results:
        print("  NO RESULTS — all splits failed", flush=True)
        return

    combined = pd.concat(all_split_results, ignore_index=True)

    # Save raw per-split results
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"nolookahead_{symbol}_{tf}"
    combined.to_csv(output_dir / f"{prefix}_all_splits.csv", index=False)

    # Aggregate: mean holdout score across splits
    agg = combined.groupby("target").agg(
        mean_holdout=("holdout_score", "mean"),
        std_holdout=("holdout_score", "std"),
        min_holdout=("holdout_score", "min"),
        max_holdout=("holdout_score", "max"),
        n_splits=("holdout_score", "count"),
        all_positive=("holdout_score", lambda x: (x > 0).all()),
        is_binary=("is_binary", "first"),
        avg_n_features=("n_features", "mean"),
    ).reset_index()

    agg = agg.sort_values("mean_holdout", ascending=False)
    agg.to_csv(output_dir / f"{prefix}_aggregated.csv", index=False)
    print(f"Saved: {output_dir / f'{prefix}_aggregated.csv'}", flush=True)

    # Report
    n_splits_total = len(splits)
    multi_split = agg[agg["n_splits"] == n_splits_total]
    single_split = agg[agg["n_splits"] < n_splits_total]

    print(f"\n  Targets in ALL {n_splits_total} splits: {len(multi_split)}", flush=True)
    print(f"  Targets in fewer splits: {len(single_split)}", flush=True)

    # Tier targets
    print(f"\n--- Targets predictable across ALL splits (ranked by mean holdout) ---",
          flush=True)
    print(f"  {'Target':<35} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} "
          f"{'all+':>5} {'n_feat':>6}", flush=True)
    print(f"  {'-'*85}", flush=True)

    tiers = {"A": [], "B": [], "C": [], "D": []}
    for _, r in multi_split.iterrows():
        h = r["mean_holdout"]
        marker = ""
        if np.isfinite(h):
            score = abs(h)
            if score > 0.03 and r["all_positive"]:
                tiers["A"].append(r)
                marker = " [A]"
            elif score > 0.01 and r["all_positive"]:
                tiers["B"].append(r)
                marker = " [B]"
            elif score > 0.005:
                tiers["C"].append(r)
                marker = " [C]"
            else:
                tiers["D"].append(r)
                marker = " [D]"

        h_str = f"{h:>+8.4f}" if np.isfinite(h) else "     NaN"
        std_str = f"{r['std_holdout']:>8.4f}" if np.isfinite(r['std_holdout']) else "     NaN"
        print(f"  {r['target']:<35} {h_str} {std_str} "
              f"{r['min_holdout']:>+8.4f} {r['max_holdout']:>+8.4f} "
              f"{'yes' if r['all_positive'] else 'no':>5} "
              f"{r['avg_n_features']:>6.0f}{marker}", flush=True)

    print(f"\n  TIER A (strong, holdout>0.03, all splits positive): "
          f"{len(tiers['A'])} targets", flush=True)
    print(f"  TIER B (moderate, holdout>0.01, all positive): "
          f"{len(tiers['B'])} targets", flush=True)
    print(f"  TIER C (weak, holdout>0.005): {len(tiers['C'])} targets", flush=True)
    print(f"  TIER D (unpredictable): {len(tiers['D'])} targets", flush=True)

    # --- Visualizations ---
    print(f"\nGenerating visualizations...", flush=True)

    # Bar chart
    plot_df = multi_split.dropna(subset=["mean_holdout"]).sort_values(
        "mean_holdout", ascending=True).tail(40)
    if len(plot_df) > 0:
        fig, ax = plt.subplots(figsize=(12, max(8, len(plot_df) * 0.3)))
        colors = ["#2ecc71" if s > 0.01 else "#f39c12" if s > 0 else "#e74c3c"
                  for s in plot_df["mean_holdout"]]
        ax.barh(range(len(plot_df)), plot_df["mean_holdout"], color=colors,
                xerr=plot_df["std_holdout"].fillna(0), capsize=2)
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(plot_df["target"], fontsize=7)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.axvline(x=0.01, color="green", linewidth=1, linestyle="--", alpha=0.5)
        ax.set_xlabel("Mean Holdout Score (across splits)")
        ax.set_title(f"No-Lookahead Pipeline: {symbol} {tf}\n"
                     f"{n_splits_total} splits, {TEST_DAYS}d test each, "
                     f"{PURGE_DAYS}d purge",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(output_dir / f"{prefix}_holdout_scores.png", dpi=120)
        plt.close(fig)
        print(f"  Saved: {prefix}_holdout_scores.png", flush=True)

    # Per-split comparison
    if len(combined) > 0 and n_splits_total > 1:
        top_tgts = multi_split.head(25)["target"].tolist()
        pivot = combined[combined["target"].isin(top_tgts)].pivot(
            index="target", columns="split", values="holdout_score")
        pivot = pivot.reindex(
            multi_split.set_index("target").loc[
                [t for t in top_tgts if t in pivot.index], "mean_holdout"
            ].sort_values(ascending=False).index
        )

        fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.35)))
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                       vmin=-0.1, vmax=0.2)
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels([f"Split {c}" for c in pivot.columns], fontsize=10)
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index, fontsize=7)
        for i in range(len(pivot)):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                            fontsize=7, color="white" if abs(val) > 0.1 else "black")
        plt.colorbar(im, ax=ax, label="Holdout Score", shrink=0.6)
        ax.set_title(f"No-Lookahead: Per-Split Holdout — {symbol} {tf}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(output_dir / f"{prefix}_split_heatmap.png", dpi=120)
        plt.close(fig)
        print(f"  Saved: {prefix}_split_heatmap.png", flush=True)

    total_time = time.time() - t0_total
    print(f"\nTotal pipeline time: {total_time:.0f}s ({total_time/60:.1f}m)", flush=True)
    print(f"Done! Results in {output_dir}/", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="No-Lookahead Walk-Forward Feature Selection & Validation")
    parser.add_argument("symbol", help="e.g. SOLUSDT")
    parser.add_argument("timeframe", help="e.g. 4h")
    parser.add_argument("--features-dir", default="./features")
    parser.add_argument("--output-dir", default="./microstructure_research/results")
    args = parser.parse_args()

    run_pipeline(args.symbol, args.timeframe,
                 Path(args.features_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
