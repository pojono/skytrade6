#!/usr/bin/env python3
"""
Walk-Forward ML Backtest — Zero Lookahead

Design:
  Fixed 12-month rolling selection window, 30-day purge, 30-day trade window.
  At each rebalance point:
    1. Take the most recent 12 months of data (selection window)
    2. Run Tier 2→4→5 from scratch on selection data only
    3. Train multi-feature Ridge/LogReg per target on selection data
    4. Generate predictions on the 30-day trade window (never seen)
    5. Evaluate predictions vs actuals

  This produces 12 independent monthly evaluations from 2 years of data.

Usage:
  python wfo_ml_backtest.py SOLUSDT 4h
  python wfo_ml_backtest.py XRPUSDT 4h
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*constant.*")
warnings.filterwarnings("ignore", message=".*Only one class.*")
warnings.filterwarnings("ignore", category=FutureWarning)

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
# WFO PARAMETERS
# ============================================================
SELECTION_DAYS = 360       # fixed 12-month rolling window
PURGE_DAYS = 30            # gap between selection and trade
TRADE_DAYS = 30            # each trade window (monthly rebalance)
MIN_DATA_DAYS = SELECTION_DAYS + PURGE_DAYS + TRADE_DAYS  # ~420d minimum

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
# DATA LOADING (same as pipeline_nolookahead.py)
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
# TIER 2: Temporal Stability (in-memory)
# ============================================================
def run_tier2(df, tf):
    feat_cols, continuous_tgts, binary_tgts = classify_columns(df)
    cpd = get_candles_per_day(tf)
    all_tgts = continuous_tgts + binary_tgts

    win_size = int(T2_WINDOW_DAYS * cpd)
    step_size = int(T2_STEP_DAYS * cpd)
    windows = []
    start = 0
    while start + win_size <= len(df):
        windows.append((start, start + win_size))
        start += step_size

    if len(windows) < 3:
        return pd.DataFrame()

    results = []
    for ti, tgt in enumerate(all_tgts):
        is_binary = tgt in binary_tgts
        tgt_vals_full = df[tgt].values

        for feat in feat_cols:
            feat_vals_full = df[feat].values

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

            wrong = [1 if s != full_sign else 0 for s in signs]
            max_wrong = 0
            cur = 0
            for w in wrong:
                if w:
                    cur += 1
                    max_wrong = max(max_wrong, cur)
                else:
                    cur = 0

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
                })

        if (ti + 1) % 10 == 0 or ti == len(all_tgts) - 1:
            print(f"      T2: [{ti+1}/{len(all_tgts)}] targets, "
                  f"{len(results)} pairs pass", flush=True)

    return pd.DataFrame(results)


# ============================================================
# TIER 4: Clustering (in-memory)
# ============================================================
def run_tier4(df, tier2_df):
    if len(tier2_df) == 0:
        return []

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

    feat_list = [f for f in feat_list if f in df.columns]
    if len(feat_list) <= 1:
        return feat_list

    feat_data = df[feat_list].replace([np.inf, -np.inf], np.nan).fillna(0)
    corr = feat_data.corr(method="spearman")

    dist = 1 - corr.abs()
    np.fill_diagonal(dist.values, 0)
    dist = (dist + dist.T) / 2
    dist = dist.clip(lower=0)

    try:
        condensed = squareform(dist.values)
        Z = linkage(condensed, method="average")
        clusters = fcluster(Z, t=1 - T4_CORR_THRESHOLD, criterion="distance")
    except Exception:
        return feat_list

    cluster_df = pd.DataFrame({"feature": feat_list, "cluster": clusters})
    cluster_df = cluster_df.merge(surv_df[["feature", "composite"]], on="feature")

    reps = []
    for cid in sorted(set(clusters)):
        members = cluster_df[cluster_df["cluster"] == cid].sort_values(
            "composite", ascending=False)
        reps.append(members.iloc[0]["feature"])

    return reps


# ============================================================
# TIER 5: Walk-Forward Single-Feature OOS (in-memory)
# ============================================================
def run_tier5(df, tf, rep_feats):
    _, continuous_tgts, binary_tgts = classify_columns(df)
    all_tgts = continuous_tgts + binary_tgts
    cpd = get_candles_per_day(tf)

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

        if done % (len(feats) * 5) < len(feats) or done == total:
            print(f"      T5: [{done}/{total}] pairs, "
                  f"{len(survivors)} survive", flush=True)

    return pd.DataFrame(survivors)


# ============================================================
# TRAIN & PREDICT: Multi-feature model per target
# ============================================================
def train_and_predict(df_selection, df_trade, survivors_df, tf):
    """Train on selection period, predict on trade window.

    Returns DataFrame with per-target predictions and actuals.
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
                     if f in df_selection.columns and f in df_trade.columns]

        if len(feat_cols) < MIN_FEATURES:
            continue

        feat_cols = feat_cols[:MAX_FEATURES]
        is_binary = tgt in binary_tgts

        # Train on full selection period
        X_train = df_selection[feat_cols].values
        y_train = df_selection[tgt].values
        X_trade = df_trade[feat_cols].values
        y_trade = df_trade[tgt].values

        tr_mask = np.all(np.isfinite(X_train), axis=1) & np.isfinite(y_train)
        te_mask = np.all(np.isfinite(X_trade), axis=1) & np.isfinite(y_trade)

        if tr_mask.sum() < 50 or te_mask.sum() < 10:
            continue

        Xtr, ytr = X_train[tr_mask], y_train[tr_mask]
        Xte, yte = X_trade[te_mask], y_trade[te_mask]

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        score = np.nan
        predictions = None
        if is_binary:
            if len(np.unique(ytr)) >= 2 and len(np.unique(yte)) >= 2:
                try:
                    model = LogisticRegression(C=LOGISTIC_C, max_iter=500,
                                               solver="lbfgs")
                    model.fit(Xtr, ytr.astype(int))
                    proba = model.predict_proba(Xte)[:, 1]
                    score = roc_auc_score(yte.astype(int), proba) - 0.5
                    predictions = proba
                except:
                    pass
        else:
            try:
                model = Ridge(alpha=RIDGE_ALPHA)
                model.fit(Xtr, ytr)
                pred = model.predict(Xte)
                score = r2_score(yte, pred)
                predictions = pred
            except:
                pass

        results.append({
            "target": tgt,
            "is_binary": is_binary,
            "n_features": len(feat_cols),
            "features": ", ".join(feat_cols),
            "trade_score": score,
            "n_train": int(tr_mask.sum()),
            "n_trade": int(te_mask.sum()),
            "metric": "AUC_dev" if is_binary else "R2",
        })

    return pd.DataFrame(results)


# ============================================================
# GENERATE REBALANCE SCHEDULE
# ============================================================
def generate_rebalance_schedule(df, tf):
    """Generate monthly rebalance points with 12-month rolling selection."""
    cpd = get_candles_per_day(tf)
    n = len(df)

    select_candles = int(SELECTION_DAYS * cpd)
    purge_candles = int(PURGE_DAYS * cpd)
    trade_candles = int(TRADE_DAYS * cpd)

    min_start = select_candles + purge_candles  # first trade can start here

    schedule = []
    trade_start = min_start
    while trade_start + trade_candles <= n:
        # Selection window: 12 months ending at purge start
        purge_start = trade_start - purge_candles
        select_end = purge_start
        select_start = select_end - select_candles

        if select_start < 0:
            break

        trade_end = min(trade_start + trade_candles, n)

        schedule.append({
            "select_start": select_start,
            "select_end": select_end,
            "purge_end": trade_start,
            "trade_start": trade_start,
            "trade_end": trade_end,
        })

        trade_start += trade_candles  # next month

    return schedule


# ============================================================
# MAIN WFO BACKTEST
# ============================================================
def run_wfo_backtest(symbol, tf, features_dir, output_dir, max_periods=0):
    t0_total = time.time()

    # Load full data
    df = load_features(features_dir, symbol, tf)
    cpd = get_candles_per_day(tf)
    print(f"\nLoaded {len(df)} candles for {symbol} {tf}", flush=True)
    print(f"  Range: {df.index[0]} -> {df.index[-1]}", flush=True)
    print(f"  Days: {len(df)/cpd:.0f}", flush=True)

    # Generate rebalance schedule
    schedule = generate_rebalance_schedule(df, tf)
    if not schedule:
        print("ERROR: Not enough data for even 1 rebalance!", flush=True)
        sys.exit(1)

    if max_periods > 0:
        schedule = schedule[:max_periods]

    print(f"\n{'='*80}", flush=True)
    print(f"WFO ML BACKTEST: {symbol} {tf}", flush=True)
    print(f"  Selection: {SELECTION_DAYS}d rolling, Purge: {PURGE_DAYS}d, "
          f"Trade: {TRADE_DAYS}d", flush=True)
    print(f"  Rebalance periods: {len(schedule)}", flush=True)
    for i, rb in enumerate(schedule):
        sel_s = df.index[rb["select_start"]].date()
        sel_e = df.index[rb["select_end"] - 1].date()
        tr_s = df.index[rb["trade_start"]].date()
        tr_e = df.index[min(rb["trade_end"] - 1, len(df) - 1)].date()
        print(f"  Period {i+1:2d}: select {sel_s} -> {sel_e}, "
              f"trade {tr_s} -> {tr_e}", flush=True)
    print(f"{'='*80}", flush=True)

    all_period_results = []

    for pi, rb in enumerate(schedule):
        t0_period = time.time()
        print(f"\n{'#'*80}", flush=True)
        print(f"# PERIOD {pi+1}/{len(schedule)}", flush=True)
        print(f"{'#'*80}", flush=True)

        df_select = df.iloc[rb["select_start"]:rb["select_end"]].copy()
        df_trade = df.iloc[rb["trade_start"]:rb["trade_end"]].copy()

        sel_days = len(df_select) / cpd
        trade_days = len(df_trade) / cpd
        print(f"  Selection: {len(df_select)} candles ({sel_days:.0f}d) "
              f"[{df_select.index[0].date()} -> {df_select.index[-1].date()}]",
              flush=True)
        print(f"  Trade:     {len(df_trade)} candles ({trade_days:.0f}d) "
              f"[{df_trade.index[0].date()} -> {df_trade.index[-1].date()}]",
              flush=True)

        # --- Tier 2 ---
        print(f"\n    --- Tier 2: Stability Scan ---", flush=True)
        t2_start = time.time()
        tier2_df = run_tier2(df_select, tf)
        n_t2_feats = tier2_df["feature"].nunique() if len(tier2_df) > 0 else 0
        n_t2_pairs = len(tier2_df)
        print(f"    T2 done: {n_t2_feats} features, {n_t2_pairs} pairs "
              f"[{time.time()-t2_start:.0f}s]", flush=True)

        if n_t2_feats == 0:
            print(f"    SKIP: No Tier 2 survivors", flush=True)
            continue

        # --- Tier 4 ---
        print(f"\n    --- Tier 4: Clustering ---", flush=True)
        t4_start = time.time()
        rep_feats = run_tier4(df_select, tier2_df)
        print(f"    T4 done: {n_t2_feats} -> {len(rep_feats)} reps "
              f"[{time.time()-t4_start:.0f}s]", flush=True)

        if len(rep_feats) == 0:
            print(f"    SKIP: No Tier 4 representatives", flush=True)
            continue

        # --- Tier 5 ---
        print(f"\n    --- Tier 5: Single-Feature WFO ---", flush=True)
        t5_start = time.time()
        survivors_df = run_tier5(df_select, tf, rep_feats)
        n_t5_feats = survivors_df["feature"].nunique() if len(survivors_df) > 0 else 0
        n_t5_tgts = survivors_df["target"].nunique() if len(survivors_df) > 0 else 0
        print(f"    T5 done: {len(rep_feats)} reps -> {n_t5_feats} features, "
              f"{n_t5_tgts} targets ({len(survivors_df)} pairs) "
              f"[{time.time()-t5_start:.0f}s]", flush=True)

        if len(survivors_df) == 0:
            print(f"    SKIP: No Tier 5 survivors", flush=True)
            continue

        # --- Train & Predict on Trade Window ---
        print(f"\n    --- Train & Predict ---", flush=True)
        t6_start = time.time()
        period_df = train_and_predict(df_select, df_trade, survivors_df, tf)
        print(f"    Done: {len(period_df)} targets predicted "
              f"[{time.time()-t6_start:.0f}s]", flush=True)

        if len(period_df) > 0:
            period_df["period"] = pi + 1
            period_df["trade_start"] = str(df_trade.index[0].date())
            period_df["trade_end"] = str(df_trade.index[-1].date())
            all_period_results.append(period_df)

            # Quick summary
            valid = period_df.dropna(subset=["trade_score"])
            n_pos = (valid["trade_score"] > 0).sum()
            mean_score = valid["trade_score"].mean()
            print(f"\n    Period {pi+1}: {n_pos}/{len(valid)} targets positive, "
                  f"mean score={mean_score:+.4f}", flush=True)

            # Top 5
            top = valid.sort_values("trade_score", ascending=False).head(5)
            for _, r in top.iterrows():
                print(f"      {r['target']:<35} score={r['trade_score']:>+.4f} "
                      f"n_feat={r['n_features']}", flush=True)

        period_time = time.time() - t0_period
        print(f"\n    Period {pi+1} time: {period_time:.0f}s ({period_time/60:.1f}m)",
              flush=True)

    # ============================================================
    # AGGREGATE RESULTS
    # ============================================================
    print(f"\n{'='*80}", flush=True)
    print(f"WFO BACKTEST RESULTS: {symbol} {tf}", flush=True)
    print(f"{'='*80}", flush=True)

    if not all_period_results:
        print("  NO RESULTS — all periods failed", flush=True)
        return

    combined = pd.concat(all_period_results, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"wfo_ml_{symbol}_{tf}"
    combined.to_csv(output_dir / f"{prefix}_all_periods.csv", index=False)

    # Per-target aggregation
    agg = combined.groupby("target").agg(
        mean_score=("trade_score", "mean"),
        std_score=("trade_score", "std"),
        min_score=("trade_score", "min"),
        max_score=("trade_score", "max"),
        n_periods=("trade_score", "count"),
        pct_positive=("trade_score", lambda x: (x > 0).mean()),
        all_positive=("trade_score", lambda x: (x > 0).all()),
        is_binary=("is_binary", "first"),
        avg_n_features=("n_features", "mean"),
    ).reset_index()

    agg = agg.sort_values("mean_score", ascending=False)
    agg.to_csv(output_dir / f"{prefix}_aggregated.csv", index=False)

    # Report
    n_periods = len(schedule)
    print(f"\n  Total periods: {n_periods}", flush=True)
    print(f"  Targets evaluated: {len(agg)}", flush=True)

    # Tier classification
    print(f"\n--- Target Performance (ranked by mean trade score) ---", flush=True)
    print(f"  {'Target':<35} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} "
          f"{'%pos':>5} {'n_per':>5} {'n_feat':>6}", flush=True)
    print(f"  {'-'*90}", flush=True)

    tier_a = []
    tier_b = []
    for _, r in agg.iterrows():
        h = r["mean_score"]
        if not np.isfinite(h):
            continue

        marker = ""
        if h > 0.03 and r["pct_positive"] >= 0.70:
            tier_a.append(r)
            marker = " [A]"
        elif h > 0.01 and r["pct_positive"] >= 0.60:
            tier_b.append(r)
            marker = " [B]"

        h_str = f"{h:>+8.4f}"
        std_str = f"{r['std_score']:>8.4f}" if np.isfinite(r['std_score']) else "     NaN"
        print(f"  {r['target']:<35} {h_str} {std_str} "
              f"{r['min_score']:>+8.4f} {r['max_score']:>+8.4f} "
              f"{r['pct_positive']:>5.0%} {r['n_periods']:>5.0f} "
              f"{r['avg_n_features']:>6.0f}{marker}", flush=True)

    print(f"\n  TIER A (mean>0.03, ≥70% positive): {len(tier_a)} targets", flush=True)
    print(f"  TIER B (mean>0.01, ≥60% positive): {len(tier_b)} targets", flush=True)

    # --- Per-period consistency ---
    print(f"\n--- Per-Period Summary ---", flush=True)
    for pi in range(1, n_periods + 1):
        period_data = combined[combined["period"] == pi]
        if len(period_data) == 0:
            print(f"  Period {pi:2d}: NO DATA", flush=True)
            continue
        valid = period_data.dropna(subset=["trade_score"])
        n_pos = (valid["trade_score"] > 0).sum()
        mean_s = valid["trade_score"].mean()
        dates = f"{period_data['trade_start'].iloc[0]} -> {period_data['trade_end'].iloc[0]}"
        print(f"  Period {pi:2d}: {dates}  {n_pos}/{len(valid)} positive, "
              f"mean={mean_s:+.4f}", flush=True)

    # --- Visualizations ---
    print(f"\nGenerating visualizations...", flush=True)

    # 1. Bar chart of mean scores
    plot_df = agg.dropna(subset=["mean_score"]).sort_values(
        "mean_score", ascending=True).tail(40)
    if len(plot_df) > 0:
        fig, ax = plt.subplots(figsize=(12, max(8, len(plot_df) * 0.3)))
        colors = ["#2ecc71" if s > 0.01 else "#f39c12" if s > 0 else "#e74c3c"
                  for s in plot_df["mean_score"]]
        ax.barh(range(len(plot_df)), plot_df["mean_score"], color=colors,
                xerr=plot_df["std_score"].fillna(0), capsize=2)
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(plot_df["target"], fontsize=7)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.axvline(x=0.01, color="green", linewidth=1, linestyle="--", alpha=0.5)
        ax.set_xlabel("Mean Trade Window Score (across periods)")
        ax.set_title(f"WFO ML Backtest: {symbol} {tf}\n"
                     f"{n_periods} periods, {SELECTION_DAYS}d selection, "
                     f"{TRADE_DAYS}d trade, {PURGE_DAYS}d purge",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(output_dir / f"{prefix}_scores.png", dpi=120)
        plt.close(fig)
        print(f"  Saved: {prefix}_scores.png", flush=True)

    # 2. Period heatmap for top targets
    if len(combined) > 0 and n_periods > 1:
        # Get targets that appear in at least half the periods
        min_periods = max(2, n_periods // 2)
        frequent = agg[agg["n_periods"] >= min_periods].head(25)["target"].tolist()

        if frequent:
            pivot = combined[combined["target"].isin(frequent)].pivot(
                index="target", columns="period", values="trade_score")
            # Reorder by mean score
            order = agg.set_index("target").loc[
                [t for t in frequent if t in pivot.index], "mean_score"
            ].sort_values(ascending=False).index
            pivot = pivot.reindex(order)

            fig, ax = plt.subplots(figsize=(max(8, n_periods * 0.8),
                                            max(6, len(pivot) * 0.35)))
            im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                           vmin=-0.1, vmax=0.2)
            ax.set_xticks(range(pivot.shape[1]))
            ax.set_xticklabels([f"P{c}" for c in pivot.columns], fontsize=8)
            ax.set_yticks(range(len(pivot)))
            ax.set_yticklabels(pivot.index, fontsize=7)
            for i in range(len(pivot)):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    if np.isfinite(val):
                        ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                                fontsize=6,
                                color="white" if abs(val) > 0.1 else "black")
            plt.colorbar(im, ax=ax, label="Trade Score", shrink=0.6)
            ax.set_title(f"WFO ML: Per-Period Scores — {symbol} {tf}",
                         fontsize=13, fontweight="bold")
            plt.tight_layout()
            fig.savefig(output_dir / f"{prefix}_period_heatmap.png", dpi=120)
            plt.close(fig)
            print(f"  Saved: {prefix}_period_heatmap.png", flush=True)

    # 3. Cumulative score over time (equity-curve style)
    if len(combined) > 0:
        # For each period, compute mean score across all targets
        period_means = combined.groupby("period")["trade_score"].mean()
        cum_score = period_means.cumsum()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Per-period bars
        colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in period_means]
        ax1.bar(period_means.index, period_means.values, color=colors)
        ax1.axhline(y=0, color="black", linewidth=0.5)
        ax1.set_ylabel("Mean Score")
        ax1.set_title(f"WFO ML: Per-Period Performance — {symbol} {tf}",
                       fontsize=13, fontweight="bold")

        # Cumulative
        ax2.plot(cum_score.index, cum_score.values, "b-o", linewidth=2)
        ax2.axhline(y=0, color="black", linewidth=0.5)
        ax2.set_xlabel("Period")
        ax2.set_ylabel("Cumulative Mean Score")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_dir / f"{prefix}_cumulative.png", dpi=120)
        plt.close(fig)
        print(f"  Saved: {prefix}_cumulative.png", flush=True)

    total_time = time.time() - t0_total
    print(f"\nTotal backtest time: {total_time:.0f}s ({total_time/60:.1f}m)",
          flush=True)
    print(f"Done! Results in {output_dir}/", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Walk-Forward ML Backtest — Zero Lookahead")
    parser.add_argument("symbol", help="e.g. SOLUSDT")
    parser.add_argument("timeframe", help="e.g. 4h")
    parser.add_argument("--features-dir", default="./features")
    parser.add_argument("--output-dir", default="./microstructure_research/results")
    parser.add_argument("--periods", type=int, default=0,
                        help="Max periods to run (0=all)")
    args = parser.parse_args()

    run_wfo_backtest(args.symbol, args.timeframe,
                     Path(args.features_dir), Path(args.output_dir),
                     max_periods=args.periods)


if __name__ == "__main__":
    main()
