#!/usr/bin/env python3
"""
Tier 6: Multi-Feature Walk-Forward Target Validation

For each (coin, target) pair:
  1. Load Tier 5 survivor features for that target
  2. Cap at top N features (by OOS score) to avoid overfitting
  3. Run expanding-window walk-forward with Ridge (continuous) or LogisticRegression (binary)
  4. Measure OOS R² or AUC per fold
  5. Rank targets by predictability

This answers: "Which targets can we actually predict with multi-feature models?"

Usage:
  python tier6_target_validation.py SOLUSDT 4h
  python tier6_target_validation.py DOGEUSDT 4h
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

# Holdout parameters
HOLDOUT_DAYS = 90              # final test set — NEVER seen during feature selection
PURGE_DAYS = 2                 # gap between train and test
INNER_TEST_DAYS = 45           # inner WFO test window for stability check
INNER_MIN_TRAIN_DAYS = 120     # inner WFO minimum train

# Multi-feature settings
MAX_FEATURES_PER_TARGET = 20   # cap features to avoid overfitting
MIN_FEATURES_PER_TARGET = 3    # skip targets with fewer survivors
RIDGE_ALPHA = 10.0             # stronger regularization for multi-feature
LOGISTIC_C = 0.1               # stronger regularization for multi-feature


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


def load_tier5_survivors(results_dir: Path, symbol: str, tf: str) -> pd.DataFrame:
    """Load per-coin Tier 5 survivor list."""
    path = results_dir / f"tier5_{symbol}_{tf}_survivors.csv"
    if not path.exists():
        print(f"ERROR: {path} not found. Run tier5_wfo_scan.py first.", flush=True)
        sys.exit(1)
    return pd.read_csv(path)


def classify_target(df, tgt):
    """Determine if a target is binary or continuous."""
    unique_vals = df[tgt].dropna().unique()
    return len(unique_vals) <= 3


def make_inner_folds(n_train_candles, candles_per_day):
    """Create expanding-window inner folds within the training period only."""
    min_train = int(INNER_MIN_TRAIN_DAYS * candles_per_day)
    test_size = int(INNER_TEST_DAYS * candles_per_day)
    purge_size = int(PURGE_DAYS * candles_per_day)

    folds = []
    test_start = min_train + purge_size
    while test_start + test_size <= n_train_candles:
        train_end = test_start - purge_size
        test_end = test_start + test_size
        folds.append((0, train_end, test_start, test_end))
        test_start += test_size

    return folds


def fit_and_score(X_train, y_train, X_test, y_test, is_binary):
    """Fit model on train, score on test. Returns score or np.nan."""
    # Remove rows with NaN/inf
    train_mask = np.all(np.isfinite(X_train), axis=1) & np.isfinite(y_train)
    test_mask = np.all(np.isfinite(X_test), axis=1) & np.isfinite(y_test)

    if train_mask.sum() < 50 or test_mask.sum() < 15:
        return np.nan, int(train_mask.sum()), int(test_mask.sum())

    X_tr, y_tr = X_train[train_mask], y_train[train_mask]
    X_te, y_te = X_test[test_mask], y_test[test_mask]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    if is_binary:
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            return np.nan, len(X_tr), len(X_te)
        try:
            model = LogisticRegression(C=LOGISTIC_C, max_iter=500, solver="lbfgs")
            model.fit(X_tr, y_tr.astype(int))
            proba = model.predict_proba(X_te)[:, 1]
            score = roc_auc_score(y_te.astype(int), proba) - 0.5
            return score, len(X_tr), len(X_te)
        except Exception:
            return np.nan, len(X_tr), len(X_te)
    else:
        try:
            model = Ridge(alpha=RIDGE_ALPHA)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            score = r2_score(y_te, pred)
            return score, len(X_tr), len(X_te)
        except Exception:
            return np.nan, len(X_tr), len(X_te)


def validate_target(df, feat_cols, tgt_col, is_binary, holdout_start, purge_size,
                    inner_folds):
    """Validate one target with holdout + inner walk-forward.

    1. HOLDOUT TEST: train on [0, holdout_start - purge), test on [holdout_start, end)
    2. INNER WFO: expanding folds within [0, holdout_start - purge) for stability

    Returns dict with holdout score + inner fold scores.
    """
    train_end = holdout_start - purge_size

    # --- Holdout test ---
    X_train_h = df[feat_cols].iloc[:train_end].values
    y_train_h = df[tgt_col].iloc[:train_end].values
    X_test_h = df[feat_cols].iloc[holdout_start:].values
    y_test_h = df[tgt_col].iloc[holdout_start:].values

    holdout_score, n_train_h, n_test_h = fit_and_score(
        X_train_h, y_train_h, X_test_h, y_test_h, is_binary
    )

    # --- Inner walk-forward (within training period) ---
    inner_scores = []
    inner_details = []
    for fold_idx, (tr_s, tr_e, te_s, te_e) in enumerate(inner_folds):
        X_tr = df[feat_cols].iloc[tr_s:tr_e].values
        y_tr = df[tgt_col].iloc[tr_s:tr_e].values
        X_te = df[feat_cols].iloc[te_s:te_e].values
        y_te = df[tgt_col].iloc[te_s:te_e].values

        score, n_tr, n_te = fit_and_score(X_tr, y_tr, X_te, y_te, is_binary)
        inner_scores.append(score)
        inner_details.append({"fold": fold_idx, "score": score,
                              "n_train": n_tr, "n_test": n_te})

    inner_arr = np.array(inner_scores)
    inner_valid = inner_arr[np.isfinite(inner_arr)]

    return {
        "holdout_score": holdout_score,
        "holdout_n_train": n_train_h,
        "holdout_n_test": n_test_h,
        "inner_scores": inner_arr,
        "inner_details": inner_details,
        "inner_mean": float(np.mean(inner_valid)) if len(inner_valid) > 0 else np.nan,
        "inner_std": float(np.std(inner_valid)) if len(inner_valid) > 0 else np.nan,
        "inner_pct_pos": float((inner_valid > 0).mean()) if len(inner_valid) > 0 else 0.0,
        "inner_n_folds": len(inner_valid),
    }


def run_tier6(df, symbol, tf, survivors_df, output_dir):
    """Run Tier 6 multi-feature validation with temporal holdout.

    Design:
      - Last HOLDOUT_DAYS are reserved as a TRUE holdout test set
      - Inner walk-forward folds run within the pre-holdout period only
      - Feature lists come from Tier 5 (which used the full dataset)
      - The holdout score is the PRIMARY metric — it's on unseen data
      - Inner scores provide stability assessment
    """

    tf_hours = {"15m": 0.25, "30m": 0.5, "1h": 1, "2h": 2, "4h": 4}
    candles_per_day = 24 / tf_hours.get(tf, 1)

    holdout_candles = int(HOLDOUT_DAYS * candles_per_day)
    purge_candles = int(PURGE_DAYS * candles_per_day)
    holdout_start = len(df) - holdout_candles
    train_end_for_inner = holdout_start - purge_candles

    inner_folds = make_inner_folds(train_end_for_inner, candles_per_day)

    # Get unique targets from survivors
    targets = sorted(survivors_df["target"].unique())

    holdout_date = df.index[holdout_start]
    train_end_date = df.index[train_end_for_inner - 1]

    print(f"\n{'='*80}", flush=True)
    print(f"Tier 6 Multi-Feature Target Validation: {symbol} {tf}", flush=True)
    print(f"  Candles:        {len(df):,}", flush=True)
    print(f"  Targets:        {len(targets)}", flush=True)
    print(f"  HOLDOUT:        last {HOLDOUT_DAYS}d = {holdout_candles} candles "
          f"({holdout_date} -> {df.index[-1]})", flush=True)
    print(f"  Train period:   {df.index[0]} -> {train_end_date} "
          f"({train_end_for_inner} candles)", flush=True)
    print(f"  Purge gap:      {PURGE_DAYS}d = {purge_candles} candles", flush=True)
    print(f"  Inner WFO:      {len(inner_folds)} folds (min {INNER_MIN_TRAIN_DAYS}d train, "
          f"{INNER_TEST_DAYS}d test)", flush=True)
    for i, (ts, te, tes, tee) in enumerate(inner_folds):
        tr_days = (te - ts) / candles_per_day
        te_days = (tee - tes) / candles_per_day
        print(f"    Inner fold {i+1}: train {te-ts} candles ({tr_days:.0f}d), "
              f"test {tee-tes} candles ({te_days:.0f}d)", flush=True)
    print(f"  Max features:   {MAX_FEATURES_PER_TARGET} per target", flush=True)
    print(f"  Min features:   {MIN_FEATURES_PER_TARGET} per target (skip if fewer)", flush=True)
    print(f"  Regularization: Ridge alpha={RIDGE_ALPHA}, Logistic C={LOGISTIC_C}", flush=True)
    print(f"{'='*80}", flush=True)

    t0 = time.time()
    results = []
    all_fold_details = []
    skipped = 0

    for i, tgt in enumerate(targets):
        # Get survivor features for this target, sorted by OOS score
        tgt_survivors = survivors_df[survivors_df["target"] == tgt].copy()
        tgt_survivors = tgt_survivors.sort_values("mean_oos", key=abs, ascending=False)

        # Filter to features present in data
        feat_cols = [f for f in tgt_survivors["feature"].tolist() if f in df.columns]

        if len(feat_cols) < MIN_FEATURES_PER_TARGET:
            skipped += 1
            continue

        # Cap at max features
        feat_cols = feat_cols[:MAX_FEATURES_PER_TARGET]

        is_binary = classify_target(df, tgt)
        result = validate_target(df, feat_cols, tgt, is_binary, holdout_start,
                                 purge_candles, inner_folds)

        if result is None:
            skipped += 1
            continue

        holdout_s = result["holdout_score"]
        inner_m = result["inner_mean"]

        results.append({
            "target": tgt,
            "is_binary": is_binary,
            "n_features": len(feat_cols),
            "features_used": ", ".join(feat_cols),
            "holdout_score": holdout_s,
            "holdout_n_train": result["holdout_n_train"],
            "holdout_n_test": result["holdout_n_test"],
            "inner_mean": inner_m,
            "inner_std": result["inner_std"],
            "inner_pct_pos": result["inner_pct_pos"],
            "inner_n_folds": result["inner_n_folds"],
            "metric": "AUC_dev" if is_binary else "R2",
        })

        # Store inner fold details
        for fd in result["inner_details"]:
            all_fold_details.append({
                "target": tgt,
                "fold": fd["fold"],
                "score": fd["score"],
                "n_train": fd["n_train"],
                "n_test": fd["n_test"],
                "type": "inner",
            })
        # Store holdout as a special fold
        all_fold_details.append({
            "target": tgt,
            "fold": "holdout",
            "score": holdout_s,
            "n_train": result["holdout_n_train"],
            "n_test": result["holdout_n_test"],
            "type": "holdout",
        })

        elapsed = time.time() - t0
        done = i + 1 - skipped
        speed = done / elapsed if elapsed > 0 else 0
        remaining = len(targets) - i - 1
        eta = remaining / speed if speed > 0 else 0
        h_str = f"{holdout_s:>+.4f}" if np.isfinite(holdout_s) else "   NaN"
        i_str = f"{inner_m:>+.4f}" if np.isfinite(inner_m) else "   NaN"
        print(f"  [{done:>3}/{len(targets)}] {tgt:<35} "
              f"n={len(feat_cols):>2}  holdout={h_str}  inner={i_str}  "
              f"[{elapsed:.0f}s, ~{eta:.0f}s ETA]", flush=True)

    res_df = pd.DataFrame(results)
    fold_df = pd.DataFrame(all_fold_details)

    # Save raw results
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"tier6_{symbol}_{tf}"

    csv_path = output_dir / f"{prefix}_target_validation.csv"
    res_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}", flush=True)

    fold_path = output_dir / f"{prefix}_fold_details.csv"
    fold_df.to_csv(fold_path, index=False)
    print(f"Saved: {fold_path}", flush=True)

    if len(res_df) == 0:
        print("  No targets validated!", flush=True)
        return res_df

    # --- Report ---
    print(f"\n{'='*80}", flush=True)
    print(f"TIER 6 RESULTS: {symbol} {tf} (holdout = last {HOLDOUT_DAYS}d)", flush=True)
    print(f"  Targets evaluated: {len(res_df)} (skipped {skipped} with "
          f"<{MIN_FEATURES_PER_TARGET} features)", flush=True)
    print(f"{'='*80}", flush=True)

    # Continuous targets ranked by holdout score
    cont_df = res_df[~res_df["is_binary"]].sort_values("holdout_score", ascending=False)
    if len(cont_df) > 0:
        print(f"\n--- Continuous Targets (ranked by HOLDOUT R²) ---", flush=True)
        print(f"  {'Target':<35} {'n_feat':>6} {'HOLDOUT':>9} {'inner_m':>9} "
              f"{'inner_std':>9} {'i_%pos':>7}", flush=True)
        print(f"  {'-'*80}", flush=True)

        n_holdout_pos = 0
        for _, r in cont_df.iterrows():
            h = r["holdout_score"]
            if np.isfinite(h) and h > 0:
                n_holdout_pos += 1
            marker = " ***" if np.isfinite(h) and h > 0.01 else ""
            h_str = f"{h:>+9.5f}" if np.isfinite(h) else "      NaN"
            print(f"  {r['target']:<35} {r['n_features']:>6} {h_str} "
                  f"{r['inner_mean']:>+9.5f} {r['inner_std']:>9.5f} "
                  f"{r['inner_pct_pos']:>6.0%}{marker}", flush=True)

        print(f"\n  Continuous: {n_holdout_pos}/{len(cont_df)} with holdout R² > 0", flush=True)
        strong = cont_df[cont_df["holdout_score"] > 0.01]
        print(f"  Strong holdout (R² > 0.01): {len(strong)} targets", flush=True)

    # Binary targets ranked by holdout AUC deviation
    bin_df = res_df[res_df["is_binary"]].copy()
    if len(bin_df) > 0:
        bin_df["abs_holdout"] = bin_df["holdout_score"].abs()
        bin_df = bin_df.sort_values("abs_holdout", ascending=False)

        print(f"\n--- Binary Targets (ranked by |HOLDOUT AUC - 0.5|) ---", flush=True)
        print(f"  {'Target':<35} {'n_feat':>6} {'HOLDOUT':>9} {'inner_m':>9} "
              f"{'inner_std':>9} {'i_%pos':>7}", flush=True)
        print(f"  {'-'*80}", flush=True)

        n_holdout_sig = 0
        for _, r in bin_df.iterrows():
            h = r["holdout_score"]
            if np.isfinite(h) and abs(h) > 0.01:
                n_holdout_sig += 1
            marker = " ***" if np.isfinite(h) and abs(h) > 0.02 else ""
            h_str = f"{h:>+9.5f}" if np.isfinite(h) else "      NaN"
            print(f"  {r['target']:<35} {r['n_features']:>6} {h_str} "
                  f"{r['inner_mean']:>+9.5f} {r['inner_std']:>9.5f} "
                  f"{r['inner_pct_pos']:>6.0%}{marker}", flush=True)

        print(f"\n  Binary: {n_holdout_sig}/{len(bin_df)} with |holdout AUC_dev| > 0.01", flush=True)

    # --- Overall Predictability Tiers ---
    print(f"\n{'='*80}", flush=True)
    print(f"PREDICTABILITY TIERS: {symbol} {tf}", flush=True)
    print(f"  (Based on HOLDOUT score — truly unseen data)", flush=True)
    print(f"{'='*80}", flush=True)

    tiers = {"A_strong": [], "B_moderate": [], "C_weak": [], "D_unpredictable": []}
    for _, r in res_df.iterrows():
        h = r["holdout_score"]
        inner_pct = r["inner_pct_pos"]
        if not np.isfinite(h):
            tiers["D_unpredictable"].append(r)
            continue
        score = abs(h)
        if score > 0.03 and inner_pct >= 0.6:
            tiers["A_strong"].append(r)
        elif score > 0.01 and inner_pct >= 0.5:
            tiers["B_moderate"].append(r)
        elif score > 0.005:
            tiers["C_weak"].append(r)
        else:
            tiers["D_unpredictable"].append(r)

    tier_labels = {
        "A_strong": "A — STRONG (holdout>0.03, inner %pos>=60%)",
        "B_moderate": "B — MODERATE (holdout>0.01, inner %pos>=50%)",
        "C_weak": "C — WEAK (holdout>0.005)",
        "D_unpredictable": "D — UNPREDICTABLE",
    }

    for tier_key, label in tier_labels.items():
        items = tiers[tier_key]
        print(f"\n  {label}: {len(items)} targets", flush=True)
        for r in sorted(items, key=lambda x: abs(x["holdout_score"])
                        if np.isfinite(x["holdout_score"]) else 0, reverse=True):
            h = r["holdout_score"]
            h_str = f"{h:>+.4f}" if np.isfinite(h) else "   NaN"
            print(f"    {r['target']:<35} holdout={h_str} "
                  f"inner={r['inner_mean']:>+.4f} n_feat={r['n_features']}", flush=True)

    # --- Visualizations ---
    print(f"\nGenerating visualizations...", flush=True)

    # Bar chart: holdout scores
    plot_df = res_df.dropna(subset=["holdout_score"]).sort_values(
        "holdout_score", ascending=True).tail(40)
    if len(plot_df) > 0:
        fig, ax = plt.subplots(figsize=(12, max(8, len(plot_df) * 0.3)))
        colors = ["#2ecc71" if s > 0.01 else "#f39c12" if s > 0 else "#e74c3c"
                  for s in plot_df["holdout_score"]]
        ax.barh(range(len(plot_df)), plot_df["holdout_score"], color=colors)
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(plot_df["target"], fontsize=7)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.axvline(x=0.01, color="green", linewidth=1, linestyle="--", alpha=0.5,
                   label="threshold=0.01")
        ax.set_xlabel("Holdout OOS Score (R² or AUC_dev)")
        ax.set_title(f"Tier 6: Target Predictability — {symbol} {tf}\n"
                     f"Holdout = last {HOLDOUT_DAYS}d (truly unseen)",
                     fontsize=13, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        bar_path = output_dir / f"{prefix}_holdout_scores.png"
        fig.savefig(bar_path, dpi=120)
        plt.close(fig)
        print(f"  Saved: {bar_path}", flush=True)

    # Scatter: holdout vs inner mean (consistency check)
    valid_df = res_df.dropna(subset=["holdout_score", "inner_mean"])
    if len(valid_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["#e74c3c" if b else "#3498db" for b in valid_df["is_binary"]]
        ax.scatter(valid_df["inner_mean"], valid_df["holdout_score"],
                   c=colors, alpha=0.6, s=40)
        # Add diagonal
        lim_min = min(valid_df["inner_mean"].min(), valid_df["holdout_score"].min()) - 0.02
        lim_max = max(valid_df["inner_mean"].max(), valid_df["holdout_score"].max()) + 0.02
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.3, label="y=x")
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.axvline(x=0, color="gray", linewidth=0.5)
        ax.set_xlabel("Inner WFO Mean Score")
        ax.set_ylabel("Holdout Score (truly unseen)")
        ax.set_title(f"Tier 6: Inner vs Holdout — {symbol} {tf}\n"
                     f"Blue=continuous, Red=binary",
                     fontsize=13, fontweight="bold")
        ax.legend()
        # Label top targets
        for _, r in valid_df.nlargest(8, "holdout_score").iterrows():
            ax.annotate(r["target"].replace("tgt_", ""), (r["inner_mean"], r["holdout_score"]),
                        fontsize=6, alpha=0.7)
        plt.tight_layout()
        scatter_path = output_dir / f"{prefix}_inner_vs_holdout.png"
        fig.savefig(scatter_path, dpi=120)
        plt.close(fig)
        print(f"  Saved: {scatter_path}", flush=True)

    # Fold heatmap: inner folds + holdout column
    inner_fold_df = fold_df[fold_df["type"] == "inner"]
    holdout_fold_df = fold_df[fold_df["type"] == "holdout"]
    if len(inner_fold_df) > 0:
        top_targets = res_df.dropna(subset=["holdout_score"]).sort_values(
            "holdout_score", key=abs, ascending=False).head(25)["target"].tolist()

        # Inner folds pivot
        pivot = inner_fold_df[inner_fold_df["target"].isin(top_targets)].pivot(
            index="target", columns="fold", values="score"
        )
        # Add holdout column
        holdout_scores = holdout_fold_df.set_index("target")["score"]
        pivot["HOLDOUT"] = pivot.index.map(lambda t: holdout_scores.get(t, np.nan))

        # Reorder by holdout score
        order = res_df.set_index("target").loc[
            [t for t in top_targets if t in pivot.index], "holdout_score"]
        pivot = pivot.reindex(order.sort_values(ascending=False).index)

        fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.35)))
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                       vmin=-0.1, vmax=0.2)
        col_labels = [f"Inner {i+1}" for i in range(pivot.shape[1] - 1)] + ["HOLDOUT"]
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index, fontsize=7)
        for i in range(len(pivot)):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                            fontsize=6, color="white" if abs(val) > 0.1 else "black")
        plt.colorbar(im, ax=ax, label="OOS Score", shrink=0.6)
        ax.set_title(f"Tier 6: Inner Folds + HOLDOUT — {symbol} {tf}\n"
                     f"Top 25 targets (holdout = last {HOLDOUT_DAYS}d, truly unseen)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        hm_path = output_dir / f"{prefix}_fold_heatmap.png"
        fig.savefig(hm_path, dpi=120)
        plt.close(fig)
        print(f"  Saved: {hm_path}", flush=True)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s", flush=True)
    print(f"Done! Results in {output_dir}/", flush=True)
    return res_df


def main():
    parser = argparse.ArgumentParser(description="Tier 6: Multi-Feature Target Validation "
                                     "(Holdout-based)")
    parser.add_argument("symbol", help="e.g. SOLUSDT")
    parser.add_argument("timeframe", help="e.g. 4h")
    parser.add_argument("--features-dir", default="./features")
    parser.add_argument("--results-dir", default="./microstructure_research/results")
    parser.add_argument("--output-dir", default="./microstructure_research/results")
    parser.add_argument("--max-features", type=int, default=None,
                        help="Override MAX_FEATURES_PER_TARGET")
    parser.add_argument("--holdout-days", type=int, default=None,
                        help="Override HOLDOUT_DAYS (default 90)")
    args = parser.parse_args()

    if args.max_features is not None:
        global MAX_FEATURES_PER_TARGET
        MAX_FEATURES_PER_TARGET = args.max_features
    if args.holdout_days is not None:
        global HOLDOUT_DAYS
        HOLDOUT_DAYS = args.holdout_days

    features_dir = Path(args.features_dir)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    # Load data
    df = load_features(features_dir, args.symbol, args.timeframe)
    print(f"Loaded {len(df)} candles for {args.symbol} {args.timeframe} "
          f"({df.index[0]} -> {df.index[-1]})", flush=True)

    # Load Tier 5 survivors
    survivors_df = load_tier5_survivors(results_dir, args.symbol, args.timeframe)
    n_targets = survivors_df["target"].nunique()
    n_feats = survivors_df["feature"].nunique()
    n_pairs = len(survivors_df)
    print(f"Tier 5 survivors: {n_feats} features across {n_targets} targets "
          f"({n_pairs} pairs)", flush=True)

    run_tier6(df, args.symbol, args.timeframe, survivors_df, output_dir)


if __name__ == "__main__":
    main()
