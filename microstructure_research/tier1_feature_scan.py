#!/usr/bin/env python3
"""
Tier 1: Univariate Feature-Target Signal Scan

For every (feature, target) pair, compute:
  - Spearman rank correlation (continuous targets)
  - AUC (binary targets)
  - Mutual Information (both)

Outputs:
  - CSV with full matrix of scores
  - Console report: top features per key target
  - Summary statistics

Usage:
  python tier1_feature_scan.py DOGEUSDT 4h
  python tier1_feature_scan.py SOLUSDT 4h --symbols SOLUSDT DOGEUSDT
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
import matplotlib.colors as mcolors
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


# Key targets to highlight in the report
KEY_TARGETS_CONTINUOUS = [
    "tgt_ret_1", "tgt_ret_3", "tgt_ret_5",
    "tgt_cum_ret_5", "tgt_cum_ret_10",
    "tgt_sharpe_5", "tgt_sharpe_10",
]

KEY_TARGETS_BINARY = [
    "tgt_profitable_long_3", "tgt_profitable_long_5",
    "tgt_profitable_short_3", "tgt_profitable_short_5",
    "tgt_direction_1", "tgt_direction_3", "tgt_direction_5",
]


def load_features(features_dir: Path, symbol: str, tf: str) -> pd.DataFrame:
    """Load all parquet files for a symbol/timeframe into one DataFrame."""
    tf_dir = features_dir / symbol / tf
    files = sorted(tf_dir.glob("*.parquet"))
    if not files:
        print(f"ERROR: No parquet files in {tf_dir}")
        sys.exit(1)

    dfs = []
    for f in files:
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    return df


def classify_columns(df: pd.DataFrame):
    """Split columns into features, targets (continuous), targets (binary)."""
    tgt_cols = [c for c in df.columns if c.startswith("tgt_")]
    feat_cols = [c for c in df.columns if not c.startswith("tgt_")]

    # Classify targets as binary or continuous
    binary_tgts = []
    continuous_tgts = []
    for c in tgt_cols:
        unique_vals = df[c].dropna().unique()
        if len(unique_vals) <= 3:  # 0, 1, (maybe NaN)
            binary_tgts.append(c)
        else:
            continuous_tgts.append(c)

    return feat_cols, continuous_tgts, binary_tgts


def compute_spearman(feat_vals: np.ndarray, tgt_vals: np.ndarray) -> float:
    """Spearman correlation, handling edge cases."""
    mask = np.isfinite(feat_vals) & np.isfinite(tgt_vals)
    if mask.sum() < 30:
        return np.nan
    x, y = feat_vals[mask], tgt_vals[mask]
    if x.std() == 0 or y.std() == 0:
        return 0.0
    r, _ = stats.spearmanr(x, y)
    return r


def compute_auc(feat_vals: np.ndarray, tgt_vals: np.ndarray) -> float:
    """AUC of a single feature as classifier for a binary target."""
    mask = np.isfinite(feat_vals) & np.isfinite(tgt_vals)
    if mask.sum() < 30:
        return np.nan
    y = tgt_vals[mask]
    x = feat_vals[mask]
    # Need both classes
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return roc_auc_score(y, x)
    except ValueError:
        return np.nan


def run_scan(df: pd.DataFrame, symbol: str, tf: str, output_dir: Path):
    """Run full Tier 1 scan."""
    feat_cols, continuous_tgts, binary_tgts = classify_columns(df)

    n_feats = len(feat_cols)
    n_cont = len(continuous_tgts)
    n_bin = len(binary_tgts)
    n_candles = len(df)

    print(f"\n{'='*70}")
    print(f"Tier 1 Scan: {symbol} {tf}")
    print(f"  Candles:           {n_candles:,}")
    print(f"  Features:          {n_feats}")
    print(f"  Continuous targets: {n_cont}")
    print(f"  Binary targets:    {n_bin}")
    print(f"  Total pairs:       {n_feats * (n_cont + n_bin):,}")
    print(f"{'='*70}")

    # --- Spearman for continuous targets ---
    print(f"\n[1/3] Computing Spearman correlations ({n_feats} × {n_cont})...")
    t0 = time.time()
    spearman_results = {}

    for i, tgt in enumerate(continuous_tgts):
        tgt_vals = df[tgt].values.astype(float)
        row = {}
        for feat in feat_cols:
            feat_vals = df[feat].values.astype(float)
            row[feat] = compute_spearman(feat_vals, tgt_vals)
        spearman_results[tgt] = row
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_cont - i - 1)
            print(f"  [{i+1}/{n_cont}] {tgt} [{elapsed:.0f}s elapsed, ~{eta:.0f}s ETA]")

    spearman_df = pd.DataFrame(spearman_results)  # features × targets
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s")

    # --- AUC for binary targets ---
    print(f"\n[2/3] Computing AUC scores ({n_feats} × {n_bin})...")
    t0 = time.time()
    auc_results = {}

    for i, tgt in enumerate(binary_tgts):
        tgt_vals = df[tgt].values.astype(float)
        row = {}
        for feat in feat_cols:
            feat_vals = df[feat].values.astype(float)
            row[feat] = compute_auc(feat_vals, tgt_vals)
        auc_results[tgt] = row
        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_bin - i - 1)
            print(f"  [{i+1}/{n_bin}] {tgt} [{elapsed:.0f}s elapsed, ~{eta:.0f}s ETA]")

    auc_df = pd.DataFrame(auc_results)  # features × targets
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s")

    # --- Mutual Information (on key targets only, it's slow) ---
    print(f"\n[3/3] Computing Mutual Information (key targets only)...")
    t0 = time.time()
    mi_results = {}

    # Prepare clean feature matrix (no NaN, no inf)
    X = df[feat_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    key_cont_present = [t for t in KEY_TARGETS_CONTINUOUS if t in continuous_tgts]
    key_bin_present = [t for t in KEY_TARGETS_BINARY if t in binary_tgts]

    for tgt in key_cont_present:
        y = df[tgt].values.astype(float)
        mask = np.isfinite(y)
        if mask.sum() < 100:
            continue
        mi = mutual_info_regression(X.loc[mask], y[mask], random_state=42, n_neighbors=5)
        mi_results[tgt] = dict(zip(feat_cols, mi))
        print(f"  MI computed for {tgt}")

    for tgt in key_bin_present:
        y = df[tgt].values.astype(float)
        mask = np.isfinite(y)
        if mask.sum() < 100:
            continue
        mi = mutual_info_classif(X.loc[mask], y[mask].astype(int), random_state=42,
                                  n_neighbors=5)
        mi_results[tgt] = dict(zip(feat_cols, mi))
        print(f"  MI computed for {tgt}")

    mi_df = pd.DataFrame(mi_results) if mi_results else pd.DataFrame()
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s")

    # --- Save CSVs ---
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"tier1_{symbol}_{tf}"

    spearman_path = output_dir / f"{prefix}_spearman.csv"
    spearman_df.to_csv(spearman_path)
    print(f"\nSaved: {spearman_path}")

    auc_path = output_dir / f"{prefix}_auc.csv"
    auc_df.to_csv(auc_path)
    print(f"Saved: {auc_path}")

    if not mi_df.empty:
        mi_path = output_dir / f"{prefix}_mi.csv"
        mi_df.to_csv(mi_path)
        print(f"Saved: {mi_path}")

    # --- Heatmaps ---
    print(f"\nGenerating heatmaps...")
    generate_heatmaps(spearman_df, auc_df, mi_df, symbol, tf, output_dir, prefix,
                      key_cont_present, key_bin_present)

    # --- Console Report ---
    print(f"\n{'='*70}")
    print(f"REPORT: Top features per key target — {symbol} {tf}")
    print(f"{'='*70}")

    # Continuous targets: top by |Spearman|
    for tgt in key_cont_present:
        if tgt not in spearman_df.columns:
            continue
        col = spearman_df[tgt].dropna().abs().sort_values(ascending=False)
        signed = spearman_df[tgt]
        print(f"\n--- {tgt} (Spearman, top 20) ---")
        print(f"{'Rank':<5} {'Feature':<50} {'Spearman':>10} {'|r|':>8}")
        for rank, (feat, absval) in enumerate(col.head(20).items(), 1):
            r = signed[feat]
            print(f"{rank:<5} {feat:<50} {r:>+10.4f} {absval:>8.4f}")

    # Binary targets: top by |AUC - 0.5|
    for tgt in key_bin_present:
        if tgt not in auc_df.columns:
            continue
        col = (auc_df[tgt].dropna() - 0.5).abs().sort_values(ascending=False)
        raw_auc = auc_df[tgt]
        print(f"\n--- {tgt} (AUC, top 20) ---")
        print(f"{'Rank':<5} {'Feature':<50} {'AUC':>8} {'|AUC-0.5|':>10}")
        for rank, (feat, dev) in enumerate(col.head(20).items(), 1):
            auc_val = raw_auc[feat]
            print(f"{rank:<5} {feat:<50} {auc_val:>8.4f} {dev:>10.4f}")

    # MI highlights
    if not mi_df.empty:
        print(f"\n--- Mutual Information highlights ---")
        for tgt in mi_df.columns:
            col = mi_df[tgt].sort_values(ascending=False)
            print(f"\n  {tgt} (MI, top 10):")
            for rank, (feat, mi_val) in enumerate(col.head(10).items(), 1):
                print(f"    {rank:>2}. {feat:<50} MI={mi_val:.5f}")

    # --- Summary statistics ---
    print(f"\n{'='*70}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*70}")

    # How many features have |Spearman| > threshold for any key target?
    for thresh in [0.02, 0.03, 0.05, 0.10]:
        n_pass = 0
        for tgt in key_cont_present:
            if tgt in spearman_df.columns:
                n_pass += (spearman_df[tgt].abs() > thresh).sum()
        print(f"  Feature-target pairs with |Spearman| > {thresh}: "
              f"{n_pass} (across {len(key_cont_present)} key continuous targets)")

    for thresh in [0.52, 0.53, 0.55]:
        n_pass = 0
        for tgt in key_bin_present:
            if tgt in auc_df.columns:
                n_pass += ((auc_df[tgt] > thresh) | (auc_df[tgt] < 1 - thresh)).sum()
        print(f"  Feature-target pairs with AUC > {thresh} or < {1-thresh:.2f}: "
              f"{n_pass} (across {len(key_bin_present)} key binary targets)")

    print(f"\nDone! Results in {output_dir}/")


def generate_heatmaps(spearman_df, auc_df, mi_df, symbol, tf, output_dir, prefix,
                      key_cont, key_bin):
    """Generate focused heatmaps for top features."""
    TOP_N = 40  # top features to show

    # --- 1. Spearman heatmap: top features × key continuous targets ---
    if key_cont:
        cont_sub = spearman_df[[t for t in key_cont if t in spearman_df.columns]].copy()
        # Rank features by max |Spearman| across key targets
        max_abs = cont_sub.abs().max(axis=1)
        top_feats = max_abs.sort_values(ascending=False).head(TOP_N).index.tolist()
        hm_data = cont_sub.loc[top_feats]
        # Clean target names for display
        hm_data.columns = [c.replace("tgt_", "") for c in hm_data.columns]

        fig, ax = plt.subplots(figsize=(max(10, len(hm_data.columns) * 1.2),
                                         max(12, TOP_N * 0.35)))
        vmax = max(0.08, hm_data.abs().max().max())
        im = ax.imshow(hm_data.values, cmap="RdBu_r", aspect="auto",
                       vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(hm_data.columns)))
        ax.set_xticklabels(hm_data.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(hm_data.index)))
        ax.set_yticklabels(hm_data.index, fontsize=8)
        ax.set_title(f"Spearman Correlation — Top {TOP_N} Features\n{symbol} {tf}",
                     fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Spearman r", shrink=0.6)
        # Annotate cells
        for i in range(hm_data.shape[0]):
            for j in range(hm_data.shape[1]):
                val = hm_data.values[i, j]
                if np.isfinite(val) and abs(val) > 0.02:
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=6, color="white" if abs(val) > vmax * 0.6 else "black")
        plt.tight_layout()
        path = output_dir / f"{prefix}_heatmap_spearman.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

    # --- 2. AUC heatmap: top features × key binary targets ---
    if key_bin:
        bin_sub = auc_df[[t for t in key_bin if t in auc_df.columns]].copy()
        # Rank by max |AUC - 0.5|
        max_dev = (bin_sub - 0.5).abs().max(axis=1)
        top_feats = max_dev.sort_values(ascending=False).head(TOP_N).index.tolist()
        hm_data = bin_sub.loc[top_feats]
        hm_data.columns = [c.replace("tgt_", "") for c in hm_data.columns]

        fig, ax = plt.subplots(figsize=(max(10, len(hm_data.columns) * 1.2),
                                         max(12, TOP_N * 0.35)))
        # Diverging colormap centered at 0.5
        vdev = max(0.04, (hm_data - 0.5).abs().max().max())
        im = ax.imshow(hm_data.values, cmap="RdBu_r", aspect="auto",
                       vmin=0.5 - vdev, vmax=0.5 + vdev)
        ax.set_xticks(range(len(hm_data.columns)))
        ax.set_xticklabels(hm_data.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(hm_data.index)))
        ax.set_yticklabels(hm_data.index, fontsize=8)
        ax.set_title(f"AUC — Top {TOP_N} Features\n{symbol} {tf}",
                     fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, label="AUC", shrink=0.6)
        for i in range(hm_data.shape[0]):
            for j in range(hm_data.shape[1]):
                val = hm_data.values[i, j]
                if np.isfinite(val) and abs(val - 0.5) > 0.01:
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if abs(val - 0.5) > vdev * 0.6 else "black")
        plt.tight_layout()
        path = output_dir / f"{prefix}_heatmap_auc.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

    # --- 3. MI heatmap (key targets only) ---
    if not mi_df.empty:
        # Top features by max MI across targets
        max_mi = mi_df.max(axis=1)
        top_feats = max_mi.sort_values(ascending=False).head(TOP_N).index.tolist()
        hm_data = mi_df.loc[top_feats]
        hm_data.columns = [c.replace("tgt_", "") for c in hm_data.columns]

        fig, ax = plt.subplots(figsize=(max(10, len(hm_data.columns) * 1.2),
                                         max(12, TOP_N * 0.35)))
        im = ax.imshow(hm_data.values, cmap="YlOrRd", aspect="auto", vmin=0)
        ax.set_xticks(range(len(hm_data.columns)))
        ax.set_xticklabels(hm_data.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(hm_data.index)))
        ax.set_yticklabels(hm_data.index, fontsize=8)
        ax.set_title(f"Mutual Information — Top {TOP_N} Features\n{symbol} {tf}",
                     fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, label="MI (nats)", shrink=0.6)
        for i in range(hm_data.shape[0]):
            for j in range(hm_data.shape[1]):
                val = hm_data.values[i, j]
                if np.isfinite(val) and val > 0.01:
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=6, color="white" if val > 0.05 else "black")
        plt.tight_layout()
        path = output_dir / f"{prefix}_heatmap_mi.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

    # --- 4. Combined overview: bar chart of top features by max signal ---
    all_scores = {}
    for feat in spearman_df.index:
        scores = {}
        # Max |Spearman| across all continuous targets
        sp_vals = spearman_df.loc[feat].dropna()
        if len(sp_vals) > 0:
            scores["max_|spearman|"] = sp_vals.abs().max()
        # Max |AUC-0.5| across all binary targets
        if feat in auc_df.index:
            auc_vals = auc_df.loc[feat].dropna()
            if len(auc_vals) > 0:
                scores["max_|auc-0.5|"] = (auc_vals - 0.5).abs().max()
        # Max MI
        if not mi_df.empty and feat in mi_df.index:
            mi_vals = mi_df.loc[feat].dropna()
            if len(mi_vals) > 0:
                scores["max_mi"] = mi_vals.max()
        if scores:
            all_scores[feat] = scores

    overview = pd.DataFrame(all_scores).T
    # Composite score: normalize each metric to [0,1] then average
    for col in overview.columns:
        cmax = overview[col].max()
        if cmax > 0:
            overview[f"{col}_norm"] = overview[col] / cmax
    norm_cols = [c for c in overview.columns if c.endswith("_norm")]
    if norm_cols:
        overview["composite"] = overview[norm_cols].mean(axis=1)
        top50 = overview.sort_values("composite", ascending=False).head(50)

        fig, ax = plt.subplots(figsize=(12, max(10, len(top50) * 0.3)))
        y_pos = range(len(top50))
        colors_map = {"max_|spearman|_norm": "#2196F3",
                      "max_|auc-0.5|_norm": "#FF9800",
                      "max_mi_norm": "#4CAF50"}
        bottom = np.zeros(len(top50))
        for col in norm_cols:
            label = col.replace("_norm", "")
            vals = top50[col].values if col in top50.columns else np.zeros(len(top50))
            ax.barh(y_pos, vals, left=bottom, label=label,
                    color=colors_map.get(col, "gray"), alpha=0.85)
            bottom += vals

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top50.index, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Normalized Score (stacked)")
        ax.set_title(f"Top 50 Features — Composite Signal Strength\n{symbol} {tf}",
                     fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        path = output_dir / f"{prefix}_overview_top50.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

        # Save overview CSV too
        overview_path = output_dir / f"{prefix}_overview.csv"
        overview.sort_values("composite", ascending=False).to_csv(overview_path)
        print(f"  Saved: {overview_path}")


def main():
    parser = argparse.ArgumentParser(description="Tier 1: Univariate Feature-Target Scan")
    parser.add_argument("symbol", help="e.g. DOGEUSDT")
    parser.add_argument("timeframe", help="e.g. 4h")
    parser.add_argument("--features-dir", default="./features")
    parser.add_argument("--output-dir", default="./microstructure_research/results")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)

    df = load_features(features_dir, args.symbol, args.timeframe)
    print(f"Loaded {len(df)} candles for {args.symbol} {args.timeframe} "
          f"({df.index[0]} -> {df.index[-1]})")

    # Quick NaN check
    total_nan = df.isna().sum().sum()
    if total_nan > 0:
        print(f"WARNING: {total_nan} NaN values in dataset!")

    run_scan(df, args.symbol, args.timeframe, output_dir)


if __name__ == "__main__":
    main()
