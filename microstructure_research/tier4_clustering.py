#!/usr/bin/env python3
"""
Tier 4: Feature Redundancy & Clustering

Per-coin clustering: takes one coin's Tier 2 survivors and:
  1. Computes feature-feature Spearman correlation matrix
  2. Hierarchical clustering to group redundant features
  3. Picks best representative per cluster (highest composite score)
  4. Outputs a clean, non-redundant feature set

Usage:
  python tier4_clustering.py SOLUSDT 4h
  python tier4_clustering.py DOGEUSDT 4h
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
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform


def load_features(features_dir: Path, symbol: str, tf: str) -> pd.DataFrame:
    tf_dir = features_dir / symbol / tf
    files = sorted(tf_dir.glob("*.parquet"))
    if not files:
        print(f"ERROR: No parquet files in {tf_dir}")
        sys.exit(1)
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def get_tier2_survivors(results_dir: Path, symbol: str, tf: str):
    """Load Tier 2 results for a single coin and find survivors."""
    path = results_dir / f"tier2_{symbol}_{tf}_stability.csv"
    if not path.exists():
        print(f"ERROR: {path} not found. Run tier2_stability_scan.py first.")
        sys.exit(1)
    df = pd.read_csv(path)
    passed = df[df["tier2_pass"] == True]

    survivors = {}
    for tgt in passed["target"].unique():
        tgt_df = passed[passed["target"] == tgt]
        for _, row in tgt_df.iterrows():
            f = row["feature"]
            if f not in survivors:
                survivors[f] = {
                    "n_targets": 0,
                    "targets": [],
                    "avg_r": [],
                    "avg_sign": [],
                    "avg_snr": [],
                }
            survivors[f]["n_targets"] += 1
            survivors[f]["targets"].append(tgt)
            survivors[f]["avg_r"].append(abs(row["full_r"]))
            survivors[f]["avg_sign"].append(row["sign_pct"])
            survivors[f]["avg_snr"].append(row["snr"])

    surv_list = []
    for f, info in survivors.items():
        surv_list.append({
            "feature": f,
            "n_targets": info["n_targets"],
            "mean_avg_r": np.mean(info["avg_r"]),
            "mean_avg_sign": np.mean(info["avg_sign"]),
            "mean_avg_snr": np.mean(info["avg_snr"]),
            "targets": ", ".join(info["targets"]),
            # Composite: strength × stability × breadth
            "composite": np.mean(info["avg_r"]) * np.mean(info["avg_sign"]) * np.log1p(info["n_targets"]),
        })

    return pd.DataFrame(surv_list).sort_values("composite", ascending=False)


def run_clustering(features_dir, results_dir, symbol, tf, output_dir, corr_threshold=0.7):
    """Run hierarchical clustering on Tier 2 survivors for a single coin."""

    # Get survivors
    surv_df = get_tier2_survivors(results_dir, symbol, tf)
    survivor_feats = surv_df["feature"].tolist()
    print(f"Tier 2 survivors for {symbol}: {len(survivor_feats)} features")

    # Load feature data
    coin_df = load_features(features_dir, symbol, tf)

    # Use only survivor features present in data
    common_feats = [f for f in survivor_feats if f in coin_df.columns]
    print(f"Features present in dataset: {len(common_feats)}")

    # Compute correlation matrix
    print(f"\nComputing {len(common_feats)}x{len(common_feats)} correlation matrix...")
    t0 = time.time()
    feat_data = coin_df[common_feats].replace([np.inf, -np.inf], np.nan).fillna(0)
    corr_matrix = feat_data.corr(method="spearman")
    print(f"  Done in {time.time()-t0:.1f}s")

    # Hierarchical clustering
    print(f"\nClustering with threshold |r| > {corr_threshold}...")
    dist_matrix = 1 - corr_matrix.abs()
    np.fill_diagonal(dist_matrix.values, 0)
    # Make symmetric and handle numerical issues
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    dist_matrix = dist_matrix.clip(lower=0)

    condensed = squareform(dist_matrix.values)
    Z = linkage(condensed, method="average")
    clusters = fcluster(Z, t=1 - corr_threshold, criterion="distance")

    # Assign clusters
    cluster_df = pd.DataFrame({
        "feature": common_feats,
        "cluster": clusters,
    }).merge(surv_df[["feature", "n_targets", "mean_avg_r", "mean_avg_sign",
                       "mean_avg_snr", "composite"]], on="feature", how="left")

    n_clusters = len(set(clusters))
    print(f"  {len(common_feats)} features → {n_clusters} clusters")

    # Pick best representative per cluster
    representatives = []
    cluster_details = []

    for cid in sorted(set(clusters)):
        members = cluster_df[cluster_df["cluster"] == cid].sort_values("composite", ascending=False)
        best = members.iloc[0]
        representatives.append(best)

        member_list = members["feature"].tolist()
        cluster_details.append({
            "cluster": cid,
            "size": len(members),
            "representative": best["feature"],
            "composite": best["composite"],
            "n_targets": best["n_targets"],
            "members": ", ".join(member_list),
        })

    rep_df = pd.DataFrame(representatives).sort_values("composite", ascending=False)
    detail_df = pd.DataFrame(cluster_details).sort_values("composite", ascending=False)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"tier4_{symbol}_{tf}"

    cluster_path = output_dir / f"{prefix}_clusters.csv"
    cluster_df.to_csv(cluster_path, index=False)
    print(f"\nSaved: {cluster_path}")

    rep_path = output_dir / f"{prefix}_representatives.csv"
    rep_df.to_csv(rep_path, index=False)
    print(f"Saved: {rep_path}")

    detail_path = output_dir / f"{prefix}_cluster_details.csv"
    detail_df.to_csv(detail_path, index=False)
    print(f"Saved: {detail_path}")

    # --- Console Report ---
    print(f"\n{'='*90}")
    print(f"TIER 4 RESULTS: {symbol} {tf} — {len(common_feats)} features -> {n_clusters} clusters")
    print(f"{'='*90}")

    print(f"\n--- Cluster Representatives (sorted by composite score) ---")
    print(f"{'#':<4} {'Representative':<40} {'Size':>5} {'#tgt':>5} {'avg|r|':>8} "
          f"{'sign%':>7} {'SNR':>6} {'comp':>8}")
    print("-" * 90)
    for i, (_, row) in enumerate(rep_df.iterrows(), 1):
        print(f"{i:<4} {row['feature']:<40} {detail_df[detail_df['representative']==row['feature']]['size'].values[0]:>5} "
              f"{row['n_targets']:>5.0f} {row['mean_avg_r']:>8.4f} "
              f"{row['mean_avg_sign']:>6.0%} {row['mean_avg_snr']:>6.2f} "
              f"{row['composite']:>8.4f}")

    # Show large clusters
    print(f"\n--- Large Clusters (size > 2) ---")
    large = detail_df[detail_df["size"] > 2].sort_values("size", ascending=False)
    for _, row in large.iterrows():
        print(f"\n  Cluster {row['cluster']} (size={row['size']}, rep={row['representative']}):")
        members = row["members"].split(", ")
        for m in members:
            marker = " ★" if m == row["representative"] else ""
            print(f"    {m}{marker}")

    # --- Dendrogram ---
    print(f"\nGenerating dendrogram...")
    fig, ax = plt.subplots(figsize=(16, max(10, len(common_feats) * 0.2)))
    dendrogram(
        Z,
        labels=common_feats,
        orientation="left",
        leaf_font_size=6,
        color_threshold=1 - corr_threshold,
        ax=ax,
    )
    ax.axvline(x=1 - corr_threshold, color="red", linestyle="--", linewidth=1,
               label=f"threshold |r|={corr_threshold}")
    ax.set_xlabel("Distance (1 - |Spearman|)")
    ax.set_title(f"Feature Clustering — {symbol} Tier 2 Survivors\n{tf} (threshold |r|>{corr_threshold})",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    dend_path = output_dir / f"{prefix}_dendrogram.png"
    fig.savefig(dend_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {dend_path}")

    # --- Correlation heatmap of representatives ---
    rep_feats = rep_df["feature"].tolist()
    if len(rep_feats) > 1:
        rep_corr = corr_matrix.loc[rep_feats, rep_feats]
        fig, ax = plt.subplots(figsize=(max(10, len(rep_feats) * 0.4),
                                         max(8, len(rep_feats) * 0.35)))
        im = ax.imshow(rep_corr.values, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(range(len(rep_feats)))
        ax.set_xticklabels(rep_feats, rotation=90, fontsize=7)
        ax.set_yticks(range(len(rep_feats)))
        ax.set_yticklabels(rep_feats, fontsize=7)
        ax.set_title(f"Correlation of Cluster Representatives\n{symbol} {tf}",
                     fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Spearman r", shrink=0.6)
        # Annotate high correlations
        for i in range(len(rep_feats)):
            for j in range(len(rep_feats)):
                val = rep_corr.values[i, j]
                if i != j and abs(val) > 0.3:
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=5, color="white" if abs(val) > 0.6 else "black")
        plt.tight_layout()
        hm_path = output_dir / f"{prefix}_rep_correlation.png"
        fig.savefig(hm_path, dpi=120)
        plt.close(fig)
        print(f"  Saved: {hm_path}")

    # --- Final summary ---
    print(f"\n{'='*90}")
    print(f"FINAL FEATURE SET: {len(rep_feats)} non-redundant features")
    print(f"{'='*90}")
    for i, (_, row) in enumerate(rep_df.iterrows(), 1):
        print(f"  {i:>2}. {row['feature']:<40} (r={row['mean_avg_r']:.4f}, "
              f"sign={row['mean_avg_sign']:.0%}, {row['n_targets']:.0f} targets)")

    print(f"\nDone! Results in {output_dir}/")
    return rep_df


def main():
    parser = argparse.ArgumentParser(description="Tier 4: Per-Coin Feature Redundancy Clustering")
    parser.add_argument("symbol", help="e.g. SOLUSDT")
    parser.add_argument("timeframe", help="e.g. 4h")
    parser.add_argument("--features-dir", default="./features")
    parser.add_argument("--results-dir", default="./microstructure_research/results")
    parser.add_argument("--output-dir", default="./microstructure_research/results")
    parser.add_argument("--corr-threshold", type=float, default=0.7,
                        help="Features with |r| > threshold are clustered together")
    args = parser.parse_args()

    run_clustering(
        Path(args.features_dir),
        Path(args.results_dir),
        args.symbol,
        args.timeframe,
        Path(args.output_dir),
        args.corr_threshold,
    )


if __name__ == "__main__":
    main()
