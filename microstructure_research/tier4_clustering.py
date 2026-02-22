#!/usr/bin/env python3
"""
Tier 4: Feature Redundancy & Clustering

Takes the Tier 2 survivors and:
  1. Computes feature-feature Spearman correlation matrix
  2. Hierarchical clustering to group redundant features
  3. Picks best representative per cluster (highest Tier 1+2 combined score)
  4. Outputs a clean, non-redundant feature set

Usage:
  python tier4_clustering.py --timeframe 4h
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


def get_tier2_survivors(results_dir: Path, tf: str):
    """Load Tier 2 results for both coins and find cross-symbol survivors."""
    doge = pd.read_csv(results_dir / f"tier2_DOGEUSDT_{tf}_stability.csv")
    sol = pd.read_csv(results_dir / f"tier2_SOLUSDT_{tf}_stability.csv")

    doge_pass = doge[doge["tier2_pass"] == True]
    sol_pass = sol[sol["tier2_pass"] == True]

    # Cross-symbol: pass on both, same sign
    survivors = {}
    for tgt in doge_pass["target"].unique():
        d = doge_pass[doge_pass["target"] == tgt].set_index("feature")
        s = sol_pass[sol_pass["target"] == tgt].set_index("feature")
        common = sorted(set(d.index) & set(s.index))
        for f in common:
            if (d.loc[f, "full_r"] * s.loc[f, "full_r"]) > 0:
                if f not in survivors:
                    survivors[f] = {
                        "n_targets": 0,
                        "targets": [],
                        "avg_r": [],
                        "avg_sign": [],
                        "avg_snr": [],
                    }
                avg_r = (abs(d.loc[f, "full_r"]) + abs(s.loc[f, "full_r"])) / 2
                avg_sign = (d.loc[f, "sign_pct"] + s.loc[f, "sign_pct"]) / 2
                avg_snr = (d.loc[f, "snr"] + s.loc[f, "snr"]) / 2
                survivors[f]["n_targets"] += 1
                survivors[f]["targets"].append(tgt)
                survivors[f]["avg_r"].append(avg_r)
                survivors[f]["avg_sign"].append(avg_sign)
                survivors[f]["avg_snr"].append(avg_snr)

    # Compute composite score per feature
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


def run_clustering(features_dir, results_dir, tf, output_dir, corr_threshold=0.7):
    """Run hierarchical clustering on Tier 2 survivors."""

    # Get survivors
    surv_df = get_tier2_survivors(results_dir, tf)
    survivor_feats = surv_df["feature"].tolist()
    print(f"Tier 2 cross-symbol survivors: {len(survivor_feats)} features")

    # Load feature data from both coins
    doge_df = load_features(features_dir, "DOGEUSDT", tf)
    sol_df = load_features(features_dir, "SOLUSDT", tf)

    # Use only survivor features present in both
    common_feats = [f for f in survivor_feats if f in doge_df.columns and f in sol_df.columns]
    print(f"Features present in both datasets: {len(common_feats)}")

    # Compute correlation matrix using DOGE data (representative)
    print(f"\nComputing {len(common_feats)}×{len(common_feats)} correlation matrix...")
    t0 = time.time()
    feat_data = doge_df[common_feats].replace([np.inf, -np.inf], np.nan).fillna(0)
    corr_matrix = feat_data.corr(method="spearman")
    print(f"  Done in {time.time()-t0:.1f}s")

    # Verify with SOL data
    print(f"Verifying with SOL data...")
    sol_feat_data = sol_df[common_feats].replace([np.inf, -np.inf], np.nan).fillna(0)
    sol_corr = sol_feat_data.corr(method="spearman")
    # Average correlation agreement
    corr_agreement = np.corrcoef(corr_matrix.values.flatten(), sol_corr.values.flatten())[0, 1]
    print(f"  DOGE-SOL correlation matrix agreement: {corr_agreement:.4f}")

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
    prefix = f"tier4_{tf}"

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
    print(f"TIER 4 RESULTS: {tf} — {len(common_feats)} features → {n_clusters} clusters")
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
    ax.set_title(f"Feature Clustering — Tier 2 Survivors\n{tf} (threshold |r|>{corr_threshold})",
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
        ax.set_title(f"Correlation of Cluster Representatives\n{tf}",
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
    parser = argparse.ArgumentParser(description="Tier 4: Feature Redundancy Clustering")
    parser.add_argument("--timeframe", default="4h", help="e.g. 4h")
    parser.add_argument("--features-dir", default="./features")
    parser.add_argument("--results-dir", default="./microstructure_research/results")
    parser.add_argument("--output-dir", default="./microstructure_research/results")
    parser.add_argument("--corr-threshold", type=float, default=0.7,
                        help="Features with |r| > threshold are clustered together")
    args = parser.parse_args()

    run_clustering(
        Path(args.features_dir),
        Path(args.results_dir),
        args.timeframe,
        Path(args.output_dir),
        args.corr_threshold,
    )


if __name__ == "__main__":
    main()
