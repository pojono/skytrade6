#!/usr/bin/env python3
"""
Tier 2: Temporal Stability Scan

For each (feature, target) pair, compute rolling Spearman/AUC in 90-day windows
and measure:
  1. Sign consistency: % of windows with same sign as full-period
  2. Signal-to-noise: |mean_r| / std_r
  3. Max wrong-sign streak: longest consecutive wrong-sign windows
  4. Regime split: signal in high-vol vs low-vol periods

Usage:
  python tier2_stability_scan.py DOGEUSDT 4h
  python tier2_stability_scan.py SOLUSDT 4h
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
from sklearn.metrics import roc_auc_score


# Key targets
KEY_TARGETS_CONTINUOUS = [
    "tgt_ret_1", "tgt_ret_3", "tgt_ret_5",
    "tgt_cum_ret_5", "tgt_cum_ret_10",
    "tgt_sharpe_5", "tgt_sharpe_10",
]

KEY_TARGETS_BINARY = [
    "tgt_profitable_long_3", "tgt_profitable_long_5",
    "tgt_profitable_short_3", "tgt_profitable_short_5",
]

# Window parameters
WINDOW_DAYS = 90
STEP_DAYS = 30
MIN_CANDLES_PER_WINDOW = 200


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


def classify_columns(df):
    tgt_cols = [c for c in df.columns if c.startswith("tgt_")]
    feat_cols = [c for c in df.columns if not c.startswith("tgt_")]
    binary_tgts = []
    continuous_tgts = []
    for c in tgt_cols:
        if len(df[c].dropna().unique()) <= 3:
            binary_tgts.append(c)
        else:
            continuous_tgts.append(c)
    return feat_cols, continuous_tgts, binary_tgts


def compute_spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30:
        return np.nan
    a, b = x[mask], y[mask]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    r, _ = stats.spearmanr(a, b)
    return r


def compute_auc_dev(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30:
        return np.nan
    yy = y[mask]
    if len(np.unique(yy)) < 2:
        return np.nan
    try:
        return roc_auc_score(yy, x[mask]) - 0.5
    except ValueError:
        return np.nan


def make_windows(df, tf):
    """Create rolling windows based on calendar days."""
    tf_hours = {"15m": 0.25, "30m": 0.5, "1h": 1, "2h": 2, "4h": 4}
    candles_per_day = 24 / tf_hours.get(tf, 1)

    window_candles = int(WINDOW_DAYS * candles_per_day)
    step_candles = int(STEP_DAYS * candles_per_day)

    windows = []
    start = 0
    while start + window_candles <= len(df):
        end = start + window_candles
        windows.append((start, end))
        start += step_candles

    return windows


def max_streak(signs, target_sign):
    """Max consecutive count of values != target_sign."""
    max_s = 0
    cur = 0
    for s in signs:
        if np.isnan(s):
            continue
        if (s > 0) != (target_sign > 0):
            cur += 1
            max_s = max(max_s, cur)
        else:
            cur = 0
    return max_s


def run_stability(df, symbol, tf, output_dir):
    feat_cols, continuous_tgts, binary_tgts = classify_columns(df)
    windows = make_windows(df, tf)

    n_feats = len(feat_cols)
    n_windows = len(windows)

    key_cont = [t for t in KEY_TARGETS_CONTINUOUS if t in continuous_tgts]
    key_bin = [t for t in KEY_TARGETS_BINARY if t in binary_tgts]
    all_key = key_cont + key_bin

    print(f"\n{'='*70}")
    print(f"Tier 2 Stability: {symbol} {tf}")
    print(f"  Candles:    {len(df):,}")
    print(f"  Features:   {n_feats}")
    print(f"  Windows:    {n_windows} ({WINDOW_DAYS}d window, {STEP_DAYS}d step)")
    print(f"  Targets:    {len(key_cont)} continuous + {len(key_bin)} binary")
    print(f"{'='*70}")

    # --- Compute vol regime ---
    if "realized_vol" in df.columns:
        vol_col = "realized_vol"
    elif "parkinson_vol" in df.columns:
        vol_col = "parkinson_vol"
    else:
        vol_col = None

    if vol_col:
        vol_median = df[vol_col].median()
        high_vol_mask = df[vol_col] >= vol_median
        low_vol_mask = df[vol_col] < vol_median
        print(f"  Vol regime: {vol_col}, median={vol_median:.6f}")
        print(f"              high-vol: {high_vol_mask.sum()} candles, low-vol: {low_vol_mask.sum()}")
    else:
        high_vol_mask = None
        low_vol_mask = None
        print(f"  Vol regime: not available")

    # --- Rolling computation ---
    results = []
    t0 = time.time()

    for ti, tgt in enumerate(all_key):
        is_binary = tgt in key_bin
        tgt_vals = df[tgt].values.astype(float)
        metric_fn = compute_auc_dev if is_binary else compute_spearman

        # Full-period score for each feature
        full_scores = {}
        for feat in feat_cols:
            full_scores[feat] = metric_fn(df[feat].values.astype(float), tgt_vals)

        # Rolling scores
        for fi, feat in enumerate(feat_cols):
            feat_vals = df[feat].values.astype(float)
            full_r = full_scores[feat]
            if np.isnan(full_r) or abs(full_r) < 0.005:
                continue  # skip features with no signal

            window_scores = []
            for (ws, we) in windows:
                x = feat_vals[ws:we]
                y = tgt_vals[ws:we]
                s = metric_fn(x, y)
                window_scores.append(s)

            window_scores = np.array(window_scores)
            valid = window_scores[~np.isnan(window_scores)]

            if len(valid) < 5:
                continue

            mean_r = np.mean(valid)
            std_r = np.std(valid)
            full_sign = 1 if full_r > 0 else -1

            # Sign consistency
            same_sign = np.sum((valid > 0) == (full_sign > 0))
            sign_pct = same_sign / len(valid)

            # Signal-to-noise
            snr = abs(mean_r) / std_r if std_r > 0 else 0.0

            # Max wrong-sign streak
            wrong_streak = max_streak(valid, full_sign)

            # Regime split
            regime_high_r = np.nan
            regime_low_r = np.nan
            if high_vol_mask is not None:
                hv_x = feat_vals[high_vol_mask.values]
                hv_y = tgt_vals[high_vol_mask.values]
                regime_high_r = metric_fn(hv_x, hv_y)
                lv_x = feat_vals[low_vol_mask.values]
                lv_y = tgt_vals[low_vol_mask.values]
                regime_low_r = metric_fn(lv_x, lv_y)

            # Regime consistency: same sign in both regimes?
            regime_consistent = False
            if not np.isnan(regime_high_r) and not np.isnan(regime_low_r):
                regime_consistent = (regime_high_r * regime_low_r) > 0

            results.append({
                "feature": feat,
                "target": tgt,
                "is_binary": is_binary,
                "full_r": full_r,
                "mean_r": mean_r,
                "std_r": std_r,
                "sign_pct": sign_pct,
                "snr": snr,
                "max_wrong_streak": wrong_streak,
                "regime_high_r": regime_high_r,
                "regime_low_r": regime_low_r,
                "regime_consistent": regime_consistent,
                "n_windows": len(valid),
                "window_scores": window_scores.tolist(),
            })

        elapsed = time.time() - t0
        eta = elapsed / (ti + 1) * (len(all_key) - ti - 1) if ti > 0 else 0
        n_pass = sum(1 for r in results if r["target"] == tgt and r["sign_pct"] >= 0.7)
        print(f"  [{ti+1}/{len(all_key)}] {tgt}: {n_pass} features pass sign>70% "
              f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s ETA]")

    res_df = pd.DataFrame([{k: v for k, v in r.items() if k != "window_scores"} for r in results])

    # --- Tier 2 pass/fail ---
    if len(res_df) > 0:
        res_df["tier2_pass"] = (
            (res_df["sign_pct"] >= 0.70) &
            (res_df["snr"] >= 0.5) &
            (res_df["max_wrong_streak"] <= 3) &
            (res_df["regime_consistent"])
        )
    else:
        res_df["tier2_pass"] = False

    # --- Save CSV ---
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"tier2_{symbol}_{tf}"
    csv_path = output_dir / f"{prefix}_stability.csv"
    res_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # --- Console Report ---
    print(f"\n{'='*70}")
    print(f"TIER 2 RESULTS: {symbol} {tf}")
    print(f"{'='*70}")

    total_pairs = len(res_df)
    n_pass = res_df["tier2_pass"].sum() if len(res_df) > 0 else 0
    print(f"  Total feature-target pairs evaluated: {total_pairs}")
    print(f"  Passed Tier 2 (sign≥70%, SNR≥0.5, streak≤4, regime OK): {n_pass}")

    for tgt in all_key:
        tgt_df = res_df[res_df["target"] == tgt]
        if len(tgt_df) == 0:
            continue
        passed = tgt_df[tgt_df["tier2_pass"]]
        metric_label = "AUC_dev" if tgt in key_bin else "Spearman"

        print(f"\n--- {tgt} ({len(passed)}/{len(tgt_df)} passed) ---")
        if len(passed) == 0:
            print(f"  No features passed Tier 2 for this target")
            continue

        # Sort by |full_r| descending
        passed_sorted = passed.sort_values("full_r", key=abs, ascending=False)
        print(f"  {'Feature':<40} {'full_r':>8} {'sign%':>6} {'SNR':>6} "
              f"{'streak':>6} {'hi_vol':>8} {'lo_vol':>8}")
        for _, row in passed_sorted.head(20).iterrows():
            print(f"  {row['feature']:<40} {row['full_r']:>+8.4f} {row['sign_pct']:>5.0%} "
                  f"{row['snr']:>6.2f} {row['max_wrong_streak']:>6.0f} "
                  f"{row['regime_high_r']:>+8.4f} {row['regime_low_r']:>+8.4f}")

    # --- Rolling plots for top features ---
    print(f"\nGenerating rolling plots...")
    generate_rolling_plots(results, all_key, symbol, tf, output_dir, prefix, windows, df)

    # --- Summary stats ---
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    for tgt in all_key:
        tgt_df = res_df[res_df["target"] == tgt]
        n_total = len(tgt_df)
        n_pass = tgt_df["tier2_pass"].sum() if n_total > 0 else 0
        n_sign70 = (tgt_df["sign_pct"] >= 0.70).sum() if n_total > 0 else 0
        n_snr05 = (tgt_df["snr"] >= 0.5).sum() if n_total > 0 else 0
        n_regime = tgt_df["regime_consistent"].sum() if n_total > 0 else 0
        print(f"  {tgt:<25} total={n_total:>4} sign≥70%={n_sign70:>4} "
              f"SNR≥0.5={n_snr05:>4} regime_ok={n_regime:>4} PASS={n_pass:>4}")

    print(f"\nDone! Results in {output_dir}/")
    return res_df


def generate_rolling_plots(results, all_key, symbol, tf, output_dir, prefix, windows, df):
    """Generate rolling correlation plots for top features per target."""
    TOP_N = 15

    # Get window center dates
    window_centers = []
    for (ws, we) in windows:
        mid = (ws + we) // 2
        if mid < len(df):
            window_centers.append(df.index[mid])
        else:
            window_centers.append(df.index[-1])

    for tgt in all_key:
        tgt_results = [r for r in results if r["target"] == tgt]
        if not tgt_results:
            continue

        # Sort by sign_pct * |full_r| (stability × strength)
        tgt_results.sort(key=lambda r: r["sign_pct"] * abs(r["full_r"]), reverse=True)
        top = tgt_results[:TOP_N]

        if not top:
            continue

        n_plots = len(top)
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, max(8, n_plots * 1.8)),
                                  sharex=True)
        if n_plots == 1:
            axes = [axes]

        for i, r in enumerate(top):
            ax = axes[i]
            scores = np.array(r["window_scores"])
            centers = window_centers[:len(scores)]

            colors = ["#2196F3" if s > 0 else "#F44336" if s < 0 else "#999"
                      for s in scores if not np.isnan(s)]
            valid_centers = [c for c, s in zip(centers, scores) if not np.isnan(s)]
            valid_scores = [s for s in scores if not np.isnan(s)]

            ax.bar(range(len(valid_scores)), valid_scores, color=colors, alpha=0.7, width=0.8)
            ax.axhline(y=0, color="black", linewidth=0.5)
            ax.axhline(y=r["full_r"], color="green", linewidth=1, linestyle="--", alpha=0.5)

            feat_short = r["feature"][:35]
            ax.set_ylabel(f"{feat_short}\nr={r['full_r']:+.3f}", fontsize=7, rotation=0,
                         labelpad=80, ha="right", va="center")
            ax.tick_params(axis="y", labelsize=7)

            # Add sign% and SNR annotation
            ax.text(0.98, 0.85, f"sign={r['sign_pct']:.0%} SNR={r['snr']:.1f}",
                    transform=ax.transAxes, fontsize=7, ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # X-axis labels on bottom plot
        if valid_centers:
            axes[-1].set_xticks(range(len(valid_centers)))
            axes[-1].set_xticklabels([c.strftime("%Y-%m") if hasattr(c, "strftime")
                                       else str(c) for c in valid_centers],
                                      rotation=45, fontsize=7)

        tgt_short = tgt.replace("tgt_", "")
        fig.suptitle(f"Rolling Signal Stability — {tgt_short}\n{symbol} {tf} "
                     f"({WINDOW_DAYS}d windows, {STEP_DAYS}d step)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = output_dir / f"{prefix}_rolling_{tgt_short}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Tier 2: Temporal Stability Scan")
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

    run_stability(df, args.symbol, args.timeframe, output_dir)


if __name__ == "__main__":
    main()
