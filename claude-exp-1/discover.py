#!/usr/bin/env python3
"""
Signal discovery: scan all 116 symbols for cross-exchange patterns that
predict large price moves (>50 bps) with enough edge to beat fees.

Approach:
1. Load each symbol, compute features
2. For each feature, measure:
   - Correlation with forward returns at multiple horizons
   - Hit rate when feature is extreme (>2σ) → does price move in predicted direction?
   - Average forward return conditional on extreme feature values
3. Aggregate across all symbols to find UNIVERSAL signals (not symbol-specific)
4. Output a ranked signal table

Usage:
    python3 discover.py [--symbols N] [--jobs N]
"""

import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from load_data import load_symbol, list_common_symbols
from features import compute_features, get_feature_columns

warnings.filterwarnings("ignore")

# Feature columns to analyze (subset — exclude raw price data and targets)
FEATURE_COLS = None  # Will be populated from first symbol


def analyze_symbol(sym: str) -> dict | None:
    """Analyze a single symbol for cross-exchange signals."""
    try:
        df = load_symbol(sym)
        if df.empty or len(df) < 5000:
            return None

        feat = compute_features(df)
        feat = feat.dropna(subset=["fwd_ret_6"])

        if len(feat) < 5000:
            return None

        fcols = get_feature_columns(feat)
        # Filter to actual signal features (not targets, not raw)
        fcols = [c for c in fcols if not c.startswith(("fwd_", "bb_fwd", "bn_fwd"))]

        results = {"symbol": sym, "n_rows": len(feat), "signals": {}}

        for col in fcols:
            valid = feat[[col, "fwd_ret_6", "fwd_ret_12"]].dropna()
            if len(valid) < 3000:
                continue

            x = valid[col]
            # Correlation with forward returns
            corr_6 = x.corr(valid["fwd_ret_6"])
            corr_12 = x.corr(valid["fwd_ret_12"])

            # Extreme value analysis: top/bottom decile
            q10 = x.quantile(0.10)
            q90 = x.quantile(0.90)

            # When feature is very HIGH (top decile) → what happens next?
            high_mask = x >= q90
            low_mask = x <= q10

            high_fwd6 = valid.loc[high_mask, "fwd_ret_6"].mean()
            low_fwd6 = valid.loc[low_mask, "fwd_ret_6"].mean()
            high_fwd12 = valid.loc[high_mask, "fwd_ret_12"].mean()
            low_fwd12 = valid.loc[low_mask, "fwd_ret_12"].mean()

            # Edge: difference between high and low decile forward returns
            edge_6 = high_fwd6 - low_fwd6
            edge_12 = high_fwd12 - low_fwd12

            # Hit rate for large moves when extreme
            # When feature is HIGH → P(fwd_ret_6 > 0)
            high_hit_rate = (valid.loc[high_mask, "fwd_ret_6"] > 0).mean()
            low_hit_rate = (valid.loc[low_mask, "fwd_ret_6"] < 0).mean()

            # Conditional mean when extreme + direction confirmed
            # "If feature very high AND price goes up, how much?"
            high_win_mean = valid.loc[high_mask & (valid["fwd_ret_6"] > 0), "fwd_ret_6"].mean()
            low_win_mean = valid.loc[low_mask & (valid["fwd_ret_6"] < 0), "fwd_ret_6"].mean()

            # Extreme tails: top/bottom 5%
            q05 = x.quantile(0.05)
            q95 = x.quantile(0.95)
            extreme_high_fwd6 = valid.loc[x >= q95, "fwd_ret_6"].mean()
            extreme_low_fwd6 = valid.loc[x <= q05, "fwd_ret_6"].mean()
            extreme_edge_6 = extreme_high_fwd6 - extreme_low_fwd6

            results["signals"][col] = {
                "corr_6": corr_6,
                "corr_12": corr_12,
                "edge_6": edge_6,
                "edge_12": edge_12,
                "extreme_edge_6": extreme_edge_6,
                "high_fwd6": high_fwd6,
                "low_fwd6": low_fwd6,
                "high_hit_rate": high_hit_rate,
                "low_hit_rate": low_hit_rate,
                "high_win_mean": high_win_mean if not np.isnan(high_win_mean) else 0,
                "low_win_mean": low_win_mean if not np.isnan(low_win_mean) else 0,
            }

        return results
    except Exception as e:
        print(f"  ERROR {sym}: {e}", file=sys.stderr)
        return None


def aggregate_results(all_results: list[dict]) -> pd.DataFrame:
    """Aggregate signal analysis across all symbols."""
    # Collect all signal metrics across symbols
    signal_data = {}

    for res in all_results:
        sym = res["symbol"]
        for sig_name, metrics in res["signals"].items():
            if sig_name not in signal_data:
                signal_data[sig_name] = []
            signal_data[sig_name].append(metrics)

    # Aggregate: for each signal, compute cross-symbol average
    rows = []
    for sig_name, metrics_list in signal_data.items():
        n_symbols = len(metrics_list)
        if n_symbols < 10:  # Need at least 10 symbols for robustness
            continue

        avg = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if not np.isnan(m[key])]
            avg[f"mean_{key}"] = np.mean(values) if values else 0
            avg[f"std_{key}"] = np.std(values) if values else 0
            # What fraction of symbols show positive edge?
            if key in ["corr_6", "edge_6", "extreme_edge_6"]:
                avg[f"pct_positive_{key}"] = np.mean([v > 0 for v in values])

        rows.append({
            "signal": sig_name,
            "n_symbols": n_symbols,
            **avg
        })

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", "-n", type=int, default=None,
                        help="Number of symbols to analyze (default: all)")
    parser.add_argument("--jobs", "-j", type=int, default=8)
    args = parser.parse_args()

    symbols = list_common_symbols()
    if args.symbols:
        symbols = symbols[:args.symbols]

    print(f"Analyzing {len(symbols)} symbols with {args.jobs} workers...")
    t0 = time.time()

    all_results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(analyze_symbol, sym): sym for sym in symbols}
        done = 0
        for fut in as_completed(futures):
            sym = futures[fut]
            done += 1
            try:
                res = fut.result()
                if res:
                    all_results.append(res)
                    if done % 10 == 0 or done == len(symbols):
                        print(f"  [{done}/{len(symbols)}] {sym}: "
                              f"{res['n_rows']} rows, {len(res['signals'])} signals "
                              f"({time.time()-t0:.0f}s)")
                else:
                    print(f"  [{done}/{len(symbols)}] {sym}: skipped")
            except Exception as e:
                print(f"  [{done}/{len(symbols)}] {sym}: ERROR {e}")

    elapsed = time.time() - t0
    print(f"\nAnalyzed {len(all_results)} symbols in {elapsed:.0f}s")

    # Aggregate
    agg = aggregate_results(all_results)
    if agg.empty:
        print("No signals found!")
        return

    # Sort by mean absolute correlation with 30m forward return
    agg["abs_corr_6"] = agg["mean_corr_6"].abs()
    agg = agg.sort_values("abs_corr_6", ascending=False)

    print(f"\n{'='*100}")
    print(f"  TOP SIGNALS BY CORRELATION WITH 30min FORWARD RETURN")
    print(f"  (averaged across {len(all_results)} symbols)")
    print(f"{'='*100}")
    print(f"{'Signal':35s} {'Syms':>4} {'r(6)':>8} {'r(12)':>8} "
          f"{'Edge6':>8} {'ExEdge6':>8} {'%pos':>6} {'HiHR':>6} {'LoHR':>6}")
    print("-" * 100)

    for _, row in agg.head(30).iterrows():
        print(f"{row['signal']:35s} {row['n_symbols']:4.0f} "
              f"{row['mean_corr_6']:+8.4f} {row['mean_corr_12']:+8.4f} "
              f"{row['mean_edge_6']:8.2f} {row['mean_extreme_edge_6']:8.2f} "
              f"{row.get('pct_positive_corr_6', 0):6.1%} "
              f"{row['mean_high_hit_rate']:6.1%} {row['mean_low_hit_rate']:6.1%}")

    # Save full results
    outpath = "signal_discovery_results.csv"
    agg.to_csv(outpath, index=False)
    print(f"\nFull results saved to {outpath}")

    # Also print edge analysis: which signals have edge > 20bps (enough to beat fees)?
    print(f"\n{'='*100}")
    print(f"  SIGNALS WITH EXTREME EDGE > 20 BPS (potential fee-beaters)")
    print(f"{'='*100}")
    profitable = agg[agg["mean_extreme_edge_6"].abs() > 20].sort_values(
        "mean_extreme_edge_6", key=abs, ascending=False)

    if profitable.empty:
        print("  None found at 20bps threshold. Checking 10bps...")
        profitable = agg[agg["mean_extreme_edge_6"].abs() > 10].sort_values(
            "mean_extreme_edge_6", key=abs, ascending=False)

    for _, row in profitable.iterrows():
        direction = "LONG" if row["mean_corr_6"] > 0 else "SHORT"
        print(f"  {row['signal']:35s} → {direction:5s}  "
              f"edge={row['mean_extreme_edge_6']:+.1f}bps  "
              f"across {row['n_symbols']:.0f} symbols  "
              f"consistent={row.get('pct_positive_edge_6', 0):.0%}")


if __name__ == "__main__":
    main()
