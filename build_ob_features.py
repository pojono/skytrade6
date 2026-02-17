#!/usr/bin/env python3
"""
Compute 5-minute orderbook features from 1-second OB snapshots.

Reads from:  parquet/{SYMBOL}/orderbook/bybit_{market}/{YYYY-MM-DD}.parquet
Writes to:   parquet/{SYMBOL}/ob_features_5m/bybit_{market}/{YYYY-MM-DD}.parquet

These features are designed to be joined with the existing 5-minute OHLCV bars
from regime_detection.py via timestamp alignment.

Feature categories:
  1. Spread features — mean, std, max spread per 5-min bar
  2. Depth features — mean depth at various bps levels
  3. Imbalance features — mean, std, trend of bid/ask imbalance
  4. Depth dynamics — rate of change of depth (absorption/replenishment)
  5. Large order detection — presence of outsized resting orders
  6. Depth slope — how fast depth decays away from mid (book shape)
  7. Microprice — volume-weighted mid price (better fair value estimate)

Usage:
  python build_ob_features.py BTCUSDT
  python build_ob_features.py BTCUSDT --market futures
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

PARQUET_WRITE_OPTS = dict(compression="snappy", use_dictionary=True)
BAR_INTERVAL_US = 300_000_000  # 5 minutes in microseconds


def compute_5m_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-second OB snapshots into 5-minute feature bars.

    Args:
        df: DataFrame with 1-second OB snapshots (from build_orderbook_parquet.py)

    Returns:
        DataFrame with one row per 5-minute bar, indexed by bar timestamp.
    """
    if df.empty:
        return pd.DataFrame()

    # Assign each row to a 5-minute bar
    df = df.sort_values("timestamp_us").copy()
    df["bar_ts"] = (df["timestamp_us"] // BAR_INTERVAL_US) * BAR_INTERVAL_US

    rows = []
    for bar_ts, grp in df.groupby("bar_ts"):
        n = len(grp)
        if n < 10:  # need at least 10 seconds of data
            continue

        row = {"timestamp_us": int(bar_ts), "ob_sample_count": n}

        # --- 1. Spread features ---
        sp = grp["spread_bps"].values
        row["ob_spread_mean"] = float(np.mean(sp))
        row["ob_spread_std"] = float(np.std(sp))
        row["ob_spread_max"] = float(np.max(sp))
        row["ob_spread_min"] = float(np.min(sp))

        # --- 2. Depth features (mean depth at each bps level) ---
        for col in grp.columns:
            if col.startswith("bid_depth_") or col.startswith("ask_depth_"):
                bps_tag = col.split("_", 2)[2]  # e.g. "1bps", "0.5bps"
                side = col.split("_")[0]  # "bid" or "ask"
                vals = grp[col].values
                row[f"ob_{side}_depth_{bps_tag}_mean"] = float(np.mean(vals))

        # Total depth
        row["ob_bid_total_mean"] = float(grp["bid_total"].mean())
        row["ob_ask_total_mean"] = float(grp["ask_total"].mean())
        row["ob_total_depth_mean"] = row["ob_bid_total_mean"] + row["ob_ask_total_mean"]

        # --- 3. Imbalance features ---
        for col in grp.columns:
            if col.startswith("imbalance_"):
                bps_tag = col.split("_", 1)[1]  # e.g. "1bps", "0.5bps"
                vals = grp[col].values
                row[f"ob_imb_{bps_tag}_mean"] = float(np.mean(vals))
                row[f"ob_imb_{bps_tag}_std"] = float(np.std(vals))
                # Imbalance trend: slope of imbalance over the bar
                if n >= 30:
                    x = np.arange(n, dtype=np.float64)
                    x -= x.mean()
                    row[f"ob_imb_{bps_tag}_trend"] = float(np.dot(x, vals) / np.dot(x, x)) if np.dot(x, x) > 0 else 0.0

        # --- 4. Depth dynamics (rate of change) ---
        # Use 2bps depth as the primary level for dynamics
        for side in ["bid", "ask"]:
            depth_col = f"{side}_depth_2bps"
            if depth_col not in grp.columns:
                depth_col = f"{side}_depth_1bps"
            if depth_col in grp.columns:
                vals = grp[depth_col].values
                # Depth change: end vs start (absorption if decreasing)
                half = n // 2
                first_half = np.mean(vals[:half])
                second_half = np.mean(vals[half:])
                if first_half > 0:
                    row[f"ob_{side}_depth_change"] = float((second_half - first_half) / first_half)
                else:
                    row[f"ob_{side}_depth_change"] = 0.0
                # Depth volatility (how much depth fluctuates)
                row[f"ob_{side}_depth_cv"] = float(np.std(vals) / max(np.mean(vals), 1e-10))

        # --- 5. Large order detection ---
        # Check if any snapshot has outsized depth at top of book
        for side in ["bid", "ask"]:
            depth_col = f"{side}_depth_0.5bps"
            if depth_col not in grp.columns:
                depth_col = f"{side}_depth_1bps"
            if depth_col in grp.columns:
                vals = grp[depth_col].values
                mean_d = np.mean(vals)
                if mean_d > 0:
                    # Max depth / mean depth — spikes indicate large resting orders
                    row[f"ob_{side}_wall_ratio"] = float(np.max(vals) / mean_d)
                    # Fraction of time depth > 3x mean
                    row[f"ob_{side}_wall_frac"] = float(np.mean(vals > 3 * mean_d))
                else:
                    row[f"ob_{side}_wall_ratio"] = 1.0
                    row[f"ob_{side}_wall_frac"] = 0.0

        # --- 6. Depth slope (book shape) ---
        # Ratio of near depth to far depth — steep = thin near, thick far
        bid_near = grp.get("bid_depth_1bps", grp.get("bid_depth_0.5bps"))
        bid_far = grp.get("bid_depth_5bps", grp.get("bid_depth_3bps"))
        if bid_near is not None and bid_far is not None:
            bn = bid_near.mean()
            bf = bid_far.mean()
            row["ob_bid_slope"] = float(bn / bf) if bf > 0 else 0.0

        ask_near = grp.get("ask_depth_1bps", grp.get("ask_depth_0.5bps"))
        ask_far = grp.get("ask_depth_5bps", grp.get("ask_depth_3bps"))
        if ask_near is not None and ask_far is not None:
            an = ask_near.mean()
            af = ask_far.mean()
            row["ob_ask_slope"] = float(an / af) if af > 0 else 0.0

        # --- 7. Microprice ---
        # Volume-weighted mid: better fair value estimate than simple mid
        bid_vwap = grp.get("bid_vwap_5bps")
        ask_vwap = grp.get("ask_vwap_5bps")
        mid = grp["mid_price"].values
        if bid_vwap is not None and ask_vwap is not None:
            bv = bid_vwap.values
            av = ask_vwap.values
            microprice = (bv + av) / 2.0
            # Microprice deviation from mid (in bps)
            row["ob_microprice_dev_bps"] = float(np.mean((microprice - mid) / mid * 10000))
        
        # Mid price movement within bar
        row["ob_mid_return_bps"] = float((mid[-1] - mid[0]) / mid[0] * 10000) if mid[0] > 0 else 0.0
        row["ob_mid_volatility"] = float(np.std(np.diff(mid) / mid[:-1]) * 10000) if len(mid) > 1 else 0.0

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result["timestamp_us"] = result["timestamp_us"].astype(np.int64)
    return result.sort_values("timestamp_us").reset_index(drop=True)


def process_market(symbol: str, market: str, parquet_dir: Path):
    """Process all OB parquet files for one market into 5-min features."""
    src_dir = parquet_dir / symbol / "orderbook" / f"bybit_{market}"
    if not src_dir.exists():
        print(f"  {market}: source not found: {src_dir}")
        return 0, 0

    files = sorted(src_dir.glob("*.parquet"))
    if not files:
        print(f"  {market}: no parquet files found")
        return 0, 0

    dest_dir = parquet_dir / symbol / "ob_features_5m" / f"bybit_{market}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    n = len(files)
    t0 = time.time()

    print(f"  {market}: {n} files → {dest_dir}")

    for i, f in enumerate(files, 1):
        date_str = f.stem  # YYYY-MM-DD
        out_path = dest_dir / f"{date_str}.parquet"

        if out_path.exists():
            skipped += 1
            continue

        t1 = time.time()
        df = pd.read_parquet(f)
        features = compute_5m_features(df)

        if features.empty:
            print(f"    [{i}/{n}] - {date_str}  (no features)")
            continue

        features.to_parquet(out_path, index=False, **PARQUET_WRITE_OPTS)
        written += 1

        elapsed = time.time() - t1
        total_elapsed = time.time() - t0
        eta = (total_elapsed / i) * (n - i) if i > 0 else 0
        print(f"    [{i}/{n}] ✓ {date_str}  bars={len(features)}  cols={len(features.columns)}  {elapsed:.1f}s  ETA={eta:.0f}s")

        del df, features

    elapsed_total = time.time() - t0
    print(f"  {market}: {written} written, {skipped} skipped, {elapsed_total:.0f}s total")
    return written, skipped


def run(args):
    symbol = args.symbol.upper()
    parquet_dir = Path(args.parquet_dir)
    markets = ["futures", "spot"] if args.market == "both" else [args.market]

    print(f"Symbol:    {symbol}")
    print(f"Parquet:   {parquet_dir.resolve()}")
    print(f"Markets:   {', '.join(markets)}")
    print("=" * 60)

    for market in markets:
        process_market(symbol, market, parquet_dir)

    # Summary
    feat_dir = parquet_dir / symbol / "ob_features_5m"
    if feat_dir.exists():
        pq_files = list(feat_dir.rglob("*.parquet"))
        total_size = sum(f.stat().st_size for f in pq_files)
        print(f"\nTotal: {len(pq_files)} feature files, {total_size / (1024*1024):.1f} MB")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Compute 5-minute orderbook features from 1-second snapshots.",
    )
    parser.add_argument("symbol", help="Trading pair symbol, e.g. BTCUSDT")
    parser.add_argument("--market", "-m", default="both",
                        choices=["futures", "spot", "both"],
                        help="Which market to process (default: both)")
    parser.add_argument("--parquet-dir", "-p", default="./parquet",
                        help="Parquet directory (default: ./parquet)")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
