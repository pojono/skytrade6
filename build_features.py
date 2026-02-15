#!/usr/bin/env python3
"""
Compute microstructure features from raw tick trades and save as daily parquet files.

Reads from:  parquet/{SYMBOL}/trades/{source}/{YYYY-MM-DD}.parquet
Writes to:   parquet/{SYMBOL}/features/{interval}/{source}/{YYYY-MM-DD}.parquet

Processes one day at a time — memory usage is bounded to ~1 day of trades.
Existing output files are skipped (incremental builds).

Usage:
  python build_features.py BTCUSDT
  python build_features.py BTCUSDT --interval 5m --sources binance_futures bybit_futures
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_microstructure_features(trades: pd.DataFrame, interval_us: int) -> pd.DataFrame:
    """Compute microstructure features from raw tick trades per interval bucket."""
    bucket = (trades["timestamp_us"].values // interval_us) * interval_us
    trades = trades.copy()
    trades["bucket"] = bucket

    features = []
    for bkt, grp in trades.groupby("bucket"):
        p = grp["price"].values
        q = grp["quantity"].values
        qq = grp["quote_quantity"].values
        s = grp["side"].values
        t = grp["timestamp_us"].values
        n = len(grp)
        if n < 2:
            continue

        buy_mask = s == 1
        sell_mask = s == -1
        buy_vol = q[buy_mask].sum()
        sell_vol = q[sell_mask].sum()
        total_vol = q.sum()
        buy_quote = qq[buy_mask].sum()
        sell_quote = qq[sell_mask].sum()

        # --- Aggression ---
        vol_imbalance = (buy_vol - sell_vol) / max(total_vol, 1e-10)
        dollar_imbalance = (buy_quote - sell_quote) / max(buy_quote + sell_quote, 1e-10)

        q90 = np.percentile(q, 90)
        large_mask = q >= q90
        large_buy = q[large_mask & buy_mask].sum()
        large_sell = q[large_mask & sell_mask].sum()
        large_imbalance = (large_buy - large_sell) / max(large_buy + large_sell, 1e-10)
        large_vol_pct = q[large_mask].sum() / max(total_vol, 1e-10)

        # --- Flow ---
        buy_count = int(buy_mask.sum())
        sell_count = int(sell_mask.sum())
        count_imbalance = (buy_count - sell_count) / max(n, 1)

        duration_s = max((t[-1] - t[0]) / 1e6, 0.001)
        arrival_rate = n / duration_s

        if n > 2:
            iti = np.diff(t).astype(np.float64)
            iti_cv = iti.std() / max(iti.mean(), 1)
            sub_buckets = np.linspace(t[0], t[-1], 6)
            sub_counts = np.histogram(t, bins=sub_buckets)[0]
            burstiness = float(sub_counts.max()) / max(n, 1)
        else:
            iti_cv = 0.0
            burstiness = 1.0

        mid_t = (t[0] + t[-1]) / 2
        first_half = int((t < mid_t).sum())
        trade_acceleration = (n - first_half - first_half) / max(n, 1)

        # --- Price impact ---
        vwap = qq.sum() / max(total_vol, 1e-10)
        price_range = (p.max() - p.min()) / max(vwap, 1e-10)
        close_vs_vwap = (p[-1] - vwap) / max(vwap, 1e-10)

        if n > 10:
            signed_vol = q * s
            price_changes = np.diff(p)
            if len(price_changes) > 1 and signed_vol[1:].std() > 0:
                kyle_lambda = float(np.corrcoef(signed_vol[1:], price_changes)[0, 1])
            else:
                kyle_lambda = 0.0
        else:
            kyle_lambda = 0.0

        ret = (p[-1] - p[0]) / max(p[0], 1e-10)
        amihud = abs(ret) / max(total_vol, 1e-10)

        # --- Volume profile ---
        price_mid = (p.max() + p.min()) / 2
        vol_above = q[p >= price_mid].sum()
        vol_below = q[p < price_mid].sum()
        vol_profile_skew = (vol_above - vol_below) / max(total_vol, 1e-10)

        # --- Candle shape ---
        open_p, close_p, high_p, low_p = p[0], p[-1], p.max(), p.min()
        full_range = high_p - low_p
        if full_range > 0:
            upper_wick = (high_p - max(open_p, close_p)) / full_range
            lower_wick = (min(open_p, close_p) - low_p) / full_range
        else:
            upper_wick = 0.0
            lower_wick = 0.0

        features.append({
            "timestamp_us": bkt,
            "vol_imbalance": vol_imbalance,
            "dollar_imbalance": dollar_imbalance,
            "large_imbalance": large_imbalance,
            "large_vol_pct": large_vol_pct,
            "count_imbalance": count_imbalance,
            "arrival_rate": arrival_rate,
            "iti_cv": iti_cv,
            "burstiness": burstiness,
            "trade_acceleration": trade_acceleration,
            "price_range": price_range,
            "close_vs_vwap": close_vs_vwap,
            "kyle_lambda": kyle_lambda,
            "amihud": amihud,
            "vol_profile_skew": vol_profile_skew,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "open": open_p,
            "close": close_p,
            "high": high_p,
            "low": low_p,
            "volume": total_vol,
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "quote_volume": buy_quote + sell_quote,
            "trade_count": n,
        })

    return pd.DataFrame(features)


# ---------------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------------

INTERVAL_MAP = {
    "1m": 60_000_000,
    "5m": 300_000_000,
    "15m": 900_000_000,
}

SOURCES = [
    "binance_futures",
    "bybit_futures",
    "okx_futures",
    "binance_spot",
    "bybit_spot",
    "okx_spot",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    symbol = args.symbol
    interval = args.interval
    interval_us = INTERVAL_MAP[interval]
    parquet_dir = Path(args.parquet_dir)
    sources = args.sources

    print(f"Symbol:   {symbol}")
    print(f"Interval: {interval} ({interval_us} us)")
    print(f"Sources:  {', '.join(sources)}")
    print(f"Input:    {parquet_dir / symbol}")
    print("=" * 60)

    total_written = 0
    total_skipped = 0

    for source in sources:
        trades_dir = parquet_dir / symbol / "trades" / source
        out_dir = parquet_dir / symbol / "features" / interval / source
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(trades_dir.glob("*.parquet")) if trades_dir.exists() else []
        if not files:
            print(f"  {source}: no trade files found")
            continue

        n = len(files)
        t0 = time.time()
        written = 0
        skipped = 0

        print(f"\n  {source} ({n} days)...")

        for i, f in enumerate(files, 1):
            date = f.stem
            dest = out_dir / f"{date}.parquet"

            # Skip existing
            if dest.exists():
                skipped += 1
                if i % 30 == 0 or i == n:
                    elapsed = time.time() - t0
                    print(f"    [{i:3d}/{n}] ⊘ skipping (exists)  "
                          f"elapsed={elapsed:.0f}s  written={written} skipped={skipped}")
                continue

            trades = pd.read_parquet(f)
            feat = compute_microstructure_features(trades, interval_us)
            del trades

            if feat.empty:
                continue

            feat.to_parquet(dest, compression="snappy", index=False)
            written += 1

            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (n - i) / rate if rate > 0 else 0

            if i % 10 == 0 or i == n:
                print(f"    [{i:3d}/{n}] ✓ {date}  bars={len(feat):>4d}  "
                      f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

        elapsed = time.time() - t0
        print(f"  {source}: {written} written, {skipped} skipped in {elapsed:.0f}s")
        total_written += written
        total_skipped += skipped

    print(f"\n{'=' * 60}")
    print(f"Done. {total_written} files written, {total_skipped} skipped.")


def main():
    parser = argparse.ArgumentParser(
        description="Compute microstructure features from tick trades.",
    )
    parser.add_argument("symbol", help="Trading pair, e.g. BTCUSDT")
    parser.add_argument("--interval", default="5m", choices=list(INTERVAL_MAP.keys()),
                        help="Aggregation interval (default: 5m)")
    parser.add_argument("--parquet-dir", default="./parquet",
                        help="Parquet root directory (default: ./parquet)")
    parser.add_argument("--sources", nargs="+", default=["binance_futures", "bybit_futures", "okx_futures"],
                        choices=SOURCES, help="Which sources to process")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
