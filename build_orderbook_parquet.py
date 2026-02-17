#!/usr/bin/env python3
"""
Convert raw Bybit orderbook snapshot+delta data into daily parquet files.

Reads from:  data/{SYMBOL}/bybit/orderbook_{market}/
Writes to:   parquet/{SYMBOL}/orderbook/{market}/{YYYY-MM-DD}.parquet

The raw data is JSON lines with "snapshot" (full 200-level book) and "delta"
(incremental updates, ~10/sec). We reconstruct the full book state and sample
at 1-second intervals, storing pre-computed summary statistics.

Output columns (per 1-second snapshot):
  timestamp_us       int64   microseconds since epoch (UTC)
  mid_price          float64 (best_bid + best_ask) / 2
  best_bid           float64
  best_ask           float64
  spread_bps         float64 spread in basis points of mid
  bid_depth_{N}bps  float64 total bid size within N bps of mid (N=0.5,1,2,3,5,10,25,50)
  ask_depth_{N}bps  float64 total ask size within N bps of mid
  imbalance_{N}bps  float64 (bid-ask)/(bid+ask) depth within N bps
  bid_vwap_5bps      float64 volume-weighted avg bid price within 5 bps
  ask_vwap_5bps      float64 volume-weighted avg ask price within 5 bps
  bid_levels         int32   number of non-zero bid levels in book
  ask_levels         int32   number of non-zero ask levels in book
  bid_total          float64 total bid size across all levels
  ask_total          float64 total ask size across all levels

Usage:
  python build_orderbook_parquet.py BTCUSDT
  python build_orderbook_parquet.py BTCUSDT --market futures
  python build_orderbook_parquet.py BTCUSDT --market spot
  python build_orderbook_parquet.py BTCUSDT --market both
"""

import argparse
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARQUET_WRITE_OPTS = dict(compression="snappy", use_dictionary=True)
# BTC futures 200-level book spans ~5bps from mid, so we need fine granularity
# For wider books (spot, altcoins), the larger levels will capture more structure
DEPTH_BPS_LEVELS = [0.5, 1, 2, 3, 5, 10, 25, 50]
SAMPLE_INTERVAL_US = 1_000_000  # 1 second


# ---------------------------------------------------------------------------
# Orderbook state manager
# ---------------------------------------------------------------------------

class OrderBook:
    """Maintains a price-level orderbook from snapshot + delta updates."""

    def __init__(self):
        self.bids = {}  # price -> size
        self.asks = {}  # price -> size
        self.initialized = False

    def apply_snapshot(self, bids, asks):
        """Replace entire book with snapshot data."""
        self.bids = {}
        self.asks = {}
        for price_str, size_str in bids:
            p, s = float(price_str), float(size_str)
            if s > 0:
                self.bids[p] = s
        for price_str, size_str in asks:
            p, s = float(price_str), float(size_str)
            if s > 0:
                self.asks[p] = s
        self.initialized = True

    def apply_delta(self, bids, asks):
        """Apply incremental updates. Size=0 means delete level."""
        for price_str, size_str in bids:
            p, s = float(price_str), float(size_str)
            if s == 0:
                self.bids.pop(p, None)
            else:
                self.bids[p] = s
        for price_str, size_str in asks:
            p, s = float(price_str), float(size_str)
            if s == 0:
                self.asks.pop(p, None)
            else:
                self.asks[p] = s

    def snapshot_stats(self):
        """Compute summary statistics from current book state.
        Returns dict of column values, or None if book is empty."""
        if not self.bids or not self.asks:
            return None

        # Sorted arrays for fast computation
        bid_prices = np.array(sorted(self.bids.keys(), reverse=True))
        bid_sizes = np.array([self.bids[p] for p in bid_prices])
        ask_prices = np.array(sorted(self.asks.keys()))
        ask_sizes = np.array([self.asks[p] for p in ask_prices])

        best_bid = bid_prices[0]
        best_ask = ask_prices[0]
        mid = (best_bid + best_ask) / 2.0

        if mid <= 0:
            return None

        spread_bps = (best_ask - best_bid) / mid * 10000.0

        row = {
            "mid_price": mid,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread_bps": spread_bps,
            "bid_levels": int(len(bid_prices)),
            "ask_levels": int(len(ask_prices)),
            "bid_total": float(bid_sizes.sum()),
            "ask_total": float(ask_sizes.sum()),
        }

        # Depth at various bps levels from mid
        bid_dist_bps = (mid - bid_prices) / mid * 10000.0
        ask_dist_bps = (ask_prices - mid) / mid * 10000.0

        for bps in DEPTH_BPS_LEVELS:
            bid_mask = bid_dist_bps <= bps
            ask_mask = ask_dist_bps <= bps
            bd = float(bid_sizes[bid_mask].sum()) if bid_mask.any() else 0.0
            ad = float(ask_sizes[ask_mask].sum()) if ask_mask.any() else 0.0
            # Use string key that handles fractional bps
            bps_key = f"{bps}" if bps == int(bps) else f"{bps}"
            row[f"bid_depth_{bps_key}bps"] = bd
            row[f"ask_depth_{bps_key}bps"] = ad

            total = bd + ad
            row[f"imbalance_{bps_key}bps"] = (bd - ad) / total if total > 0 else 0.0

        # VWAP within 5 bps (where most depth lives for BTC)
        bid_5_mask = bid_dist_bps <= 5
        ask_5_mask = ask_dist_bps <= 5
        if bid_5_mask.any():
            bp = bid_prices[bid_5_mask]
            bs = bid_sizes[bid_5_mask]
            row["bid_vwap_5bps"] = float(np.average(bp, weights=bs))
        else:
            row["bid_vwap_5bps"] = best_bid

        if ask_5_mask.any():
            ap = ask_prices[ask_5_mask]
            asizes = ask_sizes[ask_5_mask]
            row["ask_vwap_5bps"] = float(np.average(ap, weights=asizes))
        else:
            row["ask_vwap_5bps"] = best_ask

        return row


# ---------------------------------------------------------------------------
# File parser
# ---------------------------------------------------------------------------

def parse_orderbook_file(zip_path: Path, sample_interval_us: int = SAMPLE_INTERVAL_US):
    """Parse one day's orderbook zip file, reconstruct book, sample at interval.

    Returns list of dicts (one per sampled snapshot).
    """
    book = OrderBook()
    snapshots = []
    last_sample_ts = 0
    records_parsed = 0
    snapshot_count = 0

    with zipfile.ZipFile(zip_path) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            # Read in chunks to handle large files
            buffer = ""
            chunk_size = 50 * 1024 * 1024  # 50MB chunks

            while True:
                raw = f.read(chunk_size)
                if not raw:
                    break
                buffer += raw.decode("utf-8", errors="replace")

                # Split into JSON objects by finding }{ boundaries
                # Each record is a complete JSON object
                parts = re.split(r"\}\s*\{", buffer)

                # Process all complete records (all except possibly the last)
                for i, part in enumerate(parts[:-1]):
                    if i == 0:
                        text = part + "}"
                    else:
                        text = "{" + part + "}"

                    try:
                        obj = json.loads(text)
                    except json.JSONDecodeError:
                        continue

                    records_parsed += 1
                    ts_ms = obj.get("ts", 0)
                    ts_us = ts_ms * 1000
                    msg_type = obj.get("type", "")
                    data = obj.get("data", {})
                    bids = data.get("b", [])
                    asks = data.get("a", [])

                    if msg_type == "snapshot":
                        book.apply_snapshot(bids, asks)
                        snapshot_count += 1
                    elif msg_type == "delta":
                        book.apply_delta(bids, asks)
                    else:
                        continue

                    # Sample at interval
                    if book.initialized and ts_us - last_sample_ts >= sample_interval_us:
                        stats = book.snapshot_stats()
                        if stats is not None:
                            stats["timestamp_us"] = ts_us
                            snapshots.append(stats)
                            last_sample_ts = ts_us

                # Keep the last (possibly incomplete) part as buffer
                buffer = "{" + parts[-1] if len(parts) > 1 else parts[-1]

            # Process the final buffer
            text = buffer.strip()
            if text:
                # Try the last part
                if not text.startswith("{"):
                    text = "{" + text
                try:
                    obj = json.loads(text)
                    ts_ms = obj.get("ts", 0)
                    ts_us = ts_ms * 1000
                    msg_type = obj.get("type", "")
                    data = obj.get("data", {})
                    bids = data.get("b", [])
                    asks = data.get("a", [])

                    if msg_type == "snapshot":
                        book.apply_snapshot(bids, asks)
                    elif msg_type == "delta":
                        book.apply_delta(bids, asks)

                    if book.initialized and ts_us - last_sample_ts >= sample_interval_us:
                        stats = book.snapshot_stats()
                        if stats is not None:
                            stats["timestamp_us"] = ts_us
                            snapshots.append(stats)
                except json.JSONDecodeError:
                    pass

    return snapshots, records_parsed, snapshot_count


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_market(symbol: str, market: str, data_dir: Path, out_dir: Path):
    """Process all orderbook files for one market (futures or spot)."""
    src_dir = data_dir / symbol / "bybit" / f"orderbook_{market}"
    if not src_dir.exists():
        print(f"  {market}: directory not found: {src_dir}")
        return 0, 0

    files = sorted(src_dir.glob("*.zip"))
    if not files:
        print(f"  {market}: no zip files found")
        return 0, 0

    dest_dir = out_dir / symbol / "orderbook" / f"bybit_{market}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    n = len(files)
    t0 = time.time()

    print(f"  {market}: {n} files, output → {dest_dir}")

    for i, f in enumerate(files, 1):
        date = re.search(r"(\d{4}-\d{2}-\d{2})", f.name)
        if not date:
            continue
        date_str = date.group(1)
        out_path = dest_dir / f"{date_str}.parquet"

        if out_path.exists():
            skipped += 1
            if i <= 3 or i == n:
                print(f"    [{i}/{n}] ⊘ {date_str}  (exists)")
            continue

        t1 = time.time()
        file_mb = f.stat().st_size / (1024 * 1024)

        try:
            snapshots, records, snap_count = parse_orderbook_file(f)
        except Exception as exc:
            print(f"    [{i}/{n}] ⚠ {date_str}: {exc}")
            continue

        if not snapshots:
            print(f"    [{i}/{n}] - {date_str}  (empty, {records} records parsed)")
            continue

        df = pd.DataFrame(snapshots)
        df["timestamp_us"] = df["timestamp_us"].astype(np.int64)
        df = df.sort_values("timestamp_us").reset_index(drop=True)

        # Write parquet
        df.to_parquet(out_path, index=False, **PARQUET_WRITE_OPTS)
        written += 1

        elapsed = time.time() - t1
        total_elapsed = time.time() - t0
        eta = (total_elapsed / i) * (n - i)
        print(f"    [{i}/{n}] ✓ {date_str}  "
              f"rows={len(df):,}  raw_records={records:,}  snapshots={snap_count}  "
              f"zip={file_mb:.0f}MB  {elapsed:.1f}s  ETA={eta:.0f}s")

        del df, snapshots

    elapsed_total = time.time() - t0
    print(f"  {market}: {written} written, {skipped} skipped, {elapsed_total:.0f}s total")
    return written, skipped


def run(args):
    symbol = args.symbol.upper()
    data_dir = Path(args.input)
    out_dir = Path(args.output)
    markets = ["futures", "spot"] if args.market == "both" else [args.market]

    print(f"Symbol:   {symbol}")
    print(f"Input:    {data_dir.resolve()}")
    print(f"Output:   {out_dir.resolve()}")
    print(f"Markets:  {', '.join(markets)}")
    print(f"Sample:   {SAMPLE_INTERVAL_US / 1_000_000:.0f}s intervals")
    print("=" * 60)

    total_written = 0
    total_skipped = 0

    for market in markets:
        w, s = process_market(symbol, market, data_dir, out_dir)
        total_written += w
        total_skipped += s

    # Summary
    print(f"\n{'='*60}")
    ob_dir = out_dir / symbol / "orderbook"
    if ob_dir.exists():
        pq_files = list(ob_dir.rglob("*.parquet"))
        total_size = sum(f.stat().st_size for f in pq_files)
        print(f"Total: {len(pq_files)} parquet files, {total_size / (1024*1024):.1f} MB")
    print(f"Written: {total_written}, Skipped: {total_skipped}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Bybit orderbook data to daily parquet files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("symbol", help="Trading pair symbol, e.g. BTCUSDT")
    parser.add_argument("--market", "-m", default="both",
                        choices=["futures", "spot", "both"],
                        help="Which market to process (default: both)")
    parser.add_argument("--input", "-i", default="./data",
                        help="Raw data directory (default: ./data)")
    parser.add_argument("--output", "-o", default="./parquet",
                        help="Parquet output directory (default: ./parquet)")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
