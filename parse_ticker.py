#!/usr/bin/env python3
"""
Parse Bybit ticker JSONL files into parquet.

Input:  data/{SYMBOL}/ticker_YYYY-MM-DD_hrHH.jsonl.gz  (5-second snapshots)
Output: parquet/{SYMBOL}/ticker/{YYYY-MM-DD}.parquet

Each record has: timestamp_us, last_price, mark_price, index_price,
                 open_interest, oi_value, funding_rate, next_funding_time,
                 bid1_price, ask1_price, bid1_size, ask1_size,
                 volume_24h, turnover_24h
"""

import sys
import gzip
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

PARQUET_DIR = Path("./parquet")
DATA_DIR = Path("./data")


def parse_ticker_files(symbol: str, force: bool = False):
    """Parse all ticker JSONL files for a symbol into daily parquet."""
    ticker_dir = DATA_DIR / symbol
    out_dir = PARQUET_DIR / symbol / "ticker"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(ticker_dir.glob("ticker_*.jsonl.gz"))
    if not files:
        print(f"  No ticker files found for {symbol}")
        return

    # Group files by date
    by_date = defaultdict(list)
    for f in files:
        # ticker_2025-05-11_hr19.jsonl.gz
        parts = f.stem.replace(".jsonl", "").split("_")
        date = parts[1]  # YYYY-MM-DD
        by_date[date].append(f)

    print(f"  {symbol}: {len(files)} files across {len(by_date)} days")

    t0 = time.time()
    total_written = 0

    for i, (date, day_files) in enumerate(sorted(by_date.items()), 1):
        out_path = out_dir / f"{date}.parquet"
        if out_path.exists() and not force:
            total_written += 1
            if i % 20 == 0 or i == len(by_date):
                elapsed = time.time() - t0
                print(f"  [{i}/{len(by_date)}] {date} | {elapsed:.0f}s (cached)", flush=True)
            continue

        records = []
        for f in sorted(day_files):
            try:
                with gzip.open(f, "rt") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        ts_ms = rec["ts"]
                        item = rec["result"]["list"][0]

                        records.append({
                            "timestamp_us": int(ts_ms) * 1000,  # ms → us
                            "last_price": float(item["lastPrice"]),
                            "mark_price": float(item["markPrice"]),
                            "index_price": float(item["indexPrice"]),
                            "open_interest": float(item["openInterest"]),
                            "oi_value": float(item["openInterestValue"]),
                            "funding_rate": float(item["fundingRate"]),
                            "next_funding_time": int(item["nextFundingTime"]),
                            "bid1_price": float(item["bid1Price"]),
                            "ask1_price": float(item["ask1Price"]),
                            "bid1_size": float(item["bid1Size"]),
                            "ask1_size": float(item["ask1Size"]),
                            "volume_24h": float(item["volume24h"]),
                            "turnover_24h": float(item["turnover24h"]),
                        })
            except Exception as exc:
                print(f"    ⚠ {f.name}: {exc}")
                continue

        if records:
            df = pd.DataFrame(records).sort_values("timestamp_us").reset_index(drop=True)
            df.to_parquet(out_path, index=False, compression="snappy")
            total_written += 1

        if i % 10 == 0 or i == len(by_date):
            elapsed = time.time() - t0
            rate = i / max(elapsed, 0.1)
            eta = (len(by_date) - i) / max(rate, 0.01)
            print(f"  [{i}/{len(by_date)}] {date} | {elapsed:.0f}s ETA={eta:.0f}s "
                  f"rows={len(records)}", flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {total_written} days written in {elapsed:.0f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "SOLUSDT", "ETHUSDT"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for sym in args.symbols:
        print(f"\n{'='*60}")
        print(f"  Parsing ticker data: {sym}")
        print(f"{'='*60}")
        parse_ticker_files(sym, force=args.force)
