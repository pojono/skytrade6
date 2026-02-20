#!/usr/bin/env python3
"""
Download historical 1h klines from Bybit v5 API for linear perpetual futures.

Endpoint: GET https://api.bybit.com/v5/market/kline
  - category=linear, interval=60, limit=1000
  - No auth required (public market data)
  - Returns up to 1000 candles per request, sorted reverse by startTime
  - Rate limit: 10 req/s (we use conservative 5 req/s)

Output: parquet/{SYMBOL}/ohlcv/1h/bybit_futures/{YYYY-MM}.parquet
  Schema: timestamp_us, open, high, low, close, volume, turnover

Usage:
  python download_bybit_klines.py                    # all default symbols
  python download_bybit_klines.py BTCUSDT ETHUSDT    # specific symbols
  python download_bybit_klines.py --start 2022-01-01 # custom start date
"""

import sys, time, argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

sys.stdout.reconfigure(line_buffering=True)

API_URL = "https://api.bybit.com/v5/market/kline"
PARQUET_DIR = Path("parquet")
INTERVAL = "60"  # 1h
LIMIT = 1000     # max per request
RATE_LIMIT_SLEEP = 0.22  # ~4.5 req/s, well under 10/s limit

# Top 20 USDT perpetual futures by volume on Bybit
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "BNBUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "MATICUSDT", "LTCUSDT", "UNIUSDT", "APTUSDT", "ARBUSDT",
    "OPUSDT", "NEARUSDT", "FILUSDT", "ATOMUSDT", "SUIUSDT",
]

DEFAULT_START = "2023-01-01"


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list:
    """Fetch up to 1000 klines from Bybit API."""
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": INTERVAL,
        "start": start_ms,
        "end": end_ms,
        "limit": LIMIT,
    }
    resp = requests.get(API_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data.get("retCode") != 0:
        raise ValueError(f"API error: {data.get('retMsg', 'unknown')}")

    return data.get("result", {}).get("list", [])


def download_symbol(symbol: str, start_date: str, end_date: str = None):
    """Download all 1h klines for a symbol from start_date to now."""
    out_dir = PARQUET_DIR / symbol / "ohlcv" / "1h" / "bybit_futures"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end_dt = datetime.now(timezone.utc)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    all_rows = []
    cursor_end = end_ms
    request_count = 0
    t0 = time.time()

    while cursor_end > start_ms:
        try:
            rows = fetch_klines(symbol, start_ms, cursor_end)
        except Exception as e:
            print(f"    ⚠ API error at {cursor_end}: {e}")
            time.sleep(2)
            continue

        request_count += 1

        if not rows:
            break

        all_rows.extend(rows)

        # Rows are sorted reverse by startTime, so oldest is last
        oldest_ms = int(rows[-1][0])
        if oldest_ms >= cursor_end:
            break  # no progress
        cursor_end = oldest_ms - 1

        # Progress
        if request_count % 5 == 0:
            elapsed = time.time() - t0
            n_bars = len(all_rows)
            oldest_dt = datetime.fromtimestamp(oldest_ms / 1000, tz=timezone.utc)
            print(f"    {request_count} requests, {n_bars:,} bars, "
                  f"oldest={oldest_dt.strftime('%Y-%m-%d')}, {elapsed:.0f}s")

        time.sleep(RATE_LIMIT_SLEEP)

    if not all_rows:
        print(f"    No data returned for {symbol}")
        return 0

    # Parse into DataFrame
    # Each row: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
    df = pd.DataFrame(all_rows, columns=[
        "startTime", "open", "high", "low", "close", "volume", "turnover"
    ])
    df["startTime"] = df["startTime"].astype(np.int64)
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(np.float64)

    # Convert startTime (ms) to timestamp_us
    df["timestamp_us"] = df["startTime"] * 1000
    df = df.drop(columns=["startTime"])

    # Remove duplicates and sort
    df = df.drop_duplicates(subset=["timestamp_us"]).sort_values("timestamp_us").reset_index(drop=True)

    # Filter to requested date range
    df = df[(df["timestamp_us"] >= start_ms * 1000) &
            (df["timestamp_us"] < end_ms * 1000)]

    # Check for existing data and only write new
    existing_files = sorted(out_dir.glob("*.parquet"))
    existing_ts = set()
    for f in existing_files:
        try:
            edf = pd.read_parquet(f, columns=["timestamp_us"])
            existing_ts.update(edf["timestamp_us"].values)
        except Exception:
            pass

    if existing_ts:
        before = len(df)
        df = df[~df["timestamp_us"].isin(existing_ts)]
        skipped = before - len(df)
        if skipped > 0:
            print(f"    Skipped {skipped:,} existing bars")

    if df.empty:
        print(f"    All bars already exist")
        return 0

    # Write monthly parquet files
    df["timestamp"] = pd.to_datetime(df["timestamp_us"], unit="us")
    df["month"] = df["timestamp"].dt.to_period("M")
    written = 0

    for month, mdf in df.groupby("month"):
        month_str = str(month)  # e.g., "2023-01"
        out_path = out_dir / f"{month_str}.parquet"

        # If file exists, merge
        write_df = mdf.drop(columns=["timestamp", "month"])
        if out_path.exists():
            try:
                old = pd.read_parquet(out_path)
                write_df = pd.concat([old, write_df], ignore_index=True)
                write_df = write_df.drop_duplicates(subset=["timestamp_us"]).sort_values("timestamp_us").reset_index(drop=True)
            except Exception:
                pass

        # Reorder columns
        cols = ["timestamp_us", "open", "high", "low", "close", "volume", "turnover"]
        write_df = write_df[cols]
        write_df.to_parquet(out_path, index=False)
        written += 1

    return len(df), written


def main():
    parser = argparse.ArgumentParser(description="Download Bybit 1h klines")
    parser.add_argument("symbols", nargs="*", default=DEFAULT_SYMBOLS,
                        help="Symbols to download (default: top 20)")
    parser.add_argument("--start", default=DEFAULT_START,
                        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START})")
    parser.add_argument("--end", default=None,
                        help="End date YYYY-MM-DD (default: now)")
    args = parser.parse_args()

    print("=" * 70)
    print("Bybit 1h Klines Downloader")
    print(f"Symbols: {len(args.symbols)}")
    print(f"Date range: {args.start} to {args.end or 'now'}")
    print("=" * 70)

    t0 = time.time()
    total_bars = 0
    total_files = 0

    for i, symbol in enumerate(args.symbols, 1):
        print(f"\n[{i}/{len(args.symbols)}] {symbol}")
        result = download_symbol(symbol, args.start, args.end)
        if result:
            bars, files = result
            total_bars += bars
            total_files += files
            print(f"    ✓ {bars:,} new bars, {files} files written")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Done in {elapsed:.0f}s. Total: {total_bars:,} bars, {total_files} files")
    print(f"Data saved to: {PARQUET_DIR.resolve()}")


if __name__ == "__main__":
    main()
