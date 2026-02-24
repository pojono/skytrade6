#!/usr/bin/env python3
"""
Download 1-minute klines around FR settlement times from Bybit API.

For each extreme negative FR settlement (|FR| >= 20 bps), download
a ±10 min window of 1m candles. Uses parallel downloads (20 threads).

Output: data_all/historical_fr/settlement_klines.parquet
"""
import sys
import os
import json
import time
from pathlib import Path
from urllib.request import urlopen, Request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

import builtins
_print = builtins.print
def print(*a, **k):
    k.setdefault("flush", True)
    _print(*a, **k)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data_all"
OUTPUT = DATA / "historical_fr" / "settlement_klines.parquet"

MIN_FR_BPS = 15  # download for |FR| >= 15 bps to have more data
WINDOW_MIN = 10  # ±10 min around settlement = 21 candles

def api_get(url, retries=3):
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if attempt == retries - 1:
                return None
            time.sleep(0.5 * (attempt + 1))

def download_kline(symbol, settle_ts_ms):
    """Download 1m klines for ±WINDOW_MIN around a settlement time."""
    start_ms = settle_ts_ms - WINDOW_MIN * 60 * 1000
    end_ms = settle_ts_ms + WINDOW_MIN * 60 * 1000
    
    url = (f"https://api.bybit.com/v5/market/kline"
           f"?category=linear&symbol={symbol}&interval=1"
           f"&start={start_ms}&end={end_ms}&limit=30")
    
    data = api_get(url)
    if data is None or data.get("retCode") != 0:
        return None
    
    rows = []
    for candle in data.get("result", {}).get("list", []):
        rows.append({
            "symbol": symbol,
            "settle_ts_ms": settle_ts_ms,
            "ts_ms": int(candle[0]),
            "open": float(candle[1]),
            "high": float(candle[2]),
            "low": float(candle[3]),
            "close": float(candle[4]),
            "volume": float(candle[5]),
        })
    return rows


def main():
    t0 = time.time()
    
    print("=" * 90)
    print("DOWNLOAD 1m KLINES AROUND FR SETTLEMENTS (Bybit API)")
    print("=" * 90)
    
    # Load settlement schedule
    print("\n1. Loading FR history...")
    # Use Bybit FR history since we'll be getting Bybit klines
    bb_fr = pd.read_parquet(DATA / "historical_fr" / "bybit_fr_history.parquet")
    print(f"   {len(bb_fr):,} records, {bb_fr['symbol'].nunique()} symbols")
    
    # Also load Binance for comparison
    bn_fr = pd.read_parquet(DATA / "historical_fr" / "binance_fr_history.parquet")
    print(f"   Binance: {len(bn_fr):,} records")
    
    # Identify extreme negative FR settlements from Bybit
    bb_fr["fr_bps"] = bb_fr["fundingRate"] * 10000
    neg = bb_fr[bb_fr["fr_bps"] <= -MIN_FR_BPS].copy()
    neg["settle_ts_ms"] = neg["fundingTime"].astype(np.int64) // 10**6
    
    print(f"   Bybit FR <= -{MIN_FR_BPS} bps: {len(neg):,} settlements, {neg['symbol'].nunique()} symbols")
    
    # Also get positive FR (rare but let's grab them too)
    pos = bb_fr[bb_fr["fr_bps"] >= MIN_FR_BPS].copy()
    pos["settle_ts_ms"] = pos["fundingTime"].astype(np.int64) // 10**6
    print(f"   Bybit FR >= +{MIN_FR_BPS} bps: {len(pos):,} settlements")
    
    # Combine
    targets = pd.concat([neg, pos])[["symbol", "settle_ts_ms", "fr_bps"]].drop_duplicates()
    print(f"   Total to download: {len(targets):,} settlement windows")
    
    # Check for existing data (resume support)
    existing_keys = set()
    if OUTPUT.exists():
        existing = pd.read_parquet(OUTPUT, columns=["symbol", "settle_ts_ms"]).drop_duplicates()
        existing_keys = set(zip(existing["symbol"], existing["settle_ts_ms"]))
        print(f"   Already downloaded: {len(existing_keys):,} windows")
    
    new_targets = targets[~targets.apply(lambda r: (r["symbol"], r["settle_ts_ms"]) in existing_keys, axis=1)]
    print(f"   New to download: {len(new_targets):,} windows")
    
    if len(new_targets) == 0:
        print("   Nothing new to download!")
        return
    
    # 2. Parallel download
    print(f"\n2. Downloading {len(new_targets):,} windows (20 threads)...")
    
    all_rows = []
    done = 0
    errors = 0
    
    jobs = list(new_targets[["symbol", "settle_ts_ms"]].itertuples(index=False))
    
    BATCH_SIZE = 500
    CHECKPOINT_EVERY = 2000
    
    with ThreadPoolExecutor(max_workers=20) as pool:
        for batch_start in range(0, len(jobs), BATCH_SIZE):
            batch = jobs[batch_start:batch_start + BATCH_SIZE]
            futures = {pool.submit(download_kline, j.symbol, j.settle_ts_ms): j for j in batch}
            
            for future in as_completed(futures):
                done += 1
                result = future.result()
                if result:
                    all_rows.extend(result)
                else:
                    errors += 1
                
                if done % 200 == 0 or done == len(jobs):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (len(jobs) - done) / rate if rate > 0 else 0
                    print(f"   [{done:,}/{len(jobs):,}] {len(all_rows):,} candles, "
                          f"{errors} errors, {elapsed:.0f}s, ~{eta:.0f}s ETA")
            
            # Checkpoint save
            if done >= CHECKPOINT_EVERY and len(all_rows) > 0 and (done % CHECKPOINT_EVERY < BATCH_SIZE):
                _save(all_rows, existing_keys)
                print(f"   >>> Checkpoint saved ({len(all_rows):,} new candles)")
    
    # 3. Final save
    print(f"\n3. Saving...")
    _save(all_rows, existing_keys)
    
    print(f"\n   Done: {done:,} windows, {len(all_rows):,} candles, {errors} errors, {time.time()-t0:.0f}s")


def _save(new_rows, existing_keys):
    """Save new candles, merging with existing data."""
    if not new_rows:
        return
    
    new_df = pd.DataFrame(new_rows)
    
    if OUTPUT.exists() and existing_keys:
        old_df = pd.read_parquet(OUTPUT)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    
    combined = combined.drop_duplicates(subset=["symbol", "settle_ts_ms", "ts_ms"])
    combined = combined.sort_values(["symbol", "settle_ts_ms", "ts_ms"])
    
    os.makedirs(OUTPUT.parent, exist_ok=True)
    combined.to_parquet(OUTPUT, index=False)
    print(f"   Saved: {len(combined):,} total candles to {OUTPUT}")


if __name__ == "__main__":
    main()
