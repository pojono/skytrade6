#!/usr/bin/env python3
"""
Download historical funding rate data from Binance, Bybit, and OKX REST APIs.
Saves to parquet files for analysis.

Rate limits:
  Binance: /fapi/v1/fundingRate — 500 req/5min shared, limit=1000 (~42 days per req)
  Bybit:   /v5/market/funding/history — 10 req/s, limit=200 (~8 days per req for 1h coins)
  OKX:     /api/v5/public/funding-rate-history — 10 req/2s per instId, limit=400
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np
import time
import os
import sys
from datetime import datetime, timezone


def make_session():
    """Create a requests session with retry and connection pooling."""
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=5, pool_maxsize=5)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

OUTPUT_DIR = "data_all/historical_fr"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target: last 200 days
DAYS_BACK = 200
NOW_MS = int(time.time() * 1000)
START_MS = NOW_MS - DAYS_BACK * 86400 * 1000


# ═══════════════════════════════════════════════════════════════════════════
# BINANCE
# ═══════════════════════════════════════════════════════════════════════════

def get_binance_symbols():
    """Get all trading USDT-M perpetual symbols."""
    r = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo")
    r.raise_for_status()
    return sorted([
        s["symbol"] for s in r.json()["symbols"]
        if s["contractType"] == "PERPETUAL" and s["status"] == "TRADING"
    ])


BN_SESSION = make_session()

def fetch_binance_fr(symbol, start_ms=START_MS, end_ms=NOW_MS):
    """Fetch all funding rate history for one Binance symbol, paginating."""
    all_records = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        r = BN_SESSION.get("https://fapi.binance.com/fapi/v1/fundingRate", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        all_records.extend(data)
        last_ts = int(data[-1]["fundingTime"])
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        time.sleep(0.62)  # ~1.6 req/s to stay under 500/5min
    return all_records


def download_binance(symbols):
    """Download FR for all Binance symbols sequentially (rate limit is per-IP)."""
    outpath = os.path.join(OUTPUT_DIR, "binance_fr_history.parquet")
    
    # Resume support
    done_syms = set()
    existing_data = []
    if os.path.exists(outpath):
        existing = pd.read_parquet(outpath)
        done_syms = set(existing["symbol"].unique())
        existing_data = existing.to_dict("records")
        print(f"  Resuming: {len(done_syms)} symbols already downloaded")
    
    remaining = [s for s in symbols if s not in done_syms]
    print(f"\n{'='*70}")
    print(f"BINANCE: Downloading FR for {len(remaining)}/{len(symbols)} symbols")
    print(f"{'='*70}")

    all_data = existing_data
    t0 = time.time()
    errors = []

    for i, sym in enumerate(remaining):
        try:
            records = fetch_binance_fr(sym)
            for rec in records:
                all_data.append({
                    "symbol": sym,
                    "fundingTime": pd.Timestamp(int(rec["fundingTime"]), unit="ms", tz="UTC"),
                    "fundingRate": float(rec["fundingRate"]),
                    "markPrice": float(rec.get("markPrice", 0)),
                })

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:>4}/{len(remaining)}] {sym:<18} {len(records):>5} records  "
                  f"({rate:.0f} sym/min, ETA {eta:.0f}min, {len(all_data):,} total)", flush=True)
            
            # Save checkpoint every 50 symbols
            if (i + 1) % 50 == 0:
                pd.DataFrame(all_data).to_parquet(outpath, index=False)
                print(f"  >>> Checkpoint saved ({len(all_data):,} records)", flush=True)
        except Exception as e:
            errors.append((sym, str(e)))
            print(f"  [{i+1:>4}/{len(remaining)}] {sym:<18} ERROR: {e}", flush=True)
            time.sleep(2)

    if errors:
        print(f"\n  Errors: {len(errors)} symbols failed")

    df = pd.DataFrame(all_data)
    if len(df) > 0:
        df.to_parquet(outpath, index=False)
        print(f"\n  Saved {len(df):,} records to {outpath}")
        print(f"  Symbols: {df['symbol'].nunique()}, date range: {df['fundingTime'].min()} -> {df['fundingTime'].max()}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# BYBIT
# ═══════════════════════════════════════════════════════════════════════════

def get_bybit_symbols():
    """Get all trading linear USDT symbols."""
    all_syms = []
    cursor = ""
    while True:
        params = {"category": "linear", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        r = requests.get("https://api.bybit.com/v5/market/instruments-info", params=params, timeout=10)
        r.raise_for_status()
        result = r.json()["result"]
        for s in result["list"]:
            if s["status"] == "Trading" and s["symbol"].endswith("USDT"):
                all_syms.append(s["symbol"])
        cursor = result.get("nextPageCursor", "")
        if not cursor:
            break
    return sorted(all_syms)


BB_SESSION = make_session()

def fetch_bybit_fr(symbol, start_ms=START_MS, end_ms=NOW_MS):
    """Fetch all funding rate history for one Bybit symbol, paginating backwards."""
    all_records = []
    cursor_end = end_ms
    while cursor_end > start_ms:
        params = {
            "category": "linear",
            "symbol": symbol,
            "startTime": start_ms,
            "endTime": cursor_end,
            "limit": 200,
        }
        r = BB_SESSION.get("https://api.bybit.com/v5/market/funding/history", params=params, timeout=20)
        r.raise_for_status()
        data = r.json().get("result", {}).get("list", [])
        if not data:
            break
        all_records.extend(data)
        # Bybit returns newest first, so oldest is last
        oldest_ts = min(int(d["fundingRateTimestamp"]) for d in data)
        if oldest_ts >= cursor_end:
            break
        cursor_end = oldest_ts - 1
        time.sleep(0.12)  # 8 req/s to stay under 10/s
    return all_records


def download_bybit(symbols):
    """Download FR for all Bybit symbols."""
    outpath = os.path.join(OUTPUT_DIR, "bybit_fr_history.parquet")
    
    # Resume support
    done_syms = set()
    existing_data = []
    if os.path.exists(outpath):
        existing = pd.read_parquet(outpath)
        done_syms = set(existing["symbol"].unique())
        existing_data = existing.to_dict("records")
        print(f"  Resuming: {len(done_syms)} symbols already downloaded")
    
    remaining = [s for s in symbols if s not in done_syms]
    print(f"\n{'='*70}")
    print(f"BYBIT: Downloading FR for {len(remaining)}/{len(symbols)} symbols")
    print(f"{'='*70}")

    all_data = existing_data
    t0 = time.time()
    errors = []

    for i, sym in enumerate(remaining):
        try:
            records = fetch_bybit_fr(sym)
            for rec in records:
                all_data.append({
                    "symbol": sym,
                    "fundingTime": pd.Timestamp(int(rec["fundingRateTimestamp"]), unit="ms", tz="UTC"),
                    "fundingRate": float(rec["fundingRate"]),
                })

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  [{i+1:>4}/{len(remaining)}] {sym:<18} {len(records):>5} records  "
                      f"({rate:.0f} sym/min, ETA {eta:.0f}min, {len(all_data):,} total)", flush=True)
            
            # Save checkpoint every 50 symbols
            if (i + 1) % 50 == 0:
                pd.DataFrame(all_data).to_parquet(outpath, index=False)
                print(f"  >>> Checkpoint saved ({len(all_data):,} records)", flush=True)
        except Exception as e:
            errors.append((sym, str(e)))
            print(f"  [{i+1:>4}/{len(remaining)}] {sym:<18} ERROR: {e}", flush=True)
            time.sleep(1)

    if errors:
        print(f"\n  Errors: {len(errors)} symbols failed")

    df = pd.DataFrame(all_data)
    if len(df) > 0:
        df.to_parquet(outpath, index=False)
        print(f"\n  Saved {len(df):,} records to {outpath}")
        print(f"  Symbols: {df['symbol'].nunique()}, date range: {df['fundingTime'].min()} -> {df['fundingTime'].max()}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# OKX
# ═══════════════════════════════════════════════════════════════════════════

def get_okx_symbols():
    """Get all trading USDT-SWAP perpetual symbols on OKX."""
    r = requests.get("https://www.okx.com/api/v5/public/instruments",
                     params={"instType": "SWAP"}, timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])
    return sorted([
        s["instId"] for s in data
        if s["instId"].endswith("-USDT-SWAP") and s["state"] == "live"
    ])


OKX_SESSION = make_session()

def fetch_okx_fr(inst_id, start_ms=START_MS, end_ms=NOW_MS):
    """Fetch all funding rate history for one OKX instrument, paginating backwards."""
    all_records = []
    cursor_after = ""
    while True:
        params = {
            "instId": inst_id,
            "limit": 400,
        }
        if cursor_after:
            params["after"] = cursor_after
        r = OKX_SESSION.get("https://www.okx.com/api/v5/public/funding-rate-history",
                            params=params, timeout=20)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            break
        # Filter to our date range
        for rec in data:
            ft = int(rec["fundingTime"])
            if ft >= start_ms:
                all_records.append(rec)
        # OKX returns newest first; oldest is last
        oldest_ts = int(data[-1]["fundingTime"])
        if oldest_ts < start_ms:
            break
        cursor_after = data[-1]["fundingTime"]
        time.sleep(0.25)  # 4 req/s to stay under 10/2s
    return all_records


def download_okx(symbols):
    """Download FR for all OKX symbols."""
    outpath = os.path.join(OUTPUT_DIR, "okx_fr_history.parquet")

    # Resume support
    done_syms = set()
    existing_data = []
    if os.path.exists(outpath):
        existing = pd.read_parquet(outpath)
        done_syms = set(existing["symbol"].unique())
        existing_data = existing.to_dict("records")
        print(f"  Resuming: {len(done_syms)} symbols already downloaded")

    remaining = [s for s in symbols if s not in done_syms]
    print(f"\n{'='*70}")
    print(f"OKX: Downloading FR for {len(remaining)}/{len(symbols)} symbols")
    print(f"{'='*70}")

    all_data = existing_data
    t0 = time.time()
    errors = []

    for i, sym in enumerate(remaining):
        try:
            records = fetch_okx_fr(sym)
            for rec in records:
                all_data.append({
                    "symbol": sym,
                    "fundingTime": pd.Timestamp(int(rec["fundingTime"]), unit="ms", tz="UTC"),
                    "fundingRate": float(rec.get("realizedRate") or rec["fundingRate"]),
                })

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  [{i+1:>4}/{len(remaining)}] {sym:<24} {len(records):>5} records  "
                      f"({rate:.0f} sym/min, ETA {eta:.0f}min, {len(all_data):,} total)", flush=True)

            # Save checkpoint every 50 symbols
            if (i + 1) % 50 == 0:
                pd.DataFrame(all_data).to_parquet(outpath, index=False)
                print(f"  >>> Checkpoint saved ({len(all_data):,} records)", flush=True)
        except Exception as e:
            errors.append((sym, str(e)))
            print(f"  [{i+1:>4}/{len(remaining)}] {sym:<24} ERROR: {e}", flush=True)
            time.sleep(1)

    if errors:
        print(f"\n  Errors: {len(errors)} symbols failed")

    df = pd.DataFrame(all_data)
    if len(df) > 0:
        df.to_parquet(outpath, index=False)
        print(f"\n  Saved {len(df):,} records to {outpath}")
        print(f"  Symbols: {df['symbol'].nunique()}, date range: {df['fundingTime'].min()} -> {df['fundingTime'].max()}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    print(f"Historical FR downloader — target: {target}")
    print(f"  Period: last {DAYS_BACK} days ({datetime.fromtimestamp(START_MS/1000, tz=timezone.utc).strftime('%Y-%m-%d')} -> now)")

    if target in ("binance", "both", "all"):
        bn_syms = get_binance_symbols()
        print(f"  Binance symbols: {len(bn_syms)}")
        bn_df = download_binance(bn_syms)

    if target in ("bybit", "both", "all"):
        bb_syms = get_bybit_symbols()
        print(f"  Bybit symbols: {len(bb_syms)}")
        bb_df = download_bybit(bb_syms)

    if target in ("okx", "all"):
        okx_syms = get_okx_symbols()
        print(f"  OKX symbols: {len(okx_syms)}")
        okx_df = download_okx(okx_syms)

    print(f"\nDone!")
