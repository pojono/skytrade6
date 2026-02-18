#!/usr/bin/env python3
"""
Download comparable data from Binance Futures API.
Fetches ticker data and long/short ratio for comparison with Bybit.
"""
import requests
import time
import json
import gzip
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Binance Futures API
BINANCE_BASE = "https://fapi.binance.com"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
LOCAL_BASE = Path("data_binance")

def get_date_ranges():
    """Split May 11 - Aug 10, 2025 into weekly chunks to reduce API calls."""
    start = datetime(2025, 5, 11, 0, 0, 0)
    end = datetime(2025, 8, 10, 23, 59, 59)
    
    ranges = []
    current = start
    while current < end:
        # Use 7-day chunks
        chunk_end = min(current + timedelta(days=7), end)
        ranges.append((current, chunk_end))
        current = chunk_end + timedelta(seconds=1)
    
    return ranges

def fetch_ticker_24hr(symbol):
    """Fetch 24hr ticker statistics."""
    endpoint = "/fapi/v1/ticker/24hr"
    params = {'symbol': symbol}
    
    try:
        response = requests.get(BINANCE_BASE + endpoint, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"    ✗ Error fetching ticker: {e}")
        return None

def fetch_open_interest(symbol):
    """Fetch current open interest."""
    endpoint = "/fapi/v1/openInterest"
    params = {'symbol': symbol}
    
    try:
        response = requests.get(BINANCE_BASE + endpoint, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"    ✗ Error fetching OI: {e}")
        return None

def fetch_funding_rate(symbol):
    """Fetch current funding rate."""
    endpoint = "/fapi/v1/premiumIndex"
    params = {'symbol': symbol}
    
    try:
        response = requests.get(BINANCE_BASE + endpoint, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"    ✗ Error fetching funding: {e}")
        return None

def fetch_longshort_ratio(symbol, period='1h', start_time=None, end_time=None, limit=500):
    """
    Fetch long/short account ratio (top trader).
    
    Binance endpoint: /futures/data/topLongShortAccountRatio
    Parameters:
    - symbol: Trading pair
    - period: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
    - startTime: Start timestamp (ms)
    - endTime: End timestamp (ms)
    - limit: Max 500
    """
    endpoint = "/futures/data/topLongShortAccountRatio"
    params = {
        'symbol': symbol,
        'period': period,
        'limit': limit
    }
    
    if start_time:
        params['startTime'] = int(start_time.timestamp() * 1000)
    if end_time:
        params['endTime'] = int(end_time.timestamp() * 1000)
    
    try:
        response = requests.get(BINANCE_BASE + endpoint, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"    ✗ Error fetching L/S ratio: {e}")
        return None

def fetch_global_longshort_ratio(symbol, period='1h', start_time=None, end_time=None, limit=500):
    """
    Fetch long/short position ratio (all accounts).
    
    Binance endpoint: /futures/data/globalLongShortAccountRatio
    """
    endpoint = "/futures/data/globalLongShortAccountRatio"
    params = {
        'symbol': symbol,
        'period': period,
        'limit': limit
    }
    
    if start_time:
        params['startTime'] = int(start_time.timestamp() * 1000)
    if end_time:
        params['endTime'] = int(end_time.timestamp() * 1000)
    
    try:
        response = requests.get(BINANCE_BASE + endpoint, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"    ✗ Error fetching global L/S: {e}")
        return None

def fetch_taker_buysell_volume(symbol, period='1h', start_time=None, end_time=None, limit=500):
    """
    Fetch taker buy/sell volume ratio.
    
    Binance endpoint: /futures/data/takerlongshortRatio
    """
    endpoint = "/futures/data/takerlongshortRatio"
    params = {
        'symbol': symbol,
        'period': period,
        'limit': limit
    }
    
    if start_time:
        params['startTime'] = int(start_time.timestamp() * 1000)
    if end_time:
        params['endTime'] = int(end_time.timestamp() * 1000)
    
    try:
        response = requests.get(BINANCE_BASE + endpoint, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"    ✗ Error fetching taker ratio: {e}")
        return None

def download_longshort_data(symbol):
    """Download all long/short ratio data for a symbol."""
    print(f"\n{'='*70}")
    print(f"Downloading Binance Long/Short Data: {symbol}")
    print(f"{'='*70}")
    
    symbol_dir = LOCAL_BASE / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    
    date_ranges = get_date_ranges()
    
    all_top_ls = []
    all_global_ls = []
    all_taker = []
    
    print(f"Fetching data for {len(date_ranges)} chunks...")
    
    for i, (start_time, end_time) in enumerate(date_ranges, 1):
        print(f"  [{i}/{len(date_ranges)}] {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}...")
        
        # Fetch top trader L/S ratio
        top_ls = fetch_longshort_ratio(symbol, period='1h', start_time=start_time, end_time=end_time)
        if top_ls:
            all_top_ls.extend(top_ls)
        
        # Fetch global L/S ratio
        global_ls = fetch_global_longshort_ratio(symbol, period='1h', start_time=start_time, end_time=end_time)
        if global_ls:
            all_global_ls.extend(global_ls)
        
        # Fetch taker buy/sell ratio
        taker = fetch_taker_buysell_volume(symbol, period='1h', start_time=start_time, end_time=end_time)
        if taker:
            all_taker.extend(taker)
        
        # Rate limiting
        time.sleep(0.3)
    
    # Save data
    if all_top_ls:
        output_file = symbol_dir / "binance_top_longshort_ratio.jsonl.gz"
        with gzip.open(output_file, 'wt') as f:
            for record in all_top_ls:
                record['source'] = 'binance'
                record['stream'] = 'top_longshort_ratio'
                f.write(json.dumps(record) + '\n')
        print(f"  ✓ Top L/S ratio: {len(all_top_ls)} records → {output_file.name}")
    
    if all_global_ls:
        output_file = symbol_dir / "binance_global_longshort_ratio.jsonl.gz"
        with gzip.open(output_file, 'wt') as f:
            for record in all_global_ls:
                record['source'] = 'binance'
                record['stream'] = 'global_longshort_ratio'
                f.write(json.dumps(record) + '\n')
        print(f"  ✓ Global L/S ratio: {len(all_global_ls)} records → {output_file.name}")
    
    if all_taker:
        output_file = symbol_dir / "binance_taker_buysell_ratio.jsonl.gz"
        with gzip.open(output_file, 'wt') as f:
            for record in all_taker:
                record['source'] = 'binance'
                record['stream'] = 'taker_buysell_ratio'
                f.write(json.dumps(record) + '\n')
        print(f"  ✓ Taker buy/sell: {len(all_taker)} records → {output_file.name}")
    
    print(f"{'='*70}")
    
    return len(all_top_ls), len(all_global_ls), len(all_taker)

def fetch_klines(symbol, interval='1h', start_time=None, end_time=None, limit=1500):
    """
    Fetch kline/candlestick data.
    
    Binance endpoint: /fapi/v1/klines
    Intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    """
    endpoint = "/fapi/v1/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    if start_time:
        params['startTime'] = int(start_time.timestamp() * 1000)
    if end_time:
        params['endTime'] = int(end_time.timestamp() * 1000)
    
    try:
        response = requests.get(BINANCE_BASE + endpoint, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"    ✗ Error fetching klines: {e}")
        return None

def download_klines(symbol, interval='1h'):
    """Download OHLCV kline data."""
    print(f"\n{'='*70}")
    print(f"Downloading Binance Klines ({interval}): {symbol}")
    print(f"{'='*70}")
    
    symbol_dir = LOCAL_BASE / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    
    start = datetime(2025, 5, 11, 0, 0, 0)
    end = datetime(2025, 8, 10, 23, 59, 59)
    
    all_klines = []
    current = start
    
    print(f"Fetching {interval} klines from {start} to {end}...")
    
    while current < end:
        chunk_end = min(current + timedelta(days=30), end)
        
        klines = fetch_klines(symbol, interval=interval, start_time=current, end_time=chunk_end, limit=1500)
        
        if klines:
            all_klines.extend(klines)
            print(f"  Fetched {len(klines)} klines up to {datetime.fromtimestamp(klines[-1][0]/1000)}")
        
        current = chunk_end
        time.sleep(0.3)
    
    # Save as CSV for easy analysis
    if all_klines:
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_base', 'taker_buy_quote']:
            df[col] = pd.to_numeric(df[col])
        
        output_file = symbol_dir / f"binance_klines_{interval}.csv.gz"
        df.to_csv(output_file, index=False, compression='gzip')
        print(f"  ✓ Saved {len(df)} klines → {output_file.name}")
    
    print(f"{'='*70}")
    
    return len(all_klines)

def compare_with_bybit(symbol):
    """Compare Binance data with Bybit data."""
    print(f"\n{'='*70}")
    print(f"Comparing Binance vs Bybit: {symbol}")
    print(f"{'='*70}")
    
    # Load Binance data
    binance_dir = LOCAL_BASE / symbol
    bybit_dir = Path("data") / symbol
    
    print(f"\nBinance data:")
    if (binance_dir / "binance_top_longshort_ratio.jsonl.gz").exists():
        with gzip.open(binance_dir / "binance_top_longshort_ratio.jsonl.gz", 'rt') as f:
            binance_top_ls = sum(1 for _ in f)
        print(f"  Top L/S ratio: {binance_top_ls} records")
    
    if (binance_dir / "binance_global_longshort_ratio.jsonl.gz").exists():
        with gzip.open(binance_dir / "binance_global_longshort_ratio.jsonl.gz", 'rt') as f:
            binance_global_ls = sum(1 for _ in f)
        print(f"  Global L/S ratio: {binance_global_ls} records")
    
    if (binance_dir / "binance_klines_1h.csv.gz").exists():
        df = pd.read_csv(binance_dir / "binance_klines_1h.csv.gz")
        print(f"  1h klines: {len(df)} records")
        print(f"  Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    print(f"\nBybit data:")
    bybit_ls_files = list(bybit_dir.glob("longshort_ratio_*.jsonl.gz"))
    if bybit_ls_files:
        total_records = 0
        for file in bybit_ls_files:
            with gzip.open(file, 'rt') as f:
                total_records += sum(1 for _ in f)
        print(f"  L/S ratio: {total_records} records")
    
    bybit_ticker_files = list(bybit_dir.glob("ticker_*.jsonl.gz"))
    if bybit_ticker_files:
        print(f"  Ticker files: {len(bybit_ticker_files)} files")
    
    print(f"{'='*70}")

def main():
    print("="*70)
    print("BINANCE FUTURES DATA DOWNLOAD")
    print("="*70)
    print(f"Period: May 11 - Aug 10, 2025")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Output: {LOCAL_BASE}/")
    print("="*70)
    
    results = {}
    
    for symbol in SYMBOLS:
        print(f"\n\nProcessing {symbol}...")
        
        # Download long/short ratio data
        top_ls, global_ls, taker = download_longshort_data(symbol)
        
        # Download 1h klines
        klines = download_klines(symbol, interval='1h')
        
        results[symbol] = {
            'top_ls': top_ls,
            'global_ls': global_ls,
            'taker': taker,
            'klines': klines
        }
        
        # Compare with Bybit
        compare_with_bybit(symbol)
        
        time.sleep(1)
    
    print(f"\n{'='*70}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    for symbol, stats in results.items():
        print(f"{symbol}:")
        print(f"  Top L/S: {stats['top_ls']} records")
        print(f"  Global L/S: {stats['global_ls']} records")
        print(f"  Taker: {stats['taker']} records")
        print(f"  Klines: {stats['klines']} records")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
