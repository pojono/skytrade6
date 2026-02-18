#!/usr/bin/env python3
"""
Download Long/Short Account Ratio data from Bybit API.
Covers May 11 - Aug 10, 2025 for BTCUSDT, ETHUSDT, SOLUSDT.
"""
import requests
import time
import json
import gzip
from datetime import datetime, timedelta
from pathlib import Path

API_BASE = "https://api.bybit.com"
ENDPOINT = "/v5/market/account-ratio"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
CATEGORY = "linear"
PERIOD = "1h"  # 1 hour resolution
LIMIT = 500  # Max per request

LOCAL_BASE = Path("data")

def get_date_ranges():
    """Split May 11 - Aug 10, 2025 into chunks to avoid API limits."""
    start = datetime(2025, 5, 11, 0, 0, 0)
    end = datetime(2025, 8, 10, 23, 59, 59)
    
    # Split into ~7 day chunks to stay under API limits
    ranges = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=7), end)
        ranges.append((current, chunk_end))
        current = chunk_end + timedelta(seconds=1)
    
    return ranges

def fetch_longshort_ratio(symbol, start_time, end_time):
    """Fetch long/short ratio data from Bybit API."""
    params = {
        'category': CATEGORY,
        'symbol': symbol,
        'period': PERIOD,
        'startTime': int(start_time.timestamp() * 1000),
        'endTime': int(end_time.timestamp() * 1000),
        'limit': LIMIT
    }
    
    all_data = []
    cursor = None
    
    while True:
        if cursor:
            params['cursor'] = cursor
        
        try:
            response = requests.get(API_BASE + ENDPOINT, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data['retCode'] != 0:
                print(f"    ✗ API error: {data['retMsg']}")
                break
            
            result = data['result']
            records = result.get('list', [])
            
            if not records:
                break
            
            all_data.extend(records)
            
            # Check for next page
            cursor = result.get('nextPageCursor')
            if not cursor or cursor == 'lastid%3D0%26lasttime%3D0':
                break
            
            # Rate limiting - be respectful
            time.sleep(0.2)
            
        except Exception as e:
            print(f"    ✗ Request failed: {e}")
            break
    
    return all_data

def save_data(symbol, data, output_file):
    """Save data as compressed JSONL."""
    if not data:
        return
    
    with gzip.open(output_file, 'wt') as f:
        for record in data:
            # Add metadata
            record['symbol'] = symbol
            record['source'] = 'bybit_api'
            record['stream'] = 'account_ratio'
            f.write(json.dumps(record) + '\n')

def download_symbol_data(symbol):
    """Download all long/short ratio data for a symbol."""
    print(f"\n{'='*70}")
    print(f"Downloading Long/Short Ratio: {symbol}")
    print(f"{'='*70}")
    
    symbol_dir = LOCAL_BASE / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    
    date_ranges = get_date_ranges()
    total_records = 0
    
    print(f"Date ranges to fetch: {len(date_ranges)}")
    
    for i, (start_time, end_time) in enumerate(date_ranges, 1):
        start_str = start_time.strftime("%Y-%m-%d")
        end_str = end_time.strftime("%Y-%m-%d")
        
        print(f"\n[{i}/{len(date_ranges)}] {start_str} to {end_str}")
        
        # Check if already downloaded
        output_file = symbol_dir / f"longshort_ratio_{start_str}_to_{end_str}.jsonl.gz"
        if output_file.exists():
            # Count existing records
            with gzip.open(output_file, 'rt') as f:
                existing_count = sum(1 for _ in f)
            print(f"  ✓ Already exists: {existing_count} records")
            total_records += existing_count
            continue
        
        # Fetch data
        print(f"  Fetching from API...")
        data = fetch_longshort_ratio(symbol, start_time, end_time)
        
        if data:
            save_data(symbol, data, output_file)
            print(f"  ✓ Downloaded: {len(data)} records → {output_file.name}")
            total_records += len(data)
        else:
            print(f"  ✗ No data returned")
        
        # Rate limiting between chunks
        time.sleep(0.5)
    
    print(f"\n{'='*70}")
    print(f"Summary for {symbol}:")
    print(f"  Total records: {total_records}")
    print(f"  Expected records: ~{92 * 24} (92 days × 24 hours)")
    print(f"  Coverage: {100 * total_records / (92 * 24):.1f}%")
    print(f"{'='*70}")
    
    return total_records

def verify_data(symbol):
    """Verify downloaded data."""
    symbol_dir = LOCAL_BASE / symbol
    files = sorted(symbol_dir.glob("longshort_ratio_*.jsonl.gz"))
    
    if not files:
        return 0
    
    total_records = 0
    timestamps = []
    
    for file in files:
        with gzip.open(file, 'rt') as f:
            for line in f:
                record = json.loads(line)
                total_records += 1
                timestamps.append(int(record['timestamp']))
    
    if timestamps:
        timestamps.sort()
        first_ts = datetime.fromtimestamp(timestamps[0] / 1000)
        last_ts = datetime.fromtimestamp(timestamps[-1] / 1000)
        
        print(f"\n{symbol} Data Verification:")
        print(f"  Total records: {total_records}")
        print(f"  First timestamp: {first_ts}")
        print(f"  Last timestamp: {last_ts}")
        print(f"  Time span: {(last_ts - first_ts).days} days")
        
        # Sample data
        with gzip.open(files[0], 'rt') as f:
            sample = json.loads(f.readline())
            print(f"  Sample data:")
            print(f"    Buy ratio: {sample['buyRatio']}")
            print(f"    Sell ratio: {sample['sellRatio']}")
            print(f"    Timestamp: {datetime.fromtimestamp(int(sample['timestamp'])/1000)}")
    
    return total_records

def main():
    print("="*70)
    print("BYBIT LONG/SHORT ACCOUNT RATIO DOWNLOAD")
    print("="*70)
    print(f"API: {API_BASE}{ENDPOINT}")
    print(f"Period: May 11 - Aug 10, 2025")
    print(f"Resolution: 1 hour")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Output: {LOCAL_BASE}/SYMBOL/")
    print("="*70)
    
    results = {}
    
    for symbol in SYMBOLS:
        try:
            records = download_symbol_data(symbol)
            results[symbol] = records
        except Exception as e:
            print(f"\n✗ Error downloading {symbol}: {e}")
            results[symbol] = 0
    
    print(f"\n{'='*70}")
    print("VERIFICATION")
    print(f"{'='*70}")
    
    for symbol in SYMBOLS:
        verify_data(symbol)
    
    print(f"\n{'='*70}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    for symbol, count in results.items():
        print(f"  {symbol}: {count} records")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
