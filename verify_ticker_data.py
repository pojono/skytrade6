#!/usr/bin/env python3
"""
Verify downloaded ticker data coverage and integrity.
"""
import gzip
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

LOCAL_BASE = Path("data")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

def get_expected_dates():
    """Get all expected dates (May 11 - Aug 10, 2025)."""
    start = datetime(2025, 5, 11)
    end = datetime(2025, 8, 10)
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates

def verify_symbol(symbol):
    """Verify data for a symbol."""
    print(f"\n{'='*70}")
    print(f"Verifying {symbol}")
    print(f"{'='*70}")
    
    symbol_dir = LOCAL_BASE / symbol
    expected_dates = get_expected_dates()
    
    # Find all ticker files
    ticker_files = sorted(symbol_dir.glob("ticker_*.jsonl.gz"))
    
    # Group by date
    date_coverage = defaultdict(list)
    for file in ticker_files:
        # Extract date from filename: ticker_2025-06-15_hr12.jsonl.gz
        parts = file.name.replace(".jsonl.gz", "").split("_")
        if len(parts) >= 3:
            date = parts[1]
            hour = parts[2].replace("hr", "")
            date_coverage[date].append(int(hour))
    
    # Check coverage
    print(f"\nDate Coverage:")
    print(f"  Expected dates: {len(expected_dates)}")
    print(f"  Dates with data: {len(date_coverage)}")
    
    missing_dates = []
    partial_dates = []
    full_dates = 0
    
    for date in expected_dates:
        if date not in date_coverage:
            missing_dates.append(date)
        elif len(date_coverage[date]) < 24:
            partial_dates.append((date, len(date_coverage[date])))
        else:
            full_dates += 1
    
    print(f"  Complete days (24 hours): {full_dates}")
    print(f"  Partial days: {len(partial_dates)}")
    print(f"  Missing days: {len(missing_dates)}")
    
    if partial_dates:
        print(f"\n  Partial days detail:")
        for date, hours in partial_dates[:5]:
            print(f"    {date}: {hours}/24 hours")
        if len(partial_dates) > 5:
            print(f"    ... and {len(partial_dates)-5} more")
    
    if missing_dates:
        print(f"\n  Missing dates: {', '.join(missing_dates[:10])}")
        if len(missing_dates) > 10:
            print(f"    ... and {len(missing_dates)-10} more")
    
    # Sample data verification
    print(f"\nData Quality Check:")
    sample_file = ticker_files[len(ticker_files)//2] if ticker_files else None
    
    if sample_file:
        print(f"  Sample file: {sample_file.name}")
        try:
            with gzip.open(sample_file, 'rt') as f:
                lines = f.readlines()
                print(f"  Lines in file: {len(lines)}")
                
                # Parse first and last line
                first = json.loads(lines[0])
                last = json.loads(lines[-1])
                
                # Extract key fields
                first_data = first['result']['list'][0]
                last_data = last['result']['list'][0]
                
                print(f"  First timestamp: {datetime.fromtimestamp(first['ts']/1000)}")
                print(f"  Last timestamp:  {datetime.fromtimestamp(last['ts']/1000)}")
                print(f"  Sample data fields:")
                print(f"    Open Interest: {first_data['openInterest']}")
                print(f"    Funding Rate: {first_data['fundingRate']}")
                print(f"    Last Price: {first_data['lastPrice']}")
                print(f"    Mark Price: {first_data['markPrice']}")
                print(f"    Volume 24h: {first_data['volume24h']}")
                print(f"  ✓ Data format valid")
        except Exception as e:
            print(f"  ✗ Error reading sample: {e}")
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in ticker_files)
    print(f"\nStorage:")
    print(f"  Total files: {len(ticker_files)}")
    print(f"  Total size: {total_size / (1024**2):.1f} MB")
    print(f"  Avg per file: {total_size / len(ticker_files) / 1024:.1f} KB")
    
    return {
        'total_files': len(ticker_files),
        'full_dates': full_dates,
        'partial_dates': len(partial_dates),
        'missing_dates': len(missing_dates),
        'total_size_mb': total_size / (1024**2)
    }

def main():
    print("="*70)
    print("TICKER DATA VERIFICATION")
    print("="*70)
    print(f"Expected period: 2025-05-11 to 2025-08-10 (92 days)")
    print(f"Expected files per symbol: ~2208 (92 days × 24 hours)")
    print("="*70)
    
    results = {}
    for symbol in SYMBOLS:
        results[symbol] = verify_symbol(symbol)
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for symbol, stats in results.items():
        print(f"{symbol}:")
        print(f"  Files: {stats['total_files']}")
        print(f"  Complete days: {stats['full_dates']}/92")
        print(f"  Size: {stats['total_size_mb']:.1f} MB")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
