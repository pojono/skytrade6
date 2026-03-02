#!/usr/bin/env python3
"""
Quick check: which dates have BOTH trade and orderbook data available.
"""
from pathlib import Path
from datetime import datetime, timedelta

TRADE_DIR = Path("/home/ubuntu/Projects/skytrade6/data_bybit/SOLUSDT/trade/dataminer/data/archive/raw")
OB_DIR = Path("/home/ubuntu/Projects/skytrade6/flow_shock_research/data_bybit/SOLUSDT/orderbook/dataminer/data/archive/raw")

def check_availability():
    start = datetime(2025, 5, 11)
    end = datetime(2025, 8, 10)
    
    dates_with_both = []
    dates_trade_only = []
    dates_ob_only = []
    dates_missing = []
    
    current = start
    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        
        trade_dir = TRADE_DIR / f"dt={date_str}"
        ob_dir = OB_DIR / f"dt={date_str}"
        
        has_trade = trade_dir.exists()
        has_ob = ob_dir.exists()
        
        if has_trade and has_ob:
            dates_with_both.append(date_str)
        elif has_trade:
            dates_trade_only.append(date_str)
        elif has_ob:
            dates_ob_only.append(date_str)
        else:
            dates_missing.append(date_str)
        
        current += timedelta(days=1)
    
    print("="*80)
    print("DATA AVAILABILITY CHECK")
    print("="*80)
    print(f"\n✅ Dates with BOTH trade + orderbook: {len(dates_with_both)}")
    if dates_with_both:
        print(f"   First: {dates_with_both[0]}")
        print(f"   Last:  {dates_with_both[-1]}")
        print(f"   Dates: {', '.join(dates_with_both[:5])}...")
    
    print(f"\n⚠️  Dates with ONLY trades: {len(dates_trade_only)}")
    if dates_trade_only:
        print(f"   {', '.join(dates_trade_only[:10])}...")
    
    print(f"\n⚠️  Dates with ONLY orderbook: {len(dates_ob_only)}")
    if dates_ob_only:
        print(f"   {', '.join(dates_ob_only[:10])}...")
    
    print(f"\n❌ Dates with NO data: {len(dates_missing)}")
    
    print(f"\n{'='*80}")
    print(f"RECOMMENDATION: Use {len(dates_with_both)} dates with complete data")
    print(f"{'='*80}")
    
    # Save list of valid dates
    output_file = Path("flow_shock_research/valid_dates.txt")
    with open(output_file, 'w') as f:
        for date in dates_with_both:
            f.write(f"{date}\n")
    print(f"\n💾 Valid dates saved to: {output_file}")
    
    return dates_with_both

if __name__ == "__main__":
    valid_dates = check_availability()
