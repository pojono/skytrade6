#!/usr/bin/env python3
"""
OOS Validation: Run Flow Pressure detector on 2 samples from different months.

Sample 1: 2025-05-18 to 2025-05-24 (7 days, in-sample period)
Sample 2: 2026-01-15 to 2026-01-21 (7 days, OOS - 8 months later)

Goal: Validate that hour 7 dominance holds OOS.
"""
import subprocess
import sys
from pathlib import Path

SAMPLES = [
    {
        'name': 'Sample 1 (May 2025)',
        'start': '2025-05-18',
        'end': '2025-05-24',
        'days': 7
    },
    {
        'name': 'Sample 2 (Jan 2026 - OOS)',
        'start': '2026-01-15',
        'end': '2026-01-21',
        'days': 7
    }
]

def download_sample_data(sample):
    """Download trade and orderbook data for a sample."""
    print("="*80)
    print(f"📥 DOWNLOADING: {sample['name']}")
    print(f"   Dates: {sample['start']} to {sample['end']}")
    print("="*80)
    
    # Download trades
    print("\n1️⃣  Downloading trade data...")
    cmd_trade = f"python3 download_trade_data.py --symbol SOLUSDT --start-date {sample['start']} --end-date {sample['end']}"
    result = subprocess.run(cmd_trade, shell=True, cwd=".")
    
    if result.returncode != 0:
        print(f"❌ Trade download failed")
        return False
    
    # Download orderbook
    print("\n2️⃣  Downloading orderbook data...")
    cmd_ob = f"python3 download_orderbook_data.py --symbol SOLUSDT --start-date {sample['start']} --end-date {sample['end']}"
    result = subprocess.run(cmd_ob, shell=True, cwd=".")
    
    if result.returncode != 0:
        print(f"❌ Orderbook download failed")
        return False
    
    print(f"\n✅ {sample['name']} data downloaded\n")
    return True

def run_detector_on_sample(sample):
    """Run Flow Pressure detector on a sample."""
    print("="*80)
    print(f"🔬 RUNNING DETECTOR: {sample['name']}")
    print("="*80)
    
    # Modify detector to use specific dates
    cmd = f"python3 research_flow_pressure_v3.py --days {sample['days']} --start-date {sample['start']}"
    
    # For now, just note that we need to modify the detector script
    print(f"\n⚠️  Need to modify research_flow_pressure_v3.py to accept date range")
    print(f"   For now, manually check dates in valid_dates.txt")
    
    return True

def main():
    print("="*80)
    print("🎯 OOS VALIDATION - 2 SAMPLES FROM DIFFERENT MONTHS")
    print("="*80)
    print("\nGoal: Validate hour 7 dominance holds out-of-sample\n")
    
    for i, sample in enumerate(SAMPLES, 1):
        print(f"\n{'='*80}")
        print(f"SAMPLE {i}: {sample['name']}")
        print(f"{'='*80}\n")
        
        # Download data
        if not download_sample_data(sample):
            print(f"❌ Failed to download {sample['name']}")
            return 1
        
        # Run detector (will implement after download)
        # run_detector_on_sample(sample)
    
    print("\n" + "="*80)
    print("✅ DATA DOWNLOAD COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Check data availability with check_data_availability.py")
    print("2. Run Flow Pressure detector on each sample")
    print("3. Compare hourly distribution between samples")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
