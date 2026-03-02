#!/usr/bin/env python3
"""
Threshold scan to find optimal z-score for 1-5 events/day target.
Tests multiple z-thresholds: 3, 5, 7, 10, 15, 20, 25, 30
"""
import gzip
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("data_bybit/SOLUSDT/trade/dataminer/data/archive/raw")
WINDOW_SECONDS = 30  # Use 30s window
Z_THRESHOLDS = [3, 5, 7, 10, 15, 20, 25, 30]  # Test range
OUTPUT_DIR = Path("results/flow_shock")

class MultiThresholdDetector:
    """Detector that tracks multiple z-thresholds simultaneously."""
    
    def __init__(self, window_seconds=30, z_thresholds=[3, 5, 10, 15, 20]):
        self.window_seconds = window_seconds
        self.window_ms = window_seconds * 1000
        self.z_thresholds = sorted(z_thresholds)
        self.trades = deque()
        
        # Events dict: {threshold: [events]}
        self.events = {z: [] for z in z_thresholds}
        self.total_trades = 0
        
    def add_trade(self, timestamp_ms, volume, price):
        """Add trade and check all thresholds."""
        self.total_trades += 1
        self.trades.append((timestamp_ms, volume, price))
        
        # Remove old trades
        cutoff = timestamp_ms - self.window_ms
        while self.trades and self.trades[0][0] < cutoff:
            self.trades.popleft()
        
        if len(self.trades) < 10:
            return
        
        # Calculate stats
        volumes = np.array([t[1] for t in self.trades])
        mean_vol = volumes.mean()
        std_vol = volumes.std()
        
        if std_vol == 0:
            return
        
        flow_z = (volume - mean_vol) / std_vol
        
        # Check each threshold
        for threshold in self.z_thresholds:
            if flow_z > threshold:
                self.events[threshold].append({
                    'timestamp': timestamp_ms,
                    'datetime': datetime.fromtimestamp(timestamp_ms / 1000).isoformat(),
                    'flow_z': flow_z,
                    'volume': volume,
                    'price': price
                })

def parse_trade_file_fast(filepath):
    """Fast parse."""
    trades = []
    try:
        with gzip.open(filepath, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'result' in data and 'data' in data['result']:
                        for trade in data['result']['data']:
                            trades.append((
                                int(trade['T']),
                                float(trade['v']),
                                float(trade['p'])
                            ))
                except:
                    continue
    except:
        pass
    return trades

def get_sample_files(start_date, end_date, sample_days=7):
    """Get sample files."""
    all_files = []
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    total_days = (end - start).days + 1
    step = max(1, total_days // sample_days)
    
    current = start
    sampled_dates = []
    while current <= end and len(sampled_dates) < sample_days:
        sampled_dates.append(current)
        current += timedelta(days=step)
    
    for date in sampled_dates:
        date_str = date.strftime('%Y-%m-%d')
        date_dir = DATA_DIR / f"dt={date_str}"
        
        if date_dir.exists():
            for hour in range(24):
                hour_file = date_dir / f"hr={hour:02d}" / "exchange=bybit" / "source=ws" / "market=linear" / "stream=trade" / "symbol=SOLUSDT" / "data.jsonl.gz"
                if hour_file.exists():
                    all_files.append(hour_file)
    
    return sorted(all_files), sampled_dates

def scan_thresholds():
    """Scan multiple z-thresholds."""
    print(f"{'='*80}")
    print(f"🔍 FLOW SHOCK THRESHOLD SCAN")
    print(f"{'='*80}")
    print(f"Window:         {WINDOW_SECONDS}s")
    print(f"Z-thresholds:   {Z_THRESHOLDS}")
    print(f"Target:         1-5 events/day")
    print(f"{'='*80}\n")
    
    # Get sample files
    print("📂 Loading sample data...")
    files, sampled_dates = get_sample_files('2025-05-11', '2025-08-10', sample_days=7)
    print(f"   Sampled: {len(files)} hours across {len(sampled_dates)} days\n")
    
    # Create detector
    detector = MultiThresholdDetector(WINDOW_SECONDS, Z_THRESHOLDS)
    
    # Process files
    print("🔄 Processing trades...")
    total_trades = 0
    
    for i, filepath in enumerate(files, 1):
        trades = parse_trade_file_fast(filepath)
        for ts, vol, price in trades:
            total_trades += 1
            detector.add_trade(ts, vol, price)
        
        if i % 20 == 0:
            print(f"   {i}/{len(files)} files ({total_trades:,} trades)...")
    
    print(f"\n✅ Processed {total_trades:,} trades\n")
    
    # Analyze results
    print(f"{'='*80}")
    print(f"📊 THRESHOLD SCAN RESULTS")
    print(f"{'='*80}\n")
    
    results = []
    
    for threshold in Z_THRESHOLDS:
        events = detector.events[threshold]
        
        if events:
            df = pd.DataFrame(events)
            df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
            df['date'] = df['datetime'].dt.date
            
            events_per_day = df.groupby('date').size()
            avg_per_day = events_per_day.mean()
            median_per_day = events_per_day.median()
            
            # Estimate for full 92 days
            est_total = len(events) * (92 / len(sampled_dates))
            est_per_day = est_total / 92
            
            # Check if in target range
            in_target = "✅" if 1 <= est_per_day <= 5 else "❌"
            
            results.append({
                'threshold': threshold,
                'sample_events': len(events),
                'sample_days': len(sampled_dates),
                'avg_per_day': avg_per_day,
                'median_per_day': median_per_day,
                'est_per_day': est_per_day,
                'est_total_92d': est_total,
                'in_target': in_target,
                'max_z': df['flow_z'].max(),
                'mean_z': df['flow_z'].mean()
            })
        else:
            results.append({
                'threshold': threshold,
                'sample_events': 0,
                'sample_days': len(sampled_dates),
                'avg_per_day': 0,
                'median_per_day': 0,
                'est_per_day': 0,
                'est_total_92d': 0,
                'in_target': '❌',
                'max_z': 0,
                'mean_z': 0
            })
    
    # Print table
    df_results = pd.DataFrame(results)
    
    print("| Z-Threshold | Sample Events | Est. Events/Day | Est. Total (92d) | Target? | Max Z | Mean Z |")
    print("|-------------|---------------|-----------------|------------------|---------|-------|--------|")
    
    for _, row in df_results.iterrows():
        print(f"| {row['threshold']:>11.0f} | {row['sample_events']:>13,} | {row['est_per_day']:>15.1f} | {row['est_total_92d']:>16.0f} | {row['in_target']:>7} | {row['max_z']:>5.1f} | {row['mean_z']:>6.1f} |")
    
    print(f"\n{'='*80}")
    
    # Find optimal threshold
    target_rows = df_results[(df_results['est_per_day'] >= 1) & (df_results['est_per_day'] <= 5)]
    
    if len(target_rows) > 0:
        optimal = target_rows.iloc[0]
        print(f"\n✅ OPTIMAL THRESHOLD FOUND: z > {optimal['threshold']:.0f}")
        print(f"   Expected: {optimal['est_per_day']:.1f} events/day ({optimal['est_total_92d']:.0f} total over 92 days)")
        print(f"   Mean flow_z: {optimal['mean_z']:.1f}, Max: {optimal['max_z']:.1f}")
    else:
        print(f"\n⚠️  No threshold in target range. Need to test higher z-values.")
        print(f"   Current range tested: {Z_THRESHOLDS[0]} - {Z_THRESHOLDS[-1]}")
        if df_results['est_per_day'].min() > 5:
            print(f"   Recommendation: Test z > {Z_THRESHOLDS[-1]}")
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "threshold_scan_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n💾 Results saved: {output_file}")
    
    return df_results

def main():
    results = scan_thresholds()
    return 0

if __name__ == "__main__":
    sys.exit(main())
