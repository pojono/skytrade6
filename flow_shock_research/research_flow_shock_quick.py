#!/usr/bin/env python3
"""
Quick Flow Shock Analysis - Sample-based for fast iteration.
Processes 1 week of data to get quick insights on event frequency.
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
WINDOW_SECONDS = [5, 10, 15, 30]
FLOW_Z_THRESHOLD = 3.0
OUTPUT_DIR = Path("results/flow_shock")

class FlowShockDetector:
    """Efficient streaming detector for volume flow shocks."""
    
    def __init__(self, window_seconds=30, z_threshold=3.0):
        self.window_seconds = window_seconds
        self.z_threshold = z_threshold
        self.window_ms = window_seconds * 1000
        self.trades = deque()
        self.events = []
        self.total_trades = 0
        
    def add_trade(self, timestamp_ms, volume, price):
        """Add a trade and check for flow shock."""
        self.total_trades += 1
        self.trades.append((timestamp_ms, volume, price))
        
        # Remove trades outside window
        cutoff = timestamp_ms - self.window_ms
        while self.trades and self.trades[0][0] < cutoff:
            self.trades.popleft()
        
        if len(self.trades) < 10:
            return None
        
        # Calculate volume statistics
        volumes = np.array([t[1] for t in self.trades])
        mean_vol = volumes.mean()
        std_vol = volumes.std()
        
        if std_vol == 0:
            return None
        
        # Z-score for current volume
        flow_z = (volume - mean_vol) / std_vol
        
        # Detect shock
        if flow_z > self.z_threshold:
            event = {
                'timestamp': timestamp_ms,
                'datetime': datetime.fromtimestamp(timestamp_ms / 1000).isoformat(),
                'flow_z': flow_z,
                'volume': volume,
                'price': price,
                'mean_vol': mean_vol,
                'std_vol': std_vol,
                'window_trades': len(self.trades),
                'window_seconds': self.window_seconds
            }
            self.events.append(event)
            return event
        
        return None

def parse_trade_file_fast(filepath):
    """Fast parse - extract all trades at once."""
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
    """Get sample of files evenly distributed across date range."""
    all_files = []
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    total_days = (end - start).days + 1
    
    # Sample evenly across the range
    step = max(1, total_days // sample_days)
    
    current = start
    sampled_dates = []
    while current <= end and len(sampled_dates) < sample_days:
        sampled_dates.append(current)
        current += timedelta(days=step)
    
    # Get all hour files for sampled dates
    for date in sampled_dates:
        date_str = date.strftime('%Y-%m-%d')
        date_dir = DATA_DIR / f"dt={date_str}"
        
        if date_dir.exists():
            for hour in range(24):
                hour_file = date_dir / f"hr={hour:02d}" / "exchange=bybit" / "source=ws" / "market=linear" / "stream=trade" / "symbol=SOLUSDT" / "data.jsonl.gz"
                if hour_file.exists():
                    all_files.append(hour_file)
    
    return sorted(all_files), sampled_dates

def analyze_quick(start_date='2025-05-11', end_date='2025-08-10', sample_days=7):
    """Quick analysis on sample data."""
    print(f"{'='*80}")
    print(f"🔍 FLOW SHOCK DETECTOR - Quick Sample Analysis")
    print(f"{'='*80}")
    print(f"Full range:     {start_date} to {end_date}")
    print(f"Sample size:    {sample_days} days (evenly distributed)")
    print(f"Window sizes:   {WINDOW_SECONDS} seconds")
    print(f"Z-threshold:    {FLOW_Z_THRESHOLD} σ")
    print(f"{'='*80}\n")
    
    # Get sample files
    print("📂 Selecting sample files...")
    files, sampled_dates = get_sample_files(start_date, end_date, sample_days)
    print(f"   Sampled dates: {', '.join([d.strftime('%Y-%m-%d') for d in sampled_dates])}")
    print(f"   Total files: {len(files)} hours\n")
    
    if not files:
        print("❌ No trade data files found!")
        return
    
    # Create detectors
    detectors = {w: FlowShockDetector(w, FLOW_Z_THRESHOLD) for w in WINDOW_SECONDS}
    
    # Process files
    print("🔄 Processing trade data...")
    total_trades = 0
    
    for i, filepath in enumerate(files, 1):
        trades = parse_trade_file_fast(filepath)
        
        for ts, vol, price in trades:
            total_trades += 1
            for detector in detectors.values():
                detector.add_trade(ts, vol, price)
        
        if i % 10 == 0:
            print(f"   Processed {i}/{len(files)} files ({total_trades:,} trades)...")
    
    print(f"\n✅ Processing complete!")
    print(f"   Total trades: {total_trades:,}\n")
    
    # Analyze results
    print(f"{'='*80}")
    print(f"📊 FLOW SHOCK EVENTS DETECTED")
    print(f"{'='*80}\n")
    
    all_results = {}
    
    for window in WINDOW_SECONDS:
        detector = detectors[window]
        events = detector.events
        
        print(f"Window: {window}s")
        print(f"  Total events:     {len(events)}")
        
        if events:
            df = pd.DataFrame(events)
            df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
            df['date'] = df['datetime'].dt.date
            df['hour'] = df['datetime'].dt.hour
            
            # Events per day
            events_per_day = df.groupby('date').size()
            print(f"  Events per day:   {events_per_day.mean():.1f} avg, {events_per_day.median():.0f} median")
            print(f"  Range:            {events_per_day.min()}-{events_per_day.max()} events/day")
            
            # Extrapolate to full period
            total_days = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days + 1
            estimated_total = len(events) * (total_days / sample_days)
            print(f"  Estimated total:  ~{estimated_total:.0f} events over {total_days} days")
            
            # Hourly distribution
            hourly = df.groupby('hour').size()
            top_hours = hourly.nlargest(5)
            print(f"  Top hours (UTC):  {', '.join([f'{h}:00 ({c})' for h, c in top_hours.items()])}")
            
            # Flow z statistics
            print(f"  Flow Z range:     {df['flow_z'].min():.1f} - {df['flow_z'].max():.1f}")
            print(f"  Avg flow Z:       {df['flow_z'].mean():.1f}")
            
            # Time between events
            df_sorted = df.sort_values('timestamp')
            time_diffs = df_sorted['timestamp'].diff() / 1000 / 60
            print(f"  Avg time between: {time_diffs.mean():.1f} minutes")
            print(f"  Median spacing:   {time_diffs.median():.1f} minutes")
            
            # Target range analysis
            target_days = events_per_day[(events_per_day >= 1) & (events_per_day <= 5)]
            print(f"  Days with 1-5:    {len(target_days)}/{len(events_per_day)} ({len(target_days)/len(events_per_day)*100:.0f}%)")
            
            all_results[window] = {
                'events': events,
                'df': df,
                'events_per_day': events_per_day
            }
        
        print()
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for window, results in all_results.items():
        df = results['df']
        output_file = OUTPUT_DIR / f"flow_shock_sample_{window}s.csv"
        df.to_csv(output_file, index=False)
        print(f"💾 Saved: {output_file}")
    
    # Generate quick report
    report_file = OUTPUT_DIR / "FINDINGS_flow_shock_quick.md"
    with open(report_file, 'w') as f:
        f.write("# Flow Shock Detector - Quick Sample Analysis\n\n")
        f.write(f"**Sample Period:** {sample_days} days from {start_date} to {end_date}\n")
        f.write(f"**Sampled Dates:** {', '.join([d.strftime('%Y-%m-%d') for d in sampled_dates])}\n")
        f.write(f"**Total Trades:** {total_trades:,}\n")
        f.write(f"**Detection Threshold:** flow_z > {FLOW_Z_THRESHOLD}σ\n\n")
        
        f.write("## Results Summary\n\n")
        f.write("| Window | Events | Avg/Day | Median/Day | Est. Total (92d) | Target Range % |\n")
        f.write("|--------|--------|---------|------------|------------------|----------------|\n")
        
        for window in WINDOW_SECONDS:
            if window in all_results:
                df = all_results[window]['df']
                epd = all_results[window]['events_per_day']
                total_days = 92  # Full period
                est_total = len(df) * (total_days / sample_days)
                target_pct = len(epd[(epd >= 1) & (epd <= 5)]) / len(epd) * 100
                f.write(f"| {window}s | {len(df)} | {epd.mean():.1f} | {epd.median():.0f} | {est_total:.0f} | {target_pct:.0f}% |\n")
        
        f.write("\n## Interpretation\n\n")
        f.write("**Target:** 1-5 high-quality events per day\n\n")
        f.write("- Shorter windows (5-10s) detect more frequent spikes\n")
        f.write("- Longer windows (30s) filter for sustained volume shocks\n")
        f.write("- Adjust z-threshold or window size to hit target range\n\n")
        f.write("**Next Steps:**\n")
        f.write("1. Analyze price behavior around detected events\n")
        f.write("2. Test different z-thresholds (2.5, 3.5, 4.0)\n")
        f.write("3. Add directional filters (buy vs sell aggression)\n")
        f.write("4. Validate on full dataset\n")
    
    print(f"📄 Report saved: {report_file}\n")
    print(f"{'='*80}")
    print("✅ Quick analysis complete!")
    print(f"{'='*80}")
    
    return all_results

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Quick Flow Shock Analysis')
    parser.add_argument('--start-date', type=str, default='2025-05-11')
    parser.add_argument('--end-date', type=str, default='2025-08-10')
    parser.add_argument('--sample-days', type=int, default=7, help='Number of days to sample')
    args = parser.parse_args()
    
    results = analyze_quick(args.start_date, args.end_date, args.sample_days)
    
    if results:
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
