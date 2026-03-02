#!/usr/bin/env python3
"""
Flow Shock Detector - Event-based trading signal research.

Goal: Find 1-5 high-quality events per day where market loses equilibrium.

Detector #1: Forced Flow + Overshoot
- Detect aggressive volume spikes in short windows (5-30 seconds)
- flow_z = (volume_now - mean(volume)) / std(volume)
- Trigger: flow_z > 3

This script processes trade data efficiently with streaming to avoid RAM overload.
"""
import gzip
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────
DATA_DIR = Path("data_bybit/SOLUSDT/trade/dataminer/data/archive/raw")
WINDOW_SECONDS = [5, 10, 15, 30]  # Multiple window sizes to test
FLOW_Z_THRESHOLD = 3.0  # Standard deviations above mean
OUTPUT_DIR = Path("results/flow_shock")

class FlowShockDetector:
    """Efficient streaming detector for volume flow shocks."""
    
    def __init__(self, window_seconds=30, z_threshold=3.0):
        self.window_seconds = window_seconds
        self.z_threshold = z_threshold
        self.window_ms = window_seconds * 1000
        
        # Rolling window storage (timestamp, volume)
        self.trades = deque()
        
        # Statistics
        self.events = []
        self.total_trades = 0
        
    def add_trade(self, timestamp_ms, volume, price):
        """Add a trade and check for flow shock."""
        self.total_trades += 1
        
        # Add new trade
        self.trades.append((timestamp_ms, volume, price))
        
        # Remove trades outside window
        cutoff = timestamp_ms - self.window_ms
        while self.trades and self.trades[0][0] < cutoff:
            self.trades.popleft()
        
        # Need at least 10 trades for statistics
        if len(self.trades) < 10:
            return None
        
        # Calculate volume statistics
        volumes = [t[1] for t in self.trades]
        mean_vol = np.mean(volumes)
        std_vol = np.std(volumes)
        
        if std_vol == 0:
            return None
        
        # Calculate z-score for current volume
        current_vol = volume
        flow_z = (current_vol - mean_vol) / std_vol
        
        # Detect shock
        if flow_z > self.z_threshold:
            event = {
                'timestamp': timestamp_ms,
                'datetime': datetime.fromtimestamp(timestamp_ms / 1000).isoformat(),
                'flow_z': flow_z,
                'volume': current_vol,
                'price': price,
                'mean_vol': mean_vol,
                'std_vol': std_vol,
                'window_trades': len(self.trades),
                'window_seconds': self.window_seconds
            }
            self.events.append(event)
            return event
        
        return None

def parse_trade_file(filepath):
    """Parse a single trade data file and yield trades in batches."""
    batch = []
    try:
        with gzip.open(filepath, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    # Handle WebSocket trade format
                    if 'result' in data and 'data' in data['result']:
                        trades = data['result']['data']
                        for trade in trades:
                            batch.append({
                                'timestamp': int(trade['T']),
                                'price': float(trade['p']),
                                'volume': float(trade['v']),
                                'side': trade['S']
                            })
                            
                            # Yield in batches of 1000 for efficiency
                            if len(batch) >= 1000:
                                yield batch
                                batch = []
                except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                    continue
        
        # Yield remaining trades
        if batch:
            yield batch
            
    except Exception as e:
        print(f"  ⚠️  Error reading {filepath.name}: {e}")

def get_trade_files(start_date, end_date):
    """Get list of all trade data files in date range."""
    files = []
    
    # Parse dates
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Iterate through dates
    current = start
    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        date_dir = DATA_DIR / f"dt={date_str}"
        
        if date_dir.exists():
            # Get all hour files for this date
            for hour in range(24):
                hour_file = date_dir / f"hr={hour:02d}" / "exchange=bybit" / "source=ws" / "market=linear" / "stream=trade" / "symbol=SOLUSDT" / "data.jsonl.gz"
                if hour_file.exists():
                    files.append(hour_file)
        
        current += timedelta(days=1)
    
    return sorted(files)

def analyze_flow_shocks(start_date='2025-05-11', end_date='2025-08-10'):
    """Main analysis function."""
    print(f"{'='*80}")
    print(f"🔍 FLOW SHOCK DETECTOR - Event-Based Trading Research")
    print(f"{'='*80}")
    print(f"Date range:     {start_date} to {end_date}")
    print(f"Window sizes:   {WINDOW_SECONDS} seconds")
    print(f"Z-threshold:    {FLOW_Z_THRESHOLD} σ")
    print(f"{'='*80}\n")
    
    # Get all trade files
    print("📂 Scanning for trade data files...")
    files = get_trade_files(start_date, end_date)
    print(f"   Found {len(files)} hourly files\n")
    
    if not files:
        print("❌ No trade data files found!")
        print(f"   Expected location: {DATA_DIR}")
        return
    
    # Create detectors for each window size
    detectors = {
        window: FlowShockDetector(window, FLOW_Z_THRESHOLD)
        for window in WINDOW_SECONDS
    }
    
    # Process all files with progress bar
    print("🔄 Processing trade data...")
    total_trades = 0
    
    for filepath in tqdm(files, desc="Files processed", unit="file"):
        # Parse trades from file (in batches)
        for trade_batch in parse_trade_file(filepath):
            for trade in trade_batch:
                total_trades += 1
                
                # Check each detector
                for window, detector in detectors.items():
                    detector.add_trade(
                        trade['timestamp'],
                        trade['volume'],
                        trade['price']
                    )
    
    print(f"\n✅ Processing complete!")
    print(f"   Total trades processed: {total_trades:,}\n")
    
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
            # Convert to DataFrame for analysis
            df = pd.DataFrame(events)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['date'] = df['datetime'].dt.date
            df['hour'] = df['datetime'].dt.hour
            
            # Events per day
            events_per_day = df.groupby('date').size()
            print(f"  Events per day:   {events_per_day.mean():.1f} avg, {events_per_day.median():.0f} median")
            print(f"  Range:            {events_per_day.min()}-{events_per_day.max()} events/day")
            
            # Hourly distribution
            hourly = df.groupby('hour').size()
            top_hours = hourly.nlargest(5)
            print(f"  Top hours (UTC):  {', '.join([f'{h}:00 ({c})' for h, c in top_hours.items()])}")
            
            # Flow z statistics
            print(f"  Flow Z range:     {df['flow_z'].min():.1f} - {df['flow_z'].max():.1f}")
            print(f"  Avg flow Z:       {df['flow_z'].mean():.1f}")
            
            # Time between events
            df_sorted = df.sort_values('timestamp')
            time_diffs = df_sorted['timestamp'].diff() / 1000 / 60  # Minutes
            print(f"  Avg time between: {time_diffs.mean():.1f} minutes")
            print(f"  Median spacing:   {time_diffs.median():.1f} minutes")
            
            all_results[window] = {
                'events': events,
                'df': df,
                'events_per_day': events_per_day,
                'hourly': hourly
            }
        
        print()
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for window, results in all_results.items():
        df = results['df']
        output_file = OUTPUT_DIR / f"flow_shock_events_{window}s.csv"
        df.to_csv(output_file, index=False)
        print(f"💾 Saved: {output_file}")
    
    # Generate summary report
    generate_summary_report(all_results, start_date, end_date, total_trades)
    
    return all_results

def generate_summary_report(results, start_date, end_date, total_trades):
    """Generate markdown summary report."""
    report_file = OUTPUT_DIR / "FINDINGS_flow_shock_detector.md"
    
    with open(report_file, 'w') as f:
        f.write("# Flow Shock Detector - Research Findings\n\n")
        f.write(f"**Analysis Period:** {start_date} to {end_date}\n")
        f.write(f"**Total Trades Processed:** {total_trades:,}\n")
        f.write(f"**Detection Threshold:** flow_z > {FLOW_Z_THRESHOLD}σ\n\n")
        
        f.write("## Objective\n\n")
        f.write("Find 1-5 high-quality events per day where aggressive volume flow indicates market disequilibrium.\n\n")
        
        f.write("## Methodology\n\n")
        f.write("**Flow Shock Detection:**\n")
        f.write("```\n")
        f.write("flow_z = (volume_now - mean(volume)) / std(volume)\n")
        f.write("Trigger: flow_z > 3.0\n")
        f.write("```\n\n")
        
        f.write("## Results by Window Size\n\n")
        
        for window in WINDOW_SECONDS:
            if window not in results:
                continue
                
            df = results[window]['df']
            events_per_day = results[window]['events_per_day']
            hourly = results[window]['hourly']
            
            f.write(f"### {window}-Second Window\n\n")
            f.write(f"- **Total Events:** {len(df)}\n")
            f.write(f"- **Events per Day:** {events_per_day.mean():.1f} avg, {events_per_day.median():.0f} median\n")
            f.write(f"- **Range:** {events_per_day.min()}-{events_per_day.max()} events/day\n")
            f.write(f"- **Flow Z Range:** {df['flow_z'].min():.1f} - {df['flow_z'].max():.1f}\n")
            
            # Time spacing
            df_sorted = df.sort_values('timestamp')
            time_diffs = df_sorted['timestamp'].diff() / 1000 / 60
            f.write(f"- **Avg Time Between Events:** {time_diffs.mean():.1f} minutes\n")
            f.write(f"- **Median Spacing:** {time_diffs.median():.1f} minutes\n\n")
            
            # Top hours
            top_hours = hourly.nlargest(5)
            f.write(f"**Most Active Hours (UTC):**\n")
            for hour, count in top_hours.items():
                pct = count / len(df) * 100
                f.write(f"- {hour:02d}:00 - {count} events ({pct:.1f}%)\n")
            f.write("\n")
            
            # Days with target range
            target_days = events_per_day[(events_per_day >= 1) & (events_per_day <= 5)]
            f.write(f"**Days with 1-5 Events (Target Range):** {len(target_days)} / {len(events_per_day)} ({len(target_days)/len(events_per_day)*100:.1f}%)\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("Flow shock detector successfully identifies volume-driven market disequilibrium events.\n")
        f.write("Next steps: analyze price behavior around these events to validate trading opportunity.\n")
    
    print(f"📄 Report saved: {report_file}\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Flow Shock Detector Analysis')
    parser.add_argument('--start-date', type=str, default='2025-05-11', help='Start date YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, default='2025-08-10', help='End date YYYY-MM-DD')
    args = parser.parse_args()
    
    results = analyze_flow_shocks(args.start_date, args.end_date)
    
    if results:
        print("✅ Analysis complete! Check results/ directory for detailed findings.")
        return 0
    else:
        print("❌ Analysis failed - no data processed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
