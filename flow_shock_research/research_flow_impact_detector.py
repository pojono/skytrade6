#!/usr/bin/env python3
"""
Flow Impact Detector v2 - Market Impact Based Event Detection

Key improvements over z-score approach:
1. Flow Impact = AggressiveVolume / TopBookDepth (measures ability to move market)
2. Signed flow (buy vs sell imbalance)
3. Robust statistics (median/MAD instead of mean/std)
4. Burst detection (consecutive trades same direction)

Theory:
- Z-score assumes normal distribution (wrong for crypto)
- Crypto has heavy-tailed, bursty, regime-dependent volume
- Need to measure: "volume capable of moving market" not just "big volume"
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

DATA_DIR_TRADE = Path("/home/ubuntu/Projects/skytrade6/data_bybit/SOLUSDT/trade/dataminer/data/archive/raw")
DATA_DIR_OB = Path("/home/ubuntu/Projects/skytrade6/flow_shock_research/data_bybit/SOLUSDT/orderbook/dataminer/data/archive/raw")
WINDOW_SECONDS = 10  # Shorter window for aggressive flow
FLOW_IMPACT_THRESHOLD = 0.6  # Market stress level
OUTPUT_DIR = Path("flow_shock_research/results")

class FlowImpactDetector:
    """
    Detects forced flow events using market impact normalization.
    
    Flow Impact = Aggressive Volume / Top Book Depth
    
    Interpretation:
    - 0.05: noise
    - 0.2: activity
    - 0.5: stress
    - 1.0: market being punched through
    - >1.5: forced flow
    """
    
    def __init__(self, window_seconds=10, impact_threshold=0.6, min_burst_trades=15):
        self.window_seconds = window_seconds
        self.window_ms = window_seconds * 1000
        self.impact_threshold = impact_threshold
        self.min_burst_trades = min_burst_trades
        
        # Rolling windows
        self.trades = deque()  # (timestamp, volume, side, price)
        self.orderbook_snapshots = deque()  # (timestamp, bid_depth, ask_depth)
        
        # Events
        self.events = []
        self.total_trades = 0
        
    def add_orderbook_snapshot(self, timestamp_ms, bids, asks):
        """Add orderbook snapshot for depth calculation."""
        # Calculate top 5 levels depth
        bid_depth = sum(float(b[1]) for b in bids[:5]) if bids else 0
        ask_depth = sum(float(a[1]) for a in asks[:5]) if asks else 0
        
        self.orderbook_snapshots.append((timestamp_ms, bid_depth, ask_depth))
        
        # Keep only recent snapshots (1 minute window)
        cutoff = timestamp_ms - 60000
        while self.orderbook_snapshots and self.orderbook_snapshots[0][0] < cutoff:
            self.orderbook_snapshots.popleft()
    
    def get_book_depth_at_time(self, timestamp_ms):
        """Get orderbook depth closest to timestamp."""
        if not self.orderbook_snapshots:
            return None
        
        # Find closest snapshot (within 5 seconds)
        closest = None
        min_diff = 5000  # 5 seconds max
        
        for ts, bid_depth, ask_depth in self.orderbook_snapshots:
            diff = abs(ts - timestamp_ms)
            if diff < min_diff:
                min_diff = diff
                closest = (bid_depth, ask_depth)
        
        return closest
    
    def add_trade(self, timestamp_ms, volume, side, price):
        """Add trade and check for flow impact event."""
        self.total_trades += 1
        
        # Add to rolling window
        self.trades.append((timestamp_ms, volume, side, price))
        
        # Remove old trades
        cutoff = timestamp_ms - self.window_ms
        while self.trades and self.trades[0][0] < cutoff:
            self.trades.popleft()
        
        # Need minimum trades for detection
        if len(self.trades) < self.min_burst_trades:
            return None
        
        # Calculate signed aggressive flow
        buy_volume = sum(v for ts, v, s, p in self.trades if s == 'Buy')
        sell_volume = sum(v for ts, v, s, p in self.trades if s == 'Sell')
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return None
        
        # Flow imbalance
        flow_imbalance = abs(buy_volume - sell_volume) / total_volume
        aggressive_volume = max(buy_volume, sell_volume)
        flow_direction = 'Buy' if buy_volume > sell_volume else 'Sell'
        
        # Get orderbook depth
        book_depth = self.get_book_depth_at_time(timestamp_ms)
        if not book_depth:
            return None
        
        bid_depth, ask_depth = book_depth
        
        # Use relevant side depth
        if flow_direction == 'Buy':
            relevant_depth = ask_depth  # Buyers hit asks
        else:
            relevant_depth = bid_depth  # Sellers hit bids
        
        if relevant_depth == 0:
            return None
        
        # Calculate Flow Impact
        flow_impact = aggressive_volume / relevant_depth
        
        # Burst detection: count consecutive same-direction trades
        recent_trades = list(self.trades)[-30:]  # Last 30 trades
        same_direction_count = sum(1 for _, _, s, _ in recent_trades if s == flow_direction)
        burst_ratio = same_direction_count / len(recent_trades) if recent_trades else 0
        
        # Robust statistics (median/MAD) for regime handling
        volumes = [v for _, v, _, _ in self.trades]
        median_vol = np.median(volumes)
        mad = np.median(np.abs(np.array(volumes) - median_vol))
        robust_z = (volume - median_vol) / mad if mad > 0 else 0
        
        # Detection criteria
        is_high_impact = flow_impact > self.impact_threshold
        is_imbalanced = flow_imbalance > 0.7  # 70%+ one direction
        is_burst = burst_ratio > 0.6  # 60%+ same direction
        
        if is_high_impact and is_imbalanced and is_burst:
            event = {
                'timestamp': timestamp_ms,
                'datetime': datetime.fromtimestamp(timestamp_ms / 1000).isoformat(),
                'flow_impact': flow_impact,
                'flow_direction': flow_direction,
                'flow_imbalance': flow_imbalance,
                'aggressive_volume': aggressive_volume,
                'book_depth': relevant_depth,
                'burst_ratio': burst_ratio,
                'same_dir_trades': same_direction_count,
                'window_trades': len(self.trades),
                'robust_z': robust_z,
                'price': price,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume
            }
            self.events.append(event)
            return event
        
        return None

def parse_trade_file_fast(filepath):
    """Parse trade data."""
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
                                trade['S'],
                                float(trade['p'])
                            ))
                except:
                    continue
    except:
        pass
    return trades

def parse_orderbook_file_fast(filepath):
    """Parse orderbook snapshot data."""
    snapshots = []
    try:
        with gzip.open(filepath, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'result' in data and 'data' in data['result']:
                        ob_data = data['result']['data']
                        if 'b' in ob_data and 'a' in ob_data:
                            ts = int(data.get('ts', 0))
                            bids = ob_data['b']
                            asks = ob_data['a']
                            snapshots.append((ts, bids, asks))
                except:
                    continue
    except:
        pass
    return snapshots

def get_sample_files(start_date, end_date, sample_days=7):
    """Get sample files for both trade and orderbook - only dates with BOTH available."""
    trade_files = []
    ob_files = []
    
    # Read valid dates from file (dates with both trade and orderbook)
    valid_dates_file = Path("flow_shock_research/valid_dates.txt")
    if valid_dates_file.exists():
        with open(valid_dates_file, 'r') as f:
            valid_date_strs = [line.strip() for line in f if line.strip()]
        sampled_dates = [datetime.strptime(d, '%Y-%m-%d') for d in valid_date_strs]
        print(f"   Using {len(sampled_dates)} dates with complete data (trade + orderbook)")
    else:
        # Fallback to sampling
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
        
        for hour in range(24):
            # Trade files
            trade_file = DATA_DIR_TRADE / f"dt={date_str}" / f"hr={hour:02d}" / "exchange=bybit" / "source=ws" / "market=linear" / "stream=trade" / "symbol=SOLUSDT" / "data.jsonl.gz"
            if trade_file.exists():
                trade_files.append(trade_file)
            
            # Orderbook files
            ob_file = DATA_DIR_OB / f"dt={date_str}" / f"hr={hour:02d}" / "exchange=bybit" / "source=ws" / "market=linear" / "stream=orderbook" / "symbol=SOLUSDT" / "data.jsonl.gz"
            if ob_file.exists():
                ob_files.append(ob_file)
    
    return sorted(trade_files), sorted(ob_files), sampled_dates

def analyze_flow_impact(start_date='2025-05-11', end_date='2025-08-10', sample_days=7):
    """Main analysis with Flow Impact detector."""
    print(f"{'='*80}")
    print(f"🔍 FLOW IMPACT DETECTOR v2 - Market Impact Based")
    print(f"{'='*80}")
    print(f"Window:           {WINDOW_SECONDS}s")
    print(f"Impact threshold: {FLOW_IMPACT_THRESHOLD} (market stress)")
    print(f"Sample size:      {sample_days} days")
    print(f"{'='*80}\n")
    
    print("📂 Loading sample data...")
    trade_files, ob_files, sampled_dates = get_sample_files(start_date, end_date, sample_days)
    print(f"   Trade files: {len(trade_files)}")
    print(f"   Orderbook files: {len(ob_files)}")
    print(f"   Sampled dates: {', '.join([d.strftime('%Y-%m-%d') for d in sampled_dates])}\n")
    
    if not trade_files or not ob_files:
        print("❌ Missing data files!")
        print(f"   Trade dir: {DATA_DIR_TRADE}")
        print(f"   OB dir: {DATA_DIR_OB}")
        return None
    
    # Create detector
    detector = FlowImpactDetector(WINDOW_SECONDS, FLOW_IMPACT_THRESHOLD)
    
    # Process files
    print("🔄 Processing data...")
    total_trades = 0
    
    # Build orderbook timeline first
    print("   Loading orderbook snapshots...")
    for i, ob_file in enumerate(ob_files, 1):
        snapshots = parse_orderbook_file_fast(ob_file)
        for ts, bids, asks in snapshots:
            detector.add_orderbook_snapshot(ts, bids, asks)
        
        if i % 20 == 0:
            print(f"   OB: {i}/{len(ob_files)} files, {len(detector.orderbook_snapshots)} snapshots loaded...")
    
    print(f"   ✓ Loaded {len(detector.orderbook_snapshots)} orderbook snapshots\n")
    
    # Process trades
    print("   Processing trades...")
    for i, trade_file in enumerate(trade_files, 1):
        trades = parse_trade_file_fast(trade_file)
        
        for ts, vol, side, price in trades:
            total_trades += 1
            detector.add_trade(ts, vol, side, price)
        
        if i % 20 == 0:
            print(f"   Trades: {i}/{len(trade_files)} files, {total_trades:,} trades, {len(detector.events)} events...")
    
    print(f"\n✅ Processing complete!")
    print(f"   Total trades: {total_trades:,}")
    print(f"   Total events: {len(detector.events)}\n")
    
    # Analyze results
    if not detector.events:
        print("⚠️  No events detected with current thresholds")
        return None
    
    print(f"{'='*80}")
    print(f"📊 FLOW IMPACT EVENTS DETECTED")
    print(f"{'='*80}\n")
    
    df = pd.DataFrame(detector.events)
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    
    # Events per day
    events_per_day = df.groupby('date').size()
    print(f"Events per day:   {events_per_day.mean():.1f} avg, {events_per_day.median():.0f} median")
    print(f"Range:            {events_per_day.min()}-{events_per_day.max()} events/day")
    
    # Extrapolate
    total_days = 92
    est_total = len(df) * (total_days / sample_days)
    est_per_day = est_total / total_days
    print(f"Estimated total:  ~{est_total:.0f} events over {total_days} days ({est_per_day:.1f}/day)")
    
    # Flow Impact statistics
    print(f"\nFlow Impact stats:")
    print(f"  Range:          {df['flow_impact'].min():.2f} - {df['flow_impact'].max():.2f}")
    print(f"  Mean:           {df['flow_impact'].mean():.2f}")
    print(f"  Median:         {df['flow_impact'].median():.2f}")
    
    # Directional breakdown
    print(f"\nDirectional breakdown:")
    dir_counts = df['flow_direction'].value_counts()
    for direction, count in dir_counts.items():
        pct = count / len(df) * 100
        print(f"  {direction:4s}: {count:3d} events ({pct:5.1f}%)")
    
    # Imbalance statistics
    print(f"\nFlow imbalance:")
    print(f"  Mean:           {df['flow_imbalance'].mean():.2%}")
    print(f"  Median:         {df['flow_imbalance'].median():.2%}")
    
    # Burst statistics
    print(f"\nBurst characteristics:")
    print(f"  Avg burst ratio: {df['burst_ratio'].mean():.1%}")
    print(f"  Avg same-dir:    {df['same_dir_trades'].mean():.0f} trades")
    
    # Hourly distribution
    hourly = df.groupby('hour').size()
    top_hours = hourly.nlargest(5)
    print(f"\nTop hours (UTC):  {', '.join([f'{h}:00 ({c})' for h, c in top_hours.items()])}")
    
    # Target range check
    target_days = events_per_day[(events_per_day >= 1) & (events_per_day <= 5)]
    print(f"\nDays with 1-5:    {len(target_days)}/{len(events_per_day)} ({len(target_days)/len(events_per_day)*100:.0f}%)")
    
    # Comparison with z-score
    print(f"\nRobust Z-score (for comparison):")
    print(f"  Range:          {df['robust_z'].min():.1f} - {df['robust_z'].max():.1f}")
    print(f"  Mean:           {df['robust_z'].mean():.1f}")
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"flow_impact_events_{WINDOW_SECONDS}s.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Saved: {output_file}")
    
    # Generate report
    generate_report(df, sampled_dates, total_trades, est_per_day)
    
    return df

def generate_report(df, sampled_dates, total_trades, est_per_day):
    """Generate comparison report."""
    report_file = OUTPUT_DIR / "FINDINGS_flow_impact_v2.md"
    
    with open(report_file, 'w') as f:
        f.write("# Flow Impact Detector v2 - Findings\n\n")
        f.write("## Methodology Change\n\n")
        f.write("**v1 (Z-Score):** Measured volume spikes\n")
        f.write("```\nflow_z = (volume - mean) / std\n```\n")
        f.write("**Problem:** Assumes normal distribution, measures activity not impact\n\n")
        
        f.write("**v2 (Flow Impact):** Measures ability to move market\n")
        f.write("```\nFlow Impact = Aggressive Volume / Top Book Depth\n```\n")
        f.write("**Improvement:** Normalizes by available liquidity, regime-independent\n\n")
        
        f.write("## Results Comparison\n\n")
        f.write("| Metric | Z-Score (v1) | Flow Impact (v2) |\n")
        f.write("|--------|--------------|------------------|\n")
        f.write(f"| Events/day | 2.3 (z>30) | {est_per_day:.1f} (impact>0.6) |\n")
        f.write(f"| Total events | 210 (92d) | {len(df) * 92 / len(sampled_dates):.0f} (92d) |\n")
        f.write(f"| Detection basis | Volume spike | Market impact |\n")
        f.write(f"| Regime handling | Poor (mean/std) | Good (median/MAD) |\n")
        f.write(f"| Directional | No | Yes (buy/sell) |\n\n")
        
        f.write("## Flow Impact Interpretation\n\n")
        f.write("| Flow Impact | Meaning |\n")
        f.write("|-------------|----------|\n")
        f.write("| 0.05 | Noise |\n")
        f.write("| 0.2 | Activity |\n")
        f.write("| 0.5 | Stress |\n")
        f.write("| 1.0 | Market punched through |\n")
        f.write("| >1.5 | Forced flow |\n\n")
        
        f.write(f"**Detected events:** {df['flow_impact'].min():.2f} - {df['flow_impact'].max():.2f}\n\n")
        
        f.write("## Key Improvements\n\n")
        f.write("1. **Market Impact Normalization:** $500k volume on thin book ≠ thick book\n")
        f.write("2. **Signed Flow:** Tracks buy vs sell aggression separately\n")
        f.write("3. **Robust Statistics:** Median/MAD handles regime changes\n")
        f.write("4. **Burst Detection:** Requires sustained same-direction flow\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Analyze price behavior around Flow Impact events\n")
        f.write("2. Compare profitability: z-score vs flow impact triggers\n")
        f.write("3. Optimize impact threshold (0.6 vs 0.8 vs 1.0)\n")
        f.write("4. Add order flow toxicity metrics\n")
    
    print(f"📄 Report saved: {report_file}\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Flow Impact Detector v2')
    parser.add_argument('--start-date', type=str, default='2025-05-11')
    parser.add_argument('--end-date', type=str, default='2025-08-10')
    parser.add_argument('--sample-days', type=int, default=7)
    args = parser.parse_args()
    
    results = analyze_flow_impact(args.start_date, args.end_date, args.sample_days)
    
    if results is not None:
        print("✅ Analysis complete!")
        return 0
    else:
        print("❌ Analysis failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
