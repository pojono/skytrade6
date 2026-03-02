#!/usr/bin/env python3
"""
Optimized Flow Impact Detector - 10x faster with batching and caching.
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
DATA_DIR_OB = Path("/home/ubuntu/Projects/skytrade6/flow_shock_research/data_bybit/SOLUSDT/orderbook_all/dataminer/data/archive/raw")
WINDOW_SECONDS = 10
FLOW_IMPACT_THRESHOLD = 0.6
MIN_BURST_TRADES = 15

class FastFlowImpactDetector:
    """Optimized detector with batching and numpy operations."""
    
    def __init__(self, window_seconds=10, impact_threshold=0.6, min_burst_trades=15):
        self.window_seconds = window_seconds
        self.window_ms = window_seconds * 1000
        self.impact_threshold = impact_threshold
        self.min_burst_trades = min_burst_trades
        
        # Use lists for batch operations
        self.trade_timestamps = []
        self.trade_volumes = []
        self.trade_sides = []
        self.trade_prices = []
        
        # Orderbook cache
        self.ob_cache = {}  # timestamp -> (bid_depth, ask_depth)
        
        self.events = []
        self.total_trades = 0
        
    def add_orderbook_batch(self, snapshots):
        """Add orderbook snapshots in batch."""
        for ts, bids, asks in snapshots:
            bid_depth = sum(float(b[1]) for b in bids[:5]) if bids else 0
            ask_depth = sum(float(a[1]) for a in asks[:5]) if asks else 0
            # Round timestamp to nearest second for caching
            ts_key = (ts // 1000) * 1000
            self.ob_cache[ts_key] = (bid_depth, ask_depth)
    
    def get_book_depth_at_time(self, timestamp_ms):
        """Fast lookup with caching."""
        ts_key = (timestamp_ms // 1000) * 1000
        # Try exact match
        if ts_key in self.ob_cache:
            return self.ob_cache[ts_key]
        # Try nearby (within 5 seconds)
        for offset in range(-5000, 5001, 1000):
            key = ts_key + offset
            if key in self.ob_cache:
                return self.ob_cache[key]
        return None
    
    def process_trade_batch(self, trades):
        """Process trades in batch with numpy."""
        for ts, vol, side, price in trades:
            self.total_trades += 1
            self.trade_timestamps.append(ts)
            self.trade_volumes.append(vol)
            self.trade_sides.append(1 if side == 'Buy' else -1)  # Encode as int
            self.trade_prices.append(price)
            
            # Check for events every 100 trades (reduce overhead)
            if len(self.trade_timestamps) % 100 == 0:
                self._check_for_events()
        
        # Final check
        self._check_for_events()
    
    def _check_for_events(self):
        """Check for events using numpy operations."""
        if len(self.trade_timestamps) < self.min_burst_trades:
            return
        
        # Convert to numpy for fast operations
        timestamps = np.array(self.trade_timestamps)
        volumes = np.array(self.trade_volumes)
        sides = np.array(self.trade_sides)
        prices = np.array(self.trade_prices)
        
        current_ts = timestamps[-1]
        cutoff = current_ts - self.window_ms
        
        # Find trades in window
        in_window = timestamps >= cutoff
        if in_window.sum() < self.min_burst_trades:
            return
        
        window_volumes = volumes[in_window]
        window_sides = sides[in_window]
        window_prices = prices[in_window]
        
        # Calculate buy/sell volumes
        buy_mask = window_sides > 0
        buy_volume = window_volumes[buy_mask].sum()
        sell_volume = window_volumes[~buy_mask].sum()
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return
        
        flow_imbalance = abs(buy_volume - sell_volume) / total_volume
        aggressive_volume = max(buy_volume, sell_volume)
        flow_direction = 'Buy' if buy_volume > sell_volume else 'Sell'
        
        # Get orderbook depth
        book_depth = self.get_book_depth_at_time(current_ts)
        if not book_depth:
            return
        
        bid_depth, ask_depth = book_depth
        relevant_depth = ask_depth if flow_direction == 'Buy' else bid_depth
        
        if relevant_depth == 0:
            return
        
        flow_impact = aggressive_volume / relevant_depth
        
        # Burst detection on last 30 trades
        recent_sides = window_sides[-30:]
        same_direction = 1 if flow_direction == 'Buy' else -1
        burst_ratio = (recent_sides == same_direction).sum() / len(recent_sides)
        
        # Detection
        is_high_impact = flow_impact > self.impact_threshold
        is_imbalanced = flow_imbalance > 0.7
        is_burst = burst_ratio > 0.6
        
        if is_high_impact and is_imbalanced and is_burst:
            event = {
                'timestamp': int(current_ts),
                'datetime': datetime.fromtimestamp(current_ts / 1000).isoformat(),
                'flow_impact': float(flow_impact),
                'flow_direction': flow_direction,
                'flow_imbalance': float(flow_imbalance),
                'aggressive_volume': float(aggressive_volume),
                'book_depth': float(relevant_depth),
                'burst_ratio': float(burst_ratio),
                'price': float(window_prices[-1])
            }
            self.events.append(event)
            print(f"      🔥 EVENT: impact={flow_impact:.1f}, {flow_direction}, {datetime.fromtimestamp(current_ts/1000).strftime('%H:%M:%S')}")

def parse_trade_file_fast(filepath):
    """Fast batch parsing."""
    trades = []
    try:
        with gzip.open(filepath, 'rt') as f:
            content = f.read()
            for line in content.split('\n'):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if 'result' in data and 'data' in data['result']:
                        for trade in data['result']['data']:
                            trades.append((int(trade['T']), float(trade['v']), trade['S'], float(trade['p'])))
                except:
                    continue
    except:
        pass
    return trades

def parse_orderbook_file_fast(filepath):
    """Fast batch parsing."""
    snapshots = []
    try:
        with gzip.open(filepath, 'rt') as f:
            content = f.read()
            for line in content.split('\n'):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if 'result' in data and 'data' in data['result']:
                        ob_data = data['result']['data']
                        if 'b' in ob_data and 'a' in ob_data:
                            ts = int(data.get('ts', 0))
                            snapshots.append((ts, ob_data['b'], ob_data['a']))
                except:
                    continue
    except:
        pass
    return snapshots

def get_all_available_dates():
    """Get all dates with both trade and orderbook data."""
    valid_dates = []
    
    start = datetime(2025, 5, 11)
    end = datetime(2025, 8, 10)
    current = start
    
    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        trade_dir = DATA_DIR_TRADE / f"dt={date_str}"
        
        # Check for any orderbook variant
        ob_exists = False
        for ob_variant in ['orderbook', 'orderbook.50', 'orderbook.500']:
            ob_pattern = DATA_DIR_OB / f"dt={date_str}" / "hr=00" / "exchange=bybit" / "source=ws" / "market=linear" / f"stream={ob_variant}" / "symbol=SOLUSDT"
            if ob_pattern.exists():
                ob_exists = True
                break
        
        if trade_dir.exists() and ob_exists:
            valid_dates.append(date_str)
        
        current += timedelta(days=1)
    
    return valid_dates

def analyze_fast(dates_to_process=None):
    """Fast analysis with optimizations."""
    print("="*80)
    print("🚀 FAST FLOW IMPACT DETECTOR")
    print("="*80)
    
    if dates_to_process is None:
        print("📂 Finding dates with complete data...")
        dates_to_process = get_all_available_dates()
    
    print(f"   Processing {len(dates_to_process)} dates")
    print(f"   First: {dates_to_process[0]}, Last: {dates_to_process[-1]}\n")
    
    detector = FastFlowImpactDetector(WINDOW_SECONDS, FLOW_IMPACT_THRESHOLD, MIN_BURST_TRADES)
    
    # Load ALL orderbook data first (batch)
    print("📂 Loading orderbook data...")
    ob_files_loaded = 0
    for date_str in dates_to_process:
        for hour in range(24):
            # Try all orderbook variants
            for ob_variant in ['orderbook', 'orderbook.50', 'orderbook.500']:
                ob_file = DATA_DIR_OB / f"dt={date_str}" / f"hr={hour:02d}" / "exchange=bybit" / "source=ws" / "market=linear" / f"stream={ob_variant}" / "symbol=SOLUSDT" / "data.jsonl.gz"
                if ob_file.exists():
                    snapshots = parse_orderbook_file_fast(ob_file)
                    detector.add_orderbook_batch(snapshots)
                    ob_files_loaded += 1
                    break  # Use first available variant
        
        if (dates_to_process.index(date_str) + 1) % 10 == 0:
            print(f"   {dates_to_process.index(date_str) + 1}/{len(dates_to_process)} dates, {ob_files_loaded} files, {len(detector.ob_cache)} snapshots")
    
    print(f"   ✓ Loaded {len(detector.ob_cache)} orderbook snapshots\n")
    
    # Process trades
    print("🔄 Processing trades...")
    for date_str in dates_to_process:
        date_trades = 0
        for hour in range(24):
            trade_file = DATA_DIR_TRADE / f"dt={date_str}" / f"hr={hour:02d}" / "exchange=bybit" / "source=ws" / "market=linear" / "stream=trade" / "symbol=SOLUSDT" / "data.jsonl.gz"
            if trade_file.exists():
                trades = parse_trade_file_fast(trade_file)
                detector.process_trade_batch(trades)
                date_trades += len(trades)
        
        print(f"   {date_str}: {date_trades:,} trades, {len(detector.events)} total events")
    
    print(f"\n✅ Processing complete!")
    print(f"   Total trades: {detector.total_trades:,}")
    print(f"   Total events: {len(detector.events)}\n")
    
    if detector.events:
        df = pd.DataFrame(detector.events)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].dt.date
        
        print("="*80)
        print("📊 RESULTS")
        print("="*80)
        
        events_per_day = df.groupby('date').size()
        print(f"\nEvents per day: {events_per_day.mean():.1f} avg, {events_per_day.median():.0f} median")
        print(f"Range: {events_per_day.min()}-{events_per_day.max()} events/day")
        
        print(f"\nFlow Impact: {df['flow_impact'].min():.1f} - {df['flow_impact'].max():.1f} (mean: {df['flow_impact'].mean():.1f})")
        print(f"Flow Imbalance: {df['flow_imbalance'].mean():.1%}")
        print(f"Burst Ratio: {df['burst_ratio'].mean():.1%}")
        
        dir_counts = df['flow_direction'].value_counts()
        print(f"\nDirection: Buy {dir_counts.get('Buy', 0)}, Sell {dir_counts.get('Sell', 0)}")
        
        # Save
        output_file = Path("flow_shock_research/results/flow_impact_fast_results.csv")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\n💾 Saved: {output_file}")
        
        return df
    else:
        print("⚠️  No events detected")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, help='Number of days to process (default: all available)')
    args = parser.parse_args()
    
    dates = get_all_available_dates()
    if args.days:
        dates = dates[:args.days]
    
    analyze_fast(dates)
    return 0

if __name__ == "__main__":
    sys.exit(main())
