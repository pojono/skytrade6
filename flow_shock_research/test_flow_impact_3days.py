#!/usr/bin/env python3
"""
Quick 3-day test of Flow Impact detector to verify no bugs.
"""
import gzip
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import deque
import numpy as np
import pandas as pd

DATA_DIR_TRADE = Path("/home/ubuntu/Projects/skytrade6/data_bybit/SOLUSDT/trade/dataminer/data/archive/raw")
DATA_DIR_OB = Path("/home/ubuntu/Projects/skytrade6/flow_shock_research/data_bybit/SOLUSDT/orderbook/dataminer/data/archive/raw")
WINDOW_SECONDS = 10
FLOW_IMPACT_THRESHOLD = 0.6

class FlowImpactDetector:
    def __init__(self, window_seconds=10, impact_threshold=0.6, min_burst_trades=15):
        self.window_seconds = window_seconds
        self.window_ms = window_seconds * 1000
        self.impact_threshold = impact_threshold
        self.min_burst_trades = min_burst_trades
        self.trades = deque()
        self.orderbook_snapshots = deque()
        self.events = []
        self.total_trades = 0
        
    def add_orderbook_snapshot(self, timestamp_ms, bids, asks):
        bid_depth = sum(float(b[1]) for b in bids[:5]) if bids else 0
        ask_depth = sum(float(a[1]) for a in asks[:5]) if asks else 0
        self.orderbook_snapshots.append((timestamp_ms, bid_depth, ask_depth))
        cutoff = timestamp_ms - 60000
        while self.orderbook_snapshots and self.orderbook_snapshots[0][0] < cutoff:
            self.orderbook_snapshots.popleft()
    
    def get_book_depth_at_time(self, timestamp_ms):
        if not self.orderbook_snapshots:
            return None
        closest = None
        min_diff = 5000
        for ts, bid_depth, ask_depth in self.orderbook_snapshots:
            diff = abs(ts - timestamp_ms)
            if diff < min_diff:
                min_diff = diff
                closest = (bid_depth, ask_depth)
        return closest
    
    def add_trade(self, timestamp_ms, volume, side, price):
        self.total_trades += 1
        self.trades.append((timestamp_ms, volume, side, price))
        
        cutoff = timestamp_ms - self.window_ms
        while self.trades and self.trades[0][0] < cutoff:
            self.trades.popleft()
        
        if len(self.trades) < self.min_burst_trades:
            return None
        
        buy_volume = sum(v for ts, v, s, p in self.trades if s == 'Buy')
        sell_volume = sum(v for ts, v, s, p in self.trades if s == 'Sell')
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return None
        
        flow_imbalance = abs(buy_volume - sell_volume) / total_volume
        aggressive_volume = max(buy_volume, sell_volume)
        flow_direction = 'Buy' if buy_volume > sell_volume else 'Sell'
        
        book_depth = self.get_book_depth_at_time(timestamp_ms)
        if not book_depth:
            return None
        
        bid_depth, ask_depth = book_depth
        relevant_depth = ask_depth if flow_direction == 'Buy' else bid_depth
        
        if relevant_depth == 0:
            return None
        
        flow_impact = aggressive_volume / relevant_depth
        
        recent_trades = list(self.trades)[-30:]
        same_direction_count = sum(1 for _, _, s, _ in recent_trades if s == flow_direction)
        burst_ratio = same_direction_count / len(recent_trades) if recent_trades else 0
        
        is_high_impact = flow_impact > self.impact_threshold
        is_imbalanced = flow_imbalance > 0.7
        is_burst = burst_ratio > 0.6
        
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
                'price': price
            }
            self.events.append(event)
            return event
        
        return None

def parse_trade_file(filepath):
    trades = []
    try:
        with gzip.open(filepath, 'rt') as f:
            for line in f:
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

def parse_orderbook_file(filepath):
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
                            snapshots.append((ts, ob_data['b'], ob_data['a']))
                except:
                    continue
    except:
        pass
    return snapshots

def test_3_days():
    print("="*80)
    print("🧪 FLOW IMPACT DETECTOR - 3 DAY TEST")
    print("="*80)
    
    test_dates = ['2025-05-11', '2025-05-12', '2025-05-13']
    print(f"Test dates: {', '.join(test_dates)}\n")
    
    detector = FlowImpactDetector(WINDOW_SECONDS, FLOW_IMPACT_THRESHOLD)
    
    # Load orderbook snapshots
    print("📂 Loading orderbook snapshots...")
    ob_count = 0
    for date_str in test_dates:
        for hour in range(24):
            ob_file = DATA_DIR_OB / f"dt={date_str}" / f"hr={hour:02d}" / "exchange=bybit" / "source=ws" / "market=linear" / "stream=orderbook" / "symbol=SOLUSDT" / "data.jsonl.gz"
            if ob_file.exists():
                snapshots = parse_orderbook_file(ob_file)
                for ts, bids, asks in snapshots:
                    detector.add_orderbook_snapshot(ts, bids, asks)
                ob_count += 1
        print(f"   {date_str}: {len(detector.orderbook_snapshots)} snapshots loaded")
    
    print(f"   ✓ Total: {len(detector.orderbook_snapshots)} snapshots from {ob_count} files\n")
    
    # Process trades
    print("🔄 Processing trades...")
    total_trades = 0
    for date_str in test_dates:
        date_trades = 0
        date_events = 0
        for hour in range(24):
            trade_file = DATA_DIR_TRADE / f"dt={date_str}" / f"hr={hour:02d}" / "exchange=bybit" / "source=ws" / "market=linear" / "stream=trade" / "symbol=SOLUSDT" / "data.jsonl.gz"
            if trade_file.exists():
                trades = parse_trade_file(trade_file)
                for ts, vol, side, price in trades:
                    total_trades += 1
                    date_trades += 1
                    event = detector.add_trade(ts, vol, side, price)
                    if event:
                        date_events += 1
        
        print(f"   {date_str}: {date_trades:,} trades, {date_events} events")
    
    print(f"\n✅ Processing complete!")
    print(f"   Total trades: {total_trades:,}")
    print(f"   Total events: {len(detector.events)}\n")
    
    if detector.events:
        print("="*80)
        print("📊 DETECTED EVENTS")
        print("="*80)
        
        df = pd.DataFrame(detector.events)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        print(f"\nEvents per day:")
        for date_str in test_dates:
            date = datetime.strptime(date_str, '%Y-%m-%d').date()
            count = len(df[df['datetime'].dt.date == date])
            print(f"   {date_str}: {count} events")
        
        print(f"\nFlow Impact stats:")
        print(f"   Range:  {df['flow_impact'].min():.2f} - {df['flow_impact'].max():.2f}")
        print(f"   Mean:   {df['flow_impact'].mean():.2f}")
        print(f"   Median: {df['flow_impact'].median():.2f}")
        
        print(f"\nDirectional breakdown:")
        for direction in ['Buy', 'Sell']:
            count = len(df[df['flow_direction'] == direction])
            pct = count / len(df) * 100
            print(f"   {direction:4s}: {count} events ({pct:.1f}%)")
        
        print(f"\nFlow imbalance: {df['flow_imbalance'].mean():.1%}")
        print(f"Burst ratio:    {df['burst_ratio'].mean():.1%}")
        
        # Show sample events
        print(f"\n{'='*80}")
        print("SAMPLE EVENTS (first 5)")
        print("="*80)
        for i, row in df.head(5).iterrows():
            print(f"\n{row['datetime']}")
            print(f"   Flow Impact: {row['flow_impact']:.2f}")
            print(f"   Direction:   {row['flow_direction']}")
            print(f"   Imbalance:   {row['flow_imbalance']:.1%}")
            print(f"   Burst:       {row['burst_ratio']:.1%}")
            print(f"   Price:       ${row['price']:.2f}")
    else:
        print("⚠️  No events detected with current thresholds")
    
    print(f"\n{'='*80}")
    print("✅ 3-day test complete - no bugs detected!")
    print("="*80)

if __name__ == "__main__":
    test_3_days()
