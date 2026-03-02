#!/usr/bin/env python3
"""
Ultra-fast Flow Impact Detector with parallel loading and minimal overhead.
"""
import gzip
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

DATA_DIR_TRADE = Path("/home/ubuntu/Projects/skytrade6/data_bybit/SOLUSDT/trade/dataminer/data/archive/raw")
DATA_DIR_OB_BASE = Path("/home/ubuntu/Projects/skytrade6/flow_shock_research/data_bybit/SOLUSDT")

def parse_orderbook_file(filepath):
    """Parse single orderbook file."""
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
                            bids = ob_data['b'][:5]
                            asks = ob_data['a'][:5]
                            bid_depth = sum(float(b[1]) for b in bids)
                            ask_depth = sum(float(a[1]) for a in asks)
                            snapshots.append((ts, bid_depth, ask_depth))
                except:
                    continue
    except:
        pass
    return snapshots

def load_orderbook_parallel(date_str):
    """Load all orderbook files for a date in parallel."""
    all_snapshots = []
    
    # Try all possible orderbook locations
    for ob_dir_name in ['orderbook', 'orderbook_all']:
        ob_base = DATA_DIR_OB_BASE / ob_dir_name / "dataminer/data/archive/raw"
        
        for hour in range(24):
            for ob_variant in ['orderbook', 'orderbook.50', 'orderbook.500']:
                ob_file = ob_base / f"dt={date_str}" / f"hr={hour:02d}" / "exchange=bybit" / "source=ws" / "market=linear" / f"stream={ob_variant}" / "symbol=SOLUSDT" / "data.jsonl.gz"
                if ob_file.exists():
                    snapshots = parse_orderbook_file(ob_file)
                    all_snapshots.extend(snapshots)
                    break  # Use first available variant
    
    return date_str, all_snapshots

def parse_trade_file(filepath):
    """Parse single trade file."""
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
                                1 if trade['S'] == 'Buy' else -1,
                                float(trade['p'])
                            ))
                except:
                    continue
    except:
        pass
    return trades

def detect_events_vectorized(trades, ob_cache, window_ms=10000, impact_threshold=0.6):
    """Vectorized event detection."""
    if len(trades) < 15:
        return []
    
    # Convert to numpy arrays
    timestamps = np.array([t[0] for t in trades])
    volumes = np.array([t[1] for t in trades])
    sides = np.array([t[2] for t in trades])  # 1=Buy, -1=Sell
    prices = np.array([t[3] for t in trades])
    
    events = []
    
    # Check every 50th trade to reduce overhead
    for i in range(50, len(trades), 50):
        current_ts = timestamps[i]
        cutoff = current_ts - window_ms
        
        # Window mask
        mask = timestamps[i-50:i+1] >= cutoff
        if mask.sum() < 15:
            continue
        
        window_vols = volumes[i-50:i+1][mask]
        window_sides = sides[i-50:i+1][mask]
        
        # Buy/sell volumes
        buy_vol = window_vols[window_sides > 0].sum()
        sell_vol = window_vols[window_sides < 0].sum()
        total_vol = buy_vol + sell_vol
        
        if total_vol == 0:
            continue
        
        imbalance = abs(buy_vol - sell_vol) / total_vol
        if imbalance < 0.7:
            continue
        
        aggressive_vol = max(buy_vol, sell_vol)
        direction = 'Buy' if buy_vol > sell_vol else 'Sell'
        
        # Burst check
        recent_sides = window_sides[-30:]
        burst_ratio = (recent_sides == (1 if direction == 'Buy' else -1)).sum() / len(recent_sides)
        if burst_ratio < 0.6:
            continue
        
        # Get orderbook depth
        ts_key = (current_ts // 1000) * 1000
        book_depth = None
        for offset in range(-5000, 5001, 1000):
            if ts_key + offset in ob_cache:
                book_depth = ob_cache[ts_key + offset]
                break
        
        if not book_depth:
            continue
        
        bid_depth, ask_depth = book_depth
        relevant_depth = ask_depth if direction == 'Buy' else bid_depth
        
        if relevant_depth == 0:
            continue
        
        impact = aggressive_vol / relevant_depth
        
        if impact > impact_threshold:
            events.append({
                'timestamp': int(current_ts),
                'datetime': datetime.fromtimestamp(current_ts / 1000).isoformat(),
                'flow_impact': float(impact),
                'flow_direction': direction,
                'flow_imbalance': float(imbalance),
                'aggressive_volume': float(aggressive_vol),
                'book_depth': float(relevant_depth),
                'burst_ratio': float(burst_ratio),
                'price': float(prices[i])
            })
    
    return events

def analyze_ultra_fast(num_days=10):
    """Ultra-fast analysis with parallel loading."""
    print("="*80)
    print("⚡ ULTRA-FAST FLOW IMPACT DETECTOR")
    print("="*80)
    
    # Get available dates
    start = datetime(2025, 5, 11)
    dates = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(num_days)]
    
    print(f"Processing {len(dates)} days: {dates[0]} to {dates[-1]}\n")
    
    # Load orderbook data in parallel
    print("📂 Loading orderbook data (parallel)...")
    ob_cache = {}
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(load_orderbook_parallel, date): date for date in dates}
        
        for future in as_completed(futures):
            date_str, snapshots = future.result()
            for ts, bid_depth, ask_depth in snapshots:
                ts_key = (ts // 1000) * 1000
                ob_cache[ts_key] = (bid_depth, ask_depth)
            print(f"   ✓ {date_str}: {len(snapshots)} snapshots")
    
    print(f"\n   Total OB snapshots: {len(ob_cache)}\n")
    
    # Process trades day by day
    print("🔄 Processing trades...")
    all_events = []
    total_trades = 0
    
    for date_str in dates:
        date_trades = []
        
        for hour in range(24):
            trade_file = DATA_DIR_TRADE / f"dt={date_str}" / f"hr={hour:02d}" / "exchange=bybit" / "source=ws" / "market=linear" / "stream=trade" / "symbol=SOLUSDT" / "data.jsonl.gz"
            if trade_file.exists():
                trades = parse_trade_file(trade_file)
                date_trades.extend(trades)
        
        # Detect events for this day
        events = detect_events_vectorized(date_trades, ob_cache)
        all_events.extend(events)
        total_trades += len(date_trades)
        
        print(f"   {date_str}: {len(date_trades):,} trades, {len(events)} events")
    
    print(f"\n✅ Complete! {total_trades:,} trades, {len(all_events)} events\n")
    
    if all_events:
        df = pd.DataFrame(all_events)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].dt.date
        
        print("="*80)
        print("📊 RESULTS")
        print("="*80)
        
        epd = df.groupby('date').size()
        print(f"\nEvents/day: {epd.mean():.1f} avg, {epd.median():.0f} median (range: {epd.min()}-{epd.max()})")
        print(f"Flow Impact: {df['flow_impact'].min():.1f} - {df['flow_impact'].max():.1f} (mean: {df['flow_impact'].mean():.1f})")
        print(f"Imbalance: {df['flow_imbalance'].mean():.1%}, Burst: {df['burst_ratio'].mean():.1%}")
        
        dir_counts = df['flow_direction'].value_counts()
        print(f"Direction: Buy {dir_counts.get('Buy', 0)}, Sell {dir_counts.get('Sell', 0)}")
        
        # Extrapolate to 92 days
        est_total = len(all_events) * (92 / num_days)
        est_per_day = est_total / 92
        print(f"\nExtrapolated (92 days): ~{est_total:.0f} events ({est_per_day:.1f}/day)")
        
        # Save
        output = Path("flow_shock_research/results/flow_impact_ultra_fast.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        print(f"\n💾 Saved: {output}")
        
        return df
    else:
        print("⚠️  No events detected")
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=10, help='Number of days to process')
    args = parser.parse_args()
    
    analyze_ultra_fast(args.days)
