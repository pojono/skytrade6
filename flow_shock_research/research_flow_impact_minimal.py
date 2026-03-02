#!/usr/bin/env python3
"""
Minimal Flow Impact Detector - sample orderbook, not load everything.
"""
import gzip
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

DATA_DIR_TRADE = Path("/home/ubuntu/Projects/skytrade6/data_bybit/SOLUSDT/trade/dataminer/data/archive/raw")
DATA_DIR_OB = Path("/home/ubuntu/Projects/skytrade6/flow_shock_research/data_bybit/SOLUSDT/orderbook/dataminer/data/archive/raw")

def sample_orderbook_file(filepath, sample_rate=10):
    """Sample every Nth snapshot instead of loading all."""
    snapshots = []
    try:
        with gzip.open(filepath, 'rt') as f:
            for i, line in enumerate(f):
                if i % sample_rate != 0:  # Sample every 10th line
                    continue
                try:
                    data = json.loads(line)
                    if 'result' in data and 'data' in data['result']:
                        ob_data = data['result']['data']
                        if 'b' in ob_data and 'a' in ob_data:
                            ts = int(data.get('ts', 0))
                            bid_depth = sum(float(b[1]) for b in ob_data['b'][:5])
                            ask_depth = sum(float(a[1]) for a in ob_data['a'][:5])
                            snapshots.append((ts, bid_depth, ask_depth))
                except:
                    continue
    except:
        pass
    return snapshots

def parse_trade_file(filepath):
    """Parse trades."""
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

def detect_events(trades, ob_cache, window_ms=10000, impact_threshold=0.6):
    """Simple event detection."""
    if len(trades) < 15:
        return []
    
    timestamps = np.array([t[0] for t in trades])
    volumes = np.array([t[1] for t in trades])
    sides = np.array([t[2] for t in trades])
    prices = np.array([t[3] for t in trades])
    
    events = []
    
    # Check every 100th trade
    for i in range(100, len(trades), 100):
        current_ts = timestamps[i]
        cutoff = current_ts - window_ms
        
        mask = (timestamps[i-100:i+1] >= cutoff)
        if mask.sum() < 15:
            continue
        
        window_vols = volumes[i-100:i+1][mask]
        window_sides = sides[i-100:i+1][mask]
        
        buy_vol = window_vols[window_sides > 0].sum()
        sell_vol = window_vols[window_sides < 0].sum()
        total_vol = buy_vol + sell_vol
        
        if total_vol == 0:
            continue
        
        imbalance = abs(buy_vol - sell_vol) / total_vol
        if imbalance < 0.7:
            continue
        
        direction = 'Buy' if buy_vol > sell_vol else 'Sell'
        aggressive_vol = max(buy_vol, sell_vol)
        
        # Burst
        recent = window_sides[-30:]
        burst = (recent == (1 if direction == 'Buy' else -1)).sum() / len(recent)
        if burst < 0.6:
            continue
        
        # Get OB depth (nearest second)
        ts_key = (current_ts // 1000) * 1000
        book_depth = None
        for offset in range(-10000, 10001, 1000):
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
                'burst_ratio': float(burst),
                'price': float(prices[i])
            })
    
    return events

def analyze_minimal(num_days=10):
    """Minimal analysis with sampled orderbook."""
    print("="*80)
    print("⚡ MINIMAL FLOW IMPACT DETECTOR (sampled orderbook)")
    print("="*80)
    
    start = datetime(2025, 5, 11)
    dates = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(num_days)]
    print(f"Processing {len(dates)} days\n")
    
    # Load sampled orderbook
    print("📂 Loading orderbook (sampled 1/10)...")
    ob_cache = {}
    
    for date_str in dates:
        date_snapshots = 0
        for hour in range(24):
            ob_file = DATA_DIR_OB / f"dt={date_str}" / f"hr={hour:02d}" / "exchange=bybit" / "source=ws" / "market=linear" / "stream=orderbook" / "symbol=SOLUSDT" / "data.jsonl.gz"
            if ob_file.exists():
                snapshots = sample_orderbook_file(ob_file, sample_rate=10)
                for ts, bid, ask in snapshots:
                    ts_key = (ts // 1000) * 1000
                    ob_cache[ts_key] = (bid, ask)
                date_snapshots += len(snapshots)
        
        print(f"   {date_str}: {date_snapshots} snapshots")
    
    print(f"\n   Total: {len(ob_cache)} snapshots\n")
    
    # Process trades
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
        
        events = detect_events(date_trades, ob_cache)
        all_events.extend(events)
        total_trades += len(date_trades)
        
        print(f"   {date_str}: {len(date_trades):,} trades, {len(events)} events")
    
    print(f"\n✅ Complete! {total_trades:,} trades, {len(all_events)} events\n")
    
    if all_events:
        df = pd.DataFrame(all_events)
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
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
        
        # Extrapolate
        est_total = len(all_events) * (92 / num_days)
        est_per_day = est_total / 92
        print(f"\nExtrapolated (92 days): ~{est_total:.0f} events ({est_per_day:.1f}/day)")
        
        # Compare with z-score
        print(f"\n{'='*80}")
        print("COMPARISON")
        print("="*80)
        print(f"Z-Score (v1):     2.3 events/day (z>30)")
        print(f"Flow Impact (v2): {est_per_day:.1f} events/day (impact>0.6)")
        print("="*80)
        
        # Save
        output = Path("flow_shock_research/results/flow_impact_minimal.csv")
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
    parser.add_argument('--days', type=int, default=10)
    args = parser.parse_args()
    
    analyze_minimal(args.days)
