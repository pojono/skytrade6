#!/usr/bin/env python3
"""
Build production event dataset with orderbook.500 data - PARALLELIZED.

Using multiprocessing to process events in parallel across 6 cores.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import json
import sys
from multiprocessing import Pool, cpu_count

DATA_DIR_TRADE = Path("data_bybit/SOLUSDT/trade/dataminer/data/archive/raw")
DATA_DIR_OB = Path("data_bybit/SOLUSDT/orderbook_all/dataminer/data/archive/raw")

def load_orderbook_snapshot(date_str, target_ts):
    """Load orderbook.500 snapshot closest to target timestamp."""
    target_dt = pd.to_datetime(target_ts, unit='ms')
    hour = target_dt.hour
    
    ob_file = DATA_DIR_OB / f"dt={date_str}" / f"hr={hour:02d}" / "exchange=bybit" / "source=ws" / "market=linear" / "stream=orderbook.500" / "symbol=SOLUSDT" / "data.jsonl.gz"
    
    if not ob_file.exists():
        return None
    
    closest_snapshot = None
    min_diff = float('inf')
    
    try:
        with gzip.open(ob_file, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'result' in data and 'data' in data['result']:
                        ob_data = data['result']['data']
                        ts = int(data['result']['ts'])
                        
                        diff = abs(ts - target_ts)
                        if diff < min_diff:
                            min_diff = diff
                            closest_snapshot = ob_data
                            
                            if diff < 100:
                                break
                except:
                    continue
    except:
        return None
    
    return closest_snapshot

def extract_orderbook_features(ob_snapshot):
    """Extract orderbook features."""
    if not ob_snapshot:
        return {}
    
    try:
        bids = ob_snapshot.get('b', [])
        asks = ob_snapshot.get('a', [])
        
        if not bids or not asks:
            return {}
        
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        
        spread = best_ask - best_bid
        spread_bps = (spread / best_bid) * 10000
        
        depth_levels = []
        for i in range(min(5, len(bids), len(asks))):
            bid_size = float(bids[i][1])
            ask_size = float(asks[i][1])
            depth_levels.append(bid_size + ask_size)
        
        while len(depth_levels) < 5:
            depth_levels.append(0)
        
        top_depth = depth_levels[0]
        total_depth = sum(depth_levels)
        
        return {
            'spread': spread,
            'spread_bps': spread_bps,
            'depth_l1': depth_levels[0],
            'depth_l2': depth_levels[1],
            'depth_l3': depth_levels[2],
            'depth_l4': depth_levels[3],
            'depth_l5': depth_levels[4],
            'top_depth': top_depth,
            'total_depth_l5': total_depth,
            'best_bid': best_bid,
            'best_ask': best_ask
        }
    except:
        return {}

def load_trades_window(date_str, start_ts, end_ts):
    """Load trades in time window."""
    trades = []
    start_dt = pd.to_datetime(start_ts, unit='ms')
    end_dt = pd.to_datetime(end_ts, unit='ms')
    
    hours = set()
    current = start_dt
    while current <= end_dt:
        hours.add(current.hour)
        current += pd.Timedelta(hours=1)
    
    for hour in sorted(hours):
        trade_file = DATA_DIR_TRADE / f"dt={date_str}" / f"hr={hour:02d}" / "exchange=bybit" / "source=ws" / "market=linear" / "stream=trade" / "symbol=SOLUSDT" / "data.jsonl.gz"
        
        if not trade_file.exists():
            continue
        
        try:
            with gzip.open(trade_file, 'rt') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if 'result' in data and 'data' in data['result']:
                            for trade in data['result']['data']:
                                ts = int(trade['T'])
                                if start_ts <= ts <= end_ts:
                                    trades.append({
                                        'timestamp': ts,
                                        'price': float(trade['p']),
                                        'volume': float(trade['v']),
                                        'side': trade['S']
                                    })
                    except:
                        continue
        except:
            continue
    
    return pd.DataFrame(trades).sort_values('timestamp') if trades else pd.DataFrame()

def calculate_multiscale_features(event_ts, event_price, date_str):
    """Calculate multi-scale regime features."""
    windows = {'10s': 10, '30s': 30, '2m': 120, '5m': 300, '15m': 900}
    features = {}
    
    for window_name, window_seconds in windows.items():
        start_ts = event_ts - (window_seconds * 1000)
        end_ts = event_ts
        
        trades_df = load_trades_window(date_str, start_ts, end_ts)
        
        if len(trades_df) < 5:
            continue
        
        prefix = f"pre_{window_name}_"
        
        if len(trades_df) > 1:
            returns = trades_df['price'].pct_change().dropna() * 10000
            features[f'{prefix}vol'] = returns.std()
        
        high = trades_df['price'].max()
        low = trades_df['price'].min()
        features[f'{prefix}range'] = (high - low) / trades_df['price'].mean() * 10000
        
        first_price = trades_df['price'].iloc[0]
        last_price = trades_df['price'].iloc[-1]
        features[f'{prefix}drift'] = (last_price - first_price) / first_price * 10000
        
        buy_vol = trades_df[trades_df['side'] == 'Buy']['volume'].sum()
        sell_vol = trades_df[trades_df['side'] == 'Sell']['volume'].sum()
        total_vol = buy_vol + sell_vol
        if total_vol > 0:
            features[f'{prefix}imbalance'] = (buy_vol - sell_vol) / total_vol
    
    return features

def process_event(row_tuple):
    """Process a single event - designed for parallel execution."""
    idx, row = row_tuple
    
    event_ts = row['timestamp']
    event_price = row['price']
    event_date = pd.to_datetime(row['datetime']).date()
    date_str = event_date.strftime('%Y-%m-%d')
    
    event = {
        'timestamp': event_ts,
        'datetime': row['datetime'],
        'date': date_str,
        'hour': pd.to_datetime(row['datetime']).hour,
        'direction': row['direction'],
        'flow_impact': row['flow_impact'],
        'imbalance': row['imbalance'],
        'max_run': row['max_run'],
        'event_price': event_price
    }
    
    # Orderbook
    ob_snapshot = load_orderbook_snapshot(date_str, event_ts)
    ob_features = extract_orderbook_features(ob_snapshot)
    event.update(ob_features)
    
    # Multi-scale features
    regime_features = calculate_multiscale_features(event_ts, event_price, date_str)
    event.update(regime_features)
    
    return event

def main():
    print("="*80, flush=True)
    print("📦 BUILDING PRODUCTION DATASET - PARALLEL", flush=True)
    print("="*80, flush=True)
    
    n_workers = 6
    print(f"\nUsing {n_workers} parallel workers", flush=True)
    print("Using July-Aug 2025 sample with orderbook.500 data", flush=True)
    print("="*80 + "\n", flush=True)
    
    # Load events
    events_df = pd.read_csv("results/sample_jul_aug_ob.csv")
    events_df['datetime'] = pd.to_datetime(events_df['datetime'])
    
    print(f"Total events: {len(events_df)}", flush=True)
    print(f"Processing in parallel with {n_workers} workers...\n", flush=True)
    
    # Process in parallel
    with Pool(n_workers) as pool:
        dataset = pool.map(process_event, events_df.iterrows())
    
    df = pd.DataFrame(dataset)
    
    print(f"\n✅ Dataset built: {len(df)} events", flush=True)
    print(f"   Columns: {len(df.columns)}", flush=True)
    
    # Check data quality
    print(f"\nData quality:", flush=True)
    if 'spread' in df.columns:
        print(f"   Events with orderbook: {df['spread'].notna().sum()} ({df['spread'].notna().mean():.1%})", flush=True)
    else:
        print(f"   Events with orderbook: 0 (no spread column)", flush=True)
    if 'pre_15m_vol' in df.columns:
        print(f"   Events with regime features: {df['pre_15m_vol'].notna().sum()} ({df['pre_15m_vol'].notna().mean():.1%})", flush=True)
    else:
        print(f"   Events with regime features: 0 (no pre_15m_vol column)", flush=True)
    
    # Save
    df.to_parquet("results/production_dataset.parquet", index=False)
    df.to_csv("results/production_dataset.csv", index=False)
    
    print(f"\n💾 Saved: results/production_dataset.parquet", flush=True)
    print(f"💾 Saved: results/production_dataset.csv", flush=True)
    
    print(f"\n{'='*80}", flush=True)
    print(f"✅ STEP 1 COMPLETE - Production dataset ready", flush=True)
    print(f"{'='*80}", flush=True)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
