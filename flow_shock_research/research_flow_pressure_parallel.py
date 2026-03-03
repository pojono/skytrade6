#!/usr/bin/env python3
"""
Forced flow detector - PARALLELIZED by date.

Process multiple dates in parallel using multiprocessing.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import json
from datetime import datetime, timedelta
from multiprocessing import Pool
import sys

DATA_DIR_TRADE = Path("data_bybit/SOLUSDT/trade/dataminer/data/archive/raw")
DATA_DIR_OB = Path("data_bybit/SOLUSDT/orderbook_all/dataminer/data/archive/raw")

# Detection parameters
FLOW_IMPACT_THRESHOLD = 70
IMBALANCE_THRESHOLD = 0.6
MIN_AGG_TRADES = 20
SAME_SIDE_SHARE = 0.75

def process_single_date(date_str):
    """Process forced flow detection for a single date."""
    print(f"[{date_str}] Starting...", flush=True)
    
    events = []
    
    # Check if orderbook data exists
    ob_date_dir = DATA_DIR_OB / f"dt={date_str}"
    if not ob_date_dir.exists():
        print(f"[{date_str}] No orderbook data, skipping", flush=True)
        return events
    
    # Get available hours
    hours = sorted([int(h.name.replace('hr=', '')) for h in ob_date_dir.iterdir() if h.name.startswith('hr=')])
    
    for hour in hours:
        # Load orderbook
        ob_file = DATA_DIR_OB / f"dt={date_str}" / f"hr={hour:02d}" / "exchange=bybit" / "source=ws" / "market=linear" / "stream=orderbook.500" / "symbol=SOLUSDT" / "data.jsonl.gz"
        
        if not ob_file.exists():
            continue
        
        orderbook = []
        try:
            with gzip.open(ob_file, 'rt') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if 'result' in data and 'data' in data['result']:
                            ob_data = data['result']['data']
                            ts = int(data['result']['ts'])
                            
                            bids = ob_data.get('b', [])
                            asks = ob_data.get('a', [])
                            
                            if bids and asks:
                                best_bid = float(bids[0][0])
                                best_ask = float(asks[0][0])
                                bid_size = float(bids[0][1])
                                ask_size = float(asks[0][1])
                                
                                orderbook.append({
                                    'timestamp': ts,
                                    'best_bid': best_bid,
                                    'best_ask': best_ask,
                                    'bid_size': bid_size,
                                    'ask_size': ask_size,
                                    'top_depth': bid_size + ask_size
                                })
                    except:
                        continue
        except:
            continue
        
        if len(orderbook) == 0:
            continue
        
        ob_df = pd.DataFrame(orderbook).sort_values('timestamp')
        
        # Load trades
        trade_file = DATA_DIR_TRADE / f"dt={date_str}" / f"hr={hour:02d}" / "exchange=bybit" / "source=ws" / "market=linear" / "stream=trade" / "symbol=SOLUSDT" / "data.jsonl.gz"
        
        if not trade_file.exists():
            continue
        
        trades = []
        try:
            with gzip.open(trade_file, 'rt') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if 'result' in data and 'data' in data['result']:
                            for trade in data['result']['data']:
                                trades.append({
                                    'timestamp': int(trade['T']),
                                    'price': float(trade['p']),
                                    'volume': float(trade['v']),
                                    'side': trade['S']
                                })
                    except:
                        continue
        except:
            continue
        
        if len(trades) == 0:
            continue
        
        trades_df = pd.DataFrame(trades).sort_values('timestamp')
        
        # Detect forced flow events
        window_ms = 1000
        
        for i in range(len(trades_df)):
            start_ts = trades_df.iloc[i]['timestamp']
            end_ts = start_ts + window_ms
            
            window_trades = trades_df[(trades_df['timestamp'] >= start_ts) & (trades_df['timestamp'] < end_ts)]
            
            if len(window_trades) < MIN_AGG_TRADES:
                continue
            
            # Calculate metrics
            buy_vol = window_trades[window_trades['side'] == 'Buy']['volume'].sum()
            sell_vol = window_trades[window_trades['side'] == 'Sell']['volume'].sum()
            total_vol = buy_vol + sell_vol
            
            if total_vol == 0:
                continue
            
            imbalance = abs(buy_vol - sell_vol) / total_vol
            
            if imbalance < IMBALANCE_THRESHOLD:
                continue
            
            # Same-side share
            dominant_side = 'Buy' if buy_vol > sell_vol else 'Sell'
            dominant_vol = buy_vol if dominant_side == 'Buy' else sell_vol
            same_side_share = dominant_vol / total_vol
            
            if same_side_share < SAME_SIDE_SHARE:
                continue
            
            # Get orderbook depth at event time
            ob_snapshot = ob_df[ob_df['timestamp'] <= start_ts].iloc[-1] if len(ob_df[ob_df['timestamp'] <= start_ts]) > 0 else None
            
            if ob_snapshot is None:
                continue
            
            top_depth = ob_snapshot['top_depth']
            
            if top_depth == 0:
                continue
            
            # Flow impact
            flow_impact = total_vol / top_depth * 100
            
            if flow_impact < FLOW_IMPACT_THRESHOLD:
                continue
            
            # Max run
            max_run = 0
            current_run = 0
            prev_side = None
            
            for _, trade in window_trades.iterrows():
                if trade['side'] == prev_side:
                    current_run += 1
                else:
                    max_run = max(max_run, current_run)
                    current_run = 1
                    prev_side = trade['side']
            max_run = max(max_run, current_run)
            
            # Event detected
            event_price = window_trades.iloc[0]['price']
            
            events.append({
                'timestamp': start_ts,
                'datetime': pd.to_datetime(start_ts, unit='ms'),
                'flow_impact': flow_impact,
                'agg_vol': total_vol,
                'top_depth': top_depth,
                'imbalance': imbalance,
                'direction': dominant_side,
                'agg_trades_count': len(window_trades),
                'same_side_share': same_side_share,
                'max_run': max_run,
                'spread': ob_snapshot['best_ask'] - ob_snapshot['best_bid'],
                'price': event_price
            })
    
    print(f"[{date_str}] Complete: {len(events)} events", flush=True)
    return events

def main():
    print("="*80)
    print("🔬 FORCED FLOW DETECTOR - PARALLEL")
    print("="*80)
    print(f"Processing all available dates with orderbook.500 data")
    print(f"Using 6 parallel workers")
    print("="*80 + "\n")
    
    # Get all available dates
    ob_dir = DATA_DIR_OB
    dates = sorted([d.name.replace('dt=', '') for d in ob_dir.iterdir() if d.name.startswith('dt=')])
    
    print(f"Found {len(dates)} dates to process")
    print(f"Date range: {dates[0]} to {dates[-1]}\n")
    
    # Process in parallel
    with Pool(6) as pool:
        results = pool.map(process_single_date, dates)
    
    # Combine all events
    all_events = []
    for events in results:
        all_events.extend(events)
    
    if len(all_events) == 0:
        print("❌ No events detected!")
        return 1
    
    # Save
    df = pd.DataFrame(all_events).sort_values('timestamp')
    df.to_csv("results/all_dates_forced_flow.csv", index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ Complete! {len(df)} events detected")
    print(f"💾 Saved: results/all_dates_forced_flow.csv")
    print(f"{'='*80}")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Total events: {len(df)}")
    print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"   Total days: {df['datetime'].dt.date.nunique()}")
    print(f"   Events/day: {len(df) / df['datetime'].dt.date.nunique():.1f}")
    
    print(f"\n   By direction:")
    for direction, count in df['direction'].value_counts().items():
        print(f"      {direction}: {count}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
