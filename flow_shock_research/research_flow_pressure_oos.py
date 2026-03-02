#!/usr/bin/env python3
"""
Flow Pressure Detector - OOS Validation on specific date ranges.
"""
import sys
import gzip
import json
from pathlib import Path
from collections import deque
import pandas as pd
import numpy as np
from datetime import datetime

# Configuration
WINDOW_SECONDS = 15
FLOW_IMPACT_THRESHOLD = 70.0
IMBALANCE_THRESHOLD = 0.6
MIN_AGG_TRADES = 20
SAME_SIDE_SHARE = 0.75

DATA_DIR_TRADE = Path("data_bybit/SOLUSDT/trade/dataminer/data/archive/raw")
DATA_DIR_OB = Path("data_bybit/SOLUSDT/orderbook_all/dataminer/data/archive/raw")

def get_available_dates(start_date, end_date):
    """Get dates where both trade and orderbook data exist."""
    dates = []
    current = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        
        # Check trade data
        trade_dir = DATA_DIR_TRADE / f"dt={date_str}"
        ob_dir = DATA_DIR_OB / f"dt={date_str}"
        
        if trade_dir.exists() and ob_dir.exists():
            dates.append(date_str)
        
        current += pd.Timedelta(days=1)
    
    return dates

def load_orderbook_sampled(date_str, sample_rate=10):
    """Load orderbook data with sampling."""
    snapshots = []
    
    for hour in range(24):
        ob_file = DATA_DIR_OB / f"dt={date_str}" / f"hr={hour:02d}" / "exchange=bybit" / "source=ws" / "market=linear" / "stream=orderbook.500" / "symbol=SOLUSDT" / "data.jsonl.gz"
        
        if not ob_file.exists():
            continue
        
        try:
            with gzip.open(ob_file, 'rt') as f:
                for i, line in enumerate(f):
                    if i % sample_rate != 0:
                        continue
                    
                    try:
                        data = json.loads(line)
                        if 'result' in data and 'data' in data['result']:
                            ob = data['result']['data']
                            ts = int(data['result']['ts'])  # Timestamp is in result, not data
                            
                            bids = ob.get('b', [])[:5]
                            asks = ob.get('a', [])[:5]
                            
                            if not bids or not asks:
                                continue
                            
                            bid_depth = sum(float(b[1]) for b in bids)
                            ask_depth = sum(float(a[1]) for a in asks)
                            
                            spread = float(asks[0][0]) - float(bids[0][0])
                            
                            snapshots.append({
                                'timestamp': ts,
                                'bid_depth': bid_depth,
                                'ask_depth': ask_depth,
                                'spread': spread
                            })
                    except:
                        continue
        except:
            continue
    
    return pd.DataFrame(snapshots).sort_values('timestamp') if snapshots else pd.DataFrame()

def process_trades(date_str, ob_df):
    """Process trades and detect flow pressure events."""
    events = []
    
    # Rolling windows
    window_ms = WINDOW_SECONDS * 1000
    trade_window = deque()
    
    # Baseline tracking
    impact_history = deque(maxlen=10000)
    
    trades_processed = 0
    
    for hour in range(24):
        trade_file = DATA_DIR_TRADE / f"dt={date_str}" / f"hr={hour:02d}" / "exchange=bybit" / "source=ws" / "market=linear" / "stream=trade" / "symbol=SOLUSDT" / "data.jsonl.gz"
        
        if not trade_file.exists():
            continue
        
        with gzip.open(trade_file, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'result' not in data or 'data' not in data['result']:
                        continue
                    
                    for trade in data['result']['data']:
                        ts = int(trade['T'])
                        price = float(trade['p'])
                        volume = float(trade['v'])
                        side = trade['S']
                        
                        trades_processed += 1
                        
                        # Progress every 10k trades
                        if trades_processed % 10000 == 0:
                            print(f"      {trades_processed:,} trades processed...", flush=True)
                        
                        trade_window.append({
                            'timestamp': ts,
                            'price': price,
                            'volume': volume,
                            'side': side
                        })
                        
                        # Remove old trades
                        while trade_window and trade_window[0]['timestamp'] < ts - window_ms:
                            trade_window.popleft()
                        
                        # Check every 50 trades
                        if len(trade_window) % 50 != 0:
                            continue
                        
                        # Get orderbook snapshot
                        ob_snapshot = ob_df[ob_df['timestamp'] <= ts].iloc[-1] if len(ob_df[ob_df['timestamp'] <= ts]) > 0 else None
                        
                        if ob_snapshot is None:
                            continue
                        
                        top_depth = ob_snapshot['bid_depth'] + ob_snapshot['ask_depth']
                        
                        if top_depth == 0:
                            continue
                        
                        # Calculate metrics
                        trades_list = list(trade_window)
                        agg_buy_vol = sum(t['volume'] for t in trades_list if t['side'] == 'Buy')
                        agg_sell_vol = sum(t['volume'] for t in trades_list if t['side'] == 'Sell')
                        agg_vol = agg_buy_vol + agg_sell_vol
                        
                        if agg_vol == 0:
                            continue
                        
                        # FlowImpact
                        flow_impact = agg_vol / top_depth
                        impact_history.append(flow_impact)
                        
                        # Imbalance
                        net_agg = abs(agg_buy_vol - agg_sell_vol)
                        imbalance = net_agg / agg_vol
                        
                        # Direction
                        direction = 'Buy' if agg_buy_vol > agg_sell_vol else 'Sell'
                        
                        # Burst detection
                        agg_trades_count = len(trades_list)
                        same_side_count = max(agg_buy_vol, agg_sell_vol) / agg_vol if agg_vol > 0 else 0
                        
                        # Run detection
                        max_run = 1
                        current_run = 1
                        for i in range(1, len(trades_list)):
                            if trades_list[i]['side'] == trades_list[i-1]['side']:
                                current_run += 1
                                max_run = max(max_run, current_run)
                            else:
                                current_run = 1
                        
                        # Robust baseline
                        if len(impact_history) >= 100:
                            median_impact = np.median(list(impact_history))
                            mad = np.median([abs(x - median_impact) for x in impact_history])
                            robust_z = (flow_impact - median_impact) / (1.4826 * mad) if mad > 0 else 0
                        else:
                            robust_z = 0
                        
                        # Event detection
                        if (flow_impact >= FLOW_IMPACT_THRESHOLD and
                            imbalance >= IMBALANCE_THRESHOLD and
                            agg_trades_count >= MIN_AGG_TRADES and
                            same_side_count >= SAME_SIDE_SHARE):
                            
                            events.append({
                                'timestamp': ts,
                                'datetime': datetime.fromtimestamp(ts/1000).strftime('%Y-%m-%d %H:%M:%S'),
                                'flow_impact': flow_impact,
                                'agg_vol': agg_vol,
                                'top_depth': top_depth,
                                'imbalance': imbalance,
                                'direction': direction,
                                'agg_trades_count': agg_trades_count,
                                'same_side_share': same_side_count,
                                'max_run': max_run,
                                'spread': ob_snapshot['spread'],
                                'robust_z': robust_z,
                                'price': price
                            })
                
                except:
                    continue
    
    return events

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', required=True)
    parser.add_argument('--end-date', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    print("="*80)
    print("⚡ FLOW PRESSURE DETECTOR - OOS VALIDATION")
    print("="*80)
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Output: {args.output}")
    print("="*80 + "\n")
    
    # Get available dates
    dates = get_available_dates(args.start_date, args.end_date)
    print(f"📅 Available dates: {len(dates)}")
    for date in dates:
        print(f"   {date}")
    print()
    
    all_events = []
    
    import time
    start_time = time.time()
    
    for i, date_str in enumerate(dates, 1):
        print(f"\n[{i}/{len(dates)}] Processing {date_str}...", flush=True)
        
        # Load orderbook
        print(f"   Loading orderbook...", flush=True)
        ob_df = load_orderbook_sampled(date_str, sample_rate=10)
        print(f"   Orderbook: {len(ob_df):,} snapshots", flush=True)
        
        if len(ob_df) == 0:
            print(f"   ⚠️  No orderbook data, skipping", flush=True)
            continue
        
        # Process trades
        print(f"   Processing trades...", flush=True)
        events = process_trades(date_str, ob_df)
        print(f"   ✅ Events detected: {len(events)}", flush=True)
        
        all_events.extend(events)
        
        # ETA
        elapsed = time.time() - start_time
        avg_per_day = elapsed / i
        remaining = (len(dates) - i) * avg_per_day
        print(f"   ⏱️  Elapsed: {elapsed:.0f}s, ETA: {remaining:.0f}s", flush=True)
    
    # Save results
    df = pd.DataFrame(all_events)
    df.to_csv(args.output, index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ Complete! {len(all_events)} events detected")
    print(f"💾 Saved: {args.output}")
    print(f"{'='*80}")
    
    # Quick stats
    if len(df) > 0:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour
        
        print(f"\n📊 Hourly distribution:")
        hourly = df.groupby('hour').size()
        for hour, count in hourly.items():
            pct = count / len(df) * 100
            print(f"   Hour {hour:02d}: {count:4d} ({pct:5.1f}%)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
