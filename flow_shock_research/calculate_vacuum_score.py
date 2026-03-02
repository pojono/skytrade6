#!/usr/bin/env python3
"""
Step 2: Calculate Liquidity Vacuum Score.

For each event, measure market emptiness in 1-10s window after forced flow:
1. Depth collapse: TopDepth / median(TopDepth, 15m)
2. Spread expansion: Spread / median(Spread, 15m)
3. Price impact per notional: |return_2s| / AggNotional_2s

VacuumScore = w1*(1-DepthDrop) + w2*(SpreadRatio-1) + w3*ImpactPerFlow
"""
import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import json
from multiprocessing import Pool

DATA_DIR_TRADE = Path("data_bybit/SOLUSDT/trade/dataminer/data/archive/raw")
DATA_DIR_OB = Path("data_bybit/SOLUSDT/orderbook_all/dataminer/data/archive/raw")

def load_orderbook_window(date_str, hour, start_ts, end_ts):
    """Load orderbook snapshots in time window."""
    ob_file = DATA_DIR_OB / f"dt={date_str}" / f"hr={hour:02d}" / "exchange=bybit" / "source=ws" / "market=linear" / "stream=orderbook.500" / "symbol=SOLUSDT" / "data.jsonl.gz"
    
    if not ob_file.exists():
        return []
    
    snapshots = []
    try:
        with gzip.open(ob_file, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'result' in data and 'data' in data['result']:
                        ts = int(data['result']['ts'])
                        if start_ts <= ts <= end_ts:
                            ob_data = data['result']['data']
                            bids = ob_data.get('b', [])
                            asks = ob_data.get('a', [])
                            
                            if bids and asks:
                                best_bid = float(bids[0][0])
                                best_ask = float(asks[0][0])
                                spread = best_ask - best_bid
                                
                                # Top depth (L1)
                                bid_size = float(bids[0][1])
                                ask_size = float(asks[0][1])
                                top_depth = bid_size + ask_size
                                
                                snapshots.append({
                                    'timestamp': ts,
                                    'spread': spread,
                                    'spread_bps': (spread / best_bid) * 10000,
                                    'top_depth': top_depth,
                                    'mid_price': (best_bid + best_ask) / 2
                                })
                except:
                    continue
    except:
        pass
    
    return snapshots

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

def calculate_vacuum_metrics(row_tuple):
    """Calculate vacuum metrics for one event."""
    idx, row = row_tuple
    
    event_ts = row['timestamp']
    event_price = row['event_price']
    date_str = row['date']
    hour = row['hour']
    
    # Baseline: 15m before event
    baseline_start = event_ts - (15 * 60 * 1000)
    baseline_end = event_ts
    
    # Post-event window: 1-10s after event
    post_start = event_ts + 1000
    post_end = event_ts + 10000
    
    metrics = {
        'depth_drop': np.nan,
        'spread_ratio': np.nan,
        'price_impact_per_flow': np.nan,
        'vacuum_score': np.nan
    }
    
    try:
        # Load baseline orderbook (15m before)
        baseline_obs = []
        for h in [hour - 1, hour]:
            if h >= 0:
                obs = load_orderbook_window(date_str, h, baseline_start, baseline_end)
                baseline_obs.extend(obs)
        
        if len(baseline_obs) < 10:
            return metrics
        
        baseline_df = pd.DataFrame(baseline_obs)
        median_depth = baseline_df['top_depth'].median()
        median_spread = baseline_df['spread_bps'].median()
        
        # Load post-event orderbook (1-10s after)
        post_obs = load_orderbook_window(date_str, hour, post_start, post_end)
        
        if len(post_obs) < 3:
            return metrics
        
        post_df = pd.DataFrame(post_obs)
        
        # 1. Depth collapse
        min_depth_post = post_df['top_depth'].min()
        depth_drop = min_depth_post / median_depth if median_depth > 0 else 1.0
        metrics['depth_drop'] = depth_drop
        
        # 2. Spread expansion
        max_spread_post = post_df['spread_bps'].max()
        spread_ratio = max_spread_post / median_spread if median_spread > 0 else 1.0
        metrics['spread_ratio'] = spread_ratio
        
        # 3. Price impact per notional (2s window)
        trades_2s = load_trades_window(date_str, event_ts, event_ts + 2000)
        
        if len(trades_2s) > 0:
            first_price = event_price
            last_price = trades_2s['price'].iloc[-1]
            price_move = abs(last_price - first_price) / first_price * 10000  # bps
            
            total_notional = (trades_2s['price'] * trades_2s['volume']).sum()
            
            if total_notional > 0:
                impact_per_flow = price_move / (total_notional / 1000)  # per $1k
                metrics['price_impact_per_flow'] = impact_per_flow
        
        # Calculate VacuumScore (equal weights for now)
        w1, w2, w3 = 1.0, 1.0, 1.0
        
        components = []
        if not np.isnan(depth_drop):
            components.append(w1 * (1 - depth_drop))
        if not np.isnan(spread_ratio):
            components.append(w2 * (spread_ratio - 1))
        if not np.isnan(metrics['price_impact_per_flow']):
            # Normalize impact (cap at 10 for reasonable scale)
            impact_norm = min(metrics['price_impact_per_flow'], 10) / 10
            components.append(w3 * impact_norm)
        
        if components:
            metrics['vacuum_score'] = sum(components) / len(components)
    
    except Exception as e:
        pass
    
    return metrics

print("="*80, flush=True)
print("📊 STEP 2: CALCULATE LIQUIDITY VACUUM SCORE", flush=True)
print("="*80, flush=True)
print("\nMeasuring market emptiness after forced flow events", flush=True)
print("Using 6 parallel workers", flush=True)
print("="*80 + "\n", flush=True)

# Load dataset
df = pd.read_parquet("results/production_dataset.parquet")
print(f"Total events: {len(df)}", flush=True)
print("Processing vacuum metrics in parallel...\n", flush=True)

# Calculate vacuum metrics in parallel
with Pool(6) as pool:
    vacuum_metrics = pool.map(calculate_vacuum_metrics, df.iterrows())

# Add to dataset
vacuum_df = pd.DataFrame(vacuum_metrics)
for col in vacuum_df.columns:
    df[col] = vacuum_df[col]

print(f"✅ Vacuum metrics calculated", flush=True)
print(f"\nData quality:", flush=True)
print(f"   Events with depth_drop: {df['depth_drop'].notna().sum()} ({df['depth_drop'].notna().mean():.1%})", flush=True)
print(f"   Events with spread_ratio: {df['spread_ratio'].notna().sum()} ({df['spread_ratio'].notna().mean():.1%})", flush=True)
print(f"   Events with price_impact: {df['price_impact_per_flow'].notna().sum()} ({df['price_impact_per_flow'].notna().mean():.1%})", flush=True)
print(f"   Events with vacuum_score: {df['vacuum_score'].notna().sum()} ({df['vacuum_score'].notna().mean():.1%})", flush=True)

# Summary stats
print(f"\n{'='*80}", flush=True)
print(f"VACUUM METRICS SUMMARY", flush=True)
print(f"{'='*80}\n", flush=True)

valid = df[df['vacuum_score'].notna()]
if len(valid) > 0:
    print(f"Valid events: {len(valid)}", flush=True)
    print(f"\nDepth Drop (lower = more vacuum):", flush=True)
    print(f"   Mean: {valid['depth_drop'].mean():.3f}", flush=True)
    print(f"   Median: {valid['depth_drop'].median():.3f}", flush=True)
    print(f"   Q25: {valid['depth_drop'].quantile(0.25):.3f}", flush=True)
    print(f"   Q75: {valid['depth_drop'].quantile(0.75):.3f}", flush=True)
    
    print(f"\nSpread Ratio (higher = more vacuum):", flush=True)
    print(f"   Mean: {valid['spread_ratio'].mean():.3f}", flush=True)
    print(f"   Median: {valid['spread_ratio'].median():.3f}", flush=True)
    print(f"   Q25: {valid['spread_ratio'].quantile(0.25):.3f}", flush=True)
    print(f"   Q75: {valid['spread_ratio'].quantile(0.75):.3f}", flush=True)
    
    print(f"\nPrice Impact per $1k (higher = more vacuum):", flush=True)
    impact_valid = valid[valid['price_impact_per_flow'].notna()]
    if len(impact_valid) > 0:
        print(f"   Mean: {impact_valid['price_impact_per_flow'].mean():.3f}", flush=True)
        print(f"   Median: {impact_valid['price_impact_per_flow'].median():.3f}", flush=True)
    
    print(f"\nVacuum Score (higher = stronger vacuum):", flush=True)
    print(f"   Mean: {valid['vacuum_score'].mean():.3f}", flush=True)
    print(f"   Median: {valid['vacuum_score'].median():.3f}", flush=True)
    print(f"   Q25: {valid['vacuum_score'].quantile(0.25):.3f}", flush=True)
    print(f"   Q50: {valid['vacuum_score'].quantile(0.50):.3f}", flush=True)
    print(f"   Q75: {valid['vacuum_score'].quantile(0.75):.3f}", flush=True)
    print(f"   Q90: {valid['vacuum_score'].quantile(0.90):.3f}", flush=True)

# Save
df.to_parquet("results/production_dataset_vacuum.parquet", index=False)
df.to_csv("results/production_dataset_vacuum.csv", index=False)

print(f"\n💾 Saved: results/production_dataset_vacuum.parquet", flush=True)
print(f"💾 Saved: results/production_dataset_vacuum.csv", flush=True)

print(f"\n{'='*80}", flush=True)
print(f"✅ STEP 2 COMPLETE - Vacuum scores ready", flush=True)
print(f"{'='*80}", flush=True)
