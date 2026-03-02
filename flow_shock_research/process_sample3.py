#!/usr/bin/env python3
"""Process Sample 3 (June 2025) - calculate returns and multi-scale features."""
import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import json
import sys

DATA_DIR_TRADE = Path("data_bybit/SOLUSDT/trade/dataminer/data/archive/raw")
HORIZONS = [5, 15, 30, 60, 120, 300]
WINDOWS = {'10s': 10, '30s': 30, '2m': 120, '5m': 300, '15m': 900}

def load_trades_for_date(date_str):
    """Load all trades for a specific date."""
    print(f"   Loading trades for {date_str}...", flush=True)
    trades = []
    
    for hour in range(24):
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
                                trades.append({
                                    'timestamp': int(trade['T']),
                                    'price': float(trade['p']),
                                    'side': trade['S']
                                })
                    except:
                        continue
        except:
            continue
    
    return pd.DataFrame(trades).sort_values('timestamp') if trades else pd.DataFrame()

def get_price_at_time(trades_df, target_ts, window_ms=1000):
    """Get price closest to target timestamp."""
    mask = (trades_df['timestamp'] >= target_ts - window_ms) & (trades_df['timestamp'] <= target_ts + window_ms)
    nearby = trades_df[mask].copy()
    
    if len(nearby) == 0:
        return None
    
    nearby['diff'] = abs(nearby['timestamp'] - target_ts)
    closest = nearby.loc[nearby['diff'].idxmin()]
    return closest['price']

# Load Sample 3
print("="*80, flush=True)
print("Processing Sample 3 (June 10-16, 2025)", flush=True)
print("="*80 + "\n", flush=True)

events_df = pd.read_csv("results/sample3_jun2025.csv")
events_df['datetime'] = pd.to_datetime(events_df['datetime'])
events_df['date'] = events_df['datetime'].dt.date

print(f"Total events: {len(events_df)}", flush=True)

# Get unique dates
unique_dates = sorted(events_df['date'].unique())
print(f"Unique dates: {len(unique_dates)}", flush=True)

# Load trades for each date
print(f"\nLoading trade data...", flush=True)
trades_by_date = {}
for date in unique_dates:
    date_str = date.strftime('%Y-%m-%d')
    trades_by_date[date_str] = load_trades_for_date(date_str)
    print(f"   {date_str}: {len(trades_by_date[date_str]):,} trades", flush=True)

# Calculate returns
print(f"\nCalculating forward returns...", flush=True)
results = []

for idx, event in events_df.iterrows():
    if (idx + 1) % 20 == 0:
        print(f"   Processed {idx + 1}/{len(events_df)} events...", flush=True)
    
    event_ts = event['timestamp']
    event_price = event['price']
    event_date = event['date']
    event_direction = event['direction']
    
    date_str = event_date.strftime('%Y-%m-%d')
    
    if date_str not in trades_by_date:
        continue
    
    trades_df = trades_by_date[date_str]
    
    returns = {
        'timestamp': event_ts,
        'datetime': event['datetime'],
        'event_price': event_price,
        'direction': event_direction,
        'flow_impact': event['flow_impact'],
        'imbalance': event['imbalance'],
        'max_run': event['max_run']
    }
    
    for horizon in HORIZONS:
        target_ts = event_ts + (horizon * 1000)
        future_price = get_price_at_time(trades_df, target_ts)
        
        if future_price:
            ret_bps = ((future_price - event_price) / event_price) * 10000
            if event_direction == 'Sell':
                ret_bps = -ret_bps
            returns[f'ret_{horizon}s'] = ret_bps
            returns[f'price_{horizon}s'] = future_price
        else:
            returns[f'ret_{horizon}s'] = None
            returns[f'price_{horizon}s'] = None
    
    results.append(returns)

df = pd.DataFrame(results)

# Classification
def classify(row):
    if pd.isna(row.get('ret_30s')):
        return None
    ret_30s = row['ret_30s']
    if ret_30s > 10:
        return 'Continuation'
    elif ret_30s < -10:
        return 'Reversal'
    else:
        return 'Exhaustion'

df['classification'] = df.apply(classify, axis=1)
df = df[df['ret_30s'].notna()].copy()

print(f"\n✅ Calculated returns for {len(df)} events", flush=True)

# Save
df.to_csv("results/sample3_with_returns.csv", index=False)
print(f"💾 Saved: results/sample3_with_returns.csv", flush=True)

# Analysis
print(f"\n{'='*80}", flush=True)
print(f"SAMPLE 3 ANALYSIS", flush=True)
print(f"{'='*80}\n", flush=True)

print(f"📋 Classification:", flush=True)
for cls, count in df['classification'].value_counts().items():
    pct = count / len(df) * 100
    print(f"   {cls:15s}: {count:4d} ({pct:5.1f}%)", flush=True)

print(f"\n📈 Returns:", flush=True)
for horizon in HORIZONS:
    col = f'ret_{horizon}s'
    mean_ret = df[col].mean()
    print(f"   t+{horizon:3d}s: {mean_ret:+7.2f} bps", flush=True)

print(f"\n📊 Hourly distribution:", flush=True)
df['hour'] = pd.to_datetime(df['datetime']).dt.hour
for hour, count in df.groupby('hour').size().items():
    pct = count / len(df) * 100
    print(f"   Hour {hour:02d}: {count:3d} ({pct:5.1f}%)", flush=True)

print(f"\n{'='*80}", flush=True)
