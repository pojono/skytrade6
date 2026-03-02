#!/usr/bin/env python3
"""Extract multi-scale features for Sample 3."""
import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import json
import sys

DATA_DIR_TRADE = Path("data_bybit/SOLUSDT/trade/dataminer/data/archive/raw")
WINDOWS = {'10s': 10, '30s': 30, '2m': 120, '5m': 300, '15m': 900}

def load_trades_window(date_str, start_ts, end_ts):
    """Load trades in a time window."""
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

def calculate_window_features(trades_df, event_price, window_name):
    """Calculate features for a specific time window."""
    if len(trades_df) == 0:
        return {}
    
    prefix = f"{window_name}_"
    features = {}
    
    # Volatility
    if len(trades_df) > 1:
        returns = trades_df['price'].pct_change().dropna() * 10000
        features[f'{prefix}vol'] = returns.std()
        features[f'{prefix}vol_mean'] = returns.mean()
    else:
        features[f'{prefix}vol'] = 0
        features[f'{prefix}vol_mean'] = 0
    
    # Price drift
    if len(trades_df) > 0:
        first_price = trades_df['price'].iloc[0]
        last_price = trades_df['price'].iloc[-1]
        features[f'{prefix}drift'] = (last_price - first_price) / first_price * 10000
        features[f'{prefix}distance'] = (event_price - last_price) / last_price * 10000
    else:
        features[f'{prefix}drift'] = 0
        features[f'{prefix}distance'] = 0
    
    # Range
    if len(trades_df) > 0:
        high = trades_df['price'].max()
        low = trades_df['price'].min()
        range_bps = (high - low) / trades_df['price'].mean() * 10000
        features[f'{prefix}range'] = range_bps
        
        if high > low:
            features[f'{prefix}position'] = (event_price - low) / (high - low)
        else:
            features[f'{prefix}position'] = 0.5
    else:
        features[f'{prefix}range'] = 0
        features[f'{prefix}position'] = 0.5
    
    # Activity
    duration_s = (trades_df['timestamp'].max() - trades_df['timestamp'].min()) / 1000
    if duration_s > 0:
        features[f'{prefix}trade_rate'] = len(trades_df) / duration_s
    else:
        features[f'{prefix}trade_rate'] = 0
    
    features[f'{prefix}volume'] = trades_df['volume'].sum()
    
    # Buy/Sell imbalance
    buy_vol = trades_df[trades_df['side'] == 'Buy']['volume'].sum()
    sell_vol = trades_df[trades_df['side'] == 'Sell']['volume'].sum()
    total_vol = buy_vol + sell_vol
    if total_vol > 0:
        features[f'{prefix}imbalance'] = (buy_vol - sell_vol) / total_vol
    else:
        features[f'{prefix}imbalance'] = 0
    
    # Trend strength
    if len(trades_df) > 10:
        trades_df_copy = trades_df.copy()
        trades_df_copy['time_norm'] = (trades_df_copy['timestamp'] - trades_df_copy['timestamp'].min()) / 1000
        slope = np.polyfit(trades_df_copy['time_norm'], trades_df_copy['price'], 1)[0]
        features[f'{prefix}slope'] = slope / event_price * 10000
    else:
        features[f'{prefix}slope'] = 0
    
    return features

print("="*80, flush=True)
print("Extracting multi-scale features for Sample 3", flush=True)
print("="*80 + "\n", flush=True)

# Load Sample 3
df = pd.read_csv("results/sample3_with_returns.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
df['date'] = df['datetime'].dt.date

print(f"Total events: {len(df)}", flush=True)

all_features = []

for idx, event in df.iterrows():
    if (idx + 1) % 20 == 0:
        print(f"   Processed {idx + 1}/{len(df)} events...", flush=True)
    
    event_ts = event['timestamp']
    event_price = event['event_price']
    event_date = event['date']
    date_str = event_date.strftime('%Y-%m-%d')
    
    event_features = {
        'timestamp': event_ts,
        'datetime': event['datetime'],
        'classification': event['classification'],
        'ret_30s': event['ret_30s'],
        'flow_impact': event['flow_impact'],
        'imbalance': event['imbalance'],
        'direction': event['direction'],
        'hour': pd.to_datetime(event['datetime']).hour
    }
    
    for window_name, window_seconds in WINDOWS.items():
        start_ts = event_ts - (window_seconds * 1000)
        end_ts = event_ts
        
        trades_df = load_trades_window(date_str, start_ts, end_ts)
        
        if len(trades_df) >= 5:
            window_features = calculate_window_features(trades_df, event_price, window_name)
            event_features.update(window_features)
        else:
            for key in ['vol', 'vol_mean', 'drift', 'distance', 'range', 'position', 
                       'trade_rate', 'volume', 'imbalance', 'slope']:
                event_features[f'{window_name}_{key}'] = np.nan
    
    all_features.append(event_features)

result_df = pd.DataFrame(all_features)
result_df.to_csv("results/sample3_multiscale.csv", index=False)

print(f"\n✅ Extracted features for {len(result_df)} events", flush=True)
print(f"💾 Saved: results/sample3_multiscale.csv", flush=True)
