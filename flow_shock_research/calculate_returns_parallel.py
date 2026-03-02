#!/usr/bin/env python3
"""
Calculate forward returns for production dataset - PARALLEL.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import json
from multiprocessing import Pool

DATA_DIR_TRADE = Path("data_bybit/SOLUSDT/trade/dataminer/data/archive/raw")

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
                                        'price': float(trade['p'])
                                    })
                    except:
                        continue
        except:
            continue
    
    return pd.DataFrame(trades).sort_values('timestamp') if trades else pd.DataFrame()

def calculate_returns(row_tuple):
    """Calculate forward returns for one event."""
    idx, row = row_tuple
    
    event_ts = row['timestamp']
    event_price = row['event_price']
    date_str = row['date']
    
    returns = {}
    
    for window_name, window_ms in [('5s', 5000), ('15s', 15000), ('30s', 30000), 
                                     ('60s', 60000), ('120s', 120000), ('300s', 300000)]:
        end_ts = event_ts + window_ms
        
        trades = load_trades_window(date_str, event_ts, end_ts)
        
        if len(trades) > 0:
            future_price = trades['price'].iloc[-1]
            ret = (future_price - event_price) / event_price * 10000
            returns[f'ret_{window_name}'] = ret
        else:
            returns[f'ret_{window_name}'] = np.nan
    
    return returns

print("Calculating returns in parallel...")

df = pd.read_parquet("results/production_dataset_vacuum.parquet")

with Pool(6) as pool:
    returns_list = pool.map(calculate_returns, df.iterrows())

returns_df = pd.DataFrame(returns_list)
for col in returns_df.columns:
    df[col] = returns_df[col]

df.to_parquet("results/production_dataset_vacuum.parquet", index=False)
print(f"✅ Returns calculated and saved")
