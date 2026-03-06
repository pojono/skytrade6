import pandas as pd
import numpy as np
from pathlib import Path

DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def analyze_trend_regime():
    kline_files = sorted(list((DATALAKE / f"bybit/BTCUSDT").glob("*_kline_1m.csv")))
    kline_files = [f for f in kline_files if "2024-01-01" <= f.name[:10] <= "2024-12-31" and "mark" not in f.name and "index" not in f.name and "premium" not in f.name]
    
    dfs = []
    for f in kline_files:
        try: dfs.append(pd.read_csv(f, usecols=['startTime', 'close'], engine='c'))
        except: pass
        
    df = pd.concat(dfs, ignore_index=True)
    df.rename(columns={'startTime': 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    if df['timestamp'].max() < 1e11: df['timestamp'] *= 1000
    df.set_index('timestamp', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')
    
    hourly = df.resample('1h').agg({'close': 'last'}).dropna()
    
    # Let's calculate a Macro Regime Filter
    # 1. 200-Hour SMA (Weekly Trend)
    hourly['sma_200'] = hourly['close'].rolling(200).mean()
    
    # 2. ADX (Average Directional Index) approximation using rolling max/min over 14 days
    hourly['rolling_max'] = hourly['close'].rolling(336).max()
    hourly['rolling_min'] = hourly['close'].rolling(336).min()
    hourly['macro_range_pct'] = (hourly['rolling_max'] - hourly['rolling_min']) / hourly['rolling_min'] * 100
    
    print("\n--- Regime Filter Diagnostics ---")
    trend_period = hourly.loc['2024-02-01':'2024-04-30']
    chop_period = hourly.loc['2024-08-01':'2024-10-31']
    
    print(f"Q1 (Trend) Avg 14-Day Range: {trend_period['macro_range_pct'].mean():.2f}%")
    print(f"Q3 (Chop) Avg 14-Day Range: {chop_period['macro_range_pct'].mean():.2f}%")
    
    print("\nLet's see if we can filter Chop by requiring a minimum Macro Range of >15%")
    chop_filtered = chop_period[chop_period['macro_range_pct'] > 15.0]
    print(f"Chop Period Days Active if Filtered: {len(chop_filtered)/24:.1f} out of {len(chop_period)/24:.1f}")

analyze_trend_regime()
