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
    
    # Feature 3: Long-term ADX / Trend Strength
    # Calculate rolling 30-day (720 hour) high and low
    hourly['hh_30d'] = hourly['close'].rolling(720).max()
    hourly['ll_30d'] = hourly['close'].rolling(720).min()
    
    # Calculate how close the current price is to the extremes (0 = at low, 1 = at high)
    hourly['range_pos'] = (hourly['close'] - hourly['ll_30d']) / (hourly['hh_30d'] - hourly['ll_30d'])
    
    # "Chop" is usually defined as being stuck in the middle of a long-term range (0.3 to 0.7) 
    # without breaking out of either side.
    
    # Let's also look at the 7-day Historical Volatility
    hourly['returns'] = hourly['close'].pct_change()
    hourly['hv_7d'] = hourly['returns'].rolling(168).std() * np.sqrt(24 * 365) * 100
    
    print("\n--- Regime Filter Diagnostics (HV & Range Pos) ---")
    trend_period = hourly.loc['2024-02-01':'2024-04-30']
    chop_period = hourly.loc['2024-08-01':'2024-10-31']
    
    print(f"Q1 (Trend) Avg 7d HV: {trend_period['hv_7d'].mean():.2f}%")
    print(f"Q3 (Chop) Avg 7d HV: {chop_period['hv_7d'].mean():.2f}%")
    
    print(f"\nQ1 (Trend) Avg Range Position: {trend_period['range_pos'].mean():.2f}")
    print(f"Q3 (Chop) Avg Range Position: {chop_period['range_pos'].mean():.2f}")
    
    # If HV < 40%, market is dead.
    print(f"\nTime spent in HV < 45%:")
    print(f"Q1: {(trend_period['hv_7d'] < 45).mean() * 100:.1f}%")
    print(f"Q3: {(chop_period['hv_7d'] < 45).mean() * 100:.1f}%")

analyze_trend_regime()
