import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def test_safe_resample(symbol="BTCUSDT"):
    start_date = "2025-01-01"
    
    kline_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_kline_1m.csv")))
    kline_files = [f for f in kline_files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name and f.name >= start_date]
    dfs = []
    for f in kline_files[:30]:
        try: dfs.append(pd.read_csv(f, usecols=['open_time', 'close'], engine='c'))
        except: pass
    kline_df = pd.concat(dfs, ignore_index=True)
    kline_df.rename(columns={'open_time': 'timestamp'}, inplace=True)
    kline_df['timestamp'] = pd.to_numeric(kline_df['timestamp'])
    if kline_df['timestamp'].max() < 1e11: kline_df['timestamp'] *= 1000
    kline_df.set_index('timestamp', inplace=True)
    kline_df.index = pd.to_datetime(kline_df.index, unit='ms')
    
    # Safe resample: The index label will be the EXACT execution minute (e.g. 08:00:00).
    # It will contain data strictly from 07:00:00 up to 07:59:00 (which closed exactly at 08:00:00).
    # Then we can safely execute starting at the 08:00:00 candle (which opens at 08:00:00).
    hourly = kline_df.resample('1h', label='right', closed='left').agg({'close': 'last'})
    
    # Check
    print("Raw 1m:")
    print(kline_df.loc['2025-01-01 07:58:00':'2025-01-01 08:02:00'])
    
    print("\nSafe Hourly:")
    print(hourly.loc['2025-01-01 07:00:00':'2025-01-01 08:00:00'])

test_safe_resample()
