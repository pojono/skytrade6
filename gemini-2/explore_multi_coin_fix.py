import pandas as pd
import numpy as np
import os
import glob
from itertools import combinations

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
START_DATE = "2025-01-01"
END_DATE = "2025-06-30"

def load_klines(symbol):
    pattern = f"{DATALAKE_DIR}/{symbol}/*_kline_1m.csv"
    files = glob.glob(pattern)
    filtered_files = [f for f in files if START_DATE <= os.path.basename(f).split('_')[0] <= END_DATE]
    if not filtered_files: return pd.DataFrame()
    
    dfs = []
    for f in filtered_files:
        try:
            df = pd.read_csv(f)
            # Remove any rows where close is small/negative (erroneous data as seen in debug)
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['startTime'] = pd.to_numeric(df['startTime'], errors='coerce')
            df = df.dropna(subset=['startTime', 'close'])
            df = df[df['close'] > 0]
            dfs.append(df)
        except: pass
            
    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df['datetime'] = pd.to_datetime(df['startTime'], unit='ms')
    df = df.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
    # Resample to 1min to align exactly, ffill missing
    df = df.resample('1min').ffill()
    return df['close']

data = {}
for sym in SYMBOLS:
    data[sym] = load_klines(sym)

prices = pd.DataFrame(data).ffill().dropna()

# Check for huge outliers in prices which cause insane returns
for sym in SYMBOLS:
    ret = prices[sym].pct_change()
    # Mask out ridiculous returns (e.g. > 10% in 1 minute)
    mask = (ret.abs() > 0.10)
    if mask.any():
        print(f"Warning: {sym} has {mask.sum()} extreme returns. Cleaning them up.")
        prices.loc[mask, sym] = np.nan
        prices[sym] = prices[sym].ffill()

returns = prices.pct_change().dropna()
returns.replace([np.inf, -np.inf], np.nan, inplace=True)
returns.dropna(inplace=True)

print(f"Loaded {len(returns)} minutes of aligned returns")

print("\n--- Return Correlation Matrix ---")
print(returns.corr().round(4))

# Explore multi-coin momentum (Breadth)
print("\n--- Breadth Strategy Exploration ---")
# Count how many coins are up in the last 15m
returns_15m = prices.pct_change(15).dropna()
breadth = (returns_15m > 0).sum(axis=1)

# Forward 15m returns for BTC
fwd_ret_15m = prices['BTCUSDT'].pct_change(15).shift(-15).dropna()

df = pd.DataFrame({'breadth': breadth, 'fwd_ret_BTC': fwd_ret_15m}).dropna()
for b in range(6):
    mean_ret = df[df['breadth'] == b]['fwd_ret_BTC'].mean() * 10000
    win_rate = (df[df['breadth'] == b]['fwd_ret_BTC'] > 0).mean() * 100
    count = len(df[df['breadth'] == b])
    print(f"Breadth = {b} ({count} samples): Next 15m BTC Return = {mean_ret:.2f} bps, WR: {win_rate:.1f}%")

