import pandas as pd
import numpy as np
import os
import glob
from itertools import combinations
import gc
import warnings
warnings.filterwarnings('ignore')

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
START_DATE = "2025-01-01"
END_DATE = "2025-02-28" # Shorter period for faster exploration

# Get all available symbols
all_symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d)) and d.endswith('USDT')]
print(f"Found {len(all_symbols)} symbols. Using subset for initial check.")

# Start with top 50 by alphabetical to save memory for initial exploration, 
# but process them in chunks.
symbols_subset = all_symbols[:50] 

def load_klines_chunk(symbol):
    pattern = f"{DATALAKE_DIR}/{symbol}/*_kline_1m.csv"
    files = glob.glob(pattern)
    filtered_files = [f for f in files if START_DATE <= os.path.basename(f).split('_')[0] <= END_DATE]
    if not filtered_files: return None
    
    dfs = []
    for f in filtered_files:
        try:
            # Only load necessary columns to save memory
            df = pd.read_csv(f, usecols=['startTime', 'close'])
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['startTime'] = pd.to_numeric(df['startTime'], errors='coerce')
            df = df.dropna()
            df = df[df['close'] > 0]
            dfs.append(df)
        except: pass
            
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True)
    df['datetime'] = pd.to_datetime(df['startTime'], unit='ms')
    df = df.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
    # Use '1min' instead of '1T' to avoid warnings
    df = df.resample('1min').ffill()
    
    # Filter extreme outliers
    ret = df['close'].pct_change()
    mask = (ret.abs() > 0.10)
    df.loc[mask, 'close'] = np.nan
    df['close'] = df['close'].ffill()
    
    return df['close'].rename(symbol)

# Load data efficiently
data_series = []
for sym in symbols_subset:
    s = load_klines_chunk(sym)
    if s is not None:
        data_series.append(s)

prices = pd.concat(data_series, axis=1).ffill().dropna()
del data_series
gc.collect()

returns = prices.pct_change().dropna()
returns.replace([np.inf, -np.inf], np.nan, inplace=True)
returns.dropna(inplace=True)

print(f"Loaded {len(returns)} minutes of aligned returns for {len(returns.columns)} coins")

# Explore multi-coin momentum (Breadth)
print("\n--- Breadth Strategy Exploration (15m window) ---")

# Compute 15m return directly using prices
returns_15m = (prices / prices.shift(15) - 1).dropna()
# Align properly
common_idx = returns_15m.index.intersection(prices.index)
returns_15m = returns_15m.loc[common_idx]
prices_aligned = prices.loc[common_idx]

breadth_pct = (returns_15m > 0).mean(axis=1) # percentage of coins that are UP

# Forward 15m returns for top coins
top_coins = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
valid_top_coins = [c for c in top_coins if c in prices_aligned.columns]

for target_coin in valid_top_coins:
    print(f"\n--- Target: {target_coin} ---")
    fwd_ret_15m = (prices_aligned[target_coin].shift(-15) / prices_aligned[target_coin] - 1).dropna()
    
    # Align again
    common_idx2 = breadth_pct.index.intersection(fwd_ret_15m.index)
    b_pct = breadth_pct.loc[common_idx2]
    f_ret = fwd_ret_15m.loc[common_idx2]
    
    df = pd.DataFrame({'breadth_pct': b_pct, 'fwd_ret': f_ret})
    
    # Bucket into deciles
    df['breadth_bucket'] = pd.qcut(df['breadth_pct'], 10, labels=False, duplicates='drop')
    
    for b in sorted(df['breadth_bucket'].unique()):
        subset = df[df['breadth_bucket'] == b]
        mean_ret = subset['fwd_ret'].mean() * 10000
        win_rate = (subset['fwd_ret'] > 0).mean() * 100
        count = len(subset)
        min_b = subset['breadth_pct'].min() * 100
        max_b = subset['breadth_pct'].max() * 100
        print(f"Decile {b} ({min_b:.1f}% - {max_b:.1f}% up) | N={count}: Fwd 15m Ret = {mean_ret:6.2f} bps | WR = {win_rate:.1f}%")

