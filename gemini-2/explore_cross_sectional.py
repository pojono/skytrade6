import pandas as pd
import numpy as np
import os
import glob
import gc
import warnings
warnings.filterwarnings('ignore')

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
START_DATE = "2025-01-01"
END_DATE = "2025-02-28"

all_symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d)) and d.endswith('USDT')]
symbols_subset = all_symbols[:50] 

def load_klines_clean(symbol):
    pattern = f"{DATALAKE_DIR}/{symbol}/*_kline_1m.csv"
    files = glob.glob(pattern)
    filtered_files = [f for f in files if START_DATE <= os.path.basename(f).split('_')[0] <= END_DATE]
    if not filtered_files: return None
    
    dfs = []
    for f in filtered_files:
        try:
            df = pd.read_csv(f, usecols=['startTime', 'close', 'volume'])
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            df['startTime'] = pd.to_numeric(df['startTime'], errors='coerce')
            df = df.dropna()
            df = df[df['close'] > 0]
            dfs.append(df)
        except: pass
            
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True)
    df['datetime'] = pd.to_datetime(df['startTime'], unit='ms')
    df = df.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
    df = df.resample('1min').ffill()
    
    # Strictly filter outliers: if any 1min return > 10%, forward fill instead
    ret = df['close'].pct_change()
    mask = (ret.abs() > 0.10)
    df.loc[mask, 'close'] = np.nan
    df['close'] = df['close'].ffill()
    
    return df['close'].rename(symbol)

print("Loading data...")
data_series = []
for sym in symbols_subset:
    s = load_klines_clean(sym)
    if s is not None:
        data_series.append(s)

prices = pd.concat(data_series, axis=1).ffill().dropna()
del data_series
gc.collect()

print(f"Loaded {len(prices)} minutes for {len(prices.columns)} coins")

# Define strategy parameters
LOOKBACK = 60 # 60 minutes
HOLDING = 15 # 15 minutes

# Calculate lookback returns
ret_lookback = (prices / prices.shift(LOOKBACK)) - 1
ret_lookback = ret_lookback.dropna()

# Calculate forward returns
ret_forward = (prices.shift(-HOLDING) / prices) - 1
ret_forward = ret_forward.dropna()

# Align
common_idx = ret_lookback.index.intersection(ret_forward.index)
ret_lookback = ret_lookback.loc[common_idx]
ret_forward = ret_forward.loc[common_idx]

# Cross-sectional ranking
# Every 'HOLDING' minutes, we rank coins and take positions
timestamps = common_idx[::HOLDING]

results = []

for t in timestamps:
    current_lookback = ret_lookback.loc[t]
    current_forward = ret_forward.loc[t]
    
    # Drop NaNs for this specific timestamp
    valid_mask = current_lookback.notna() & current_forward.notna()
    if valid_mask.sum() < 10: # Need at least 10 coins
        continue
        
    rankings = current_lookback[valid_mask].rank(pct=True)
    
    # Long top 20%, Short bottom 20%
    longs = rankings[rankings >= 0.8].index
    shorts = rankings[rankings <= 0.2].index
    
    long_ret = current_forward[longs].mean() if len(longs) > 0 else 0
    short_ret = current_forward[shorts].mean() if len(shorts) > 0 else 0
    
    portfolio_ret = 0.5 * long_ret - 0.5 * short_ret
    
    results.append({
        'datetime': t,
        'long_ret': long_ret,
        'short_ret': short_ret,
        'portfolio_ret': portfolio_ret
    })

res_df = pd.DataFrame(results).set_index('datetime')

# Fee calculation
# Assume we flip all coins every HOLDING period
# 0.1% taker fee per leg = 0.2% round trip = 20 bps
FEE_BPS = 20
fee_pct = FEE_BPS / 10000

res_df['net_ret'] = res_df['portfolio_ret'] - fee_pct

print(f"\n--- Strategy Performance (Lookback={LOOKBACK}m, Holding={HOLDING}m) ---")
print(f"Total Trades: {len(res_df)}")
print(f"Gross Mean Return per trade: {res_df['portfolio_ret'].mean() * 10000:.2f} bps")
print(f"Net Mean Return per trade: {res_df['net_ret'].mean() * 10000:.2f} bps")
print(f"Win Rate (Gross): {(res_df['portfolio_ret'] > 0).mean() * 100:.1f}%")
print(f"Win Rate (Net): {(res_df['net_ret'] > 0).mean() * 100:.1f}%")
print(f"Cumulative Return (Net): {(1 + res_df['net_ret']).prod() - 1:.2%}")

