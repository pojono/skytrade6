import pandas as pd
import numpy as np
import os
import glob
from itertools import combinations

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
START_DATE = "2025-01-01"
END_DATE = "2025-06-30"

def load_klines(symbol, is_spot=False):
    pattern = f"{DATALAKE_DIR}/{symbol}/*_kline_1m{'_spot' if is_spot else ''}.csv"
    files = glob.glob(pattern)
    filtered_files = [f for f in files if START_DATE <= os.path.basename(f).split('_')[0] <= END_DATE]
    if not filtered_files: return pd.DataFrame()
    
    dfs = []
    for f in filtered_files:
        try:
            df = pd.read_csv(f)
            for col in ['startTime', 'close', 'volume']:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            dfs.append(df)
        except: pass
            
    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True).dropna(subset=['startTime', 'close'])
    df['datetime'] = pd.to_datetime(df['startTime'], unit='ms')
    return df.sort_values('startTime').drop_duplicates('startTime').set_index('datetime')

# Load all futures data
data = {}
for sym in SYMBOLS:
    df = load_klines(sym, is_spot=False)
    if not df.empty:
        data[sym] = df['close'].pct_change()

ret_df = pd.DataFrame(data).dropna()
print(f"Loaded {len(ret_df)} minutes of aligned returns")

# Correlation matrix
print("\n--- Return Correlation Matrix ---")
print(ret_df.corr().round(4))

# Lead-lag relationships across coins (e.g. BTC leads ALTs)
print("\n--- Cross-Coin Lead-Lag (1m) ---")
for coin1, coin2 in combinations(ret_df.columns, 2):
    c1_leads_c2 = ret_df[coin1].shift(1).corr(ret_df[coin2])
    c2_leads_c1 = ret_df[coin2].shift(1).corr(ret_df[coin1])
    if abs(c1_leads_c2) > 0.05 or abs(c2_leads_c1) > 0.05:
        print(f"{coin1}(t-1) -> {coin2}(t): {c1_leads_c2:.4f}")
        print(f"{coin2}(t-1) -> {coin1}(t): {c2_leads_c1:.4f}")

# Cross-coin Lead-Lag (5m smoothed to 1m ahead)
print("\n--- Cross-Coin Lead-Lag (5m rolling -> 1m) ---")
ret_5m = ret_df.rolling(5).sum()
for coin1, coin2 in combinations(ret_df.columns, 2):
    c1_leads_c2 = ret_5m[coin1].shift(1).corr(ret_df[coin2])
    c2_leads_c1 = ret_5m[coin2].shift(1).corr(ret_df[coin1])
    if abs(c1_leads_c2) > 0.05 or abs(c2_leads_c1) > 0.05:
        print(f"{coin1}(5m prev) -> {coin2}(t): {c1_leads_c2:.4f}")
        print(f"{coin2}(5m prev) -> {coin1}(t): {c2_leads_c1:.4f}")

