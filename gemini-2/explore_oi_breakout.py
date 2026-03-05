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

def load_data(symbol):
    print(f"Loading {symbol}...")
    # Klines
    pattern_kline = f"{DATALAKE_DIR}/{symbol}/*_kline_1m.csv"
    files_kline = glob.glob(pattern_kline)
    files_kline = [f for f in files_kline if START_DATE <= os.path.basename(f).split('_')[0] <= END_DATE]
    
    dfs_k = []
    for f in files_kline:
        try:
            df = pd.read_csv(f, usecols=['startTime', 'close', 'volume'])
            df['startTime'] = pd.to_numeric(df['startTime'], errors='coerce')
            dfs_k.append(df)
        except: pass
    if not dfs_k: return None
    kline = pd.concat(dfs_k, ignore_index=True).dropna()
    kline['datetime'] = pd.to_datetime(kline['startTime'], unit='ms')
    kline = kline.set_index('datetime').resample('5min').agg({'close': 'last', 'volume': 'sum'}).ffill()
    
    # Open Interest
    pattern_oi = f"{DATALAKE_DIR}/{symbol}/*_open_interest_5min.csv"
    files_oi = glob.glob(pattern_oi)
    files_oi = [f for f in files_oi if START_DATE <= os.path.basename(f).split('_')[0] <= END_DATE]
    
    dfs_o = []
    for f in files_oi:
        try:
            df = pd.read_csv(f)
            # Find timestamp column. It might be 'timestamp' or 'startTime'
            ts_col = 'timestamp' if 'timestamp' in df.columns else 'startTime' if 'startTime' in df.columns else None
            if not ts_col: continue
            df[ts_col] = pd.to_numeric(df[ts_col], errors='coerce')
            # OI column
            oi_col = 'openInterest' if 'openInterest' in df.columns else 'OpenInterest' if 'OpenInterest' in df.columns else None
            if not oi_col: continue
            df = df[[ts_col, oi_col]].dropna()
            df.columns = ['startTime', 'oi']
            dfs_o.append(df)
        except: pass
    
    if not dfs_o: return None
    oi_df = pd.concat(dfs_o, ignore_index=True)
    oi_df['datetime'] = pd.to_datetime(oi_df['startTime'], unit='ms')
    oi_df = oi_df.set_index('datetime').resample('5min').last().ffill()
    
    # Merge
    merged = pd.concat([kline, oi_df['oi']], axis=1).ffill().dropna()
    return merged

# Test on a few volatile coins
symbols = ['BTCUSDT', 'SOLUSDT', 'DOGEUSDT', 'WIFUSDT'] # try some active ones

results = []

for sym in symbols:
    df = load_data(sym)
    if df is None or df.empty: continue
    
    df['ret_1h'] = df['close'].pct_change(12) # 12 * 5m = 1h
    df['oi_change_1h'] = df['oi'].pct_change(12)
    
    df['fwd_ret_1h'] = df['close'].shift(-12) / df['close'] - 1
    df = df.dropna()
    
    # Filter for large OI increase + large price increase
    mask_bull_squeeze = (df['oi_change_1h'] > 0.05) & (df['ret_1h'] > 0.02)
    mask_bear_squeeze = (df['oi_change_1h'] > 0.05) & (df['ret_1h'] < -0.02)
    
    fwd_bull = df[mask_bull_squeeze]['fwd_ret_1h']
    fwd_bear = df[mask_bear_squeeze]['fwd_ret_1h']
    
    print(f"\n--- {sym} ---")
    if len(fwd_bull) > 0:
        print(f"OI Up + Price Up (N={len(fwd_bull)}): Fwd 1h Ret = {fwd_bull.mean()*10000:.2f} bps, WR = {(fwd_bull > 0).mean()*100:.1f}%")
    if len(fwd_bear) > 0:
        print(f"OI Up + Price Down (N={len(fwd_bear)}): Fwd 1h Ret = {fwd_bear.mean()*10000:.2f} bps, WR = {(fwd_bear < 0).mean()*100:.1f}%")

