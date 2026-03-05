import pandas as pd
import numpy as np
import os
import glob
import gc
import warnings
from concurrent.futures import ProcessPoolExecutor
warnings.filterwarnings('ignore')

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
START_DATE = "2025-01-01"
END_DATE = "2025-02-28"

def get_all_symbols():
    return [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d)) and d.endswith('USDT')]

def load_symbol_data(symbol):
    # Klines
    pattern_kline = f"{DATALAKE_DIR}/{symbol}/*_kline_1m.csv"
    files_kline = glob.glob(pattern_kline)
    files_kline = [f for f in files_kline if START_DATE <= os.path.basename(f).split('_')[0] <= END_DATE]
    
    dfs_k = []
    for f in files_kline:
        try:
            df = pd.read_csv(f, usecols=['startTime', 'close', 'volume'])
            df['startTime'] = pd.to_numeric(df['startTime'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df = df[df['close'] > 0]
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
            ts_col = 'timestamp' if 'timestamp' in df.columns else 'startTime' if 'startTime' in df.columns else None
            oi_col = 'openInterest' if 'openInterest' in df.columns else 'OpenInterest' if 'OpenInterest' in df.columns else None
            if not ts_col or not oi_col: continue
            df = df[[ts_col, oi_col]].dropna()
            df.columns = ['startTime', 'oi']
            df['startTime'] = pd.to_numeric(df['startTime'], errors='coerce')
            df['oi'] = pd.to_numeric(df['oi'], errors='coerce')
            dfs_o.append(df)
        except: pass
    
    if not dfs_o: return None
    oi_df = pd.concat(dfs_o, ignore_index=True).dropna()
    oi_df['datetime'] = pd.to_datetime(oi_df['startTime'], unit='ms')
    oi_df = oi_df.set_index('datetime').resample('5min').last().ffill()
    
    merged = pd.concat([kline, oi_df['oi']], axis=1).ffill().dropna()
    merged['symbol'] = symbol
    return merged

if __name__ == '__main__':
    symbols = get_all_symbols()[:80] # Test on first 80 coins
    print(f"Processing {len(symbols)} symbols...")
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(load_symbol_data, symbols))
        
    valid_dfs = [r for r in results if r is not None and not r.empty]
    print(f"Loaded data for {len(valid_dfs)} symbols")
    
    all_data = pd.concat(valid_dfs)
    del valid_dfs
    gc.collect()
    
    # Evaluate parameters
    lookbacks = [3, 6, 12] # 15m, 30m, 1h
    holdings = [3, 6, 12, 24] # 15m, 30m, 1h, 2h
    oi_thresholds = [0.03, 0.05, 0.10]
    px_thresholds = [0.02, 0.03, 0.05]
    
    best_combo = None
    best_pnl = -9999
    
    print("\n--- Strategy Exploration: OI + Momentum ---")
    print("Fees assumed: 20 bps roundtrip")
    
    # Calculate returns grouped by symbol
    for lb in lookbacks:
        all_data[f'ret_{lb}'] = all_data.groupby('symbol')['close'].pct_change(lb)
        all_data[f'oi_chg_{lb}'] = all_data.groupby('symbol')['oi'].pct_change(lb)
    
    for hold in holdings:
        all_data[f'fwd_{hold}'] = all_data.groupby('symbol')['close'].shift(-hold) / all_data['close'] - 1
        
    all_data = all_data.dropna()
    
    for lb in lookbacks:
        for hold in holdings:
            for oi_thresh in oi_thresholds:
                for px_thresh in px_thresholds:
                    
                    # Longs: OI up + Price up
                    long_mask = (all_data[f'oi_chg_{lb}'] > oi_thresh) & (all_data[f'ret_{lb}'] > px_thresh)
                    long_ret = all_data.loc[long_mask, f'fwd_{hold}']
                    
                    # Shorts: OI up + Price down
                    short_mask = (all_data[f'oi_chg_{lb}'] > oi_thresh) & (all_data[f'ret_{lb}'] < -px_thresh)
                    short_ret = -all_data.loc[short_mask, f'fwd_{hold}']
                    
                    total_trades = len(long_ret) + len(short_ret)
                    if total_trades < 50:
                        continue
                        
                    # Calculate net PnL (subtract 20 bps = 0.0020)
                    all_returns = pd.concat([long_ret, short_ret])
                    net_returns = all_returns - 0.0020
                    
                    mean_net_bps = net_returns.mean() * 10000
                    win_rate = (net_returns > 0).mean() * 100
                    
                    if mean_net_bps > 0:
                        print(f"LB: {lb*5}m, Hold: {hold*5}m, OI>{oi_thresh*100}%, Px>{px_thresh*100}% | N={total_trades} | Net PnL/Trade: {mean_net_bps:.1f} bps | WR: {win_rate:.1f}%")

