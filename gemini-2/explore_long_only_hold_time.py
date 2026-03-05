import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"

def get_all_symbols():
    return sorted([d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d)) and d.endswith('USDT')])

def process_symbol(symbol):
    try:
        files_kline = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_kline_1m.csv")
        if not files_kline: return None
        
        dfs_k = []
        for f in files_kline:
            try:
                df = pd.read_csv(f, usecols=['startTime', 'open', 'high', 'low', 'close', 'volume'])
                dfs_k.append(df)
            except: pass
        if not dfs_k: return None
        kline = pd.concat(dfs_k, ignore_index=True).dropna()
        kline['startTime'] = pd.to_numeric(kline['startTime'], errors='coerce')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            kline[col] = pd.to_numeric(kline[col], errors='coerce')
        kline = kline[kline['close'] > 0]
        kline['datetime'] = pd.to_datetime(kline['startTime'], unit='ms')
        kline = kline.set_index('datetime').sort_index()
        kline = kline[~kline.index.duplicated(keep='first')]
        kline_5m = kline.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).ffill()
        
        files_oi = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_open_interest_5min.csv")
        dfs_o = []
        for f in files_oi:
            try:
                df = pd.read_csv(f)
                ts_col = 'timestamp' if 'timestamp' in df.columns else 'startTime' if 'startTime' in df.columns else None
                oi_col = 'openInterest' if 'openInterest' in df.columns else 'OpenInterest' if 'OpenInterest' in df.columns else None
                if not ts_col or not oi_col: continue
                df = df[[ts_col, oi_col]].dropna()
                df.columns = ['startTime', 'oi']
                dfs_o.append(df)
            except: pass
        
        if dfs_o:
            oi_df = pd.concat(dfs_o, ignore_index=True)
            oi_df['startTime'] = pd.to_numeric(oi_df['startTime'], errors='coerce')
            oi_df['oi'] = pd.to_numeric(oi_df['oi'], errors='coerce')
            oi_df['datetime'] = pd.to_datetime(oi_df['startTime'], unit='ms')
            oi_df = oi_df.set_index('datetime').sort_index()
            oi_df = oi_df[~oi_df.index.duplicated(keep='first')]
            oi_df = oi_df.resample('5min').last().ffill()
            df = pd.concat([kline_5m, oi_df['oi']], axis=1).ffill().dropna()
        else:
            return None
            
        df['ret_15m'] = df['close'].pct_change(3)
        df['oi_chg_15m'] = df['oi'].pct_change(3)
        
        # Original baseline that generated 147 trades
        df['sig_long'] = ((df['oi_chg_15m'] < -0.10) & (df['ret_15m'] < -0.05)).astype(int)
        
        trades = []
        indices = np.where(df['sig_long'] == 1)[0]
        
        for i in indices:
            # We want to trace paths for up to 48 hours (576 bars of 5m) to see the optimal hold
            if i + 576 >= len(df): continue 
            
            entry_time = df.index[i]
            entry_price = df['close'].iloc[i]
            
            fwd_window = df.iloc[i+1 : i+577]
            
            trades.append({
                'symbol': symbol,
                'entry_time': entry_time,
                'entry_price': entry_price,
                'path_high': fwd_window['high'].values.tolist(),
                'path_low': fwd_window['low'].values.tolist(),
                'path_close': fwd_window['close'].values.tolist()
            })

        return trades
    except Exception as e:
        return None

if __name__ == '__main__':
    symbols = get_all_symbols()
    print(f"Extracting path data...")
    
    trades = []
    with Pool(processes=12) as pool:
        for res_list in tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols)):
            if res_list:
                trades.extend(res_list)
                
    if not trades:
        print("No trades found.")
        exit(0)
        
    df_trades = pd.DataFrame(trades)
    print(f"Extracted {len(df_trades)} base trades.")
    
    FEE_BPS = 20
    
    # We will test different hold times without TP/SL to find the raw underlying drift
    # Holds in hours: 1, 2, 4, 8, 12, 24, 48
    hold_hours = [1, 2, 4, 6, 8, 12, 24, 48]
    
    print("\n=== Pure Hold Time Analysis (No TP/SL, Net of Fees) ===")
    
    paths_c = np.array([t['path_close'] for _, t in df_trades.iterrows()])
    entries = np.array(df_trades['entry_price'].tolist()).reshape(-1, 1)
    ret_c = (paths_c - entries) / entries
    
    for h in hold_hours:
        idx = h * 12 - 1 # 12 5-min bars in an hour
        final_ret = ret_c[:, idx] - (FEE_BPS / 10000)
        mean_bps = final_ret.mean() * 10000
        wr = (final_ret > 0).mean() * 100
        
        print(f"Hold {h:2d}h: Mean = {mean_bps:7.2f} bps | WR = {wr:.1f}%")

