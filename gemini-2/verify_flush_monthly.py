import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool
import warnings
import time
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
        
        df['signal'] = ((df['oi_chg_15m'] < -0.10) & (df['ret_15m'] < -0.05)).astype(int)
        
        trades = []
        signal_indices = np.where(df['signal'] == 1)[0]
        
        for i in signal_indices:
            if i + 48 >= len(df): continue 
            
            entry_time = df.index[i]
            entry_price = df['close'].iloc[i]
            
            fwd_window = df.iloc[i+1 : i+49]
            
            trade_info = {
                'symbol': symbol,
                'entry_time': entry_time,
                'entry_price': entry_price,
                'path_high': fwd_window['high'].values.tolist(),
                'path_low': fwd_window['low'].values.tolist(),
                'path_close': fwd_window['close'].values.tolist()
            }
            trades.append(trade_info)

        return trades
    except Exception as e:
        return None

if __name__ == '__main__':
    symbols = get_all_symbols()
    print(f"Extracting data to analyze monthly consistency...")
    
    trades = []
    with Pool(processes=12) as pool:
        for res_list in tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols), desc="Extracting"):
            if res_list:
                trades.extend(res_list)
                
    if not trades:
        print("No trades found.")
        exit(0)
        
    FEE_BPS = 20 # 0.2% total
    TP = 0.20
    SL = -0.15
    HOLD = 48 # 4 hours
    
    results = []
    for t in trades:
        ep = t['entry_price']
        highs = t['path_high'][:HOLD]
        lows = t['path_low'][:HOLD]
        closes = t['path_close'][:HOLD]
        
        exit_price = closes[-1]
        
        for step in range(HOLD):
            ret_h = (highs[step] - ep) / ep
            ret_l = (lows[step] - ep) / ep
            
            if ret_l <= SL:
                exit_price = ep * (1 + SL)
                break
            elif ret_h >= TP:
                exit_price = ep * (1 + TP)
                break
                
        net_ret = (exit_price - ep) / ep - (FEE_BPS/10000)
        
        results.append({
            'time': t['entry_time'],
            'symbol': t['symbol'],
            'net_ret': net_ret
        })
        
    df_res = pd.DataFrame(results)
    df_res['month'] = df_res['time'].dt.to_period('M')
    
    monthly = df_res.groupby('month').agg(
        trades=('net_ret', 'count'),
        win_rate=('net_ret', lambda x: (x > 0).mean() * 100),
        mean_ret_bps=('net_ret', lambda x: x.mean() * 10000)
    ).reset_index()
    
    print("\n--- Monthly Performance Breakdown ---")
    print(f"Strategy: Long Flush (OI < -10%, Ret < -5% in 15m)")
    print(f"Params: TP=20%, SL=-15%, Hold=4h")
    print(monthly.to_string(index=False))

