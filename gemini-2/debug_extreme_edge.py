import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"

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
        
        df['sig_long'] = ((df['oi_chg_15m'] < -0.10) & (df['ret_15m'] < -0.05)).astype(int)
        
        trades = []
        indices = np.where(df['sig_long'] == 1)[0]
        
        for i in indices:
            if i + 48 >= len(df): continue 
            
            entry_time = df.index[i]
            entry_price = df['close'].iloc[i]
            
            fwd_window = df.iloc[i+1 : i+49]
            
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
    symbols = sorted([d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d)) and d.endswith('USDT')])
    
    trades = []
    with Pool(processes=12) as pool:
        for res_list in pool.imap_unordered(process_symbol, symbols):
            if res_list: trades.extend(res_list)
                
    df_trades = pd.DataFrame(trades)
    print(f"Extracted {len(df_trades)} base trades.")
    
    FEE_BPS = 20
    HOLD = 24 # 2 hours
    
    # 1. Pure Hold
    pure_rets = []
    for _, t in df_trades.iterrows():
        ep = t['entry_price']
        exit_price = t['path_close'][HOLD-1]
        pure_rets.append((exit_price - ep) / ep - (FEE_BPS/10000))
        
    pure_rets = np.array(pure_rets)
    print(f"\n--- Pure Hold {HOLD*5}m ---")
    print(f"Mean Net: {pure_rets.mean()*10000:.2f} bps")
    print(f"Median Net: {np.median(pure_rets)*10000:.2f} bps")
    print(f"Win Rate: {(pure_rets > 0).mean()*100:.1f}%")
    
    # 2. With TP/SL
    tp = 0.20
    sl = -0.10
    
    tpsl_rets = []
    exit_reasons = {'TP': 0, 'SL': 0, 'Time': 0}
    
    for _, t in df_trades.iterrows():
        ep = t['entry_price']
        highs = t['path_high'][:HOLD]
        lows = t['path_low'][:HOLD]
        closes = t['path_close'][:HOLD]
        
        exit_price = closes[-1]
        reason = 'Time'
        
        for j in range(len(highs)):
            ret_h = (highs[j] - ep) / ep
            ret_l = (lows[j] - ep) / ep
            
            if ret_l <= sl:
                exit_price = ep * (1 + sl)
                reason = 'SL'
                break
            elif ret_h >= tp:
                exit_price = ep * (1 + tp)
                reason = 'TP'
                break
                
        tpsl_rets.append((exit_price - ep) / ep - (FEE_BPS/10000))
        exit_reasons[reason] += 1
        
    tpsl_rets = np.array(tpsl_rets)
    print(f"\n--- With TP {tp*100}% / SL {sl*100}% ---")
    print(f"Mean Net: {tpsl_rets.mean()*10000:.2f} bps")
    print(f"Median Net: {np.median(tpsl_rets)*10000:.2f} bps")
    print(f"Win Rate: {(tpsl_rets > 0).mean()*100:.1f}%")
    print(f"Exit Reasons: {exit_reasons}")
