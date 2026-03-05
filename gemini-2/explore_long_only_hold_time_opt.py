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
        
        # Optimize entry for the sweet spot between freq and edge
        df['sig_long'] = ((df['oi_chg_15m'] < -0.10) & (df['ret_15m'] < -0.06)).astype(int)
        
        trades = []
        indices = np.where(df['sig_long'] == 1)[0]
        
        for i in indices:
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
    
    trades = []
    with Pool(processes=12) as pool:
        for res_list in pool.imap_unordered(process_symbol, symbols):
            if res_list: trades.extend(res_list)
                
    df_trades = pd.DataFrame(trades)
    
    FEE_BPS = 20
    holds = [12, 24, 36, 48, 72, 144, 288] # 1h, 2h, 3h, 4h, 6h, 12h, 24h
    tps = [0.15, 0.20, 0.25, 0.30, 0.40]
    sls = [-0.08, -0.10, -0.15, -0.20]
    
    results_grid = []
    
    for hold in holds:
        for tp in tps:
            for sl in sls:
                total_pnl = []
                for _, t in df_trades.iterrows():
                    ep = t['entry_price']
                    highs = t['path_high'][:hold]
                    lows = t['path_low'][:hold]
                    closes = t['path_close'][:hold]
                    
                    exit_price = closes[-1]
                    for j in range(len(highs)):
                        if (lows[j] - ep) / ep <= sl:
                            exit_price = ep * (1 + sl)
                            break
                        elif (highs[j] - ep) / ep >= tp:
                            exit_price = ep * (1 + tp)
                            break
                            
                    net_ret = (exit_price - ep) / ep - (FEE_BPS/10000)
                    total_pnl.append(net_ret)
                    
                rets = np.array(total_pnl)
                results_grid.append({
                    'Hold': hold/12,
                    'TP': tp,
                    'SL': sl,
                    'Mean_Bps': rets.mean() * 10000,
                    'WR': (rets > 0).mean() * 100,
                    'Sharpe': rets.mean() / (rets.std() + 1e-9) * np.sqrt(len(rets))
                })
                
    res_df = pd.DataFrame(results_grid)
    print("\n--- Optimized Exit Rules (Sorted by Sharpe) ---")
    print(res_df.sort_values('Sharpe', ascending=False).head(20).to_string(index=False))

