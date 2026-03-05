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
                df = pd.read_csv(f, usecols=['startTime', 'open', 'high', 'low', 'close'])
                dfs_k.append(df)
            except: pass
        if not dfs_k: return None
        kline = pd.concat(dfs_k, ignore_index=True).dropna()
        kline['startTime'] = pd.to_numeric(kline['startTime'], errors='coerce')
        for col in ['open', 'high', 'low', 'close']:
            kline[col] = pd.to_numeric(kline[col], errors='coerce')
        kline = kline[kline['close'] > 0]
        kline['datetime'] = pd.to_datetime(kline['startTime'], unit='ms')
        kline = kline.set_index('datetime').sort_index()
        kline = kline[~kline.index.duplicated(keep='first')]
        
        kline_5m = kline.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).ffill()
        
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
        
        if not dfs_o: return None
        
        oi_df = pd.concat(dfs_o, ignore_index=True)
        oi_df['startTime'] = pd.to_numeric(oi_df['startTime'], errors='coerce')
        oi_df['oi'] = pd.to_numeric(oi_df['oi'], errors='coerce')
        oi_df['datetime'] = pd.to_datetime(oi_df['startTime'], unit='ms')
        oi_df = oi_df.set_index('datetime').sort_index()
        oi_df = oi_df[~oi_df.index.duplicated(keep='first')]
        oi_df = oi_df.resample('5min').last().ffill()
        
        df_sig = pd.concat([kline_5m, oi_df['oi']], axis=1).ffill().dropna()
        
        df_sig['ret_15m'] = df_sig['close'] / df_sig['close'].shift(3) - 1
        df_sig['oi_chg_15m'] = df_sig['oi'] / df_sig['oi'].shift(3) - 1
        
        # SHORT FLUSH: Price pumps > 5%, OI drops > 10% (Short liquidations)
        # We want to SHORT the top.
        df_sig['signal'] = ((df_sig['oi_chg_15m'] < -0.10) & (df_sig['ret_15m'] > 0.05)).astype(int)
        
        trades = []
        signal_times = df_sig[df_sig['signal'] == 1].index
        
        for t_sig in signal_times:
            entry_price = df_sig.loc[t_sig, 'close']
            
            end_time = t_sig + pd.Timedelta(hours=4)
            path_df = kline.loc[t_sig + pd.Timedelta(minutes=1) : end_time]
            
            if len(path_df) == 0: continue
            
            trades.append({
                'symbol': symbol,
                'entry_time': t_sig,
                'entry_price': entry_price,
                'path_high': path_df['high'].values.tolist(),
                'path_low': path_df['low'].values.tolist(),
                'path_close': path_df['close'].values.tolist(),
            })

        return trades
    except Exception as e:
        return None

if __name__ == '__main__':
    symbols = get_all_symbols()
    print(f"Extracting Short Flush data on {len(symbols)} symbols...")
    
    trades = []
    with Pool(processes=12) as pool:
        for res_list in tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols), desc="Extracting"):
            if res_list:
                trades.extend(res_list)
                
    if not trades:
        print("No trades found.")
        exit(0)
        
    print(f"Found {len(trades)} trades.")
    
    FEE_BPS = 20
    
    tps = [0.05, 0.10, 0.15, 0.20] 
    sls = [-0.03, -0.05, -0.08, -0.10, -0.15] 
    holds = [12, 24, 36, 48] # 1h, 2h, 3h, 4h
    
    results_grid = []
    n_trades = len(trades)
    
    paths_h = np.array([t['path_high'] for t in trades])
    paths_l = np.array([t['path_low'] for t in trades])
    paths_c = np.array([t['path_close'] for t in trades])
    entries = np.array([t['entry_price'] for t in trades]).reshape(-1, 1)
    
    # FOR SHORTS:
    # High prices = negative return (drawdown)
    # Low prices = positive return (profit)
    ret_h = (entries - paths_h) / entries  # e.g. entry 100, high 110 -> ret = -0.10 (loss)
    ret_l = (entries - paths_l) / entries  # e.g. entry 100, low 90 -> ret = +0.10 (profit)
    ret_c = (entries - paths_c) / entries
    
    for tp in tps:
        for sl in sls:
            for hold in holds:
                
                # Check hits within holding period
                hit_tp = (ret_l[:, :hold] >= tp) # Profit when price goes LOW
                hit_sl = (ret_h[:, :hold] <= sl) # Loss when price goes HIGH
                
                tp_idx = np.where(hit_tp.any(axis=1), hit_tp.argmax(axis=1), 999)
                sl_idx = np.where(hit_sl.any(axis=1), hit_sl.argmax(axis=1), 999)
                
                final_ret = ret_c[:, hold-1].copy()
                
                sl_first_mask = (sl_idx <= tp_idx) & (sl_idx < 999)
                tp_first_mask = (tp_idx < sl_idx) & (tp_idx < 999)
                
                final_ret[sl_first_mask] = sl
                final_ret[tp_first_mask] = tp
                
                net_ret = final_ret - (FEE_BPS/10000)
                
                mean_pnl = net_ret.mean() * 10000
                wr = (net_ret > 0).mean() * 100
                sharpe = net_ret.mean() / (net_ret.std() + 1e-9) * np.sqrt(n_trades)
                
                results_grid.append({
                    'TP': tp,
                    'SL': sl,
                    'Hold_h': hold/12,
                    'Net_bps': mean_pnl,
                    'WR': wr,
                    'Sharpe': sharpe
                })

    df_grid = pd.DataFrame(results_grid)
    print("\nTop 15 Configurations for SHORT FLUSH:")
    valid_grid = df_grid[df_grid['WR'] >= 40]
    if len(valid_grid) > 0:
        print(valid_grid.sort_values('Sharpe', ascending=False).head(15).to_string(index=False))
    else:
        print("No configurations with WR > 40%. Here are the best overall:")
        print(df_grid.sort_values('Net_bps', ascending=False).head(10).to_string(index=False))

