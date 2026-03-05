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
        
        df_sig = kline_5m.dropna()
        if len(df_sig) < 100: return None
        
        df_sig['ret_15m'] = df_sig['close'] / df_sig['close'].shift(3) - 1
        df_sig['vol_sma_4h'] = df_sig['volume'].rolling(48).mean()
        df_sig['vol_spike_15m'] = df_sig['volume'].rolling(3).sum() / (df_sig['vol_sma_4h'] * 3 + 1e-9)
        
        df_sig['signal'] = ((df_sig['vol_spike_15m'] > 3.0) & (df_sig['ret_15m'] > 0.03)).astype(int)
        
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
                'path_times': path_df.index.tolist()
            })

        return trades
    except Exception as e:
        return None

if __name__ == '__main__':
    symbols = get_all_symbols()
    print(f"Running volume spike momentum extraction on {len(symbols)} symbols...")
    
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
    
    results = []
    
    # Analyze multiple holding periods with strict path calculation (no TP/SL to avoid path dependency artifacts first)
    for hold_m in [15, 30, 60, 120, 240]:
        total_ret = []
        for t in trades:
            c_path = t['path_close']
            if len(c_path) < hold_m:
                exit_price = c_path[-1] if len(c_path) > 0 else t['entry_price']
            else:
                exit_price = c_path[hold_m - 1]
                
            net_ret = (exit_price - t['entry_price']) / t['entry_price'] - (FEE_BPS/10000)
            total_ret.append(net_ret)
            
        r = np.array(total_ret)
        print(f"\nHold {hold_m}m:")
        print(f"Mean Net: {r.mean()*10000:.2f} bps | WR: {(r>0).mean()*100:.1f}%")
        print(f"Robust Net (5-95%): {r[(r > np.percentile(r, 5)) & (r < np.percentile(r, 95))].mean()*10000:.2f} bps")

