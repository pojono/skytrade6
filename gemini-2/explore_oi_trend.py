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
        kline_1h = kline.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).ffill()
        
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
            oi_df = oi_df.resample('1h').last().ffill()
            df = pd.concat([kline_1h, oi_df['oi']], axis=1).ffill().dropna()
        else:
            return None
            
        df['ret_4h'] = df['close'].pct_change(4)
        df['oi_chg_4h'] = df['oi'].pct_change(4)
        
        df['fwd_ret_24h'] = df['close'].shift(-24) / df['close'] - 1
        df['fwd_ret_48h'] = df['close'].shift(-48) / df['close'] - 1
        
        df = df.dropna(subset=['fwd_ret_48h'])
        
        results = []
        
        # Trend strategy: price up, oi up consistently over 4h => momentum long
        mask_bull = (df['oi_chg_4h'] > 0.05) & (df['ret_4h'] > 0.05)
        for idx, row in df[mask_bull].iterrows():
            results.append({
                'symbol': symbol,
                'datetime': idx,
                'setup': 'Momentum_Long',
                'ret_4h': row['ret_4h'],
                'oi_chg_4h': row['oi_chg_4h'],
                'fwd_ret_24h': row['fwd_ret_24h'],
                'fwd_ret_48h': row['fwd_ret_48h']
            })

        # Trend exhaust: price up, oi down => short
        mask_exhaust = (df['oi_chg_4h'] < -0.05) & (df['ret_4h'] > 0.05)
        for idx, row in df[mask_exhaust].iterrows():
            results.append({
                'symbol': symbol,
                'datetime': idx,
                'setup': 'Exhaust_Short',
                'ret_4h': row['ret_4h'],
                'oi_chg_4h': row['oi_chg_4h'],
                'fwd_ret_24h': -row['fwd_ret_24h'], # short return
                'fwd_ret_48h': -row['fwd_ret_48h']
            })

        return results
    except Exception as e:
        return None

if __name__ == '__main__':
    symbols = get_all_symbols()
    print(f"Extracting trend data...")
    
    trades = []
    with Pool(processes=12) as pool:
        for res_list in tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols), desc="Extracting"):
            if res_list:
                trades.extend(res_list)
                
    if not trades:
        print("No trades found.")
        exit(0)
        
    df_trades = pd.DataFrame(trades)
    
    FEE_BPS = 20
    print("\n--- Strategy: 4H Momentum & Exhaustion ---")
    
    for setup in ['Momentum_Long', 'Exhaust_Short']:
        setup_df = df_trades[df_trades['setup'] == setup]
        if len(setup_df) == 0: continue
        
        net_24h = setup_df['fwd_ret_24h'] - (FEE_BPS/10000)
        net_48h = setup_df['fwd_ret_48h'] - (FEE_BPS/10000)
        
        print(f"\n{setup}:")
        print(f"Trades: {len(setup_df)}")
        print(f"Net Return 24h: {net_24h.mean()*10000:.2f} bps | WR: {(net_24h > 0).mean()*100:.1f}%")
        print(f"Net Return 48h: {net_48h.mean()*10000:.2f} bps | WR: {(net_48h > 0).mean()*100:.1f}%")
        
        sharpe_24h = net_24h.mean() / (net_24h.std() + 1e-9) * np.sqrt(365) # daily trades assumed
        print(f"Pseudo-Sharpe 24h: {sharpe_24h:.2f}")

