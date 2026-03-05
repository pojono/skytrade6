import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool
import warnings
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
        kline = kline.set_index('datetime').resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).ffill()
        
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
            oi_df = oi_df.set_index('datetime').resample('5min').last().ffill()
            df = pd.concat([kline, oi_df['oi']], axis=1).ffill().dropna()
        else:
            return None
            
        df['ret_15m'] = df['close'].pct_change(3)
        df['oi_chg_15m'] = df['oi'].pct_change(3)
        
        df['fwd_ret_1h'] = df['close'].shift(-12) / df['close'] - 1
        df['fwd_ret_4h'] = df['close'].shift(-48) / df['close'] - 1
        
        # Max excursion for analysis
        df['fwd_max_1h'] = df['high'].shift(-12).rolling(12, min_periods=1).max() / df['close'] - 1
        df['fwd_min_1h'] = df['low'].shift(-12).rolling(12, min_periods=1).min() / df['close'] - 1
        
        df = df.dropna(subset=['fwd_ret_4h', 'ret_15m', 'oi_chg_15m'])
        
        results = []
        
        # Setup 1: Long Flush
        mask_long_flush = (df['oi_chg_15m'] < -0.05) & (df['ret_15m'] < -0.03)
        subset = df[mask_long_flush]
        for idx, row in subset.iterrows():
            results.append({
                'symbol': symbol,
                'datetime': idx,
                'setup': 'Long_Flush',
                'ret_15m': row['ret_15m'],
                'oi_chg_15m': row['oi_chg_15m'],
                'fwd_ret_1h': row['fwd_ret_1h'],
                'fwd_ret_4h': row['fwd_ret_4h'],
                'fwd_max_1h': row['fwd_max_1h'],
                'fwd_min_1h': row['fwd_min_1h']
            })
            
        # Setup 2: Short Squeeze Continuation (Price up, OI down -> buy the continuation)
        mask_short_squeeze = (df['oi_chg_15m'] < -0.05) & (df['ret_15m'] > 0.03)
        subset2 = df[mask_short_squeeze]
        for idx, row in subset2.iterrows():
            results.append({
                'symbol': symbol,
                'datetime': idx,
                'setup': 'Short_Squeeze_Continuation',
                'ret_15m': row['ret_15m'],
                'oi_chg_15m': row['oi_chg_15m'],
                'fwd_ret_1h': row['fwd_ret_1h'],
                'fwd_ret_4h': row['fwd_ret_4h'],
                'fwd_max_1h': row['fwd_max_1h'],
                'fwd_min_1h': row['fwd_min_1h']
            })

        return results
    except Exception as e:
        return None

if __name__ == '__main__':
    symbols = get_all_symbols()
    print(f"Starting detailed analysis on {len(symbols)} symbols...")
    
    with Pool(processes=12) as pool:
        all_results = pool.map(process_symbol, symbols)
        
    trades = []
    for res_list in all_results:
        if res_list:
            trades.extend(res_list)
            
    if not trades:
        print("No trades found.")
        exit(0)
        
    df_trades = pd.DataFrame(trades)
    print(f"Total trades found: {len(df_trades)}")
    
    FEE_BPS = 20
    
    for setup in ['Long_Flush', 'Short_Squeeze_Continuation']:
        print(f"\n--- {setup} ---")
        if 'setup' not in df_trades.columns:
            continue
        setup_df = df_trades[df_trades['setup'] == setup]
        if len(setup_df) == 0: continue
        
        # Calculate net returns
        net_1h = setup_df['fwd_ret_1h'] - (FEE_BPS/10000)
        net_4h = setup_df['fwd_ret_4h'] - (FEE_BPS/10000)
        
        print(f"Total Trades: {len(setup_df)} (across {setup_df['symbol'].nunique()} symbols)")
        print(f"Top 3 Symbols by Trade Count: {setup_df['symbol'].value_counts().head(3).to_dict()}")
        
        print(f"Net Return 1h : {net_1h.mean()*10000:.2f} bps | WR: {(net_1h > 0).mean()*100:.1f}%")
        print(f"Net Return 4h : {net_4h.mean()*10000:.2f} bps | WR: {(net_4h > 0).mean()*100:.1f}%")
        
        # Max excursion
        print(f"Mean Max Excursion 1h (MFE): {setup_df['fwd_max_1h'].mean()*10000:.2f} bps")
        print(f"Mean Min Excursion 1h (MAE): {setup_df['fwd_min_1h'].mean()*10000:.2f} bps")
        
        # Outlier robust mean
        p95_1h = net_1h.quantile(0.95)
        p05_1h = net_1h.quantile(0.05)
        robust_1h = net_1h[(net_1h >= p05_1h) & (net_1h <= p95_1h)]
        print(f"Robust Net Return 1h (5-95th percentile): {robust_1h.mean()*10000:.2f} bps")
        
        # Sort trades by time to see if they are clustered
        counts_by_day = setup_df.groupby(setup_df['datetime'].dt.date).size()
        print(f"Max trades in a single day: {counts_by_day.max()} on {counts_by_day.idxmax()}")

