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
                df = pd.read_csv(f, usecols=['startTime', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                dfs_k.append(df)
            except: pass
        if not dfs_k: return None
        kline = pd.concat(dfs_k, ignore_index=True).dropna()
        kline['startTime'] = pd.to_numeric(kline['startTime'], errors='coerce')
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            kline[col] = pd.to_numeric(kline[col], errors='coerce')
        kline = kline[kline['close'] > 0]
        kline['datetime'] = pd.to_datetime(kline['startTime'], unit='ms')
        kline = kline.set_index('datetime').resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'turnover': 'sum'}).ffill()
        
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
            
        # Strategy Features
        df['ret_15m'] = df['close'].pct_change(3)
        df['oi_chg_15m'] = df['oi'].pct_change(3)
        df['vol_4h_sma'] = df['volume'].rolling(48).mean()
        df['vol_spike'] = df['volume'].rolling(3).sum() / (df['vol_4h_sma'] * 3 + 1e-9)
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(24).std() * np.sqrt(288) # roughly daily vol
        
        # Targets
        df['fwd_ret_1h'] = df['close'].shift(-12) / df['close'] - 1
        df['fwd_ret_2h'] = df['close'].shift(-24) / df['close'] - 1
        df['fwd_ret_4h'] = df['close'].shift(-48) / df['close'] - 1
        
        df = df.dropna(subset=['fwd_ret_4h', 'ret_15m', 'oi_chg_15m', 'volatility'])
        
        results = []
        
        # Long Flush Variant 1: Base
        mask_base = (df['oi_chg_15m'] < -0.05) & (df['ret_15m'] < -0.03)
        
        # Long Flush Variant 2: High Vol Spike
        mask_vol_spike = mask_base & (df['vol_spike'] > 2.0)
        
        # Long Flush Variant 3: Extreme Flush
        mask_extreme = (df['oi_chg_15m'] < -0.10) & (df['ret_15m'] < -0.05)
        
        for idx, row in df[mask_base].iterrows():
            results.append({
                'symbol': symbol,
                'datetime': idx,
                'setup': 'Base_Flush',
                'ret_15m': row['ret_15m'],
                'oi_chg_15m': row['oi_chg_15m'],
                'vol_spike': row['vol_spike'],
                'fwd_ret_1h': row['fwd_ret_1h'],
                'fwd_ret_2h': row['fwd_ret_2h'],
                'fwd_ret_4h': row['fwd_ret_4h']
            })
            
        for idx, row in df[mask_vol_spike].iterrows():
            results.append({
                'symbol': symbol,
                'datetime': idx,
                'setup': 'VolSpike_Flush',
                'ret_15m': row['ret_15m'],
                'oi_chg_15m': row['oi_chg_15m'],
                'vol_spike': row['vol_spike'],
                'fwd_ret_1h': row['fwd_ret_1h'],
                'fwd_ret_2h': row['fwd_ret_2h'],
                'fwd_ret_4h': row['fwd_ret_4h']
            })
            
        for idx, row in df[mask_extreme].iterrows():
            results.append({
                'symbol': symbol,
                'datetime': idx,
                'setup': 'Extreme_Flush',
                'ret_15m': row['ret_15m'],
                'oi_chg_15m': row['oi_chg_15m'],
                'vol_spike': row['vol_spike'],
                'fwd_ret_1h': row['fwd_ret_1h'],
                'fwd_ret_2h': row['fwd_ret_2h'],
                'fwd_ret_4h': row['fwd_ret_4h']
            })

        return results
    except Exception as e:
        return None

if __name__ == '__main__':
    symbols = get_all_symbols()
    print(f"Starting detailed flush analysis on {len(symbols)} symbols...")
    
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
    print(f"Total entries found: {len(df_trades)}")
    
    FEE_BPS = 20
    
    for setup in ['Base_Flush', 'VolSpike_Flush', 'Extreme_Flush']:
        print(f"\n======================")
        print(f"--- {setup} ---")
        setup_df = df_trades[df_trades['setup'] == setup]
        if len(setup_df) == 0: continue
        
        # Calculate net returns
        net_1h = setup_df['fwd_ret_1h'] - (FEE_BPS/10000)
        net_2h = setup_df['fwd_ret_2h'] - (FEE_BPS/10000)
        net_4h = setup_df['fwd_ret_4h'] - (FEE_BPS/10000)
        
        print(f"Total Trades: {len(setup_df)} (across {setup_df['symbol'].nunique()} symbols)")
        print(f"Top 3 Symbols: {setup_df['symbol'].value_counts().head(3).to_dict()}")
        
        print(f"Net Return 1h : {net_1h.mean()*10000:7.2f} bps | WR: {(net_1h > 0).mean()*100:.1f}% | Median: {net_1h.median()*10000:7.2f} bps")
        print(f"Net Return 2h : {net_2h.mean()*10000:7.2f} bps | WR: {(net_2h > 0).mean()*100:.1f}% | Median: {net_2h.median()*10000:7.2f} bps")
        print(f"Net Return 4h : {net_4h.mean()*10000:7.2f} bps | WR: {(net_4h > 0).mean()*100:.1f}% | Median: {net_4h.median()*10000:7.2f} bps")
        
        # Win rate is okay, but median is often lower than mean which implies a long right tail.
        p25 = net_4h.quantile(0.25) * 10000
        p75 = net_4h.quantile(0.75) * 10000
        print(f"4h Net Returns 25th-75th Percentile: {p25:.2f} bps to {p75:.2f} bps")

