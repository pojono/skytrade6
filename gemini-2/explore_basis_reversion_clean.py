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
        # Futures
        files_fut = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_kline_1m.csv")
        if not files_fut: return None
        
        dfs_f = []
        for f in files_fut:
            try:
                df = pd.read_csv(f, usecols=['startTime', 'close'])
                dfs_f.append(df)
            except: pass
        if not dfs_f: return None
        fut = pd.concat(dfs_f, ignore_index=True).dropna()
        fut['startTime'] = pd.to_numeric(fut['startTime'], errors='coerce')
        fut['close'] = pd.to_numeric(fut['close'], errors='coerce')
        fut = fut[fut['close'] > 0]
        fut['datetime'] = pd.to_datetime(fut['startTime'], unit='ms')
        fut = fut.set_index('datetime').sort_index()
        fut = fut[~fut.index.duplicated(keep='first')]
        
        # Spot
        files_spot = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_kline_1m_spot.csv")
        if not files_spot: return None
        
        dfs_s = []
        for f in files_spot:
            try:
                df = pd.read_csv(f, usecols=['startTime', 'close'])
                dfs_s.append(df)
            except: pass
        if not dfs_s: return None
        spot = pd.concat(dfs_s, ignore_index=True).dropna()
        spot['startTime'] = pd.to_numeric(spot['startTime'], errors='coerce')
        spot['close'] = pd.to_numeric(spot['close'], errors='coerce')
        spot = spot[spot['close'] > 0]
        spot['datetime'] = pd.to_datetime(spot['startTime'], unit='ms')
        spot = spot.set_index('datetime').sort_index()
        spot = spot[~spot.index.duplicated(keep='first')]
        
        df = pd.DataFrame({'fut': fut['close'], 'spot': spot['close']}).dropna()
        
        # Filter extreme data errors (e.g. flash crashes that are just bad data, return > 50% in 1m)
        df['fut_ret'] = df['fut'].pct_change()
        df['spot_ret'] = df['spot'].pct_change()
        mask_bad = (df['fut_ret'].abs() > 0.30) | (df['spot_ret'].abs() > 0.30)
        df.loc[mask_bad, 'fut'] = np.nan
        df.loc[mask_bad, 'spot'] = np.nan
        df = df.ffill()

        df_5m = df.resample('5min').last().dropna()
        if len(df_5m) < 100: return None
        
        df_5m['basis_bps'] = (df_5m['fut'] - df_5m['spot']) / df_5m['spot'] * 10000
        
        # Exclude extreme basis > 10% (1000 bps) which is likely data error
        df_5m = df_5m[df_5m['basis_bps'].abs() < 1000]
        
        df_5m['fwd_ret_1h'] = df_5m['fut'].shift(-12) / df_5m['fut'] - 1
        df_5m['fwd_ret_4h'] = df_5m['fut'].shift(-48) / df_5m['fut'] - 1
        
        df_5m = df_5m.dropna()
        
        trades = []
        for threshold in [-10, -20, -30, -50]:
            mask = df_5m['basis_bps'] < threshold
            subset = df_5m[mask]
            
            for idx, row in subset.iterrows():
                trades.append({
                    'symbol': symbol,
                    'datetime': idx,
                    'basis_threshold': threshold,
                    'basis_bps': row['basis_bps'],
                    'fwd_ret_1h': row['fwd_ret_1h'],
                    'fwd_ret_4h': row['fwd_ret_4h']
                })

        return trades
    except Exception as e:
        return None

if __name__ == '__main__':
    symbols = get_all_symbols()
    print(f"Extracting clean basis data on {len(symbols)} symbols...")
    
    trades = []
    with Pool(processes=12) as pool:
        for res_list in tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols), desc="Extracting"):
            if res_list:
                trades.extend(res_list)
                
    if not trades:
        print("No trades found.")
        exit(0)
        
    df_trades = pd.DataFrame(trades)
    print("\n--- Clean Basis Reversion Strategy (Buy Futures when heavily discounted) ---")
    FEE_BPS = 20
    
    for thresh in [-10, -20, -30, -50]:
        setup_df = df_trades[df_trades['basis_threshold'] == thresh]
        if len(setup_df) == 0: continue
        
        net_1h = setup_df['fwd_ret_1h'] - (FEE_BPS/10000)
        net_4h = setup_df['fwd_ret_4h'] - (FEE_BPS/10000)
        
        print(f"\nBasis < {thresh} bps:")
        print(f"Signals: {len(setup_df)} (across {setup_df['symbol'].nunique()} symbols)")
        
        # Outlier robust mean
        r1h = setup_df['fwd_ret_1h']
        r1h_clean = r1h[(r1h > r1h.quantile(0.01)) & (r1h < r1h.quantile(0.99))] - (FEE_BPS/10000)
        r4h = setup_df['fwd_ret_4h']
        r4h_clean = r4h[(r4h > r4h.quantile(0.01)) & (r4h < r4h.quantile(0.99))] - (FEE_BPS/10000)
        
        print(f"Robust Net Return 1h: {r1h_clean.mean()*10000:.2f} bps | WR: {(r1h_clean > 0).mean()*100:.1f}%")
        print(f"Robust Net Return 4h: {r4h_clean.mean()*10000:.2f} bps | WR: {(r4h_clean > 0).mean()*100:.1f}%")

