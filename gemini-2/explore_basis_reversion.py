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
        fut_5m = fut.resample('5min').last().ffill()
        
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
        spot_5m = spot.resample('5min').last().ffill()
        
        df = pd.DataFrame({'fut': fut_5m['close'], 'spot': spot_5m['close']}).dropna()
        if len(df) < 100: return None
        
        df['basis_bps'] = (df['fut'] - df['spot']) / df['spot'] * 10000
        
        # We want to buy futures when basis is very negative
        df['fwd_ret_1h'] = df['fut'].shift(-12) / df['fut'] - 1
        df['fwd_ret_4h'] = df['fut'].shift(-48) / df['fut'] - 1
        
        df = df.dropna()
        
        trades = []
        for threshold in [-10, -20, -30, -50]:
            mask = df['basis_bps'] < threshold
            subset = df[mask]
            
            # To avoid overlapping trades in the stats, we can just record all signals
            # and do group analysis.
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
    print(f"Extracting basis data on {len(symbols)} symbols...")
    
    trades = []
    with Pool(processes=12) as pool:
        for res_list in tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols), desc="Extracting"):
            if res_list:
                trades.extend(res_list)
                
    if not trades:
        print("No trades found.")
        exit(0)
        
    df_trades = pd.DataFrame(trades)
    print("\n--- Basis Reversion Strategy (Buy Futures when heavily discounted) ---")
    FEE_BPS = 20
    
    for thresh in [-10, -20, -30, -50]:
        setup_df = df_trades[df_trades['basis_threshold'] == thresh]
        if len(setup_df) == 0: continue
        
        net_1h = setup_df['fwd_ret_1h'] - (FEE_BPS/10000)
        net_4h = setup_df['fwd_ret_4h'] - (FEE_BPS/10000)
        
        print(f"\nBasis < {thresh} bps:")
        print(f"Trades: {len(setup_df)} (across {setup_df['symbol'].nunique()} symbols)")
        print(f"Net Return 1h: {net_1h.mean()*10000:.2f} bps | WR: {(net_1h > 0).mean()*100:.1f}%")
        print(f"Net Return 4h: {net_4h.mean()*10000:.2f} bps | WR: {(net_4h > 0).mean()*100:.1f}%")

