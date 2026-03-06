import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def load_klines_and_oi(symbol, exchange='bybit', start_date="2025-01-01"):
    try:
        # Load Klines
        kline_files = sorted(list((DATALAKE / f"{exchange}/{symbol}").glob("*_kline_1m.csv")))
        kline_files = [f for f in kline_files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name and f.name >= start_date]
        
        if not kline_files: return None
        
        dfs = []
        time_col = 'startTime' if exchange == 'bybit' else 'open_time'
        for f in kline_files:
            try:
                df = pd.read_csv(f, usecols=[time_col, 'close', 'volume', 'high', 'low'], engine='c')
                dfs.append(df)
            except: pass
            
        kline_df = pd.concat(dfs, ignore_index=True)
        kline_df.rename(columns={time_col: 'timestamp'}, inplace=True)
        kline_df['timestamp'] = pd.to_numeric(kline_df['timestamp'])
        if kline_df['timestamp'].max() < 1e11: kline_df['timestamp'] *= 1000
        kline_df.set_index('timestamp', inplace=True)
        
        # Load OI (Open Interest is 5min resolution usually)
        oi_files = sorted(list((DATALAKE / f"{exchange}/{symbol}").glob("*_open_interest_5min.csv")))
        oi_files = [f for f in oi_files if f.name >= start_date]
        
        if not oi_files: return None
        
        dfs = []
        for f in oi_files:
            try:
                df = pd.read_csv(f, engine='c')
                # Format varies, usually has timestamp and openInterest
                ts_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
                val_col = 'openInterest' if 'openInterest' in df.columns else 'sumOpenInterest' if 'sumOpenInterest' in df.columns else df.columns[1]
                
                df = df[[ts_col, val_col]]
                df.columns = ['timestamp', 'oi']
                dfs.append(df)
            except: pass
            
        oi_df = pd.concat(dfs, ignore_index=True)
        oi_df['timestamp'] = pd.to_numeric(oi_df['timestamp'])
        if oi_df['timestamp'].max() < 1e11: oi_df['timestamp'] *= 1000
        oi_df.set_index('timestamp', inplace=True)
        
        # Merge on exact index, forward fill OI (since it's 5min)
        merged = kline_df.join(oi_df, how='left')
        merged['oi'] = merged['oi'].ffill()
        merged = merged.dropna(subset=['close', 'oi'])
        merged = merged[~merged.index.duplicated(keep='last')]
        
        return merged
    except Exception as e:
        return None

def analyze_oi_flush(symbol):
    # Hypothesis: A massive drop in Open Interest (Flush) usually indicates liquidations.
    # Liquidations force the price to overextend. Once the flush is over, price should mean-revert.
    df = load_klines_and_oi(symbol, exchange='bybit')
    if df is None or len(df) < 1000:
        return None
        
    # Calculate 5-minute rolling OI change
    df['oi_change_5m'] = df['oi'].pct_change(periods=5)
    df['price_change_5m'] = df['close'].pct_change(periods=5)
    
    # Calculate forward 60-minute return
    df['fwd_ret_60m'] = df['close'].shift(-60) / df['close'] - 1
    
    # Identify Flushes (e.g. OI drops by more than 2% in 5 minutes)
    flush_mask = df['oi_change_5m'] < -0.02
    
    if flush_mask.sum() < 5:
        return None
        
    flushes = df[flush_mask].copy()
    
    # Did the price drop during the flush? (Long liquidations)
    long_liq = flushes[flushes['price_change_5m'] < 0]
    
    # Did the price rise during the flush? (Short liquidations)
    short_liq = flushes[flushes['price_change_5m'] > 0]
    
    res = {
        'symbol': symbol,
        'total_flushes': len(flushes),
        'long_liq_events': len(long_liq),
        'long_liq_fwd_ret_bps': long_liq['fwd_ret_60m'].mean() * 10000 if len(long_liq) > 0 else 0,
        'long_liq_win_rate': (long_liq['fwd_ret_60m'] > 0).mean() if len(long_liq) > 0 else 0,
        
        'short_liq_events': len(short_liq),
        'short_liq_fwd_ret_bps': short_liq['fwd_ret_60m'].mean() * 10000 if len(short_liq) > 0 else 0,
        'short_liq_win_rate': (short_liq['fwd_ret_60m'] < 0).mean() if len(short_liq) > 0 else 0 # Shorting after short liq
    }
    return res


def load_klines_and_premium(symbol, exchange='bybit', start_date="2025-01-01"):
    try:
        # Load Klines
        kline_files = sorted(list((DATALAKE / f"{exchange}/{symbol}").glob("*_kline_1m.csv")))
        kline_files = [f for f in kline_files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name and f.name >= start_date]
        if not kline_files: return None
        
        dfs = []
        time_col = 'startTime' if exchange == 'bybit' else 'open_time'
        for f in kline_files:
            try: dfs.append(pd.read_csv(f, usecols=[time_col, 'close'], engine='c'))
            except: pass
        kline_df = pd.concat(dfs, ignore_index=True)
        kline_df.rename(columns={time_col: 'timestamp'}, inplace=True)
        kline_df['timestamp'] = pd.to_numeric(kline_df['timestamp'])
        if kline_df['timestamp'].max() < 1e11: kline_df['timestamp'] *= 1000
        kline_df.set_index('timestamp', inplace=True)
        
        # Load Premium Index
        prem_files = sorted(list((DATALAKE / f"{exchange}/{symbol}").glob("*_premium_index_kline_1m.csv")))
        prem_files = [f for f in prem_files if f.name >= start_date]
        if not prem_files: return None
        
        dfs = []
        for f in prem_files:
            try:
                df = pd.read_csv(f, usecols=[time_col, 'close'], engine='c')
                dfs.append(df)
            except: pass
        prem_df = pd.concat(dfs, ignore_index=True)
        prem_df.rename(columns={time_col: 'timestamp', 'close': 'premium'}, inplace=True)
        prem_df['timestamp'] = pd.to_numeric(prem_df['timestamp'])
        if prem_df['timestamp'].max() < 1e11: prem_df['timestamp'] *= 1000
        prem_df.set_index('timestamp', inplace=True)
        
        merged = kline_df.join(prem_df, how='inner')
        merged = merged[~merged.index.duplicated(keep='last')]
        return merged
    except:
        return None

def analyze_premium_extreme(symbol):
    # Hypothesis: Extreme Premium Index means retail is overwhelmingly long/short. 
    # Price will likely move in the opposite direction over the next 4 hours.
    df = load_klines_and_premium(symbol, exchange='bybit')
    if df is None or len(df) < 1000:
        return None
        
    df['premium_z'] = (df['premium'] - df['premium'].rolling(1440).mean()) / df['premium'].rolling(1440).std()
    
    # Forward 4 hour return
    df['fwd_ret_4h'] = df['close'].shift(-240) / df['close'] - 1
    
    extreme_long = df[df['premium_z'] > 3.0] # Massive premium (Futures highly overpriced compared to spot)
    extreme_short = df[df['premium_z'] < -3.0] # Massive discount (Futures highly underpriced)
    
    if len(extreme_long) < 5 and len(extreme_short) < 5:
        return None
        
    res = {
        'symbol': symbol,
        'extreme_long_events': len(extreme_long),
        'extreme_long_fwd_ret_bps': extreme_long['fwd_ret_4h'].mean() * 10000 if len(extreme_long) > 0 else 0,
        'extreme_long_short_wr': (extreme_long['fwd_ret_4h'] < 0).mean() if len(extreme_long) > 0 else 0, # Short the top
        
        'extreme_short_events': len(extreme_short),
        'extreme_short_fwd_ret_bps': extreme_short['fwd_ret_4h'].mean() * 10000 if len(extreme_short) > 0 else 0,
        'extreme_short_long_wr': (extreme_short['fwd_ret_4h'] > 0).mean() if len(extreme_short) > 0 else 0 # Long the bottom
    }
    return res

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'SUIUSDT', 'WLDUSDT', 'PEPEUSDT', 'DYDXUSDT', 'ENAUSDT']
    
    print("--- 1. Running Open Interest Flush Analysis ---")
    oi_results = []
    with Pool(min(4, os.cpu_count() or 4)) as p:
        for res in p.imap_unordered(analyze_oi_flush, symbols):
            if res: oi_results.append(res)
            
    if oi_results:
        oi_df = pd.DataFrame(oi_results)
        print(oi_df.to_string(index=False))
    
    print("\n--- 2. Running Premium Index Extremes Analysis ---")
    prem_results = []
    with Pool(min(4, os.cpu_count() or 4)) as p:
        for res in p.imap_unordered(analyze_premium_extreme, symbols):
            if res: prem_results.append(res)
            
    if prem_results:
        prem_df = pd.DataFrame(prem_results)
        print(prem_df.to_string(index=False))

