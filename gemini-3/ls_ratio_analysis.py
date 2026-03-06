import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def load_klines_and_ls(symbol, start_date="2025-01-01"):
    try:
        # Load Binance Metrics (which contains LS ratio)
        metrics_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_metrics.csv")))
        metrics_files = [f for f in metrics_files if f.name >= start_date]
        if not metrics_files: return None
        
        dfs = []
        for f in metrics_files:
            try:
                df = pd.read_csv(f, usecols=['create_time', 'count_long_short_ratio'], engine='c')
                df.rename(columns={'create_time': 'timestamp', 'count_long_short_ratio': 'long_short_ratio'}, inplace=True)
                dfs.append(df)
            except: pass
            
        if not dfs: return None
        
        ls_df = pd.concat(dfs, ignore_index=True)
        # Parse datetime string to timestamp
        try:
            ls_df['timestamp'] = pd.to_datetime(ls_df['timestamp']).astype(np.int64) // 10**6
        except:
            ls_df['timestamp'] = pd.to_numeric(ls_df['timestamp'])
            
        if ls_df['timestamp'].max() < 1e11: ls_df['timestamp'] *= 1000
        ls_df.set_index('timestamp', inplace=True)
        
        # Load Klines
        kline_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_kline_1m.csv")))
        kline_files = [f for f in kline_files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name and f.name >= start_date]
        
        if not kline_files: return None
        
        dfs = []
        for f in kline_files:
            try:
                df = pd.read_csv(f, usecols=['open_time', 'close'], engine='c')
                dfs.append(df)
            except: pass
            
        kline_df = pd.concat(dfs, ignore_index=True)
        kline_df.rename(columns={'open_time': 'timestamp'}, inplace=True)
        kline_df['timestamp'] = pd.to_numeric(kline_df['timestamp'])
        if kline_df['timestamp'].max() < 1e11: kline_df['timestamp'] *= 1000
        kline_df.set_index('timestamp', inplace=True)
        
        # Merge on exact index, forward fill LS (since it's usually 5m or 15m)
        merged = kline_df.join(ls_df, how='left')
        merged['long_short_ratio'] = merged['long_short_ratio'].ffill()
        merged = merged.dropna(subset=['close', 'long_short_ratio'])
        merged = merged[~merged.index.duplicated(keep='last')]
        
        return merged
    except Exception as e:
        return None

def analyze_ls_extremes(symbol):
    # Hypothesis: When the Long/Short ratio hits extreme highs, the crowd is overwhelmingly long.
    # The crowd is usually wrong, so this should precede a price drop (mean reversion).
    df = load_klines_and_ls(symbol)
    if df is None or len(df) < 1000:
        return None
        
    # Calculate 24h rolling Z-score of LS ratio
    df['ls_z'] = (df['long_short_ratio'] - df['long_short_ratio'].rolling(1440).mean()) / df['long_short_ratio'].rolling(1440).std()
    
    # Calculate forward 4h return
    df['fwd_ret_4h'] = df['close'].shift(-240) / df['close'] - 1
    
    # Extreme Longs (Crowd is bullish)
    ext_long = df[df['ls_z'] > 2.5]
    
    # Extreme Shorts (Crowd is bearish)
    ext_short = df[df['ls_z'] < -2.5]
    
    if len(ext_long) < 10 and len(ext_short) < 10:
        return None
        
    res = {
        'symbol': symbol,
        'crowd_long_events': len(ext_long),
        'crowd_long_fwd_ret_bps': ext_long['fwd_ret_4h'].mean() * 10000 if len(ext_long) > 0 else 0,
        'crowd_long_fade_wr': (ext_long['fwd_ret_4h'] < 0).mean() if len(ext_long) > 0 else 0, # Short when crowd is long
        
        'crowd_short_events': len(ext_short),
        'crowd_short_fwd_ret_bps': ext_short['fwd_ret_4h'].mean() * 10000 if len(ext_short) > 0 else 0,
        'crowd_short_fade_wr': (ext_short['fwd_ret_4h'] > 0).mean() if len(ext_short) > 0 else 0 # Long when crowd is short
    }
    return res

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'SUIUSDT', 'WLDUSDT', 'PEPEUSDT', 'DYDXUSDT', 'ENAUSDT']
    
    print("--- Running Long/Short Ratio Extremes Analysis (Binance) ---")
    results = []
    with Pool(min(4, os.cpu_count() or 4)) as p:
        for res in p.imap_unordered(analyze_ls_extremes, symbols):
            if res: results.append(res)
            
    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))

