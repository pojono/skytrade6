import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def load_kline_returns(symbol, exchange='binance', start_date="2025-01-01"):
    try:
        kline_files = sorted(list((DATALAKE / f"{exchange}/{symbol}").glob("*_kline_1m.csv")))
        kline_files = [f for f in kline_files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name and f.name >= start_date]
        
        if not kline_files: return None
        
        dfs = []
        time_col = 'startTime' if exchange == 'bybit' else 'open_time'
        for f in kline_files:
            try: dfs.append(pd.read_csv(f, usecols=[time_col, 'close'], engine='c'))
            except: pass
            
        if not dfs: return None
        
        df = pd.concat(dfs, ignore_index=True)
        df.rename(columns={time_col: 'timestamp'}, inplace=True)
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        if df['timestamp'].max() < 1e11: df['timestamp'] *= 1000
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='last')]
        
        df.index = pd.to_datetime(df.index, unit='ms')
        return df
    except:
        return None

def analyze_hourly_seasonality(symbol):
    # Hypothesis: Crypto is heavily algorithmic. Algorithms often execute VWAP/TWAP orders
    # at the start of every hour. Does the price reliably pump or dump in the first 5 minutes of an hour?
    df = load_kline_returns(symbol)
    if df is None: return None
    
    # Calculate 5-minute forward return for every minute
    df['fwd_ret_5m_bps'] = (df['close'].shift(-5) / df['close'] - 1) * 10000
    
    df['minute'] = df.index.minute
    
    # Group by the minute of the hour
    seasonal = df.groupby('minute')['fwd_ret_5m_bps'].mean()
    
    # We want to see if the return from minute 00 to 05 is significantly different from average
    start_of_hour_bps = seasonal[0] # Return from HH:00 to HH:05
    end_of_hour_bps = seasonal[55]  # Return from HH:55 to HH:00
    avg_5m_bps = df['fwd_ret_5m_bps'].mean()
    
    # Identify the best and worst minute to buy (holding for 5 mins)
    best_minute = seasonal.idxmax()
    worst_minute = seasonal.idxmin()
    
    return {
        'symbol': symbol,
        'avg_5m_bps': avg_5m_bps,
        'HH:00_to_HH:05_bps': start_of_hour_bps,
        'HH:55_to_HH:00_bps': end_of_hour_bps,
        'best_entry_minute': best_minute,
        'best_entry_bps': seasonal[best_minute],
        'worst_entry_minute': worst_minute,
        'worst_entry_bps': seasonal[worst_minute]
    }

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'SUIUSDT', 'WLDUSDT', 'PEPEUSDT', 'DYDXUSDT', 'ENAUSDT']
    
    print("--- Intra-Hour Seasonality (5-minute returns) ---")
    results = []
    with Pool(min(4, os.cpu_count() or 4)) as p:
        for res in p.imap_unordered(analyze_hourly_seasonality, symbols):
            if res: results.append(res)
            
    if results:
        df = pd.DataFrame(results)
        cols = ['symbol', 'HH:00_to_HH:05_bps', 'HH:55_to_HH:00_bps', 'best_entry_minute', 'best_entry_bps', 'worst_entry_minute', 'worst_entry_bps']
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        print(df[cols].to_string(index=False))

