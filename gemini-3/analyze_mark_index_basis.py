import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def load_mark_and_index(symbol, exchange='binance', start_date="2025-01-01"):
    try:
        # Load Mark Price
        mark_files = sorted(list((DATALAKE / f"{exchange}/{symbol}").glob("*_mark_price_kline_1m.csv")))
        mark_files = [f for f in mark_files if f.name >= start_date]
        if not mark_files: return None
        
        dfs = []
        time_col = 'startTime' if exchange == 'bybit' else 'open_time'
        for f in mark_files:
            try: dfs.append(pd.read_csv(f, usecols=[time_col, 'close'], engine='c'))
            except: pass
            
        if not dfs: return None
        
        mark_df = pd.concat(dfs, ignore_index=True)
        mark_df.rename(columns={time_col: 'timestamp', 'close': 'mark_price'}, inplace=True)
        mark_df['timestamp'] = pd.to_numeric(mark_df['timestamp'])
        if mark_df['timestamp'].max() < 1e11: mark_df['timestamp'] *= 1000
        mark_df.set_index('timestamp', inplace=True)
        mark_df = mark_df[~mark_df.index.duplicated(keep='last')]
        
        # Load Index Price
        index_files = sorted(list((DATALAKE / f"{exchange}/{symbol}").glob("*_index_price_kline_1m.csv")))
        index_files = [f for f in index_files if f.name >= start_date]
        if not index_files: return None
        
        dfs = []
        for f in index_files:
            try: dfs.append(pd.read_csv(f, usecols=[time_col, 'close'], engine='c'))
            except: pass
            
        if not dfs: return None
        
        index_df = pd.concat(dfs, ignore_index=True)
        index_df.rename(columns={time_col: 'timestamp', 'close': 'index_price'}, inplace=True)
        index_df['timestamp'] = pd.to_numeric(index_df['timestamp'])
        if index_df['timestamp'].max() < 1e11: index_df['timestamp'] *= 1000
        index_df.set_index('timestamp', inplace=True)
        index_df = index_df[~index_df.index.duplicated(keep='last')]
        
        # Load Futures Klines to get actual tradable returns
        kline_files = sorted(list((DATALAKE / f"{exchange}/{symbol}").glob("*_kline_1m.csv")))
        kline_files = [f for f in kline_files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name and f.name >= start_date]
        if not kline_files: return None
        
        dfs = []
        for f in kline_files:
            try: dfs.append(pd.read_csv(f, usecols=[time_col, 'close'], engine='c'))
            except: pass
            
        if not dfs: return None
        
        kline_df = pd.concat(dfs, ignore_index=True)
        kline_df.rename(columns={time_col: 'timestamp', 'close': 'trade_price'}, inplace=True)
        kline_df['timestamp'] = pd.to_numeric(kline_df['timestamp'])
        if kline_df['timestamp'].max() < 1e11: kline_df['timestamp'] *= 1000
        kline_df.set_index('timestamp', inplace=True)
        kline_df = kline_df[~kline_df.index.duplicated(keep='last')]
        
        merged = mark_df.join(index_df, how='inner').join(kline_df, how='inner')
        return merged
    except Exception as e:
        return None

def analyze_basis_reversion(symbol):
    # Hypothesis: Mark Price is the local exchange futures price (smoothed). 
    # Index Price is the global spot average. 
    # If Mark heavily deviates from Index, it's a localized squeeze.
    # The tradable futures price should revert to the Index Price over the next 10 to 60 mins.
    df = load_mark_and_index(symbol, exchange='binance')
    if df is None or len(df) < 1000:
        return None
        
    df['basis_bps'] = (df['mark_price'] - df['index_price']) / df['index_price'] * 10000
    df['basis_z'] = (df['basis_bps'] - df['basis_bps'].rolling(1440).mean()) / df['basis_bps'].rolling(1440).std()
    
    # Track returns for 5m, 15m, 60m
    for lag in [5, 15, 60]:
        df[f'fwd_ret_{lag}m_bps'] = (df['trade_price'].shift(-lag) / df['trade_price'] - 1) * 10000
        
    df = df.dropna()
    
    # Extreme Positive Basis (Futures highly overpriced locally vs global spot)
    ext_pos = df[df['basis_z'] > 3.5]
    # Extreme Negative Basis (Futures highly underpriced locally vs global spot)
    ext_neg = df[df['basis_z'] < -3.5]
    
    if len(ext_pos) < 5 and len(ext_neg) < 5:
        return None
        
    res = {'symbol': symbol}
    
    if len(ext_pos) > 0:
        res['ext_pos_events'] = len(ext_pos)
        res['pos_basis_fwd_5m_bps'] = ext_pos['fwd_ret_5m_bps'].mean()
        res['pos_basis_fwd_15m_bps'] = ext_pos['fwd_ret_15m_bps'].mean()
        res['pos_basis_fwd_60m_bps'] = ext_pos['fwd_ret_60m_bps'].mean()
        res['pos_basis_short_wr_15m'] = (ext_pos['fwd_ret_15m_bps'] < 0).mean()
        
    if len(ext_neg) > 0:
        res['ext_neg_events'] = len(ext_neg)
        res['neg_basis_fwd_5m_bps'] = ext_neg['fwd_ret_5m_bps'].mean()
        res['neg_basis_fwd_15m_bps'] = ext_neg['fwd_ret_15m_bps'].mean()
        res['neg_basis_fwd_60m_bps'] = ext_neg['fwd_ret_60m_bps'].mean()
        res['neg_basis_long_wr_15m'] = (ext_neg['fwd_ret_15m_bps'] > 0).mean()
        
    return res

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'SUIUSDT', 'WLDUSDT', 'PEPEUSDT', 'DYDXUSDT', 'ENAUSDT']
    
    print("--- Running Mark vs Index Price Basis Reversion Analysis ---")
    results = []
    with Pool(min(4, os.cpu_count() or 4)) as p:
        for res in p.imap_unordered(analyze_basis_reversion, symbols):
            if res: results.append(res)
            
    if results:
        df = pd.DataFrame(results)
        cols = ['symbol', 'ext_pos_events', 'pos_basis_fwd_15m_bps', 'pos_basis_short_wr_15m', 
                'ext_neg_events', 'neg_basis_fwd_15m_bps', 'neg_basis_long_wr_15m']
        cols = [c for c in cols if c in df.columns]
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        print(df[cols].to_string(index=False))

