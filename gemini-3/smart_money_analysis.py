import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def load_metrics_and_klines(symbol, start_date="2025-01-01"):
    try:
        # Load Binance Metrics
        metrics_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_metrics.csv")))
        metrics_files = [f for f in metrics_files if f.name >= start_date]
        if not metrics_files: return None
        
        dfs = []
        for f in metrics_files:
            try:
                # 'count_toptrader_long_short_ratio' is the RATIO OF ACCOUNTS (Retail/Crowd)
                # 'sum_toptrader_long_short_ratio' is the RATIO OF MARGIN/VOLUME (Whales/Smart Money)
                df = pd.read_csv(f, usecols=['create_time', 'count_toptrader_long_short_ratio', 'sum_toptrader_long_short_ratio'], engine='c')
                dfs.append(df)
            except: pass
            
        if not dfs: return None
        
        m_df = pd.concat(dfs, ignore_index=True)
        m_df.rename(columns={'create_time': 'timestamp', 
                             'count_toptrader_long_short_ratio': 'count_ls',
                             'sum_toptrader_long_short_ratio': 'vol_ls'}, inplace=True)
        
        try:
            m_df['timestamp'] = pd.to_datetime(m_df['timestamp']).astype(np.int64) // 10**6
        except:
            m_df['timestamp'] = pd.to_numeric(m_df['timestamp'])
            
        if m_df['timestamp'].max() < 1e11: m_df['timestamp'] *= 1000
        m_df.set_index('timestamp', inplace=True)
        
        # Load Klines
        kline_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_kline_1m.csv")))
        kline_files = [f for f in kline_files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name and f.name >= start_date]
        
        if not kline_files: return None
        
        dfs = []
        for f in kline_files:
            try: dfs.append(pd.read_csv(f, usecols=['open_time', 'close'], engine='c'))
            except: pass
            
        if not dfs: return None
        kline_df = pd.concat(dfs, ignore_index=True)
        kline_df.rename(columns={'open_time': 'timestamp'}, inplace=True)
        kline_df['timestamp'] = pd.to_numeric(kline_df['timestamp'])
        if kline_df['timestamp'].max() < 1e11: kline_df['timestamp'] *= 1000
        kline_df.set_index('timestamp', inplace=True)
        
        merged = kline_df.join(m_df, how='left')
        merged['count_ls'] = merged['count_ls'].ffill()
        merged['vol_ls'] = merged['vol_ls'].ffill()
        merged = merged.dropna(subset=['close', 'count_ls', 'vol_ls'])
        merged = merged[~merged.index.duplicated(keep='last')]
        
        return merged
    except Exception as e:
        return None

def analyze_smart_money_divergence(symbol):
    # Hypothesis: When Count LS (Number of Accounts) is extremely high, retail is long.
    # When Volume LS (Margin Size) is low at the same time, Whales are short.
    # The divergence (Retail Long, Whales Short) is a massive bearish signal.
    df = load_metrics_and_klines(symbol)
    if df is None or len(df) < 1000: return None
    
    # Calculate rolling Z-scores
    df['count_z'] = (df['count_ls'] - df['count_ls'].rolling(1440).mean()) / df['count_ls'].rolling(1440).std()
    df['vol_z'] = (df['vol_ls'] - df['vol_ls'].rolling(1440).mean()) / df['vol_ls'].rolling(1440).std()
    
    # Calculate 4-hour forward return
    df['fwd_ret_4h_bps'] = (df['close'].shift(-240) / df['close'] - 1) * 10000
    
    df = df.dropna()
    
    # Divergence 1: Retail Long (Count Z > 1.5), Whales Short (Vol Z < -1.5) -> We should Short
    bear_div = df[(df['count_z'] > 1.5) & (df['vol_z'] < -1.5)]
    
    # Divergence 2: Retail Short (Count Z < -1.5), Whales Long (Vol Z > 1.5) -> We should Long
    bull_div = df[(df['count_z'] < -1.5) & (df['vol_z'] > 1.5)]
    
    res = {'symbol': symbol}
    
    if len(bear_div) > 0:
        res['bear_div_events'] = len(bear_div)
        res['bear_div_fwd_ret_bps'] = bear_div['fwd_ret_4h_bps'].mean()
        res['bear_div_short_wr'] = (bear_div['fwd_ret_4h_bps'] < 0).mean()
        
    if len(bull_div) > 0:
        res['bull_div_events'] = len(bull_div)
        res['bull_div_fwd_ret_bps'] = bull_div['fwd_ret_4h_bps'].mean()
        res['bull_div_long_wr'] = (bull_div['fwd_ret_4h_bps'] > 0).mean()
        
    return res if (len(bear_div) > 0 or len(bull_div) > 0) else None

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'SUIUSDT', 'WLDUSDT', 'PEPEUSDT', 'DYDXUSDT', 'ENAUSDT']
    
    print("--- Smart Money Divergence (Account Count vs Volume Margin) ---")
    results = []
    with Pool(min(4, os.cpu_count() or 4)) as p:
        for res in p.imap_unordered(analyze_smart_money_divergence, symbols):
            if res: results.append(res)
            
    if results:
        df = pd.DataFrame(results).fillna(0)
        cols = ['symbol', 'bear_div_events', 'bear_div_fwd_ret_bps', 'bear_div_short_wr', 
                'bull_div_events', 'bull_div_fwd_ret_bps', 'bull_div_long_wr']
        cols = [c for c in cols if c in df.columns]
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        print(df[cols].to_string(index=False))

