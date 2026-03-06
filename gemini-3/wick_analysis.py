import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def load_klines(symbol, start_date="2025-01-01"):
    try:
        kline_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_kline_1m.csv")))
        kline_files = [f for f in kline_files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name and f.name >= start_date]
        
        if not kline_files: return None
        
        dfs = []
        for f in kline_files:
            try: dfs.append(pd.read_csv(f, usecols=['open_time', 'open', 'high', 'low', 'close', 'volume'], engine='c'))
            except: pass
            
        if not dfs: return None
        
        df = pd.concat(dfs, ignore_index=True)
        df.rename(columns={'open_time': 'timestamp'}, inplace=True)
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        if df['timestamp'].max() < 1e11: df['timestamp'] *= 1000
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='last')]
        
        return df
    except:
        return None

def analyze_wick_rejections(symbol):
    # Hypothesis: Massive wicks (where high-close or close-low is huge relative to the body)
    # represent liquidity grabs or stop runs. 
    # If there is a massive upper wick on high volume, it's a bearish rejection.
    df = load_klines(symbol)
    if df is None or len(df) < 1000: return None
    
    # Calculate body and wicks
    df['body'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['total_range'] = df['high'] - df['low']
    
    # Avoid zero division
    df = df[df['total_range'] > 0]
    
    # Calculate rolling volume to find "high volume" spikes
    df['vol_ma'] = df['volume'].rolling(60).mean()
    df['vol_spike'] = df['volume'] / df['vol_ma']
    
    # Calculate forward 60-min return
    df['fwd_ret_60m_bps'] = (df['close'].shift(-60) / df['close'] - 1) * 10000
    
    df = df.dropna()
    
    # Extreme Upper Wick Rejection: 
    # Upper wick is >70% of total range, body is small, volume is >3x average
    bear_rejections = df[(df['upper_wick'] / df['total_range'] > 0.7) & 
                         (df['vol_spike'] > 3.0)]
                         
    # Extreme Lower Wick Rejection:
    # Lower wick is >70% of total range, volume >3x average
    bull_rejections = df[(df['lower_wick'] / df['total_range'] > 0.7) & 
                         (df['vol_spike'] > 3.0)]
                         
    res = {'symbol': symbol}
    
    if len(bear_rejections) > 10:
        res['bear_rej_events'] = len(bear_rejections)
        res['bear_rej_fwd_ret_bps'] = bear_rejections['fwd_ret_60m_bps'].mean()
        res['bear_rej_short_wr'] = (bear_rejections['fwd_ret_60m_bps'] < 0).mean()
        
    if len(bull_rejections) > 10:
        res['bull_rej_events'] = len(bull_rejections)
        res['bull_rej_fwd_ret_bps'] = bull_rejections['fwd_ret_60m_bps'].mean()
        res['bull_rej_long_wr'] = (bull_rejections['fwd_ret_60m_bps'] > 0).mean()
        
    return res if (len(bear_rejections) > 10 or len(bull_rejections) > 10) else None

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'SUIUSDT', 'WLDUSDT', 'PEPEUSDT', 'DYDXUSDT', 'ENAUSDT']
    
    print("--- 1-Minute High-Volume Wick Rejections ---")
    results = []
    with Pool(min(4, os.cpu_count() or 4)) as p:
        for res in p.imap_unordered(analyze_wick_rejections, symbols):
            if res: results.append(res)
            
    if results:
        df = pd.DataFrame(results).fillna(0)
        cols = ['symbol', 'bear_rej_events', 'bear_rej_fwd_ret_bps', 'bear_rej_short_wr', 
                'bull_rej_events', 'bull_rej_fwd_ret_bps', 'bull_rej_long_wr']
        cols = [c for c in cols if c in df.columns]
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        print(df[cols].to_string(index=False))

