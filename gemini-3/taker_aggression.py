import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def load_klines_and_taker(symbol, start_date="2025-01-01"):
    try:
        # Load Binance Metrics (which contains Taker Long/Short Vol Ratio)
        metrics_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_metrics.csv")))
        metrics_files = [f for f in metrics_files if f.name >= start_date]
        if not metrics_files: return None
        
        dfs = []
        for f in metrics_files:
            try:
                # 'sum_taker_long_short_vol_ratio' represents the ratio of Taker Buy Volume to Taker Sell Volume
                df = pd.read_csv(f, usecols=['create_time', 'sum_taker_long_short_vol_ratio'], engine='c')
                dfs.append(df)
            except: pass
            
        if not dfs: return None
        
        m_df = pd.concat(dfs, ignore_index=True)
        m_df.rename(columns={'create_time': 'timestamp', 'sum_taker_long_short_vol_ratio': 'taker_ratio'}, inplace=True)
        
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
        merged['taker_ratio'] = merged['taker_ratio'].ffill()
        merged = merged.dropna(subset=['close', 'taker_ratio'])
        merged = merged[~merged.index.duplicated(keep='last')]
        
        return merged
    except Exception as e:
        return None

def analyze_taker_exhaustion(symbol):
    # Hypothesis: Taker Buy/Sell ratio measures pure market aggression. 
    # If it is extremely high (Taker Buys >> Taker Sells) but the price fails to make a new high,
    # it indicates hidden limit sell walls (absorption). Takers will exhaust, and price will reverse down.
    # Conversely, extreme selling into limit buys indicates bottoming.
    
    df = load_klines_and_taker(symbol)
    if df is None or len(df) < 1000: return None
    
    # Smooth the taker ratio to find trends (usually 5m resolution, let's use 60m smoothing for mean)
    df['taker_z'] = (df['taker_ratio'] - df['taker_ratio'].rolling(288).mean()) / df['taker_ratio'].rolling(288).std()
    
    # Price trend over the same 288 periods (24 hours if 5m, but here index is 1m so it's ~4.8 hours)
    # Let's align to rolling 4 hours (240 mins)
    df['taker_z'] = (df['taker_ratio'] - df['taker_ratio'].rolling(240).mean()) / df['taker_ratio'].rolling(240).std()
    
    # Calculate 4-hour forward return
    df['fwd_ret_4h_bps'] = (df['close'].shift(-240) / df['close'] - 1) * 10000
    
    df = df.dropna()
    
    # Taker Aggression Exhaustion: Takers are massively buying (Z > 3.0), but price is about to drop
    ext_buy = df[df['taker_z'] > 3.0]
    
    # Taker Panic Exhaustion: Takers are massively selling (Z < -3.0), but price is about to bounce
    ext_sell = df[df['taker_z'] < -3.0]
    
    res = {'symbol': symbol}
    
    if len(ext_buy) > 5:
        res['ext_buy_events'] = len(ext_buy)
        res['buy_exhaust_fwd_ret_bps'] = ext_buy['fwd_ret_4h_bps'].mean()
        res['buy_exhaust_short_wr'] = (ext_buy['fwd_ret_4h_bps'] < 0).mean()
        
    if len(ext_sell) > 5:
        res['ext_sell_events'] = len(ext_sell)
        res['sell_exhaust_fwd_ret_bps'] = ext_sell['fwd_ret_4h_bps'].mean()
        res['sell_exhaust_long_wr'] = (ext_sell['fwd_ret_4h_bps'] > 0).mean()
        
    return res if (len(ext_buy) > 5 or len(ext_sell) > 5) else None

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'SUIUSDT', 'WLDUSDT', 'PEPEUSDT', 'DYDXUSDT', 'ENAUSDT']
    
    print("--- Taker Aggression Exhaustion (Taker Buy/Sell Vol Ratio) ---")
    results = []
    with Pool(min(4, os.cpu_count() or 4)) as p:
        for res in p.imap_unordered(analyze_taker_exhaustion, symbols):
            if res: results.append(res)
            
    if results:
        df = pd.DataFrame(results).fillna(0)
        cols = ['symbol', 'ext_buy_events', 'buy_exhaust_fwd_ret_bps', 'buy_exhaust_short_wr', 
                'ext_sell_events', 'sell_exhaust_fwd_ret_bps', 'sell_exhaust_long_wr']
        cols = [c for c in cols if c in df.columns]
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        print(df[cols].to_string(index=False))

