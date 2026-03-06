import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def load_data(symbol, start_date="2025-01-01"):
    try:
        # Load Binance Metrics for Open Interest (USD Value)
        metrics_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_metrics.csv")))
        metrics_files = [f for f in metrics_files if f.name >= start_date]
        if not metrics_files: return None
        
        dfs = []
        for f in metrics_files:
            try:
                df = pd.read_csv(f, usecols=['create_time', 'sum_open_interest_value'], engine='c')
                dfs.append(df)
            except: pass
            
        if not dfs: return None
        
        oi_df = pd.concat(dfs, ignore_index=True)
        oi_df.rename(columns={'create_time': 'timestamp', 'sum_open_interest_value': 'oi_usd'}, inplace=True)
        
        try:
            oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp']).astype(np.int64) // 10**6
        except:
            oi_df['timestamp'] = pd.to_numeric(oi_df['timestamp'])
            
        if oi_df['timestamp'].max() < 1e11: oi_df['timestamp'] *= 1000
        oi_df.set_index('timestamp', inplace=True)
        
        # Load Bybit Funding Rate (since Binance isn't directly in standard csv format in our datalake)
        bb_fr_files = sorted(list((DATALAKE / f"bybit/{symbol}").glob("*_funding_rate.csv")))
        bb_fr_files = [f for f in bb_fr_files if f.name >= start_date]
        if not bb_fr_files: return None
        
        dfs = []
        for f in bb_fr_files:
            try:
                df = pd.read_csv(f, engine='c')
                ts_col = 'fundingTime' if 'fundingTime' in df.columns else 'calcTime' if 'calcTime' in df.columns else df.columns[0]
                val_col = 'fundingRate' if 'fundingRate' in df.columns else df.columns[2]
                df = df[[ts_col, val_col]]
                df.columns = ['timestamp', 'funding_rate']
                dfs.append(df)
            except: pass
            
        if not dfs: return None
        fr_df = pd.concat(dfs, ignore_index=True)
        fr_df['timestamp'] = pd.to_numeric(fr_df['timestamp'])
        if fr_df['timestamp'].max() < 1e11: fr_df['timestamp'] *= 1000
        fr_df.set_index('timestamp', inplace=True)
        fr_df = fr_df[~fr_df.index.duplicated(keep='last')]
        
        # Load Klines for returns
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
        
        merged = kline_df.join(oi_df, how='left').join(fr_df, how='left')
        merged['oi_usd'] = merged['oi_usd'].ffill()
        merged['funding_rate'] = merged['funding_rate'].ffill()
        merged = merged.dropna(subset=['close', 'oi_usd', 'funding_rate'])
        merged = merged[~merged.index.duplicated(keep='last')]
        
        return merged
    except:
        return None

def analyze_leverage_heat(symbol):
    # Hypothesis: If Open Interest is at a multi-day high (high leverage in system)
    # AND Funding Rate is at a multi-day high (bias is entirely LONG),
    # the market is a "Powder Keg". A small drop will trigger massive long liquidations,
    # leading to highly negative 24-hour forward returns.
    
    df = load_data(symbol)
    if df is None or len(df) < 10000: return None
    
    # We use a 7-day rolling window (10080 minutes) to find Z-scores
    df['oi_z'] = (df['oi_usd'] - df['oi_usd'].rolling(10080).mean()) / df['oi_usd'].rolling(10080).std()
    df['fr_z'] = (df['funding_rate'] - df['funding_rate'].rolling(10080).mean()) / df['funding_rate'].rolling(10080).std()
    
    # Calculate 12-hour (720 min) and 24-hour (1440 min) forward returns
    df['fwd_ret_12h_bps'] = (df['close'].shift(-720) / df['close'] - 1) * 10000
    df['fwd_ret_24h_bps'] = (df['close'].shift(-1440) / df['close'] - 1) * 10000
    
    df = df.dropna()
    
    # The Powder Keg (Max Leverage, Max Long)
    powder_keg = df[(df['oi_z'] > 2.0) & (df['fr_z'] > 2.0)]
    
    # The Despair Pit (Max Leverage, Max Short)
    despair_pit = df[(df['oi_z'] > 2.0) & (df['fr_z'] < -2.0)]
    
    res = {'symbol': symbol}
    
    if len(powder_keg) > 10:
        res['pk_events'] = len(powder_keg)
        res['pk_fwd_12h_bps'] = powder_keg['fwd_ret_12h_bps'].mean()
        res['pk_fwd_24h_bps'] = powder_keg['fwd_ret_24h_bps'].mean()
        res['pk_short_wr_24h'] = (powder_keg['fwd_ret_24h_bps'] < 0).mean()
        
    if len(despair_pit) > 10:
        res['dp_events'] = len(despair_pit)
        res['dp_fwd_12h_bps'] = despair_pit['fwd_ret_12h_bps'].mean()
        res['dp_fwd_24h_bps'] = despair_pit['fwd_ret_24h_bps'].mean()
        res['dp_long_wr_24h'] = (despair_pit['fwd_ret_24h_bps'] > 0).mean()
        
    return res if (len(powder_keg) > 10 or len(despair_pit) > 10) else None

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'SUIUSDT', 'WLDUSDT', 'PEPEUSDT', 'DYDXUSDT', 'ENAUSDT']
    
    print("--- Leverage Heat (Open Interest + Funding Rate Extremes) ---")
    results = []
    with Pool(min(4, os.cpu_count() or 4)) as p:
        for res in p.imap_unordered(analyze_leverage_heat, symbols):
            if res: results.append(res)
            
    if results:
        df = pd.DataFrame(results).fillna(0)
        cols = ['symbol', 'pk_events', 'pk_fwd_24h_bps', 'pk_short_wr_24h', 
                'dp_events', 'dp_fwd_24h_bps', 'dp_long_wr_24h']
        cols = [c for c in cols if c in df.columns]
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        print(df[cols].to_string(index=False))

