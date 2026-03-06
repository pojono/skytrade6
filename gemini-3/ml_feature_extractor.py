import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def extract_features(symbol, start_date="2025-01-01"):
    try:
        # Load Binance Metrics (LS Ratio & OI)
        metrics_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_metrics.csv")))
        metrics_files = [f for f in metrics_files if f.name >= start_date]
        if not metrics_files: return None
        
        dfs = []
        for f in metrics_files:
            try: dfs.append(pd.read_csv(f, usecols=[
                'create_time', 'count_toptrader_long_short_ratio', 
                'sum_toptrader_long_short_ratio', 'sum_open_interest_value',
                'sum_taker_long_short_vol_ratio'
            ], engine='c'))
            except: pass
        if not dfs: return None
        
        m_df = pd.concat(dfs, ignore_index=True)
        m_df.rename(columns={
            'create_time': 'timestamp', 
            'count_toptrader_long_short_ratio': 'count_ls',
            'sum_toptrader_long_short_ratio': 'vol_ls',
            'sum_open_interest_value': 'oi_usd',
            'sum_taker_long_short_vol_ratio': 'taker_ratio'
        }, inplace=True)
        try: m_df['timestamp'] = pd.to_datetime(m_df['timestamp']).astype(np.int64) // 10**6
        except: m_df['timestamp'] = pd.to_numeric(m_df['timestamp'])
        if m_df['timestamp'].max() < 1e11: m_df['timestamp'] *= 1000
        m_df.set_index('timestamp', inplace=True)
        
        # Load Bybit Funding Rate
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
        
        # Load Premium Index
        prem_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_premium_index_kline_1m.csv")))
        prem_files = [f for f in prem_files if f.name >= start_date]
        if not prem_files: return None
        
        dfs = []
        for f in prem_files:
            try: dfs.append(pd.read_csv(f, usecols=['open_time', 'close'], engine='c'))
            except: pass
        if not dfs: return None
        prem_df = pd.concat(dfs, ignore_index=True)
        prem_df.rename(columns={'open_time': 'timestamp', 'close': 'premium'}, inplace=True)
        prem_df['timestamp'] = pd.to_numeric(prem_df['timestamp'])
        if prem_df['timestamp'].max() < 1e11: prem_df['timestamp'] *= 1000
        prem_df.set_index('timestamp', inplace=True)
        
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
        
        # Merge all
        merged = kline_df.join(m_df, how='left').join(fr_df, how='left').join(prem_df, how='left')
        merged = merged.ffill()
        merged = merged.dropna()
        merged = merged[~merged.index.duplicated(keep='last')]
        
        # Resample to Hourly
        merged.index = pd.to_datetime(merged.index, unit='ms')
        hourly = merged.resample('1h').agg({
            'close': 'last',
            'count_ls': 'last',
            'vol_ls': 'last',
            'oi_usd': 'last',
            'taker_ratio': 'last',
            'funding_rate': 'last',
            'premium': 'last'
        }).dropna()
        
        # Feature Engineering (Z-Scores)
        # 7-day rolling for OI and Funding (168h)
        hourly['oi_z'] = (hourly['oi_usd'] - hourly['oi_usd'].rolling(168).mean()) / hourly['oi_usd'].rolling(168).std()
        hourly['fr_z'] = (hourly['funding_rate'] - hourly['funding_rate'].rolling(168).mean()) / hourly['funding_rate'].rolling(168).std()
        
        # 1-day rolling for Sentiment (24h)
        hourly['count_z'] = (hourly['count_ls'] - hourly['count_ls'].rolling(24).mean()) / hourly['count_ls'].rolling(24).std()
        hourly['vol_z'] = (hourly['vol_ls'] - hourly['vol_ls'].rolling(24).mean()) / hourly['vol_ls'].rolling(24).std()
        hourly['taker_z'] = (hourly['taker_ratio'] - hourly['taker_ratio'].rolling(24).mean()) / hourly['taker_ratio'].rolling(24).std()
        hourly['prem_z'] = (hourly['premium'] - hourly['premium'].rolling(24).mean()) / hourly['premium'].rolling(24).std()
        
        # Momentum
        hourly['mom_4h'] = hourly['close'] / hourly['close'].shift(4) - 1
        hourly['mom_24h'] = hourly['close'] / hourly['close'].shift(24) - 1
        
        # Forward Return Target (Hold 24h)
        hourly['fwd_ret_24h'] = hourly['close'].shift(-24) / hourly['close'] - 1
        
        hourly = hourly.dropna()
        hourly['symbol'] = symbol
        
        return hourly
    except: return None

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'SUIUSDT', 'WLDUSDT', 'PEPEUSDT', 'DYDXUSDT', 'ENAUSDT']
    print("--- Extracting ML Features ---")
    
    df_list = []
    with Pool(min(4, os.cpu_count() or 4)) as p:
        for df in p.imap_unordered(extract_features, symbols):
            if df is not None: df_list.append(df)
            
    if df_list:
        final_df = pd.concat(df_list)
        final_df.to_csv("ml_dataset.csv", index=False)
        print(f"Saved dataset with {len(final_df)} rows and {len(final_df.columns)} columns.")

