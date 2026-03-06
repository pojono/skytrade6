import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings
from backtest_core import run_backtest

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def load_data(symbol, start_date="2025-01-01"):
    try:
        # Load Binance Metrics (Open Interest)
        metrics_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_metrics.csv")))
        metrics_files = [f for f in metrics_files if f.name >= start_date]
        if not metrics_files: return None
        
        dfs = []
        for f in metrics_files:
            try: dfs.append(pd.read_csv(f, usecols=['create_time', 'sum_open_interest_value'], engine='c'))
            except: pass
        if not dfs: return None
        
        oi_df = pd.concat(dfs, ignore_index=True)
        oi_df.rename(columns={'create_time': 'timestamp', 'sum_open_interest_value': 'oi_usd'}, inplace=True)
        try: oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp']).astype(np.int64) // 10**6
        except: oi_df['timestamp'] = pd.to_numeric(oi_df['timestamp'])
        if oi_df['timestamp'].max() < 1e11: oi_df['timestamp'] *= 1000
        oi_df.set_index('timestamp', inplace=True)
        
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
        
        # Merge and Resample to 1-hour to speed up calculation
        merged = kline_df.join(oi_df, how='left').join(fr_df, how='left')
        merged['oi_usd'] = merged['oi_usd'].ffill()
        merged['funding_rate'] = merged['funding_rate'].ffill()
        merged = merged.dropna(subset=['close', 'oi_usd', 'funding_rate'])
        merged = merged[~merged.index.duplicated(keep='last')]
        
        merged.index = pd.to_datetime(merged.index, unit='ms')
        hourly = merged.resample('1h').agg({
            'close': 'last',
            'oi_usd': 'mean',
            'funding_rate': 'mean'
        }).dropna()
        
        return hourly
    except: return None

def generate_signals(df):
    # 7-day rolling window = 168 hours
    df['oi_z'] = (df['oi_usd'] - df['oi_usd'].rolling(168).mean()) / df['oi_usd'].rolling(168).std()
    df['fr_z'] = (df['funding_rate'] - df['funding_rate'].rolling(168).mean()) / df['funding_rate'].rolling(168).std()
    
    # Initialize signal
    df['signal'] = 0
    
    # Powder Keg (Short)
    df.loc[(df['oi_z'] > 2.0) & (df['fr_z'] > 2.0), 'signal'] = -1
    
    # Despair Pit (Long)
    df.loc[(df['oi_z'] > 2.0) & (df['fr_z'] < -2.0), 'signal'] = 1
    
    # We only take the FIRST signal in a sequence to avoid overlapping trade entries.
    # If signal is the same as the previous minute, ignore it.
    df['signal_shifted'] = df['signal'].shift(1).fillna(0)
    df.loc[df['signal'] == df['signal_shifted'], 'signal'] = 0
    
    return df

def run_strat_3(symbol):
    df = load_data(symbol)
    if df is None or len(df) < 500: return None
    
    df = generate_signals(df)
    
    # Hold for 24 hours (24 periods, since it's hourly data)
    # Taker fee 5 bps
    results = run_backtest(df, hold_periods=24, fee_bps=5.0)
    
    if results['events'] > 0:
        results['symbol'] = symbol
        return results
    return None

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'SUIUSDT', 'WLDUSDT', 'PEPEUSDT', 'DYDXUSDT', 'ENAUSDT']
    print("--- Backtesting Strategy 3: The Powder Keg Swing (24h Hold, 5bps fee) ---")
    
    res_list = []
    with Pool(min(4, os.cpu_count() or 4)) as p:
        for r in p.imap_unordered(run_strat_3, symbols):
            if r: res_list.append(r)
            
    if res_list:
        res_df = pd.DataFrame(res_list)
        cols = ['symbol', 'events', 'win_rate', 'total_net_ret_%', 'avg_net_ret_bps', 'max_drawdown_%', 'sharpe']
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        print(res_df[cols].to_string(index=False))
        
        print("\nAggregate Portfolio Performance:")
        print(f"Total Trades: {res_df['events'].sum()}")
        print(f"Avg Win Rate: {res_df['win_rate'].mean():.2%}")
        print(f"Total Net Return: {res_df['total_net_ret_%'].sum():.2f}%")
