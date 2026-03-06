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
        # Load Binance Metrics (LS Ratio)
        metrics_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_metrics.csv")))
        metrics_files = [f for f in metrics_files if f.name >= start_date]
        if not metrics_files: return None
        
        dfs = []
        for f in metrics_files:
            try: dfs.append(pd.read_csv(f, usecols=['create_time', 'count_toptrader_long_short_ratio', 'sum_toptrader_long_short_ratio'], engine='c'))
            except: pass
        if not dfs: return None
        
        m_df = pd.concat(dfs, ignore_index=True)
        m_df.rename(columns={'create_time': 'timestamp', 
                             'count_toptrader_long_short_ratio': 'count_ls',
                             'sum_toptrader_long_short_ratio': 'vol_ls'}, inplace=True)
        try: m_df['timestamp'] = pd.to_datetime(m_df['timestamp']).astype(np.int64) // 10**6
        except: m_df['timestamp'] = pd.to_numeric(m_df['timestamp'])
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
        
        # Merge and Resample to 1-hour
        merged = kline_df.join(m_df, how='left')
        merged['count_ls'] = merged['count_ls'].ffill()
        merged['vol_ls'] = merged['vol_ls'].ffill()
        merged = merged.dropna(subset=['close', 'count_ls', 'vol_ls'])
        merged = merged[~merged.index.duplicated(keep='last')]
        
        merged.index = pd.to_datetime(merged.index, unit='ms')
        hourly = merged.resample('1h').agg({
            'close': 'last',
            'count_ls': 'mean',
            'vol_ls': 'mean'
        }).dropna()
        
        return hourly
    except: return None

def generate_signals(df):
    # 24-hour rolling window (24 periods) for Z-scores
    df['count_z'] = (df['count_ls'] - df['count_ls'].rolling(24).mean()) / df['count_ls'].rolling(24).std()
    df['vol_z'] = (df['vol_ls'] - df['vol_ls'].rolling(24).mean()) / df['vol_ls'].rolling(24).std()
    
    # Initialize signal
    df['signal'] = 0
    
    # Bearish Divergence (Retail Long, Whales Short) -> Short
    df.loc[(df['count_z'] > 1.5) & (df['vol_z'] < -1.5), 'signal'] = -1
    
    # Bullish Divergence (Retail Short, Whales Long) -> Long
    df.loc[(df['count_z'] < -1.5) & (df['vol_z'] > 1.5), 'signal'] = 1
    
    # Avoid overlapping entries
    df['signal_shifted'] = df['signal'].shift(1).fillna(0)
    df.loc[df['signal'] == df['signal_shifted'], 'signal'] = 0
    
    return df

def run_strat_2(symbol):
    df = load_data(symbol)
    if df is None or len(df) < 500: return None
    
    df = generate_signals(df)
    
    # Hold for 4 hours (4 periods, since it's hourly data)
    # Taker fee 5 bps
    results = run_backtest(df, hold_periods=4, fee_bps=5.0)
    
    if results['events'] > 0:
        results['symbol'] = symbol
        return results
    return None

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'SUIUSDT', 'WLDUSDT', 'PEPEUSDT', 'DYDXUSDT', 'ENAUSDT']
    print("--- Backtesting Strategy 2: The Whale Shadow (4h Hold, 5bps fee) ---")
    
    res_list = []
    with Pool(min(4, os.cpu_count() or 4)) as p:
        for r in p.imap_unordered(run_strat_2, symbols):
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
