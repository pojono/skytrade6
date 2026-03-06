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
        # Load Mark Price
        mark_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_mark_price_kline_1m.csv")))
        mark_files = [f for f in mark_files if f.name >= start_date]
        if not mark_files: return None
        dfs = []
        for f in mark_files:
            try: dfs.append(pd.read_csv(f, usecols=['open_time', 'close'], engine='c'))
            except: pass
        if not dfs: return None
        mark_df = pd.concat(dfs, ignore_index=True)
        mark_df.rename(columns={'open_time': 'timestamp', 'close': 'mark_price'}, inplace=True)
        mark_df['timestamp'] = pd.to_numeric(mark_df['timestamp'])
        if mark_df['timestamp'].max() < 1e11: mark_df['timestamp'] *= 1000
        mark_df.set_index('timestamp', inplace=True)
        
        # Load Index Price
        index_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_index_price_kline_1m.csv")))
        index_files = [f for f in index_files if f.name >= start_date]
        if not index_files: return None
        dfs = []
        for f in index_files:
            try: dfs.append(pd.read_csv(f, usecols=['open_time', 'close'], engine='c'))
            except: pass
        if not dfs: return None
        index_df = pd.concat(dfs, ignore_index=True)
        index_df.rename(columns={'open_time': 'timestamp', 'close': 'index_price'}, inplace=True)
        index_df['timestamp'] = pd.to_numeric(index_df['timestamp'])
        if index_df['timestamp'].max() < 1e11: index_df['timestamp'] *= 1000
        index_df.set_index('timestamp', inplace=True)
        
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
        
        # Load Open Interest (Bybit has 5m OI, we can ffill to 1m)
        oi_files = sorted(list((DATALAKE / f"bybit/{symbol}").glob("*_open_interest_5min.csv")))
        oi_files = [f for f in oi_files if f.name >= start_date]
        if not oi_files: return None
        dfs = []
        for f in oi_files:
            try:
                df = pd.read_csv(f, engine='c')
                ts_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
                val_col = 'openInterest' if 'openInterest' in df.columns else 'sumOpenInterest' if 'sumOpenInterest' in df.columns else df.columns[1]
                df = df[[ts_col, val_col]]
                df.columns = ['timestamp', 'oi']
                dfs.append(df)
            except: pass
        if not dfs: return None
        oi_df = pd.concat(dfs, ignore_index=True)
        oi_df['timestamp'] = pd.to_numeric(oi_df['timestamp'])
        if oi_df['timestamp'].max() < 1e11: oi_df['timestamp'] *= 1000
        oi_df.set_index('timestamp', inplace=True)
        
        # Merge exactly on 1-minute (no resampling needed, Strategy 1 is high freq)
        merged = kline_df.join(mark_df, how='left').join(index_df, how='left').join(oi_df, how='left')
        merged['mark_price'] = merged['mark_price'].ffill()
        merged['index_price'] = merged['index_price'].ffill()
        merged['oi'] = merged['oi'].ffill()
        
        merged = merged.dropna(subset=['close', 'mark_price', 'index_price', 'oi'])
        merged = merged[~merged.index.duplicated(keep='last')]
        
        return merged
    except: return None

def generate_signals(df):
    # Basis Score
    df['basis_bps'] = (df['mark_price'] - df['index_price']) / df['index_price'] * 10000
    df['basis_z'] = (df['basis_bps'] - df['basis_bps'].rolling(1440).mean()) / df['basis_bps'].rolling(1440).std()
    
    # Open Interest Flush (Drop of >2% in 5 minutes)
    df['oi_change_5m'] = df['oi'].pct_change(periods=5)
    
    # Initialize signal
    df['signal'] = 0
    
    # Local Flash Crash (Buy the Blood)
    # Long if Basis is extremely negative (Mark Price crashed below Index) AND OI dropped heavily (Long liquidations)
    df.loc[(df['basis_z'] < -3.5) & (df['oi_change_5m'] < -0.02), 'signal'] = 1
    
    # Local Pump Squeeze (Short the Spike)
    # Short if Basis is extremely positive (Mark Price pumped above Index) AND OI dropped heavily (Short liquidations)
    df.loc[(df['basis_z'] > 3.5) & (df['oi_change_5m'] < -0.02), 'signal'] = -1
    
    # Avoid overlapping entries
    df['signal_shifted'] = df['signal'].shift(1).fillna(0)
    df.loc[df['signal'] == df['signal_shifted'], 'signal'] = 0
    
    return df

def run_strat_1(symbol):
    df = load_data(symbol)
    if df is None or len(df) < 2000: return None
    
    df = generate_signals(df)
    
    # Hold for 15 minutes
    # Taker fee 5 bps
    results = run_backtest(df, hold_periods=15, fee_bps=5.0)
    
    if results['events'] > 0:
        results['symbol'] = symbol
        return results
    return None

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'SUIUSDT', 'WLDUSDT', 'PEPEUSDT', 'DYDXUSDT', 'ENAUSDT']
    print("--- Backtesting Strategy 1: The Cascade Sniper (15m Hold, 5bps fee) ---")
    
    res_list = []
    with Pool(min(4, os.cpu_count() or 4)) as p:
        for r in p.imap_unordered(run_strat_1, symbols):
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
