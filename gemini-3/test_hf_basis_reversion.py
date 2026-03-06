import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")
COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'LINKUSDT', 'AVAXUSDT', 'NEARUSDT', 'WLDUSDT', 'DOGEUSDT', 'XRPUSDT', 'BNBUSDT']

def analyze_basis_fade(symbol):
    try:
        start_date = "2025-01-01"
        
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
        
        # Load Klines for execution
        kline_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_kline_1m.csv")))
        kline_files = [f for f in kline_files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name and f.name >= start_date]
        
        dfs = []
        for f in kline_files:
            try: dfs.append(pd.read_csv(f, usecols=['open_time', 'close'], engine='c'))
            except: pass
        if not dfs: return None
        
        kline_df = pd.concat(dfs, ignore_index=True)
        kline_df.rename(columns={'open_time': 'timestamp', 'close': 'exec_price'}, inplace=True)
        kline_df['timestamp'] = pd.to_numeric(kline_df['timestamp'])
        if kline_df['timestamp'].max() < 1e11: kline_df['timestamp'] *= 1000
        kline_df.set_index('timestamp', inplace=True)
        
        # Merge
        df = mark_df.join(index_df, how='inner').join(kline_df, how='inner')
        df = df[~df.index.duplicated(keep='last')]
        df.index = pd.to_datetime(df.index, unit='ms')
        
        # Calculate Basis Spread
        df['basis_bps'] = (df['mark_price'] - df['index_price']) / df['index_price'] * 10000
        
        # Rolling 4h Z-score of Basis
        df['basis_z'] = (df['basis_bps'] - df['basis_bps'].rolling(240).mean()) / df['basis_bps'].rolling(240).std()
        
        # Signal: Fade the deviation (If Mark > Index heavily, short futures)
        df['signal'] = 0
        df.loc[df['basis_z'] > 4.0, 'signal'] = -1
        df.loc[df['basis_z'] < -4.0, 'signal'] = 1
        
        df['trade_signal'] = df['signal'].shift(1).fillna(0)
        df['signal_changed'] = df['trade_signal'] != df['trade_signal'].shift(1)
        trades = df[(df['trade_signal'] != 0) & (df['signal_changed'])].copy()
        
        if len(trades) == 0: return None
        
        # Reversion usually happens fast, hold 15m
        df['fwd_15m'] = df['exec_price'].shift(-15)
        trades['exit_price'] = df.loc[trades.index, 'fwd_15m']
        trades = trades.dropna(subset=['exit_price'])
        
        trades['gross_ret'] = (trades['exit_price'] - trades['exec_price']) / trades['exec_price'] * trades['trade_signal']
        trades['net_ret'] = trades['gross_ret'] - (10.0 / 10000) # 5bps * 2
        
        trades['symbol'] = symbol
        return trades[['symbol', 'trade_signal', 'exec_price', 'exit_price', 'net_ret']]
        
    except Exception as e:
        return None

if __name__ == "__main__":
    print(f"Running High-Frequency Basis Reversion on {len(COINS)} coins...")
    
    all_trades = []
    with Pool(min(8, os.cpu_count() or 8)) as p:
        for res in p.imap_unordered(analyze_basis_fade, COINS):
            if res is not None:
                all_trades.append(res)
                
    if not all_trades:
        print("No trades found.")
        exit(0)
        
    final_df = pd.concat(all_trades)
    
    wins = (final_df['net_ret'] > 0).sum()
    total = len(final_df)
    win_rate = wins / total * 100
    
    print("\n=== High-Frequency Basis Strategy Results (Hold 15m) ===")
    print(f"Total Trades: {total}")
    print(f"Win Rate:     {win_rate:.2f}%")
    print(f"Avg Net Ret:  {final_df['net_ret'].mean() * 10000:.2f} bps")
    print(f"Total Return: {final_df['net_ret'].sum() * 100:.2f}%")
    print(f"Sharpe Ratio: {np.sqrt(total) * (final_df['net_ret'].mean() / final_df['net_ret'].std()):.2f}")
    
    print("\nBreakdown by Symbol:")
    breakdown = final_df.groupby('symbol').agg(
        trades=('net_ret', 'count'),
        win_rate=('net_ret', lambda x: (x > 0).mean() * 100),
        avg_bps=('net_ret', lambda x: x.mean() * 10000)
    ).round(2)
    print(breakdown)
