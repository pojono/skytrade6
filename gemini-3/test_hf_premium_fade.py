import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

# Let's test on the Golden Cluster + a few more highly liquid coins
COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'LINKUSDT', 'AVAXUSDT', 'NEARUSDT', 'WLDUSDT', 'DOGEUSDT', 'XRPUSDT', 'BNBUSDT']

def analyze_premium_fade(symbol):
    try:
        start_date = "2025-01-01"
        
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
        
        dfs = []
        for f in kline_files:
            try: dfs.append(pd.read_csv(f, usecols=['open_time', 'close'], engine='c'))
            except: pass
        if not dfs: return None
        
        kline_df = pd.concat(dfs, ignore_index=True)
        kline_df.rename(columns={'open_time': 'timestamp', 'close': 'price'}, inplace=True)
        kline_df['timestamp'] = pd.to_numeric(kline_df['timestamp'])
        if kline_df['timestamp'].max() < 1e11: kline_df['timestamp'] *= 1000
        kline_df.set_index('timestamp', inplace=True)
        
        # Merge on 1m
        df = kline_df.join(prem_df, how='inner')
        df = df[~df.index.duplicated(keep='last')]
        df.index = pd.to_datetime(df.index, unit='ms')
        
        # Calculate rolling 24h (1440m) mean and std of premium
        df['prem_mean'] = df['premium'].rolling(1440).mean()
        df['prem_std'] = df['premium'].rolling(1440).std()
        df['prem_z'] = (df['premium'] - df['prem_mean']) / df['prem_std']
        
        # Generate Signals: Short when premium Z > 3.5, Long when Z < -3.5
        df['signal'] = 0
        df.loc[df['prem_z'] > 3.5, 'signal'] = -1
        df.loc[df['prem_z'] < -3.5, 'signal'] = 1
        
        # Shift signal to avoid lookahead (trade the next minute)
        df['trade_signal'] = df['signal'].shift(1).fillna(0)
        
        # Filter for unique events (don't enter 10 times in 10 minutes)
        df['signal_changed'] = df['trade_signal'] != df['trade_signal'].shift(1)
        trades = df[(df['trade_signal'] != 0) & (df['signal_changed'])].copy()
        
        if len(trades) == 0: return None
        
        # We will hold for exactly 60 minutes
        df['fwd_60m'] = df['price'].shift(-60)
        trades['exit_price'] = df.loc[trades.index, 'fwd_60m']
        trades = trades.dropna(subset=['exit_price'])
        
        trades['gross_ret'] = (trades['exit_price'] - trades['price']) / trades['price'] * trades['trade_signal']
        trades['net_ret'] = trades['gross_ret'] - (10.0 / 10000) # 5bps taker * 2
        
        trades['symbol'] = symbol
        return trades[['symbol', 'trade_signal', 'price', 'exit_price', 'net_ret']]
        
    except Exception as e:
        return None

if __name__ == "__main__":
    print(f"Running High-Frequency Premium Index Reversion on {len(COINS)} coins...")
    
    all_trades = []
    with Pool(min(8, os.cpu_count() or 8)) as p:
        for res in p.imap_unordered(analyze_premium_fade, COINS):
            if res is not None:
                all_trades.append(res)
                
    if not all_trades:
        print("No trades found.")
        exit(0)
        
    final_df = pd.concat(all_trades)
    
    wins = (final_df['net_ret'] > 0).sum()
    total = len(final_df)
    win_rate = wins / total * 100
    
    print("\n=== High-Frequency Strategy Results (Hold 60m) ===")
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

