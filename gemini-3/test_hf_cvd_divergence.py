import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

# Let's test a very specific known edge from earlier: 
# "Buy the Blood" - Liquidations during a down move.
COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'LINKUSDT', 'AVAXUSDT', 'NEARUSDT', 'WLDUSDT', 'DOGEUSDT', 'XRPUSDT', 'BNBUSDT']

def analyze_buy_the_blood(symbol):
    try:
        start_date = "2025-01-01"
        
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
        
        kline_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_kline_1m.csv")))
        kline_files = [f for f in kline_files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name and f.name >= start_date]
        if not kline_files: return None
        
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
        
        df = kline_df.join(oi_df, how='inner')
        df = df[~df.index.duplicated(keep='last')]
        df.index = pd.to_datetime(df.index, unit='ms')
        
        # Calculate 5-minute rolling changes
        df['price_pct_5m'] = df['price'].pct_change(5)
        df['oi_pct_5m'] = df['oi_usd'].pct_change(5)
        
        # Buy the Blood: Price drops > 0.5% AND OI drops > 1.5% in 5 minutes (massive long liquidations)
        df['signal'] = 0
        df.loc[(df['price_pct_5m'] < -0.005) & (df['oi_pct_5m'] < -0.015), 'signal'] = 1
        
        # Fade the Squeeze (Short): Price pumps > 0.5% AND OI drops > 1.5% in 5 minutes (massive short liquidations)
        df.loc[(df['price_pct_5m'] > 0.005) & (df['oi_pct_5m'] < -0.015), 'signal'] = -1
        
        df['trade_signal'] = df['signal'].shift(1).fillna(0)
        df['signal_changed'] = df['trade_signal'] != df['trade_signal'].shift(1)
        trades = df[(df['trade_signal'] != 0) & (df['signal_changed'])].copy()
        
        if len(trades) == 0: return None
        
        # Hold 60m
        df['fwd_60m'] = df['price'].shift(-60)
        trades['exit_price'] = df.loc[trades.index, 'fwd_60m']
        trades = trades.dropna(subset=['exit_price'])
        
        trades['gross_ret'] = (trades['exit_price'] - trades['price']) / trades['price'] * trades['trade_signal']
        trades['net_ret'] = trades['gross_ret'] - (10.0 / 10000) 
        
        trades['symbol'] = symbol
        return trades[['symbol', 'trade_signal', 'price', 'exit_price', 'net_ret']]
        
    except Exception as e:
        return None

if __name__ == "__main__":
    print(f"Running High-Frequency Liquidation Fade on {len(COINS)} coins...")
    
    all_trades = []
    with Pool(min(8, os.cpu_count() or 8)) as p:
        for res in p.imap_unordered(analyze_buy_the_blood, COINS):
            if res is not None:
                all_trades.append(res)
                
    if not all_trades:
        print("No trades found.")
        exit(0)
        
    final_df = pd.concat(all_trades)
    
    wins = (final_df['net_ret'] > 0).sum()
    total = len(final_df)
    win_rate = wins / total * 100
    
    print("\n=== High-Frequency Liquidation Fade Results (Hold 60m) ===")
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

