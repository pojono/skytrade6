import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

# High-liquidity coins
COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT']

def analyze_premium_fade_v2(symbol):
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
            try: dfs.append(pd.read_csv(f, usecols=['open_time', 'high', 'low', 'close'], engine='c'))
            except: pass
        if not dfs: return None
        
        kline_df = pd.concat(dfs, ignore_index=True)
        kline_df.rename(columns={'open_time': 'timestamp', 'close': 'price'}, inplace=True)
        kline_df['timestamp'] = pd.to_numeric(kline_df['timestamp'])
        if kline_df['timestamp'].max() < 1e11: kline_df['timestamp'] *= 1000
        kline_df.set_index('timestamp', inplace=True)
        
        # Merge
        df = kline_df.join(prem_df, how='inner')
        df = df[~df.index.duplicated(keep='last')]
        df.index = pd.to_datetime(df.index, unit='ms')
        
        # 4h rolling mean for premium (240 mins)
        df['prem_mean'] = df['premium'].rolling(240).mean()
        df['prem_std'] = df['premium'].rolling(240).std()
        df['prem_z'] = (df['premium'] - df['prem_mean']) / df['prem_std']
        
        # Extreme Z-Score > 4.0
        df['signal'] = 0
        df.loc[df['prem_z'] > 4.0, 'signal'] = -1
        df.loc[df['prem_z'] < -4.0, 'signal'] = 1
        
        df['trade_signal'] = df['signal'].shift(1).fillna(0)
        df['signal_changed'] = df['trade_signal'] != df['trade_signal'].shift(1)
        trades = df[(df['trade_signal'] != 0) & (df['signal_changed'])].copy()
        
        if len(trades) == 0: return None
        
        # Vectorized Execution Path (Hold max 60 mins, TP 30 bps, SL 30 bps)
        results = []
        for idx, row in trades.iterrows():
            entry_price = row['price']
            sig = row['trade_signal']
            
            end_time = idx + pd.Timedelta(minutes=60)
            path = df.loc[idx:end_time]
            if len(path) <= 1: continue
            
            path = path.iloc[1:] # Drop entry minute
            
            exit_price = path.iloc[-1]['price']
            
            for t_idx, p_row in path.iterrows():
                high = p_row['high']
                low = p_row['low']
                
                if sig == 1:
                    if ((high - entry_price)/entry_price)*10000 >= 30: # TP 30 bps
                        exit_price = entry_price * 1.003
                        break
                    if ((low - entry_price)/entry_price)*10000 <= -30: # SL 30 bps
                        exit_price = entry_price * 0.997
                        break
                else:
                    if ((entry_price - low)/entry_price)*10000 >= 30:
                        exit_price = entry_price * 0.997
                        break
                    if ((entry_price - high)/entry_price)*10000 <= -30:
                        exit_price = entry_price * 1.003
                        break
                        
            gross = (exit_price - entry_price) / entry_price * sig
            net = gross - (10.0 / 10000) # 10bps round trip
            results.append({'symbol': symbol, 'net_ret': net})
            
        return pd.DataFrame(results)
        
    except Exception as e:
        return None

if __name__ == "__main__":
    print(f"Running HF Premium Reversion (TP/SL 30bps) on {len(COINS)} coins...")
    
    all_trades = []
    with Pool(min(8, os.cpu_count() or 8)) as p:
        for res in p.imap_unordered(analyze_premium_fade_v2, COINS):
            if res is not None:
                all_trades.append(res)
                
    if not all_trades:
        print("No trades found.")
        exit(0)
        
    final_df = pd.concat(all_trades)
    
    wins = (final_df['net_ret'] > 0).sum()
    total = len(final_df)
    win_rate = wins / total * 100
    
    print("\n=== High-Frequency Premium Strategy Results ===")
    print(f"Total Trades: {total}")
    print(f"Win Rate:     {win_rate:.2f}%")
    print(f"Avg Net Ret:  {final_df['net_ret'].mean() * 10000:.2f} bps")
    
    breakdown = final_df.groupby('symbol').agg(
        trades=('net_ret', 'count'),
        win_rate=('net_ret', lambda x: (x > 0).mean() * 100),
        avg_bps=('net_ret', lambda x: x.mean() * 10000)
    ).round(2)
    print("\nBreakdown by Symbol:")
    print(breakdown)
