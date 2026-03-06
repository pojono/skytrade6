import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def load_kline_returns(symbol, exchange='binance', start_date="2025-01-01"):
    try:
        kline_files = sorted(list((DATALAKE / f"{exchange}/{symbol}").glob("*_kline_1m.csv")))
        kline_files = [f for f in kline_files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name and f.name >= start_date]
        
        if not kline_files: return None
        
        dfs = []
        time_col = 'startTime' if exchange == 'bybit' else 'open_time'
        for f in kline_files:
            try:
                df = pd.read_csv(f, usecols=[time_col, 'close', 'volume'], engine='c')
                dfs.append(df)
            except: pass
            
        if not dfs: return None
        
        df = pd.concat(dfs, ignore_index=True)
        df.rename(columns={time_col: 'timestamp'}, inplace=True)
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        if df['timestamp'].max() < 1e11: df['timestamp'] *= 1000
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='last')].sort_index()
        
        # Calculate 1-min return
        df[f'{symbol}_ret'] = df['close'].pct_change()
        return df[[f'{symbol}_ret']]
    except:
        return None

def analyze_cross_asset():
    print("Loading BTC data...")
    btc_df = load_kline_returns('BTCUSDT', 'binance')
    if btc_df is None: return
    
    # Identify massive 1-min BTC moves (e.g. > 0.3% in a single minute)
    btc_df['btc_move'] = np.where(btc_df['BTCUSDT_ret'] > 0.003, 1, 
                         np.where(btc_df['BTCUSDT_ret'] < -0.003, -1, 0))
                         
    spike_idx = btc_df[btc_df['btc_move'] != 0].index
    print(f"Found {len(spike_idx)} massive 1-min BTC moves.")
    
    alts = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'WLDUSDT', 'SUIUSDT', 'PEPEUSDT']
    
    results = []
    for alt in alts:
        print(f"Analyzing {alt} lag...")
        alt_df = load_kline_returns(alt, 'binance')
        if alt_df is None: continue
        
        # Merge on timestamp
        merged = btc_df.join(alt_df, how='inner')
        
        # We want to see how the Altcoin returns in minute T+1, T+2, T+3
        # based on BTC's massive move at minute T.
        
        for lag in [1, 2, 3, 5]:
            merged[f'alt_fwd_{lag}m'] = merged[f'{alt}_ret'].shift(-lag)
            
        spikes = merged[merged['btc_move'] != 0]
        
        # Buy altcoin if BTC spiked up, Short if BTC spiked down
        for lag in [1, 2, 3, 5]:
            spikes[f'trade_ret_{lag}m'] = spikes[f'alt_fwd_{lag}m'] * spikes['btc_move']
            
            avg_bps = spikes[f'trade_ret_{lag}m'].mean() * 10000
            win_rate = (spikes[f'trade_ret_{lag}m'] > 0).mean()
            
            results.append({
                'altcoin': alt,
                'lag_minutes': lag,
                'events': len(spikes),
                'avg_bps': avg_bps,
                'win_rate': win_rate
            })
            
    df_res = pd.DataFrame(results)
    print("\n--- Cross-Asset Lead-Lag (Trade Alt at T+1 after massive BTC move at T) ---")
    print(df_res.pivot(index='altcoin', columns='lag_minutes', values='avg_bps').round(2))

if __name__ == "__main__":
    analyze_cross_asset()
