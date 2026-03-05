import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm
import time

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
MAKER_FEE = 0.0004
TAKER_FEE = 0.0010

def load_universe():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    # Take top 30 by volume to ensure liquidity (simplified list based on previous runs)
    top_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'ATOMUSDT', 'NEARUSDT', 'APTUSDT', 'OPUSDT', 'ARBUSDT', 'SUIUSDT', 'INJUSDT', 'RNDRUSDT']
    
    valid_symbols = [s for s in top_symbols if s in symbols]
    return valid_symbols

def load_symbol_data(symbol):
    files = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_kline_1m.csv")
    if not files: return None
        
    df_list = []
    for f in files:
        try: df_list.append(pd.read_csv(f))
        except: pass
    if not df_list: return None
        
    df = pd.concat(df_list, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df.get('startTime', df.get('timestamp', None)), unit='ms')
         
    df = df[(df['close'] > 0) & (df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0)]
    df = df[df['high'] >= df['low']]
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['ret'] = df['close'].pct_change()
    df = df[(df['ret'] > -0.5) & (df['ret'] < 1.0)]
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp')
    
    # 1H candles
    df = df.resample('1h').agg({'close': 'last'})
    df.dropna(inplace=True)
    
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
    
    return df[['close']].rename(columns={'close': symbol})

def main():
    symbols = load_universe()
    print(f"Step 1/4: Loading data for {len(symbols)} liquid coins in parallel...")
    
    dfs = []
    # Force process count to avoid hanging and show progress properly
    pool = multiprocessing.Pool(processes=min(8, multiprocessing.cpu_count() - 1))
    
    for df in tqdm(pool.imap_unordered(load_symbol_data, symbols), total=len(symbols), desc="Loading Data"):
        if df is not None:
            dfs.append(df)
            
    pool.close()
    pool.join()
                
    if not dfs:
        print("No data.")
        return
        
    print(f"\nStep 2/4: Merging {len(dfs)} datasets into a single matrix...")
    start_merge = time.time()
    # Much faster to concat all at once than join in a loop
    master_df = pd.concat(dfs, axis=1)
    master_df.ffill(inplace=True)
    master_df.dropna(inplace=True)
    print(f"Matrix built in {time.time() - start_merge:.2f}s. Shape: {master_df.shape}")
    
    if 'BTCUSDT' not in master_df.columns or 'ETHUSDT' not in master_df.columns:
        print("Need BTC and ETH. Exiting.")
        return
        
    print("\nStep 3/4: Calculating statistical pairs indicators (Z-Scores)...")
    # Calculate Spread (Ratio)
    df = master_df[['BTCUSDT', 'ETHUSDT']].copy()
    df['ratio'] = df['ETHUSDT'] / df['BTCUSDT']
    
    # Rolling stats for Z-Score (using 7 days = 168 hours)
    df['ratio_mean'] = df['ratio'].rolling(168).mean()
    df['ratio_std'] = df['ratio'].rolling(168).std()
    df['z_score'] = (df['ratio'] - df['ratio_mean']) / df['ratio_std']
    
    # Forward Returns (Next 24H)
    df['btc_fwd'] = df['BTCUSDT'].shift(-24) / df['BTCUSDT'] - 1
    df['eth_fwd'] = df['ETHUSDT'].shift(-24) / df['ETHUSDT'] - 1
    
    print("\nStep 4/4: Simulating Pairs Trading Strategy...")
    trades = []
    
    # Walk forward, step every 24 hours
    times = sorted(df.index.unique())
    trade_steps = list(range(168, len(times) - 24, 24))
    
    for i in tqdm(trade_steps, desc="Backtesting"):
        t = times[i]
        row = df.loc[t]
        
        z = row['z_score']
        if pd.isna(z): continue
            
        if z > 2.0:
            # ETH is overvalued relative to BTC. Short ETH, Long BTC
            # Portfolio Return = (0.5 * BTC_Fwd) + (0.5 * -ETH_Fwd)
            gross_pnl = (0.5 * row['btc_fwd']) + (0.5 * -row['eth_fwd'])
            net_pnl = gross_pnl - (TAKER_FEE * 2) # Two Taker legs
            trades.append({'timestamp': t, 'pnl': net_pnl, 'type': 'short_eth_long_btc', 'z': z})
            
        elif z < -2.0:
            # ETH is undervalued relative to BTC. Long ETH, Short BTC
            gross_pnl = (0.5 * row['eth_fwd']) + (0.5 * -row['btc_fwd'])
            net_pnl = gross_pnl - (TAKER_FEE * 2)
            trades.append({'timestamp': t, 'pnl': net_pnl, 'type': 'long_eth_short_btc', 'z': z})
            
    if not trades:
        print("No trades found.")
        return
        
    df_results = pd.DataFrame(trades).set_index('timestamp')
    df_results['cumulative'] = df_results['pnl'].cumsum()
    
    print("\n" + "="*50)
    print("--- Strategy Results (BTC/ETH Pairs Trading) ---")
    print("="*50)
    print(f"Total Trades: {len(df_results)}")
    print(f"Win Rate: {(df_results['pnl'] > 0).mean():.2%}")
    print(f"Avg Net PnL per Trade: {df_results['pnl'].mean():.4%}")
    print(f"Total Cumulative PnL (1x Lev): {df_results['cumulative'].iloc[-1]:.4%}")
    
    monthly_group = df_results.resample('1ME').agg(
        trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).mean() if len(x)>0 else 0),
        total_pnl=('pnl', 'sum'),
    )
    
    print("\n--- Monthly Distribution ---")
    print(monthly_group)

if __name__ == "__main__":
    main()
