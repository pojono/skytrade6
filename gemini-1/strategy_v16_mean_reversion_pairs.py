import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
MAKER_FEE = 0.0004
TAKER_FEE = 0.0010

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
    df = df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df.dropna(inplace=True)
    
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
    
    if df.empty or len(df) < (24 * 7): return None
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return None
    _, df = data
    
    df['symbol'] = symbol
    
    # Formation: 24 Hour Return Z-Score
    df['ret_24h'] = df['close'] / df['close'].shift(24) - 1
    
    # Volume filter
    df['vol_ma'] = df['volume'].rolling(24 * 7).mean()
    
    # Holding Period: 24 Hours Forward
    df['fwd_ret_24h'] = df['close'].shift(-24) / df['close'] - 1
    
    return df.reset_index()[['timestamp', 'symbol', 'close', 'ret_24h', 'volume', 'vol_ma', 'fwd_ret_24h']]

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    print(f"Testing 24H Cross-Sectional Mean Reversion (Buy worst 5, Short best 5)...")
    
    all_dfs = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        for df in tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols)):
            if df is not None:
                all_dfs.append(df)
                
    if not all_dfs:
        print("No data.")
        return
        
    master_df = pd.concat(all_dfs, ignore_index=True)
    master_df.dropna(inplace=True)
    master_df = master_df[(master_df['fwd_ret_24h'] > -0.9) & (master_df['fwd_ret_24h'] < 2.0)]
    
    # Step every 24 hours
    master_df['hour'] = master_df['timestamp'].dt.hour
    daily_df = master_df[master_df['hour'] == 0].copy()
    
    portfolio_pnl = []
    
    for date, group in daily_df.groupby('timestamp'):
        # Filter: Volume spike to ensure liquidity
        eligible = group[group['volume'] > group['vol_ma']]
        
        # Calculate Z-score cross-sectionally for the day
        if len(eligible) < 20: 
            portfolio_pnl.append({'timestamp': date, 'pnl': 0.0, 'trades': 0})
            continue
            
        eligible['ret_z'] = (eligible['ret_24h'] - eligible['ret_24h'].mean()) / eligible['ret_24h'].std()
        
        # Mean Reversion: 
        # Long the ones that dumped the most (Z < -1.5)
        # Short the ones that pumped the most (Z > 1.5)
        
        longs = eligible[eligible['ret_z'] < -1.5].head(5)
        shorts = eligible[eligible['ret_z'] > 1.5].head(5)
        
        if len(longs) == 0 and len(shorts) == 0:
            portfolio_pnl.append({'timestamp': date, 'pnl': 0.0, 'trades': 0})
            continue
            
        long_ret = longs['fwd_ret_24h'].mean() - (TAKER_FEE * 2) if len(longs) > 0 else 0
        short_ret = (-1 * shorts['fwd_ret_24h'].mean()) - (TAKER_FEE * 2) if len(shorts) > 0 else 0
        
        # Equal weight
        net_ret = (long_ret + short_ret) / 2
        portfolio_pnl.append({'timestamp': date, 'pnl': net_ret, 'trades': len(longs) + len(shorts)})
        
    df_results = pd.DataFrame(portfolio_pnl).set_index('timestamp')
    df_results['cumulative'] = df_results['pnl'].cumsum()
    
    print("\n--- Strategy Results (24H Mean Reversion) ---")
    active_days = df_results[df_results['trades'] > 0]
    print(f"Total Rebalance Events: {len(active_days)}")
    print(f"Win Rate (Events): {(active_days['pnl'] > 0).mean():.2%}")
    print(f"Avg PnL per Event: {active_days['pnl'].mean():.4%}")
    print(f"Total Cumulative PnL (1x Lev): {df_results['cumulative'].iloc[-1]:.4%}")
    
    monthly_group = df_results.resample('1ME').agg(
        events=('trades', lambda x: (x > 0).sum()),
        win_rate=('pnl', lambda x: (x[x != 0] > 0).mean() if len(x[x != 0]) > 0 else 0),
        total_pnl=('pnl', 'sum'),
    )
    
    print("\n--- Monthly Distribution ---")
    print(monthly_group)

if __name__ == "__main__":
    main()
