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
    
    # Filter out impossible 1m flash crashes (data bugs)
    df['ret'] = df['close'].pct_change()
    df = df[(df['ret'] > -0.5) & (df['ret'] < 1.0)]
    
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp')
    
    # Resample to 1D to filter all noise and only look at daily structural prices
    df = df.resample('1d').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df.dropna(inplace=True)
    
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
    
    if df.empty or len(df) < 30: return None
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return None
    _, df = data
    
    df['symbol'] = symbol
    
    # Formation Period: 7 Days
    df['mom_7d'] = df['close'] / df['close'].shift(7) - 1
    
    # Volume Filter: Must be trading above 30-day average volume to ensure liquidity
    df['vol_ma_30d'] = df['volume'].rolling(30).mean()
    
    # Holding Period Return: 3 Days Forward
    df['fwd_ret_3d'] = df['close'].shift(-3) / df['close'] - 1
    
    return df.reset_index()[['timestamp', 'symbol', 'close', 'mom_7d', 'volume', 'vol_ma_30d', 'fwd_ret_3d']]

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    print(f"Testing Cross-Sectional Momentum (Long Top 3 / Short Bottom 3) ...")
    
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
    
    # Filter data bugs out of forward returns (e.g. 5000% moves in 3 days are usually bugs)
    master_df = master_df[(master_df['fwd_ret_3d'] > -0.9) & (master_df['fwd_ret_3d'] < 2.0)]
    
    # We step every 3 days to simulate a 3-day holding period without overlapping trades
    dates = sorted(master_df['timestamp'].unique())
    trade_dates = dates[::3] 
    
    portfolio_pnl = []
    
    for date in trade_dates:
        group = master_df[master_df['timestamp'] == date]
        
        # Filter for liquidity
        eligible = group[group['volume'] > group['vol_ma_30d']]
        
        if len(eligible) < 20: 
            portfolio_pnl.append({'timestamp': date, 'pnl': 0.0, 'trades': 0})
            continue
            
        # Sort by Momentum
        ranked = eligible.sort_values('mom_7d', ascending=False)
        
        top_3 = ranked.head(3)
        bottom_3 = ranked.tail(3)
        
        # Calculate Returns
        # Longs: Fwd Ret - (2 * Taker Fee)
        long_pnl = top_3['fwd_ret_3d'].mean() - (TAKER_FEE * 2)
        
        # Shorts: (-1 * Fwd Ret) - (2 * Taker Fee)
        short_pnl = (-1 * bottom_3['fwd_ret_3d'].mean()) - (TAKER_FEE * 2)
        
        # Total Event PnL (50% Long, 50% Short allocation)
        net_pnl = (long_pnl + short_pnl) / 2
        
        portfolio_pnl.append({'timestamp': date, 'pnl': net_pnl, 'trades': 6})
        
    df_results = pd.DataFrame(portfolio_pnl).set_index('timestamp')
    df_results['cumulative'] = df_results['pnl'].cumsum()
    
    print("\n--- Strategy Results (Market Neutral Momentum) ---")
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
