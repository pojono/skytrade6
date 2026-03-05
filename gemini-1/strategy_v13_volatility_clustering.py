import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
MAKER_FEE = 0.0004
TAKER_FEE = 0.0010

# Hypothesis: Volatility clusters. High volatility today predicts high volatility tomorrow.
# The previous funding arbitrage strategy showed positive PnL in the pure long/short setup but failed when mixed.
# Let's pivot entirely to a pure Relative Momentum strategy on the Daily timeframe. 
# Buy the strongest 5 coins every day, hold for 24 hours, rotate.

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
    
    # 1D candles, align to midnight UTC
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
    df['ret_1d'] = df['close'] / df['close'].shift(1) - 1
    df['ret_7d'] = df['close'] / df['close'].shift(7) - 1
    df['vol_ma'] = df['volume'].rolling(7).mean()
    
    # Fwd return (from today's close to tomorrow's close)
    df['fwd_ret_1d'] = df['close'].shift(-1) / df['close'] - 1
    
    return df.reset_index()[['timestamp', 'symbol', 'close', 'ret_1d', 'ret_7d', 'volume', 'vol_ma', 'fwd_ret_1d']]

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    print(f"Testing Daily Cross-Sectional Momentum Rotation...")
    
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
    
    # Strategy: Every day at UTC midnight, rank coins by 7D return.
    # Keep only coins with Volume > 7D Avg Volume (active momentum)
    # Buy the Top 5. Hold for 1 day. (Deduct 0.20% round trip Taker fees for daily rebalancing)
    
    daily_returns = []
    
    for date, group in master_df.groupby('timestamp'):
        # Filter: Volume spike
        eligible = group[group['volume'] > group['vol_ma']]
        if len(eligible) < 5:
            daily_returns.append({'timestamp': date, 'pnl': 0.0})
            continue
            
        # Top 5 by 7-day momentum
        top_5 = eligible.sort_values('ret_7d', ascending=False).head(5)
        
        # Calculate return (equally weighted, minus 0.20% Taker fee)
        gross_ret = top_5['fwd_ret_1d'].mean()
        net_ret = gross_ret - (TAKER_FEE * 2)
        
        daily_returns.append({'timestamp': date, 'pnl': net_ret})
        
    df_results = pd.DataFrame(daily_returns).set_index('timestamp')
    df_results['cumulative'] = df_results['pnl'].cumsum()
    
    print("\n--- Strategy Results ---")
    print(f"Total Days Traded: {len(df_results)}")
    print(f"Win Rate (Days): {(df_results['pnl'] > 0).mean():.2%}")
    print(f"Avg Daily PnL: {df_results['pnl'].mean():.4%}")
    print(f"Total Cumulative PnL (1x Lev): {df_results['cumulative'].iloc[-1]:.4%}")
    
    monthly_group = df_results.resample('1ME').agg(
        days=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).mean()),
        total_pnl=('pnl', 'sum'),
    )
    
    print("\n--- Monthly Distribution ---")
    print(monthly_group)

if __name__ == "__main__":
    main()
