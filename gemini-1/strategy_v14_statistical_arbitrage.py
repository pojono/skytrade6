import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
MAKER_FEE = 0.0004
TAKER_FEE = 0.0010

def load_btc():
    files = glob.glob(f"{DATALAKE_DIR}/BTCUSDT/*_kline_1m.csv")
    if not files: return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['timestamp'] = pd.to_datetime(df.get('startTime', df.get('timestamp', None)), unit='ms')
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).set_index('timestamp')
    df = df.resample('1h').agg({'close': 'last'})
    df.dropna(inplace=True)
    df['btc_ret'] = df['close'].pct_change()
    return df[['btc_ret']]

BTC_REF = load_btc()

def load_symbol_data(symbol):
    if symbol == 'BTCUSDT': return None
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
    
    if BTC_REF is not None:
        df = df.join(BTC_REF, how='inner')
        
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
    df['ret_1h'] = df['close'].pct_change()
    
    # Simple relative momentum instead of complex beta to speed up and stabilize computation
    # We measure how much the coin outperformed/underperformed BTC over the last 24h
    df['ret_24h'] = df['close'] / df['close'].shift(24) - 1
    
    # BTC 24h return
    # Need to rebuild btc_ret_24h manually since we only brought in 1h
    df['btc_ret_24h'] = (1 + df['btc_ret']).rolling(24).apply(np.prod, raw=True) - 1
    
    # Relative outperformance
    df['rel_perf_24h'] = df['ret_24h'] - df['btc_ret_24h']
    
    # Volume filter
    df['vol_ma'] = df['volume'].rolling(24 * 7).mean()
    
    # Fwd return (from today's close to tomorrow's close)
    df['fwd_ret_24h'] = df['close'].shift(-24) / df['close'] - 1
    
    return df.reset_index()[['timestamp', 'symbol', 'rel_perf_24h', 'volume', 'vol_ma', 'fwd_ret_24h']]

def main():
    if BTC_REF is None:
        print("Need BTC data.")
        return
        
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    print(f"Testing Daily Cross-Sectional Statistical Arbitrage (Relative Momentum vs BTC)...")
    
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
    
    # Strategy: 
    # Every day at midnight UTC, buy the 5 coins that UNDERPERFORMED BTC the most (Mean Reversion)
    # AND have high liquidity (volume > 7d ma)
    
    # Filter to midnight only
    master_df['hour'] = master_df['timestamp'].dt.hour
    daily_df = master_df[master_df['hour'] == 0].copy()
    
    daily_returns = []
    
    for date, group in daily_df.groupby('timestamp'):
        # Filter: Volume spike to ensure liquidity
        eligible = group[group['volume'] > group['vol_ma']]
        if len(eligible) < 5:
            daily_returns.append({'timestamp': date, 'pnl': 0.0})
            continue
            
        # Top 5 most oversold vs BTC (Mean Reversion)
        top_5 = eligible.sort_values('rel_perf_24h', ascending=True).head(5)
        
        # Calculate return (equally weighted, minus 0.20% Taker fee)
        gross_ret = top_5['fwd_ret_24h'].mean()
        net_ret = gross_ret - (TAKER_FEE * 2)
        
        daily_returns.append({'timestamp': date, 'pnl': net_ret})
        
    df_results = pd.DataFrame(daily_returns).set_index('timestamp')
    df_results['cumulative'] = df_results['pnl'].cumsum()
    
    print("\n--- Strategy Results (Mean Reversion against BTC) ---")
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
