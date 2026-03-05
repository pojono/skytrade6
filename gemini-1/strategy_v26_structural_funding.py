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

def load_symbol_data(symbol):
    files = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_kline_1m.csv")
    files_funding = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_funding_rate.csv")
    
    if not files or not files_funding: return None
        
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
    
    df = df.resample('1d').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df.dropna(inplace=True)
    
    fund_list = []
    for f in files_funding:
        try: df_f = pd.read_csv(f); fund_list.append(df_f)
        except: pass
        
    if fund_list:
        df_fund = pd.concat(fund_list, ignore_index=True)
        if 'fundingRateTimestamp' in df_fund.columns:
            df_fund['timestamp'] = pd.to_datetime(df_fund['fundingRateTimestamp'], unit='ms')
        elif 'timestamp' in df_fund.columns and df_fund['timestamp'].dtype != 'datetime64[ns]':
            df_fund['timestamp'] = pd.to_datetime(df_fund['timestamp'], unit='ms')
        df_fund = df_fund.drop_duplicates(subset=['timestamp']).sort_values('timestamp').set_index('timestamp')
        
        # Calculate daily sum of funding rate
        df_fund = df_fund.resample('1d').sum()
        df = df.join(df_fund[['fundingRate']], how='left').fillna(0)
    else: return None
    
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
    
    # Structural Funding Strategy
    # If daily funding > 0.003 (30 bps a day), it costs > 100% APR to hold long.
    # We want to SHORT these coins.
    # If daily funding < -0.003, we want to LONG these coins.
    # We hold for exactly 3 days and collect the funding + mean reversion price action.
    
    df['fwd_ret_3d'] = df['close'].shift(-3) / df['close'] - 1
    
    # We also collect 3 days of funding
    df['fwd_funding_3d'] = df['fundingRate'].shift(-1) + df['fundingRate'].shift(-2) + df['fundingRate'].shift(-3)
    
    return df.reset_index()[['timestamp', 'symbol', 'close', 'fundingRate', 'fwd_ret_3d', 'fwd_funding_3d']]

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    print(f"Testing Structural Funding Carry & Mean Reversion...")
    
    all_dfs = []
    pool = multiprocessing.Pool(processes=min(8, multiprocessing.cpu_count()))
    
    for df in tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols), desc="Processing"):
        if df is not None:
            all_dfs.append(df)
            
    pool.close()
    pool.join()
                
    if not all_dfs:
        print("No data.")
        return
        
    master_df = pd.concat(all_dfs, ignore_index=True)
    master_df.dropna(inplace=True)
    master_df = master_df[(master_df['fwd_ret_3d'] > -0.9) & (master_df['fwd_ret_3d'] < 2.0)]
    
    dates = sorted(master_df['timestamp'].unique())
    trade_dates = dates[::3]
    
    portfolio_pnl = []
    
    for date in trade_dates:
        group = master_df[master_df['timestamp'] == date]
        
        # Shorts: Coins where long funding is very expensive (> 30 bps per day)
        shorts = group[group['fundingRate'] > 0.003]
        
        # Longs: Coins where short funding is very expensive (< -30 bps per day)
        longs = group[group['fundingRate'] < -0.003]
        
        total_trades = len(shorts) + len(longs)
        if total_trades == 0:
            portfolio_pnl.append({'timestamp': date, 'pnl': 0.0, 'trades': 0})
            continue
            
        short_pnl = 0
        if len(shorts) > 0:
            # We short the asset. We gain when price falls (-1 * fwd_ret). 
            # We RECEIVE the funding rate (+fwd_funding_3d).
            # We pay 2 Taker fees.
            price_pnl = -1 * shorts['fwd_ret_3d'].mean()
            fund_pnl = shorts['fwd_funding_3d'].mean()
            short_pnl = price_pnl + fund_pnl - (TAKER_FEE * 2)
            
        long_pnl = 0
        if len(longs) > 0:
            # We long the asset. We gain when price rises.
            # We RECEIVE the funding rate because funding is negative (shorts pay longs).
            # The API gives negative funding when shorts pay longs, so we subtract it to make it a gain.
            price_pnl = longs['fwd_ret_3d'].mean()
            fund_pnl = -1 * longs['fwd_funding_3d'].mean()
            long_pnl = price_pnl + fund_pnl - (TAKER_FEE * 2)
            
        if len(shorts) > 0 and len(longs) > 0:
            net_pnl = (short_pnl + long_pnl) / 2
        elif len(shorts) > 0:
            net_pnl = short_pnl
        else:
            net_pnl = long_pnl
            
        portfolio_pnl.append({'timestamp': date, 'pnl': net_pnl, 'trades': total_trades})
        
    df_results = pd.DataFrame(portfolio_pnl).set_index('timestamp')
    df_results['cumulative'] = df_results['pnl'].cumsum()
    
    print("\n" + "="*50)
    print("--- Strategy Results (Structural Funding Carry) ---")
    print("="*50)
    active_days = df_results[df_results['trades'] > 0]
    print(f"Total Rebalance Events: {len(active_days)}")
    print(f"Win Rate (Events): {(active_days['pnl'] > 0).mean():.2%}")
    print(f"Avg PnL per Event: {active_days['pnl'].mean():.4%}")
    print(f"Total Cumulative PnL: {df_results['cumulative'].iloc[-1]:.4%}")
    
    monthly_group = df_results.resample('1ME').agg(
        events=('trades', lambda x: (x > 0).sum()),
        win_rate=('pnl', lambda x: (x[x != 0] > 0).mean() if len(x[x != 0]) > 0 else 0),
        total_pnl=('pnl', 'sum'),
    )
    
    print("\n--- Monthly Distribution ---")
    print(monthly_group)

if __name__ == "__main__":
    main()
