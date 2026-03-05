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
    
    # We found that purely going long during capitulation isn't consistent enough.
    # What if we just harvest the extreme negative funding rates (shorts paying longs)?
    # When the market crashes, funding goes severely negative. 
    # We buy the asset to collect the funding, and we hold for the mean reversion bounce.
    
    df['fwd_ret_5d'] = df['close'].shift(-5) / df['close'] - 1
    
    # 5 days of funding collection
    df['fwd_funding_5d'] = df['fundingRate'].shift(-1).rolling(5).sum()
    
    # Require decent volume
    df['vol_ma'] = df['volume'].rolling(30).mean()
    
    return df.reset_index()[['timestamp', 'symbol', 'close', 'fundingRate', 'fwd_ret_5d', 'fwd_funding_5d', 'volume', 'vol_ma']]

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    print(f"Testing Pure Negative Funding Harvesting...")
    
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
    master_df = master_df[(master_df['fwd_ret_5d'] > -0.9) & (master_df['fwd_ret_5d'] < 2.0)]
    
    dates = sorted(master_df['timestamp'].unique())
    trade_dates = dates[::5] # Step 5 days to avoid overlap
    
    portfolio_pnl = []
    
    for date in trade_dates:
        group = master_df[master_df['timestamp'] == date]
        
        # Filter for liquidity
        eligible = group[group['volume'] > group['vol_ma']]
        
        # We want to LONG the coins where funding is extremely negative
        # Meaning: Shorts are paying Longs a massive premium.
        # Threshold: < -0.005 (-0.5% per day)
        
        longs = eligible[eligible['fundingRate'] < -0.005]
        
        if len(longs) == 0:
            portfolio_pnl.append({'timestamp': date, 'pnl': 0.0, 'trades': 0})
            continue
            
        # We buy them, hold 5 days, get the price action + the funding
        # Because it's negative, shorts pay longs, so we subtract to make it a positive PnL
        price_pnl = longs['fwd_ret_5d'].mean()
        fund_pnl = -1 * longs['fwd_funding_5d'].mean() 
        net_pnl = price_pnl + fund_pnl - (TAKER_FEE * 2)
        
        portfolio_pnl.append({'timestamp': date, 'pnl': net_pnl, 'trades': len(longs)})
        
    df_results = pd.DataFrame(portfolio_pnl).set_index('timestamp')
    df_results['cumulative'] = df_results['pnl'].cumsum()
    
    print("\n" + "="*50)
    print("--- NEGATIVE FUNDING HARVESTING RESULTS ---")
    print("="*50)
    active_days = df_results[df_results['trades'] > 0]
    print(f"Total Rebalance Events: {len(active_days)}")
    
    if len(active_days) > 0:
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
    else:
        print("Strategy never triggered.")

if __name__ == "__main__":
    main()
