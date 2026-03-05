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
        
        # Calculate daily sum of funding rate. 
        # Using left alignment so 2025-01-01 holds the sum of funding on that day.
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
    
    # EXACT Forward logic with ZERO Lookahead.
    # At the close of day `i`, we see `fundingRate[i]`.
    # If `fundingRate[i] < -0.005`, we want to execute.
    # We will execute on the OPEN of day `i+1` (which is tomorrow).
    
    # We hold for exactly 3 days. 
    # So we exit on the CLOSE of day `i+3`.
    # We collect the funding rate on day `i+1`, `i+2`, and `i+3`.
    
    df['fwd_ret'] = df['close'].shift(-3) / df['open'].shift(-1) - 1
    df['fwd_funding'] = df['fundingRate'].shift(-1) + df['fundingRate'].shift(-2) + df['fundingRate'].shift(-3)
    
    df['vol_ma'] = df['volume'].rolling(30).mean()
    
    return df.reset_index()[['timestamp', 'symbol', 'close', 'fundingRate', 'fwd_ret', 'fwd_funding', 'volume', 'vol_ma']]

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    
    print(f"Testing True Zero-Lookahead Negative Funding Harvesting (3-Day Hold)...")
    
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
    
    # Filter wild data bugs
    master_df = master_df[(master_df['fwd_ret'] > -0.9) & (master_df['fwd_ret'] < 2.0)]
    
    dates = sorted(master_df['timestamp'].unique())
    # Step by 3 days so we don't have overlapping trades in the portfolio modeling
    trade_dates = dates[::3]
    
    portfolio_pnl = []
    
    for date in trade_dates:
        group = master_df[master_df['timestamp'] == date]
        
        # We need volume for liquidity
        eligible = group[group['volume'] > group['vol_ma']]
        
        # We want to LONG the coins where TODAY'S funding is extremely negative
        longs = eligible[eligible['fundingRate'] < -0.005]
        
        if len(longs) == 0:
            portfolio_pnl.append({'timestamp': date, 'pnl': 0.0, 'trades': 0})
            continue
            
        # We buy at TOMORROW's Open. We get the fwd_ret (Tomorrow's Open to Exit Close)
        # We get the fwd_funding (Sum of next 3 days). Funding is negative, so shorts pay longs.
        price_pnl = longs['fwd_ret'].mean()
        fund_pnl = -1 * longs['fwd_funding'].mean() 
        net_pnl = price_pnl + fund_pnl - (TAKER_FEE * 2)
        
        portfolio_pnl.append({'timestamp': date, 'pnl': net_pnl, 'trades': len(longs)})
        
    df_results = pd.DataFrame(portfolio_pnl).set_index('timestamp')
    
    # Model actual risk. E.g. risk 5% of account per trade event.
    df_results['equity_pnl'] = df_results['pnl'] * 0.05
    df_results['cumulative_equity'] = 1.0 + df_results['equity_pnl'].cumsum()
    
    print("\n" + "="*50)
    print("--- TRUE NEGATIVE FUNDING HARVESTING ---")
    print("="*50)
    active_days = df_results[df_results['trades'] > 0]
    print(f"Total Rebalance Events: {len(active_days)}")
    
    if len(active_days) > 0:
        print(f"Win Rate (Events): {(active_days['pnl'] > 0).mean():.2%}")
        print(f"Avg Net PnL per Event: {active_days['pnl'].mean():.4%}")
        print(f"Total Portfolio Return (5% Risk/Trade): {(df_results['cumulative_equity'].iloc[-1] - 1):.2%}")
        
        monthly_group = df_results.resample('1ME').agg(
            events=('trades', lambda x: (x > 0).sum()),
            win_rate=('pnl', lambda x: (x[x != 0] > 0).mean() if len(x[x != 0]) > 0 else 0),
            avg_event_pnl=('pnl', lambda x: x[x != 0].mean() if len(x[x != 0]) > 0 else 0),
            equity_return=('equity_pnl', 'sum'),
        )
        print("\n--- Monthly Distribution ---")
        print(monthly_group)
    else:
        print("Strategy never triggered.")

if __name__ == "__main__":
    main()
