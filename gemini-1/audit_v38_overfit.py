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
        
        df_fund = df_fund.resample('1d').sum()
        df = df.join(df_fund[['fundingRate']], how='left').fillna(0)
    else: return None
    
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
    
    if df.empty or len(df) < 30: return None
    return symbol, df

def process_symbol(args):
    symbol, params = args
    data = load_symbol_data(symbol)
    if data is None: return None
    _, df = data
    
    df['symbol'] = symbol
    hold_days = params['hold_days']
    
    # EXACT FORWARD LOGIC
    df['fwd_ret'] = df['close'].shift(-hold_days) / df['close'] - 1
    
    # We collect the funding starting from tomorrow (index i+1) up to the exit day (index i+hold_days)
    df['fwd_funding'] = df['fundingRate'].rolling(hold_days).sum().shift(-hold_days)
    
    df['vol_ma'] = df['volume'].rolling(30).mean()
    
    return df.reset_index()[['timestamp', 'symbol', 'close', 'fundingRate', 'fwd_ret', 'fwd_funding', 'volume', 'vol_ma']]

def run_scenario(symbols, hold_days, funding_threshold):
    params = {'hold_days': hold_days}
    tasks = [(s, params) for s in symbols]
    
    all_dfs = []
    with multiprocessing.Pool(processes=min(8, multiprocessing.cpu_count())) as pool:
        for df in pool.imap_unordered(process_symbol, tasks):
            if df is not None:
                all_dfs.append(df)
                
    if not all_dfs: return 0, 0, 0
    
    master_df = pd.concat(all_dfs, ignore_index=True)
    master_df.dropna(inplace=True)
    master_df = master_df[(master_df['fwd_ret'] > -0.9) & (master_df['fwd_ret'] < 2.0)]
    
    dates = sorted(master_df['timestamp'].unique())
    # Instead of step size, we can just log every trade and look at average edge.
    # To be perfectly realistic about portfolio, we assume 1 trade per symbol per day.
    
    all_pnls = []
    
    for date, group in master_df.groupby('timestamp'):
        eligible = group[group['volume'] > group['vol_ma']]
        longs = eligible[eligible['fundingRate'] < funding_threshold]
        
        for _, row in longs.iterrows():
            price_pnl = row['fwd_ret']
            fund_pnl = -1 * row['fwd_funding']
            net_pnl = price_pnl + fund_pnl - (TAKER_FEE * 2)
            all_pnls.append(net_pnl)
            
    if len(all_pnls) == 0: return 0, 0, 0
    
    win_rate = sum(1 for x in all_pnls if x > 0) / len(all_pnls)
    avg_pnl = np.mean(all_pnls)
    return len(all_pnls), win_rate, avg_pnl

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    # Subset to save time in matrix
    test_symbols = symbols[:60]
    
    print("Running Parameter Sensitivity Matrix (No Lookahead)...")
    
    scenarios = [
        {'hold': 3, 'thresh': -0.005, 'name': '3-Day, -0.5%'},
        {'hold': 5, 'thresh': -0.005, 'name': '5-Day, -0.5% (Base)'},
        {'hold': 7, 'thresh': -0.005, 'name': '7-Day, -0.5%'},
        
        {'hold': 5, 'thresh': -0.003, 'name': '5-Day, -0.3% (Looser)'},
        {'hold': 5, 'thresh': -0.010, 'name': '5-Day, -1.0% (Extreme)'},
    ]
    
    print(f"{'Scenario':<25} | {'Trades':<8} | {'Win Rate':<10} | {'Avg PnL':<10}")
    print("-" * 65)
    
    for s in scenarios:
        trades, wr, pnl = run_scenario(test_symbols, s['hold'], s['thresh'])
        print(f"{s['name']:<25} | {trades:<8} | {wr:<10.2%} | {pnl:<10.2%}")

if __name__ == "__main__":
    main()
