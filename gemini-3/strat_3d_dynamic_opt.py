import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings
from backtest_core_tp_sl import run_backtest_tp_sl
from strat_3_powder_keg import load_data, generate_signals

warnings.filterwarnings('ignore')

def run_strat_3_dynamic_opt(symbol):
    df = load_data(symbol)
    if df is None or len(df) < 500: return None
    
    kline_files = sorted(list(Path("/home/ubuntu/Projects/skytrade6/datalake/binance").glob(f"{symbol}/*_kline_1m.csv")))
    kline_files = [f for f in kline_files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name and f.name >= "2025-01-01"]
    
    dfs = []
    for f in kline_files:
        try: dfs.append(pd.read_csv(f, usecols=['open_time', 'high', 'low'], engine='c'))
        except: pass
        
    if not dfs: return None
    hl_df = pd.concat(dfs, ignore_index=True)
    hl_df.rename(columns={'open_time': 'timestamp'}, inplace=True)
    hl_df['timestamp'] = pd.to_numeric(hl_df['timestamp'])
    if hl_df['timestamp'].max() < 1e11: hl_df['timestamp'] *= 1000
    hl_df.set_index('timestamp', inplace=True)
    hl_df.index = pd.to_datetime(hl_df.index, unit='ms')
    
    df = generate_signals(df)
    
    df = df.join(hl_df.resample('1h').agg({'high': 'max', 'low': 'min'}), how='left')
    df['high'] = df['high'].fillna(df['close'])
    df['low'] = df['low'].fillna(df['close'])
    
    # Try a tighter stop (2%) with moderate TP (6%) to ensure strict risk management
    results = run_backtest_tp_sl(df, tp_pct=0.06, sl_pct=0.02, fee_bps=5.0)
    
    if results['events'] > 0:
        results['symbol'] = symbol
        return results
    return None

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'SUIUSDT', 'WLDUSDT', 'PEPEUSDT', 'DYDXUSDT', 'ENAUSDT']
    print("--- Backtesting Strategy 3: Powder Keg with Strict Risk (6% TP / 2% SL) ---")
    
    res_list = []
    with Pool(min(4, os.cpu_count() or 4)) as p:
        for r in p.imap_unordered(run_strat_3_dynamic_opt, symbols):
            if r: res_list.append(r)
            
    if res_list:
        res_df = pd.DataFrame(res_list)
        cols = ['symbol', 'events', 'win_rate', 'total_net_ret_%', 'avg_net_ret_bps', 'max_drawdown_%', 'sharpe']
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        print(res_df[cols].to_string(index=False))
        
        print("\nAggregate Portfolio Performance:")
        print(f"Total Trades: {res_df['events'].sum()}")
        print(f"Avg Win Rate: {res_df['win_rate'].mean():.2%}")
        print(f"Total Net Return: {res_df['total_net_ret_%'].sum():.2f}%")
