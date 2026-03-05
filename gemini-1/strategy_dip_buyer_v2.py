import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
import time
from tqdm import tqdm

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"

# Refined Strategy Parameters for deeper drops
DROP_PCT_15M = -0.06
DROP_PCT_60M = -0.08
TP_PCT = 0.03
SL_PCT = -0.05
TIME_STOP_MINS = 180
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
    if 'startTime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['startTime'], unit='ms')
    elif 'timestamp' in df.columns and df['timestamp'].dtype != 'datetime64[ns]':
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
         
    df = df[df['close'] > 0]
    df = df[df['high'] >= df['low']]
    df = df.drop_duplicates(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.set_index('timestamp', inplace=True)
    
    df = df.resample('1min').ffill(limit=5)
    df.dropna(inplace=True)
    
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
        
    if df.empty or len(df) < 60:
        return None
        
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    return backtest_dip_buyer(df, symbol)

def backtest_dip_buyer(df, symbol):
    df['ret_15m'] = df['close'] / df['close'].shift(15) - 1
    df['ret_60m'] = df['close'] / df['close'].shift(60) - 1
    df['ret_1m'] = df['close'].pct_change()
    
    # We want sharp, recent panic. Filter data bugs (<-50% in 1 min).
    raw_signal = (df['ret_15m'] < DROP_PCT_15M) & (df['ret_60m'] < DROP_PCT_60M) & (df['ret_1m'] > -0.5)
    
    timestamps = df.index.values
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    signals = raw_signal.values
    
    trades = []
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    cooldown_until = 0
    n = len(df)
    
    for i in range(60, n - 1):
        if not in_position:
            if signals[i] and i > cooldown_until:
                in_position = True
                entry_price = closes[i]
                entry_time = timestamps[i]
                entry_idx = i
        else:
            target_price = entry_price * (1 + TP_PCT)
            stop_price = entry_price * (1 + SL_PCT)
            
            if lows[i] <= stop_price:
                exit_price = stop_price
                net_pnl = ((exit_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append({'symbol': symbol, 'pnl': net_pnl, 'type': 'sl', 'hold_mins': i - entry_idx})
                in_position = False
                cooldown_until = i + 120
                
            elif highs[i] >= target_price:
                exit_price = target_price
                net_pnl = ((exit_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE
                trades.append({'symbol': symbol, 'pnl': net_pnl, 'type': 'tp', 'hold_mins': i - entry_idx})
                in_position = False
                cooldown_until = i + 120
                
            elif (i - entry_idx) >= TIME_STOP_MINS:
                exit_price = closes[i]
                net_pnl = ((exit_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append({'symbol': symbol, 'pnl': net_pnl, 'type': 'time', 'hold_mins': i - entry_idx})
                in_position = False
                cooldown_until = i + 120

    return trades

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    print(f"Testing more extreme drops: 15m < {DROP_PCT_15M*100}%, 60m < {DROP_PCT_60M*100}%")
    
    all_trades = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        results = list(tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols)))
        for trades in results: all_trades.extend(trades)
            
    if not all_trades:
        print("No trades generated.")
        return
        
    trades_df = pd.DataFrame(all_trades)
    print("\n=== V2 RESULTS ===")
    print(f"Total Trades: {len(trades_df)}")
    print(f"Win Rate:     {(trades_df['pnl'] > 0).mean():.2%}")
    print(f"Average PnL:  {trades_df['pnl'].mean():.4%}")
    print(f"Total PnL:    {trades_df['pnl'].sum():.4%}")
    print("\nExit Types:\n", trades_df['type'].value_counts())

if __name__ == "__main__":
    main()
