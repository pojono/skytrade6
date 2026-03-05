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
    
    # Filter out wild bad data points
    df['ret'] = df['close'].pct_change()
    df = df[(df['ret'] > -0.10) & (df['ret'] < 0.10)]
    
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp')
    df.dropna(inplace=True)
    
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
    
    if df.empty or len(df) < (60 * 24 * 7): return None
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    
    # Replicating CONFIG 2: Best Return - Aggressive No Stop Loss
    # Entry: 0.15% Offset limit
    # TP: 0.12% Limit (Maker)
    # SL: NONE
    # Max Hold: 60 mins (Taker)
    
    df['vol_ma'] = df['volume'].rolling(60).mean()
    df['vol_spike'] = df['volume'] > (df['vol_ma'] * 3.0)
    
    df['drop_pct'] = (df['low'] - df['open']) / df['open']
    df['pump_pct'] = (df['high'] - df['open']) / df['open']
    
    df['cascade_down'] = (df['drop_pct'] <= -0.0010) & df['vol_spike']
    df['cascade_up']   = (df['pump_pct'] >= 0.0010) & df['vol_spike']
    
    signals_down = df['cascade_down'].values
    signals_up = df['cascade_up'].values
    
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    trades = []
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    cooldown_until = 0
    n = len(df)
    pos_type = 0
    
    for i in range(60, n - 60):
        if not in_position:
            if i > cooldown_until:
                if signals_down[i]:
                    limit_price = closes[i] * (1 - 0.0015)
                    filled = False
                    for j in range(1, 4):
                        if lows[i+j] <= limit_price:
                            filled = True
                            in_position = True
                            pos_type = 1
                            entry_price = limit_price
                            entry_idx = i + j
                            break
                    if not filled: cooldown_until = i + 5
                            
                elif signals_up[i]:
                    limit_price = closes[i] * (1 + 0.0015)
                    filled = False
                    for j in range(1, 4):
                        if highs[i+j] >= limit_price:
                            filled = True
                            in_position = True
                            pos_type = -1
                            entry_price = limit_price
                            entry_idx = i + j
                            break
                    if not filled: cooldown_until = i + 5
        else:
            if pos_type == 1: # LONG
                target_price = entry_price * (1 + 0.0012) # TP: 0.12%
                
                if highs[i] >= target_price:
                    net_pnl = ((target_price - entry_price) / entry_price) - (MAKER_FEE * 2)
                    trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'tp_long'})
                    in_position = False
                    cooldown_until = i + 5
                elif (i - entry_idx) >= 60: # 60 min time stop
                    net_pnl = ((closes[i] - entry_price) / entry_price) - MAKER_FEE - TAKER_FEE
                    trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'time_long'})
                    in_position = False
                    cooldown_until = i + 5
                    
            elif pos_type == -1: # SHORT
                target_price = entry_price * (1 - 0.0012)
                
                if lows[i] <= target_price:
                    net_pnl = ((entry_price - target_price) / entry_price) - (MAKER_FEE * 2)
                    trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'tp_short'})
                    in_position = False
                    cooldown_until = i + 5
                elif (i - entry_idx) >= 60:
                    net_pnl = ((entry_price - closes[i]) / entry_price) - MAKER_FEE - TAKER_FEE
                    trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'time_short'})
                    in_position = False
                    cooldown_until = i + 5

    return trades

def main():
    symbols = ['DOGEUSDT', 'SOLUSDT', 'ETHUSDT', 'XRPUSDT']
    
    print(f"Executing EXACT Replica of Config 2 (Aggressive No-SL) on Top 4 Coins...")
    start_time = time.time()
    
    all_trades = []
    pool = multiprocessing.Pool(processes=min(4, multiprocessing.cpu_count()))
    
    for trades in tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols), desc="Backtesting"):
        all_trades.extend(trades)
            
    pool.close()
    pool.join()
    
    print(f"Completed in {time.time() - start_time:.2f}s")
                
    if not all_trades:
        print("No trades generated.")
        return
        
    df_trades = pd.DataFrame(all_trades)
    df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
    df_time = df_trades.set_index('exit_time')
    df_time.sort_index(inplace=True)
    df_time['cumulative'] = df_time['pnl'].cumsum()
    
    monthly_group = df_time.resample('1ME').agg(
        trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).mean() if len(x)>0 else 0),
        total_pnl=('pnl', 'sum'),
        avg_pnl=('pnl', 'mean')
    )
    
    print("\n" + "="*50)
    print("--- Strategy Results (Config 2: No SL) ---")
    print("="*50)
    print(f"Total Trades: {len(df_trades)}")
    print(f"Win Rate: {(df_trades['pnl'] > 0).mean():.2%}")
    print(f"Avg Net PnL per Trade: {df_trades['pnl'].mean():.4%}")
    print(f"Total Cumulative PnL: {df_time['cumulative'].iloc[-1]:.4%}")
    
    print("\n--- Exit Types ---")
    print(df_trades['type'].value_counts())
    
    print("\n--- Monthly Distribution ---")
    print(monthly_group)

if __name__ == "__main__":
    main()
