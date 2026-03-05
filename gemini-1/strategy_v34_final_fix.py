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
    
    df['ret'] = df['close'].pct_change()
    df = df[(df['ret'] > -0.15) & (df['ret'] < 0.15)]
    
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
    
    # We must balance the equation.
    # The original research used limit entries. 
    # If the gross target is +0.12%, net is +0.04%. 
    # The ONLY way this works is if the Stop Loss is EXTREMELY TIGHT (e.g. -0.20% max).
    # Config 1 used -0.50% SL, which meant 1 loss = 12 wins. 
    
    # Let's fix this mechanically by taking bigger targets. 
    # Cascades easily snap back 0.50% to 1.00%.
    
    ENTRY_OFFSET = 0.0015
    TP_PCT = 0.0050 # Target 0.50% snapback (0.42% net)
    SL_PCT = 0.0100 # Stop at -1.00% (-1.14% net) -> Risk Reward is 1:3, needs 75% WR to break even
    MAX_HOLD = 60
    
    for i in range(60, n - MAX_HOLD):
        if not in_position:
            if i > cooldown_until:
                if signals_down[i]:
                    limit_price = closes[i] * (1 - ENTRY_OFFSET)
                    filled = False
                    for j in range(1, 6):
                        if lows[i+j] <= limit_price:
                            filled = True
                            in_position = True
                            pos_type = 1
                            entry_price = limit_price
                            entry_idx = i + j
                            break
                    if not filled: cooldown_until = i + 5
                            
                elif signals_up[i]:
                    limit_price = closes[i] * (1 + ENTRY_OFFSET)
                    filled = False
                    for j in range(1, 6):
                        if highs[i+j] >= limit_price:
                            filled = True
                            in_position = True
                            pos_type = -1
                            entry_price = limit_price
                            entry_idx = i + j
                            break
                    if not filled: cooldown_until = i + 5
        else:
            if pos_type == 1: 
                target_price = entry_price * (1 + TP_PCT)
                stop_price = entry_price * (1 - SL_PCT)
                
                if lows[i] <= stop_price:
                    net_pnl = ((stop_price - entry_price) / entry_price) - MAKER_FEE - TAKER_FEE
                    trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'sl'})
                    in_position = False
                    cooldown_until = i + 5
                    
                elif highs[i] >= target_price:
                    net_pnl = ((target_price - entry_price) / entry_price) - (MAKER_FEE * 2)
                    trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'tp'})
                    in_position = False
                    cooldown_until = i + 5
                    
                elif (i - entry_idx) >= MAX_HOLD:
                    net_pnl = ((closes[i] - entry_price) / entry_price) - MAKER_FEE - TAKER_FEE
                    trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'time'})
                    in_position = False
                    cooldown_until = i + 5
                    
            elif pos_type == -1:
                target_price = entry_price * (1 - TP_PCT)
                stop_price = entry_price * (1 + SL_PCT)
                
                if highs[i] >= stop_price:
                    net_pnl = ((entry_price - stop_price) / entry_price) - MAKER_FEE - TAKER_FEE
                    trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'sl'})
                    in_position = False
                    cooldown_until = i + 5
                    
                elif lows[i] <= target_price:
                    net_pnl = ((entry_price - target_price) / entry_price) - (MAKER_FEE * 2)
                    trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'tp'})
                    in_position = False
                    cooldown_until = i + 5
                    
                elif (i - entry_idx) >= MAX_HOLD:
                    net_pnl = ((entry_price - closes[i]) / entry_price) - MAKER_FEE - TAKER_FEE
                    trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'time'})
                    in_position = False
                    cooldown_until = i + 5

    return trades

def main():
    symbols = ['DOGEUSDT', 'SOLUSDT', 'ETHUSDT', 'XRPUSDT']
    
    print(f"Executing Fixed Macro-Target Cascade on Top 4 Coins...")
    
    all_trades = []
    pool = multiprocessing.Pool(processes=min(4, multiprocessing.cpu_count()))
    
    for trades in tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols), desc="Backtesting"):
        all_trades.extend(trades)
            
    pool.close()
    pool.join()
                
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
        total_pnl=('pnl', 'sum')
    )
    
    print("\n" + "="*50)
    print("--- FIXED SCRIPT RESULTS (Balanced TP/SL) ---")
    print("="*50)
    print(f"Total Trades: {len(df_trades)}")
    print(f"Win Rate: {(df_trades['pnl'] > 0).mean():.2%}")
    print(f"Avg PnL: {df_trades['pnl'].mean():.4%}")
    print(f"Total Raw Strategy Return (1x Lev): {df_time['cumulative'].iloc[-1]:.4%}")
    
    print("\n--- Exit Types ---")
    print(df_trades['type'].value_counts())
    
    print("\n--- Monthly Distribution ---")
    print(monthly_group)

if __name__ == "__main__":
    main()
