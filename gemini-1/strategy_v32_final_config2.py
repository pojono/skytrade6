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
    
    # Clean anomalies
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
    
    # We are directly implementing "Config 2: BEST RETURN — Aggressive No Stop Loss"
    # from the user's previously validated ACTIONABLE_TOP3_CONFIGS.md file.
    
    # 1. Cascade Trigger: Extreme displacement (>10 bps) + Volume
    df['vol_ma'] = df['volume'].rolling(60).mean()
    df['vol_spike'] = df['volume'] > (df['vol_ma'] * 3.0) # Proxy for P95 liq event
    
    df['drop_pct'] = (df['low'] - df['open']) / df['open']
    df['pump_pct'] = (df['high'] - df['open']) / df['open']
    
    # Displacement >= 10 bps (0.0010)
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
    
    # Config 2 Exact Parameters
    ENTRY_OFFSET = 0.0015  # Limit order at 0.15% offset
    TP_PCT = 0.0012        # Take Profit at 0.12% (limit order, maker fee)
    MAX_HOLD = 60          # 60 minutes time stop (market exit, taker fee)
    # SL = NONE
    
    for i in range(60, n - MAX_HOLD):
        if not in_position:
            if i > cooldown_until:
                if signals_down[i]:
                    # Fade the cascade down -> Place limit BUY 0.15% below the close
                    limit_price = closes[i] * (1 - ENTRY_OFFSET)
                    filled = False
                    
                    # Assume order is open for 5 minutes
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
                    # Fade the cascade up -> Place limit SELL 0.15% above the close
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
            if pos_type == 1: # LONG POSITION
                target_price = entry_price * (1 + TP_PCT)
                
                # Check TP
                if highs[i] >= target_price:
                    # Both entry and exit were Maker limits
                    net_pnl = ((target_price - entry_price) / entry_price) - (MAKER_FEE * 2)
                    trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'tp_long'})
                    in_position = False
                    cooldown_until = i + 5
                    
                # Check Time Stop (60 minutes)
                elif (i - entry_idx) >= MAX_HOLD:
                    # Entry was Maker, Exit is Taker
                    net_pnl = ((closes[i] - entry_price) / entry_price) - MAKER_FEE - TAKER_FEE
                    trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'time_long'})
                    in_position = False
                    cooldown_until = i + 5
                    
            elif pos_type == -1: # SHORT POSITION
                target_price = entry_price * (1 - TP_PCT)
                
                # Check TP
                if lows[i] <= target_price:
                    net_pnl = ((entry_price - target_price) / entry_price) - (MAKER_FEE * 2)
                    trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'tp_short'})
                    in_position = False
                    cooldown_until = i + 5
                    
                # Check Time Stop (60 minutes)
                elif (i - entry_idx) >= MAX_HOLD:
                    net_pnl = ((entry_price - closes[i]) / entry_price) - MAKER_FEE - TAKER_FEE
                    trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'time_short'})
                    in_position = False
                    cooldown_until = i + 5

    return trades

def main():
    # As explicitly instructed in Config 2
    symbols = ['DOGEUSDT', 'SOLUSDT', 'ETHUSDT', 'XRPUSDT']
    
    print(f"Executing Final Live Deployment Script (Config 2)...")
    
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
    
    # Real Portfolio Simulation: Risking 5% of bankroll per trade
    # Because we don't have a stop loss, we control risk via position sizing
    POSITION_SIZE = 0.05 
    df_time['equity_pnl'] = df_time['pnl'] * POSITION_SIZE
    df_time['cumulative_equity'] = 1.0 + df_time['equity_pnl'].cumsum()
    
    df_time['peak'] = df_time['cumulative_equity'].cummax()
    df_time['drawdown'] = (df_time['cumulative_equity'] - df_time['peak']) / df_time['peak']
    max_dd = df_time['drawdown'].min()
    
    monthly_group = df_time.resample('1ME').agg(
        trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).mean() if len(x)>0 else 0),
        raw_pnl=('pnl', 'sum'),
        equity_return=('equity_pnl', 'sum')
    )
    
    print("\n" + "="*50)
    print("--- FINAL SCRIPT RESULTS (Config 2: No Stop Loss) ---")
    print("="*50)
    print(f"Total Trades Executed: {len(df_trades)}")
    print(f"Global Win Rate: {(df_trades['pnl'] > 0).mean():.2%}")
    print(f"Total Raw Strategy Return (1x Lev): {df_time['pnl'].sum():.2%}")
    print(f"Portfolio Return (5% Pos Size): {(df_time['cumulative_equity'].iloc[-1] - 1):.2%}")
    print(f"Maximum Portfolio Drawdown: {max_dd:.2%}")
    
    print("\n--- Exit Types ---")
    print(df_trades['type'].value_counts())
    
    print("\n--- Monthly Distribution ---")
    print(monthly_group[['trades', 'win_rate', 'equity_return']])

if __name__ == "__main__":
    main()
