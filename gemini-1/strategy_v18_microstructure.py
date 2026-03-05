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

# We need consistent returns across ALL months. 
# We found in ACTIONABLE_TOP3_CONFIGS.md that fading high-frequency liquidation cascades
# using Limit Orders provides an 87%+ win rate and positive returns almost every month.
# Since we only have 1-minute data here (not tick data), we will simulate the cascade 
# by looking for extreme 1-minute volume spikes combined with large 1-minute wicks.

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
    
    # Filter dirty 1m ticks
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
    
    # We are simulating the "Displacement" Filter from Config 1
    # We look for a 1-minute candle that drops heavily (liquidation cascade down)
    # AND has volume > 5x the 60-minute average
    
    df['vol_ma'] = df['volume'].rolling(60).mean()
    df['vol_spike'] = df['volume'] > (df['vol_ma'] * 5)
    
    # Cascade down: Open to Low drop > 0.5% in 1 minute
    df['drop_pct'] = (df['low'] - df['open']) / df['open']
    df['cascade_down'] = (df['drop_pct'] < -0.005) & df['vol_spike']
    
    # Execution Rules from Config 1:
    # OFFSET: Limit order 0.15% below market
    # TP: 0.15% (Maker)
    # SL: 0.50% (Taker)
    # MAX HOLD: 60 mins
    
    signals = df['cascade_down'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    trades = []
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    cooldown_until = 0
    n = len(df)
    
    # Simulate a Limit Order fill
    for i in range(60, n - 60):
        if not in_position:
            if signals[i] and i > cooldown_until:
                # Signal fired. We place a Limit Order 0.15% below the CLOSE of this cascade candle.
                # In real life, the cascade happens, we detect it, and place order.
                limit_price = closes[i] * (1 - 0.0015)
                
                # We check the NEXT 5 candles to see if we get filled.
                filled = False
                for j in range(1, 6):
                    if lows[i+j] <= limit_price:
                        filled = True
                        in_position = True
                        entry_price = limit_price
                        entry_idx = i + j
                        break
                        
                if not filled:
                    cooldown_until = i + 5
        else:
            target_price = entry_price * (1 + 0.0015)
            stop_price = entry_price * (1 - 0.0050)
            
            # Pessimistic: check SL first
            if lows[i] <= stop_price:
                net_pnl = ((stop_price - entry_price) / entry_price) - (MAKER_FEE + TAKER_FEE)
                trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'sl'})
                in_position = False
                cooldown_until = i + 5
                
            elif highs[i] >= target_price:
                net_pnl = ((target_price - entry_price) / entry_price) - (MAKER_FEE * 2) # Maker Entry, Maker Exit
                trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'tp'})
                in_position = False
                cooldown_until = i + 5
                
            elif (i - entry_idx) >= 60:
                net_pnl = ((closes[i] - entry_price) / entry_price) - (MAKER_FEE + TAKER_FEE)
                trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'time'})
                in_position = False
                cooldown_until = i + 5

    return trades

def main():
    # Only trade the Top 4 from Config 1
    symbols = ['DOGEUSDT', 'SOLUSDT', 'ETHUSDT', 'XRPUSDT']
    
    print(f"Step 1: Running Microstructure Cascade Fade on {symbols}...")
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
        total_pnl=('pnl', 'sum')
    )
    
    print("\n" + "="*50)
    print("--- Strategy Results (Config 1 Replicated on 1m Data) ---")
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
