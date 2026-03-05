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

# We need a strategy that generates consistent positive edge across all months.
# Mean reversion on 1m data is failing because Taker fees (0.10% each way) destroy the tiny profit margins.
# Since we must pay 0.20% round trip for Taker entries/exits, we need a strategy that targets larger moves.
# Let's build a Multi-Day Breakout Strategy.
# Breakouts are directional, have high gross targets (10-20%), and easily absorb the 0.20% fee.

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
    df = df[(df['ret'] > -0.5) & (df['ret'] < 1.0)]
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp')
    
    # 4H candles to filter noise, good for multi-day swing trades
    df = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df.dropna(inplace=True)
    
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
    
    if df.empty or len(df) < (6 * 30): return None
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    
    # 14-Day Donchian Channels (Breakout)
    df['high_14d'] = df['high'].rolling(6 * 14).max().shift(1)
    df['low_14d'] = df['low'].rolling(6 * 14).min().shift(1)
    
    # Volume Confirmation: breakout candle must have > 2x average volume
    df['vol_ma'] = df['volume'].rolling(6 * 14).mean().shift(1)
    df['vol_spike'] = df['volume'] > (df['vol_ma'] * 2.0)
    
    # Signals
    df['long_signal'] = (df['close'] > df['high_14d']) & df['vol_spike']
    df['short_signal'] = (df['close'] < df['low_14d']) & df['vol_spike']
    
    signals_long = df['long_signal'].values
    signals_short = df['short_signal'].values
    
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
    pos_type = 0
    
    # We use a wide trailing stop or a time stop to let runners run
    TP_PCT = 0.20 # 20% massive target
    SL_PCT = 0.10 # 10% wide stop
    TIME_STOP_CANDLES = 6 * 7 # 7 days
    
    for i in range(6 * 14, n - 1):
        if not in_position:
            if i > cooldown_until:
                if signals_long[i]:
                    in_position = True
                    pos_type = 1
                    entry_price = opens[i+1] # Taker Entry next candle
                    entry_idx = i + 1
                elif signals_short[i]:
                    in_position = True
                    pos_type = -1
                    entry_price = opens[i+1]
                    entry_idx = i + 1
        else:
            if pos_type == 1:
                target_price = entry_price * (1 + TP_PCT)
                stop_price = entry_price * (1 - SL_PCT)
                
                if lows[i] <= stop_price:
                    net_pnl = ((stop_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                    trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'sl_long'})
                    in_position = False
                    cooldown_until = i + 6
                elif highs[i] >= target_price:
                    net_pnl = ((target_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE
                    trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'tp_long'})
                    in_position = False
                    cooldown_until = i + 6
                elif (i - entry_idx) >= TIME_STOP_CANDLES:
                    net_pnl = ((closes[i] - entry_price) / entry_price) - (TAKER_FEE * 2)
                    trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'time_long'})
                    in_position = False
                    cooldown_until = i + 6
                    
            elif pos_type == -1:
                target_price = entry_price * (1 - TP_PCT)
                stop_price = entry_price * (1 + SL_PCT)
                
                if highs[i] >= stop_price:
                    net_pnl = ((entry_price - stop_price) / entry_price) - (TAKER_FEE * 2)
                    trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'sl_short'})
                    in_position = False
                    cooldown_until = i + 6
                elif lows[i] <= target_price:
                    net_pnl = ((entry_price - target_price) / entry_price) - TAKER_FEE - MAKER_FEE
                    trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'tp_short'})
                    in_position = False
                    cooldown_until = i + 6
                elif (i - entry_idx) >= TIME_STOP_CANDLES:
                    net_pnl = ((entry_price - closes[i]) / entry_price) - (TAKER_FEE * 2)
                    trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'time_short'})
                    in_position = False
                    cooldown_until = i + 6

    return trades

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    
    print(f"Testing 14-Day Breakout Strategy on all coins...")
    start_time = time.time()
    
    all_trades = []
    pool = multiprocessing.Pool(processes=min(8, multiprocessing.cpu_count()))
    
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
    print("--- Strategy Results (14D Breakout) ---")
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
