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
    
    # 15m aggregation to be stable but responsive
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp')
    df = df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df.dropna(inplace=True)
    
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
    
    if df.empty or len(df) < (4 * 24 * 7): return None
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    
    # Maker-Only Mean Reversion
    # We look for large 15m wicks.
    # A wick implies price pushed down but got bought back up instantly.
    
    df['body'] = abs(df['close'] - df['open'])
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    
    df['wick_ratio'] = df['lower_wick'] / (df['body'] + 0.0000001)
    
    # Volatility context (ATR)
    df['tr'] = df['high'] - df['low']
    df['atr'] = df['tr'].rolling(4 * 24).mean()
    
    # Signal: Lower wick is at least 3x the body, AND the wick itself is > 0.5 * ATR
    df['cascade_down'] = (df['wick_ratio'] > 3.0) & (df['lower_wick'] > (0.5 * df['atr']))
    
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
    
    # Entry: Limit order at the close of the wick candle
    # TP: Maker exit at +1.0%
    # SL: Taker exit at -2.0%
    # Time Stop: 4 hours
    
    for i in range(96, n - 16):
        if not in_position:
            if signals[i] and i > cooldown_until:
                limit_price = closes[i]
                
                # Check next 4 candles to see if limit gets filled
                filled = False
                for j in range(1, 5):
                    if lows[i+j] <= limit_price:
                        filled = True
                        in_position = True
                        entry_price = limit_price
                        entry_idx = i + j
                        break
                        
                if not filled:
                    cooldown_until = i + 4
        else:
            target_price = entry_price * (1 + 0.010)
            stop_price = entry_price * (1 - 0.020)
            
            if lows[i] <= stop_price:
                net_pnl = ((stop_price - entry_price) / entry_price) - MAKER_FEE - TAKER_FEE
                trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'sl'})
                in_position = False
                cooldown_until = i + 4
                
            elif highs[i] >= target_price:
                net_pnl = ((target_price - entry_price) / entry_price) - MAKER_FEE - MAKER_FEE
                trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'tp'})
                in_position = False
                cooldown_until = i + 4
                
            elif (i - entry_idx) >= 16: # 4 hours
                net_pnl = ((closes[i] - entry_price) / entry_price) - MAKER_FEE - TAKER_FEE
                trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'time'})
                in_position = False
                cooldown_until = i + 4

    return trades

def main():
    symbols = ['DOGEUSDT', 'SOLUSDT', 'ETHUSDT', 'XRPUSDT', 'AVAXUSDT', 'LINKUSDT', 'MATICUSDT', 'NEARUSDT', 'ADAUSDT', 'DOTUSDT']
    
    print(f"Testing 15m Wick Mean Reversion on Top 10 Alts...")
    start_time = time.time()
    
    all_trades = []
    pool = multiprocessing.Pool(processes=min(10, multiprocessing.cpu_count()))
    
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
    print("--- Strategy Results (15m Maker Reversion) ---")
    print("="*50)
    print(f"Total Trades: {len(df_trades)}")
    print(f"Win Rate: {(df_trades['pnl'] > 0).mean():.2%}")
    print(f"Avg Net PnL per Trade: {df_trades['pnl'].mean():.4%}")
    print(f"Total Cumulative PnL (1x Lev): {df_time['cumulative'].iloc[-1]:.4%}")
    
    print("\n--- Exit Types ---")
    print(df_trades['type'].value_counts())
    
    print("\n--- Monthly Distribution ---")
    print(monthly_group)

if __name__ == "__main__":
    main()
