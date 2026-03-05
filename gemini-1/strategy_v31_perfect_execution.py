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
    
    # We found that relying strictly on Limit entries still yielded a tiny negative edge because
    # the 60-minute time stops (which use Taker fees) act as a massive drag.
    # What if we eliminate Time Stops and use a VERY WIDE Maker-only structural reversal target?
    #
    # Actually, we need to adapt our initial successful "Confirmed Capitulation" (v8/v9) 
    # to this micro-structure framework to ensure consistency. 
    # We will use 1-hour candles.
    
    df_1h = df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df_1h.dropna(inplace=True)
    
    # 24H Trend Filter
    df_1h['sma_24'] = df_1h['close'].rolling(24).mean()
    df_1h['macro_bull'] = df_1h['close'] > df_1h['sma_24']
    
    # 4H Flush detection
    df_1h['ret_4h'] = df_1h['close'] / df_1h['close'].shift(4) - 1
    df_1h['vol_ma'] = df_1h['volume'].rolling(24).mean()
    
    # Flush: price drops > 4% in 4 hours on 2x average volume
    df_1h['flush'] = (df_1h['ret_4h'] < -0.04) & (df_1h['volume'] > df_1h['vol_ma'] * 2.0)
    
    # Confirmation: 1H bounce > 1%
    df_1h['ret_1h'] = df_1h['close'] / df_1h['close'].shift(1) - 1
    df_1h['bounce'] = df_1h['ret_1h'] > 0.01
    
    # Signal: Valid flush within the last 4 hours, and we just got a bounce confirmation
    df_1h['flush_recent'] = df_1h['flush'].rolling(4).max().shift(1).fillna(0).astype(bool)
    df_1h['signal'] = df_1h['flush_recent'] & df_1h['bounce'] & df_1h['macro_bull']
    
    signals = df_1h['signal'].values
    opens = df_1h['open'].values
    highs = df_1h['high'].values
    lows = df_1h['low'].values
    closes = df_1h['close'].values
    
    trades = []
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    cooldown_until = 0
    n = len(df_1h)
    
    TP_PCT = 0.08
    SL_PCT = -0.04
    TIME_STOP_CANDLES = 24 * 3
    
    for i in range(24, n - 1):
        if not in_position:
            if signals[i] and i > cooldown_until:
                in_position = True
                entry_price = opens[i+1] # Taker Entry
                entry_idx = i + 1
        else:
            target_price = entry_price * (1 + TP_PCT)
            stop_price = entry_price * (1 + SL_PCT)
            
            if lows[i] <= stop_price:
                net_pnl = ((stop_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append({'exit_time': df_1h.index[i], 'pnl': net_pnl, 'type': 'sl'})
                in_position = False
                cooldown_until = i + 4
                
            elif highs[i] >= target_price:
                net_pnl = ((target_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE
                trades.append({'exit_time': df_1h.index[i], 'pnl': net_pnl, 'type': 'tp'})
                in_position = False
                cooldown_until = i + 4
                
            elif (i - entry_idx) >= TIME_STOP_CANDLES:
                net_pnl = ((closes[i] - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append({'exit_time': df_1h.index[i], 'pnl': net_pnl, 'type': 'time'})
                in_position = False
                cooldown_until = i + 4

    return trades

def main():
    symbols = ['DOGEUSDT', 'SOLUSDT', 'ETHUSDT', 'XRPUSDT', 'AVAXUSDT', 'LINKUSDT', 'MATICUSDT', 'NEARUSDT', 'ADAUSDT', 'DOTUSDT']
    
    print(f"Testing Confirmed 4H Flushes on Top 10 Alts...")
    
    all_trades = []
    pool = multiprocessing.Pool(processes=min(10, multiprocessing.cpu_count()))
    
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
        total_pnl=('pnl', 'sum'),
    )
    
    print("\n" + "="*50)
    print("--- Strategy Results (Confirmed 4H Flushes) ---")
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
