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
    
    # Resample to 4H candles to capture larger trends smoothly
    df = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df.dropna(inplace=True)
    
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
    
    if df.empty or len(df) < (6 * 7): # At least 7 days of 4H data
        return None
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    
    # 24H Return (6 periods of 4H)
    df['ret_24h'] = df['close'] / df['close'].shift(6) - 1
    
    # Volume relative to 7-day moving average
    df['vol_ma_7d'] = df['volume'].shift(1).rolling(6 * 7).mean()
    df['vol_spike'] = df['volume'] > (df['vol_ma_7d'] * 2.0)
    
    # Trend alignment (price above 7-day MA)
    df['sma_7d'] = df['close'].shift(1).rolling(6 * 7).mean()
    df['uptrend'] = df['close'] > df['sma_7d']
    
    # Signal: Up > 15% in 24h, massive volume, in uptrend
    signals = ((df['ret_24h'] > 0.15) & df['vol_spike'] & df['uptrend']).values
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
    
    # Execution
    # Because momentum can reverse sharply, we use a relatively tight trailing stop
    # or fixed TP. Let's try Fixed TP/SL first to establish baseline.
    TP_PCT = 0.20
    SL_PCT = -0.10
    TIME_STOP_CANDLES = 6 * 3 # 3 days (18 candles)
    
    for i in range(42, n - 1):
        if not in_position:
            if signals[i] and i > cooldown_until:
                in_position = True
                entry_price = opens[i+1] # Enter on NEXT candle open
                entry_idx = i + 1
        else:
            target_price = entry_price * (1 + TP_PCT)
            stop_price = entry_price * (1 + SL_PCT)
            
            if lows[i] <= stop_price:
                net_pnl = ((stop_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'sl'})
                in_position = False
                cooldown_until = i + 6 # Wait 24h
                
            elif highs[i] >= target_price:
                net_pnl = ((target_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE
                trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'tp'})
                in_position = False
                cooldown_until = i + 6
                
            elif (i - entry_idx) >= TIME_STOP_CANDLES:
                net_pnl = ((closes[i] - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'time'})
                in_position = False
                cooldown_until = i + 6

    return trades

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    print(f"Testing Daily Momentum Continuation Strategy (4H timeframe)...")
    
    all_trades = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        for pnl_list in tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols)):
            all_trades.extend(pnl_list)
                
    if not all_trades:
        print("No trades generated.")
        return
        
    df_trades = pd.DataFrame(all_trades)
    df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
    df_time = df_trades.set_index('exit_time')
    
    monthly_group = df_time.resample('1ME').agg(
        trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).mean() if len(x) > 0 else 0),
        total_pnl=('pnl', 'sum'),
        avg_pnl=('pnl', 'mean')
    )
    
    print("\n--- Strategy Results ---")
    print(f"Total Trades: {len(df_trades)}")
    print(f"Win Rate: {(df_trades['pnl'] > 0).mean():.2%}")
    print(f"Avg PnL: {df_trades['pnl'].mean():.4%}")
    print(f"Total PnL (1x Lev): {df_trades['pnl'].sum():.4%}")
    
    print("\n--- Exit Types ---")
    print(df_trades['type'].value_counts())
    
    print("\n--- Monthly Distribution ---")
    print(monthly_group)

if __name__ == "__main__":
    main()
