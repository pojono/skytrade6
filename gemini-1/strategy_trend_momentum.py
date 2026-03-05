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
    
    # Let's resample to 1H to capture larger structural moves, filtering out noise
    df = df.resample('1H').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df.dropna(inplace=True)
    
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
    
    # We need at least 30 days of data for the EMAs
    if df.empty or len(df) < (24 * 30): return None
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    
    # Momentum / Trend Following Strategy
    # Using EMAs to define the trend
    df['ema_fast'] = df['close'].ewm(span=24, adjust=False).mean() # 1 day
    df['ema_slow'] = df['close'].ewm(span=168, adjust=False).mean() # 1 week
    df['ema_macro'] = df['close'].ewm(span=720, adjust=False).mean() # 30 days
    
    # Trend Condition: Stacked EMAs
    df['trend_up'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_slow'] > df['ema_macro'])
    
    # Pullback Condition: Price touches the fast EMA from above
    df['pullback'] = (df['low'] <= df['ema_fast']) & (df['close'] > df['ema_fast'])
    
    # Volume Confirmation: Above average volume on the pullback candle
    df['vol_ma'] = df['volume'].rolling(24).mean()
    df['vol_spike'] = df['volume'] > df['vol_ma']
    
    signals = (df['trend_up'] & df['pullback'] & df['vol_spike']).values
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
    
    # Trend following needs higher targets to offset lower win rates
    TP_PCT = 0.15
    SL_PCT = -0.05
    
    for i in range(720, n - 1):
        if not in_position:
            if signals[i] and i > cooldown_until:
                in_position = True
                entry_price = opens[i+1] # Next candle open
                entry_idx = i + 1
        else:
            target_price = entry_price * (1 + TP_PCT)
            stop_price = entry_price * (1 + SL_PCT)
            
            # Pessimistic execution: SL evaluated first
            if lows[i] <= stop_price:
                net_pnl = ((stop_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append({
                    'exit_time': df.index[i],
                    'pnl': net_pnl,
                    'type': 'sl'
                })
                in_position = False
                cooldown_until = i + 12 # wait 12h
                
            elif highs[i] >= target_price:
                net_pnl = ((target_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE
                trades.append({
                    'exit_time': df.index[i],
                    'pnl': net_pnl,
                    'type': 'tp'
                })
                in_position = False
                cooldown_until = i + 12
                
            elif (i - entry_idx) >= 24 * 7: # 1 week time stop
                net_pnl = ((closes[i] - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append({
                    'exit_time': df.index[i],
                    'pnl': net_pnl,
                    'type': 'time'
                })
                in_position = False
                cooldown_until = i + 12

    return trades

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    print(f"Testing Trend Momentum Strategy (1H Pullbacks in Upgrends)...")
    
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
    
    print("\n--- Monthly Distribution ---")
    print(monthly_group)

if __name__ == "__main__":
    main()
