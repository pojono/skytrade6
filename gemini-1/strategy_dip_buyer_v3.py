import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"

# Strategy Parameters
# Focus on identifying capitulation rather than just arbitrary drops.
# Use RSI to find overbought/oversold and Volume to confirm.
DROP_PCT_15M = -0.05
TP_PCT = 0.05      # Bigger target to justify risk
SL_PCT = -0.05     # Wider stop to avoid noise
TIME_STOP_MINS = 360 # 6 hours

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
        
    if df.empty or len(df) < 100:
        return None
        
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    return backtest_dip_buyer(df, symbol)

def calc_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=period - 1, adjust=False).mean()
    ema_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

def backtest_dip_buyer(df, symbol):
    # Features
    df['ret_15m'] = df['close'] / df['close'].shift(15) - 1
    df['ret_1m'] = df['close'].pct_change()
    df['rsi_14'] = calc_rsi(df['close'], 14)
    
    # Calculate volume surge (requires volume column)
    if 'volume' in df.columns:
        df['vol_ma'] = df['volume'].rolling(60).mean()
        df['vol_surge'] = df['volume'] > (df['vol_ma'] * 3)
    else:
        df['vol_surge'] = True # Bypass if no volume
        
    # We want sharp panic drop (15m < -5%), RSI heavily oversold (< 20), high volume (capitulation)
    raw_signal = (df['ret_15m'] < DROP_PCT_15M) & (df['ret_1m'] > -0.5) & (df['rsi_14'] < 20) & df['vol_surge']
    
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
    
    for i in range(100, n - 1):
        if not in_position:
            if signals[i] and i > cooldown_until:
                in_position = True
                # Use Limit order entry if possible, we assume we try to catch falling knife
                # For simplicity and worst-case, assume Taker entry
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
                # Limit exit
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
    print(f"Testing Panic Capitulation (RSI + Volume Surge) strategy...")
    
    all_trades = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        results = list(tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols)))
        for trades in results: all_trades.extend(trades)
            
    if not all_trades:
        print("No trades generated.")
        return
        
    trades_df = pd.DataFrame(all_trades)
    print("\n=== V3 (RSI/VOL) RESULTS ===")
    print(f"Total Trades: {len(trades_df)}")
    print(f"Win Rate:     {(trades_df['pnl'] > 0).mean():.2%}")
    print(f"Average PnL:  {trades_df['pnl'].mean():.4%}")
    print(f"Total PnL:    {trades_df['pnl'].sum():.4%}")
    print("\nExit Types:\n", trades_df['type'].value_counts())

if __name__ == "__main__":
    main()
