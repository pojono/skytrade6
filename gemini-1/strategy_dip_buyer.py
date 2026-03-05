import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import multiprocessing
import time
from tqdm import tqdm

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"

# Strategy Parameters
DROP_PCT_15M = -0.04
DROP_PCT_60M = -0.05
TP_PCT = 0.02
SL_PCT = -0.05
TIME_STOP_MINS = 240
MAKER_FEE = 0.0004
TAKER_FEE = 0.0010

def load_symbol_data(symbol):
    files = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_kline_1m.csv")
    if not files:
        return None
        
    df_list = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except:
            pass
            
    if not df_list:
        return None
        
    df = pd.concat(df_list, ignore_index=True)
    if 'startTime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['startTime'], unit='ms')
    elif 'timestamp' in df.columns and df['timestamp'].dtype != 'datetime64[ns]':
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
         
    # Extremely important data cleanup to avoid false signals
    # There seem to be overlapping or duplicate files creating 0 prices
    df = df[df['close'] > 0]
    df = df[df['high'] >= df['low']]
    
    # Drop duplicates by timestamp
    df = df.drop_duplicates(subset=['timestamp'])
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.set_index('timestamp', inplace=True)
    
    # Forward fill gaps up to 5 minutes
    df = df.resample('1min').ffill(limit=5)
    df.dropna(inplace=True)
    
    # Keep only last 180 days
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
        
    if df.empty or len(df) < 60:
        return None
        
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None:
        return []
    
    _, df = data
    return backtest_dip_buyer(df, symbol)

def backtest_dip_buyer(df, symbol):
    df['ret_15m'] = df['close'] / df['close'].shift(15) - 1
    df['ret_60m'] = df['close'] / df['close'].shift(60) - 1
    
    # Filter out extreme anomalies where prices drop > 95% in 1m (usually data bugs)
    df['ret_1m'] = df['close'].pct_change()
    
    # The signal logic
    raw_signal = (df['ret_15m'] < DROP_PCT_15M) & (df['ret_60m'] < DROP_PCT_60M) & (df['ret_1m'] > -0.5)
    
    timestamps = df.index.values
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    signals = raw_signal.values
    
    trades = []
    in_position = False
    entry_price = 0.0
    entry_time = None
    entry_idx = 0
    cooldown_until = 0
    
    n = len(df)
    
    for i in range(60, n - 1):
        if not in_position:
            if signals[i] and i > cooldown_until:
                in_position = True
                # Taker entry on the close of the candle
                entry_price = closes[i]
                entry_time = timestamps[i]
                entry_idx = i
        else:
            target_price = entry_price * (1 + TP_PCT)
            stop_price = entry_price * (1 + SL_PCT)
            
            # Check Stop Loss First (Pessimistic)
            if lows[i] <= stop_price:
                exit_price = stop_price
                gross_pnl = (exit_price - entry_price) / entry_price
                net_pnl = gross_pnl - (TAKER_FEE * 2)
                trades.append({
                    'symbol': symbol,
                    'entry_time': entry_time,
                    'exit_time': timestamps[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'type': 'sl',
                    'hold_mins': i - entry_idx
                })
                in_position = False
                cooldown_until = i + 60
                
            # Check Take Profit
            elif highs[i] >= target_price:
                exit_price = target_price
                gross_pnl = (exit_price - entry_price) / entry_price
                net_pnl = gross_pnl - TAKER_FEE - MAKER_FEE
                trades.append({
                    'symbol': symbol,
                    'entry_time': entry_time,
                    'exit_time': timestamps[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'type': 'tp',
                    'hold_mins': i - entry_idx
                })
                in_position = False
                cooldown_until = i + 60
                
            # Check Time Stop
            elif (i - entry_idx) >= TIME_STOP_MINS:
                exit_price = closes[i]
                gross_pnl = (exit_price - entry_price) / entry_price
                net_pnl = gross_pnl - (TAKER_FEE * 2)
                trades.append({
                    'symbol': symbol,
                    'entry_time': entry_time,
                    'exit_time': timestamps[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'type': 'time',
                    'hold_mins': i - entry_idx
                })
                in_position = False
                cooldown_until = i + 60

    return trades

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    print(f"Found {len(symbols)} symbols. Starting parallel processing...")
    
    all_trades = []
    start_time = time.time()
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        results = list(tqdm(pool.imap_unordered(process_symbol, symbols), 
                            total=len(symbols), 
                            desc="Backtesting Symbols",
                            unit="sym"))
        
    for trades in results:
        all_trades.extend(trades)
            
    elapsed = time.time() - start_time
    print(f"\nProcessing complete in {elapsed:.1f} seconds.")
    
    if not all_trades:
        print("No trades generated.")
        return
        
    trades_df = pd.DataFrame(all_trades)
    
    win_rate = (trades_df['pnl'] > 0).mean()
    avg_pnl = trades_df['pnl'].mean()
    total_pnl = trades_df['pnl'].sum()
    
    print("\n=== STRATEGY RESULTS ===")
    print(f"Total Trades: {len(trades_df)}")
    print(f"Win Rate:     {win_rate:.2%}")
    print(f"Average PnL:  {avg_pnl:.4%}")
    print(f"Total PnL:    {total_pnl:.4%}")
    print(f"Median PnL:   {trades_df['pnl'].median():.4%}")
    
    print("\nTrades by Exit Type:")
    print(trades_df['type'].value_counts())
    
    print("\nAverage Hold Mins:", trades_df['hold_mins'].mean())
    print("\nTrades by Symbol (Top 10):")
    print(trades_df['symbol'].value_counts().head(10))
    
    os.makedirs("/home/ubuntu/Projects/skytrade6/gemini-1", exist_ok=True)
    trades_df.to_csv("/home/ubuntu/Projects/skytrade6/gemini-1/dip_buyer_trades.csv", index=False)
    print("Saved trades to gemini-1/dip_buyer_trades.csv")

if __name__ == "__main__":
    main()
