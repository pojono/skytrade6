import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"

# Final Strategy Execution for Liquidation Cascade Edge
# We found a massive structural edge:
# When Price Drops > 8% in 4H AND Open Interest Drops > 10% in 4H
# Forward returns over 12H average +6.9% (Median +1.5%), with 58% win rate.
# Let's map this into a pure executable time-based strategy with catastrophic stop loss.

MAKER_FEE = 0.0004
TAKER_FEE = 0.0010

def load_symbol_data(symbol):
    files = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_kline_1m.csv")
    files_oi = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_open_interest_5min.csv")
    
    if not files or not files_oi: return None
        
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
         
    # Filter out impossible prices & flash crash artifacts
    df = df[(df['close'] > 0) & (df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0)]
    df = df[df['high'] >= df['low']]
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['ret'] = df['close'].pct_change()
    df = df[(df['ret'] > -0.5) & (df['ret'] < 1.0)]
    
    df = df.drop_duplicates(subset=['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    df = df.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df.dropna(inplace=True)
    
    oi_list = []
    for f in files_oi:
        try: oi_list.append(pd.read_csv(f))
        except: pass
    if oi_list:
        df_oi = pd.concat(oi_list, ignore_index=True)
        if 'timestamp' in df_oi.columns:
            if df_oi['timestamp'].dtype != 'datetime64[ns]':
                df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'], unit='ms')
            df_oi = df_oi.drop_duplicates(subset=['timestamp']).sort_values('timestamp').set_index('timestamp')
            df_oi = df_oi.resample('5min').ffill()
            df = df.join(df_oi[['openInterest']], how='left').ffill()
        else:
            return None
    else:
        return None
    
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
        
    if df.empty or len(df) < (12 * 24):
        return None
        
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    return backtest_strategy(df, symbol)

def backtest_strategy(df, symbol):
    # Signal Generation
    df['ret_4h'] = df['close'] / df['close'].shift(48) - 1
    if 'openInterest' not in df.columns: return []
    df['oi_4h'] = df['openInterest'] / df['openInterest'].shift(48) - 1
    
    # We want a flush: Price dumps AND OI dumps
    raw_signal = (df['ret_4h'] < -0.08) & (df['ret_4h'] > -0.50) & (df['oi_4h'] < -0.10)
    
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
    
    # Execution Rules
    # Catastrophic SL only (-15%). Otherwise, pure Time Exit at 12 hours (144 candles).
    SL_PCT = -0.15
    TIME_STOP_CANDLES = 144
    
    for i in range(48, n - 1):
        if not in_position:
            if signals[i] and i > cooldown_until:
                in_position = True
                entry_price = closes[i] # Taker Entry
                entry_time = timestamps[i]
                entry_idx = i
        else:
            stop_price = entry_price * (1 + SL_PCT)
            
            if lows[i] <= stop_price:
                exit_price = stop_price
                net_pnl = ((exit_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append({
                    'symbol': symbol, 
                    'entry_time': entry_time,
                    'exit_time': timestamps[i],
                    'pnl': net_pnl, 
                    'type': 'sl', 
                    'hold_mins': (i - entry_idx) * 5
                })
                in_position = False
                cooldown_until = i + 48 # wait 4h
                
            elif (i - entry_idx) >= TIME_STOP_CANDLES:
                exit_price = closes[i] # Taker Exit (market close)
                net_pnl = ((exit_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append({
                    'symbol': symbol, 
                    'entry_time': entry_time,
                    'exit_time': timestamps[i],
                    'pnl': net_pnl, 
                    'type': 'time', 
                    'hold_mins': (i - entry_idx) * 5
                })
                in_position = False
                cooldown_until = i + 48

    return trades

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    print(f"Testing Final Executable Liquidation Strategy (Time Exit 12H)...")
    
    all_trades = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        results = list(tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols)))
        for trades in results: all_trades.extend(trades)
            
    if not all_trades:
        print("No events found.")
        return
        
    trades_df = pd.DataFrame(all_trades)
    
    # Calculate daily returns (rough)
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    daily_pnl = trades_df.set_index('entry_time').resample('1D')['pnl'].sum()
    sharpe = np.sqrt(365) * (daily_pnl.mean() / daily_pnl.std()) if daily_pnl.std() > 0 else 0
    
    print("\n=== FINAL EXECUTABLE STRATEGY RESULTS ===")
    print(f"Total Trades: {len(trades_df)}")
    print(f"Win Rate:     {(trades_df['pnl'] > 0).mean():.2%}")
    print(f"Average Net PnL (per trade):  {trades_df['pnl'].mean():.4%}")
    print(f"Median Net PnL (per trade):   {trades_df['pnl'].median():.4%}")
    print(f"Total Net PnL (1x leverage):  {trades_df['pnl'].sum():.4%}")
    print(f"Sharpe Ratio (Daily): {sharpe:.2f}")
    
    print("\nExit Types:\n", trades_df['type'].value_counts())
    
    os.makedirs("/home/ubuntu/Projects/skytrade6/gemini-1", exist_ok=True)
    trades_df.to_csv("/home/ubuntu/Projects/skytrade6/gemini-1/final_liquidation_trades.csv", index=False)
    print("Saved trades to gemini-1/final_liquidation_trades.csv")

if __name__ == "__main__":
    main()
