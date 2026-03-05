import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"

# Strategy Parameters
# Focus on Extreme Volatility Expansion and Mean Reversion.
# When a coin drops heavily AND Open Interest decreases heavily (liquidations),
# it often creates a V-shape recovery. 

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
         
    df = df[df['close'] > 0]
    df = df[df['high'] >= df['low']]
    df = df.drop_duplicates(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.set_index('timestamp', inplace=True)
    
    # 5m resampling
    df = df.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df.dropna(inplace=True)
    
    # Open Interest
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
    return backtest_liquidation(df, symbol)

def backtest_liquidation(df, symbol):
    # Detect Liquidation Cascades
    # 1. Price drops > 8% in 4 hours
    df['ret_4h'] = df['close'] / df['close'].shift(48) - 1
    
    # 2. Open Interest drops > 10% in 4 hours (massive wipeout)
    if 'openInterest' not in df.columns:
        return []
        
    df['oi_4h'] = df['openInterest'] / df['openInterest'].shift(48) - 1
    
    # We want a flush: Price dumps AND OI dumps
    raw_signal = (df['ret_4h'] < -0.08) & (df['oi_4h'] < -0.10)
    
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
    
    TP_PCT = 0.08 # 8% target
    SL_PCT = -0.08 # 8% stop
    TIME_STOP = 12 * 24 # 24 hours
    
    for i in range(48, n - 1):
        if not in_position:
            if signals[i] and i > cooldown_until:
                in_position = True
                entry_price = closes[i]
                entry_time = timestamps[i]
                entry_idx = i
        else:
            target_price = entry_price * (1 + TP_PCT)
            stop_price = entry_price * (1 + SL_PCT)
            
            if lows[i] <= stop_price:
                exit_price = stop_price
                net_pnl = ((exit_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append({'symbol': symbol, 'pnl': net_pnl, 'type': 'sl', 'hold_candles': i - entry_idx})
                in_position = False
                cooldown_until = i + 48 # wait 4h
                
            elif highs[i] >= target_price:
                exit_price = target_price
                net_pnl = ((exit_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE
                trades.append({'symbol': symbol, 'pnl': net_pnl, 'type': 'tp', 'hold_candles': i - entry_idx})
                in_position = False
                cooldown_until = i + 48
                
            elif (i - entry_idx) >= TIME_STOP:
                exit_price = closes[i]
                net_pnl = ((exit_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append({'symbol': symbol, 'pnl': net_pnl, 'type': 'time', 'hold_candles': i - entry_idx})
                in_position = False
                cooldown_until = i + 48

    return trades

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    print(f"Testing True Liquidation flush (Price drop >8% AND OI Drop >10% in 4h)...")
    
    all_trades = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        results = list(tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols)))
        for trades in results: all_trades.extend(trades)
            
    if not all_trades:
        print("No trades generated.")
        return
        
    trades_df = pd.DataFrame(all_trades)
    print("\n=== LIQUIDATION CASCADE RESULTS ===")
    print(f"Total Trades: {len(trades_df)}")
    print(f"Win Rate:     {(trades_df['pnl'] > 0).mean():.2%}")
    print(f"Average PnL:  {trades_df['pnl'].mean():.4%}")
    print(f"Total PnL:    {trades_df['pnl'].sum():.4%}")
    print("\nExit Types:\n", trades_df['type'].value_counts())

if __name__ == "__main__":
    main()
