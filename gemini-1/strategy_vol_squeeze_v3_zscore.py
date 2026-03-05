import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"

# Strategy Parameters
# Focus on Z-score extreme over-extensions + Funding to find TRUE bottoms
# Using a 1H timeframe approach internally to filter noise, holding for 24-48 hours.

MAKER_FEE = 0.0004
TAKER_FEE = 0.0010

def load_symbol_data(symbol):
    files = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_kline_1m.csv")
    fund_files = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_funding_rate.csv")
    
    if not files or not fund_files: return None
        
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
    
    # 15m resampling to get strong structural signals
    df = df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df.dropna(inplace=True)
    
    # Funding
    f_list = []
    for f in fund_files:
        try: f_list.append(pd.read_csv(f))
        except: pass
    if f_list:
        df_f = pd.concat(f_list, ignore_index=True)
        if 'fundingRateTimestamp' in df_f.columns:
            df_f['timestamp'] = pd.to_datetime(df_f['fundingRateTimestamp'], unit='ms')
        elif 'timestamp' in df_f.columns and df_f['timestamp'].dtype != 'datetime64[ns]':
            df_f['timestamp'] = pd.to_datetime(df_f['timestamp'], unit='ms')
        df_f = df_f.drop_duplicates(subset=['timestamp']).sort_values('timestamp').set_index('timestamp')
        df_f = df_f.resample('15min').ffill()
        df = df.join(df_f[['fundingRate']], how='left').ffill()
    
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
        
    if df.empty or len(df) < (4 * 24 * 7): # 7 days
        return None
        
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    return backtest_zscore(df, symbol)

def backtest_zscore(df, symbol):
    # Calculate rolling Z-score of Returns (24h = 96 candles)
    df['ret'] = df['close'].pct_change()
    df['ret_mean'] = df['ret'].rolling(96).mean()
    df['ret_std'] = df['ret'].rolling(96).std()
    
    # 4-hour drop Z-score
    df['ret_4h'] = df['close'] / df['close'].shift(16) - 1
    df['ret_4h_mean'] = df['ret_4h'].rolling(96).mean()
    df['ret_4h_std'] = df['ret_4h'].rolling(96).std()
    
    df['z_score_4h'] = (df['ret_4h'] - df['ret_4h_mean']) / df['ret_4h_std']
    
    # Negative funding as fuel
    if 'fundingRate' in df.columns:
        df['extreme_funding'] = df['fundingRate'] < -0.0005 # Less than -0.05% per 8h
    else:
        df['extreme_funding'] = False
        
    # We want Z-score < -3 (very extreme 4h drop) AND extreme negative funding
    raw_signal = (df['z_score_4h'] < -3.0) & df['extreme_funding']
    
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
    
    # Wider stops and massive targets to beat fees
    TP_PCT = 0.15 # 15% bounce
    SL_PCT = -0.10 # 10% stop loss
    TIME_STOP = 96 * 2 # 48 hours
    
    for i in range(96, n - 1):
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
                cooldown_until = i + 16 # wait 4h
                
            elif highs[i] >= target_price:
                exit_price = target_price
                net_pnl = ((exit_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE
                trades.append({'symbol': symbol, 'pnl': net_pnl, 'type': 'tp', 'hold_candles': i - entry_idx})
                in_position = False
                cooldown_until = i + 16
                
            elif (i - entry_idx) >= TIME_STOP:
                exit_price = closes[i]
                net_pnl = ((exit_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append({'symbol': symbol, 'pnl': net_pnl, 'type': 'time', 'hold_candles': i - entry_idx})
                in_position = False
                cooldown_until = i + 16

    return trades

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    print(f"Testing Z-Score Extreme Reversion + Extreme Funding (15m timeframe)...")
    
    all_trades = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        results = list(tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols)))
        for trades in results: all_trades.extend(trades)
            
    if not all_trades:
        print("No trades generated. Constraints might be too tight.")
        return
        
    trades_df = pd.DataFrame(all_trades)
    print("\n=== Z-SCORE RESULTS ===")
    print(f"Total Trades: {len(trades_df)}")
    print(f"Win Rate:     {(trades_df['pnl'] > 0).mean():.2%}")
    print(f"Average PnL:  {trades_df['pnl'].mean():.4%}")
    print(f"Total PnL:    {trades_df['pnl'].sum():.4%}")
    print("\nExit Types:\n", trades_df['type'].value_counts())

if __name__ == "__main__":
    main()
