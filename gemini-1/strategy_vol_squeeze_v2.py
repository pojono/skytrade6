import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"

# Refined Squeeze Strategy
# Instead of static TPs and SLs, we trail a stop or hold for a short defined burst.
# Squeezes are fast. We want to enter on high momentum + negative funding, and exit after 1 hour or if momentum breaks.

MAKER_FEE = 0.0004
TAKER_FEE = 0.0010

def load_symbol_data(symbol):
    files_kline = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_kline_1m.csv")
    files_funding = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_funding_rate.csv")
    files_oi = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_open_interest_5min.csv")
    
    if not files_kline or not files_funding: return None
        
    df_list = []
    for f in files_kline:
        try: df_list.append(pd.read_csv(f))
        except: pass
    if not df_list: return None
    
    df = pd.concat(df_list, ignore_index=True)
    if 'startTime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['startTime'], unit='ms')
    elif 'timestamp' in df.columns and df['timestamp'].dtype != 'datetime64[ns]':
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
    df = df[df['close'] > 0]
    df = df.drop_duplicates(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.set_index('timestamp', inplace=True)
    df = df.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df.dropna(inplace=True)
    
    # Funding Rate
    fund_list = []
    for f in files_funding:
        try: fund_list.append(pd.read_csv(f))
        except: pass
    if fund_list:
        df_fund = pd.concat(fund_list, ignore_index=True)
        if 'fundingRateTimestamp' in df_fund.columns:
            df_fund['timestamp'] = pd.to_datetime(df_fund['fundingRateTimestamp'], unit='ms')
        elif 'timestamp' in df_fund.columns and df_fund['timestamp'].dtype != 'datetime64[ns]':
            df_fund['timestamp'] = pd.to_datetime(df_fund['timestamp'], unit='ms')
        df_fund = df_fund.drop_duplicates(subset=['timestamp']).sort_values('timestamp').set_index('timestamp')
        df_fund = df_fund.resample('5min').ffill()
        df = df.join(df_fund[['fundingRate']], how='left').ffill()
        
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
            df['openInterest'] = np.nan
    else:
        df['openInterest'] = np.nan
        
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
        
    if df.empty or len(df) < 1000: return None
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    return backtest_squeeze(df, symbol)

def backtest_squeeze(df, symbol):
    # Short squeeze setup:
    # 1. Very negative funding (<-0.05% per 8h)
    # 2. Open interest is rising over last 24h
    # 3. Price breaks out (24h high)
    
    if 'fundingRate' not in df.columns or 'openInterest' not in df.columns:
        return []
        
    df['is_neg_funding'] = df['fundingRate'] < -0.0005
    
    df['oi_24h_ago'] = df['openInterest'].shift(288) # 24h * 12
    df['oi_rising'] = df['openInterest'] > (df['oi_24h_ago'] * 1.1) # OI up 10%
    
    df['high_24h'] = df['high'].shift(1).rolling(288).max()
    df['price_breakout'] = df['close'] > df['high_24h']
    
    raw_signal = df['is_neg_funding'] & df['oi_rising'] & df['price_breakout']
    
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
    
    # Adaptive trailing stop
    highest_seen = 0.0
    
    for i in range(288, n - 1):
        if not in_position:
            if signals[i] and i > cooldown_until:
                in_position = True
                entry_price = closes[i]
                entry_time = timestamps[i]
                entry_idx = i
                highest_seen = entry_price
        else:
            highest_seen = max(highest_seen, highs[i])
            
            # Trailing stop: 5% from highest seen
            stop_price = highest_seen * 0.95
            
            # Or absolute target: 10%
            target_price = entry_price * 1.10
            
            if lows[i] <= stop_price:
                exit_price = stop_price
                net_pnl = ((exit_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append({'symbol': symbol, 'pnl': net_pnl, 'type': 'trail_sl', 'hold_candles': i - entry_idx})
                in_position = False
                cooldown_until = i + 12 # wait 1 hour
                
            elif highs[i] >= target_price:
                exit_price = target_price
                net_pnl = ((exit_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE
                trades.append({'symbol': symbol, 'pnl': net_pnl, 'type': 'tp', 'hold_candles': i - entry_idx})
                in_position = False
                cooldown_until = i + 12
                
            elif (i - entry_idx) >= 144: # 12 hours time stop
                exit_price = closes[i]
                net_pnl = ((exit_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append({'symbol': symbol, 'pnl': net_pnl, 'type': 'time', 'hold_candles': i - entry_idx})
                in_position = False
                cooldown_until = i + 12

    return trades

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    print(f"Testing True Squeeze (Neg Funding + Rising OI + Price Breakout)...")
    
    all_trades = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        results = list(tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols)))
        for trades in results: all_trades.extend(trades)
            
    if not all_trades:
        print("No trades generated.")
        return
        
    trades_df = pd.DataFrame(all_trades)
    print("\n=== TRUE SQUEEZE RESULTS ===")
    print(f"Total Trades: {len(trades_df)}")
    print(f"Win Rate:     {(trades_df['pnl'] > 0).mean():.2%}")
    print(f"Average PnL:  {trades_df['pnl'].mean():.4%}")
    print(f"Total PnL:    {trades_df['pnl'].sum():.4%}")
    print("\nExit Types:\n", trades_df['type'].value_counts())

if __name__ == "__main__":
    main()
