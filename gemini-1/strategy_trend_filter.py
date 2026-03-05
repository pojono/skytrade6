import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"

# Strategy Parameters
DROP_PCT_60M = -0.05
TP_PCT = 0.05
SL_PCT = -0.05
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
    
    # Fast 5m resampling
    df = df.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df.dropna(inplace=True)
    
    if not df.empty:
        # Keep 180 days only to speed up processing and standardizing
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
        
    if df.empty or len(df) < (72 * 12):
        return None
        
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    return backtest_strategy(df, symbol)

def backtest_strategy(df, symbol):
    # Features (on 5m timeframe)
    # 1 hour drop = 12 candles
    df['ret_1h'] = df['close'] / df['close'].shift(12) - 1
    
    # 24h (288 candles) and 72h (864 candles) SMA to determine trend
    df['sma_24h'] = df['close'].shift(1).rolling(288).mean()
    df['sma_72h'] = df['close'].shift(1).rolling(864).mean()
    
    # Signal: Overall trend is strongly up, price above HTF support, flash crash occurred
    df['trend_up'] = (df['sma_24h'] > df['sma_72h']) & (df['close'] > df['sma_72h'])
    df['flash_crash'] = df['ret_1h'] < DROP_PCT_60M
    
    raw_signal = df['trend_up'] & df['flash_crash']
    
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
    
    for i in range(864, n - 1):
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
                cooldown_until = i + 24 # wait 2 hours
                
            elif highs[i] >= target_price:
                exit_price = target_price
                net_pnl = ((exit_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE
                trades.append({'symbol': symbol, 'pnl': net_pnl, 'type': 'tp', 'hold_candles': i - entry_idx})
                in_position = False
                cooldown_until = i + 24
                
            elif (i - entry_idx) >= (TIME_STOP_MINS / 5):
                exit_price = closes[i]
                net_pnl = ((exit_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append({'symbol': symbol, 'pnl': net_pnl, 'type': 'time', 'hold_candles': i - entry_idx})
                in_position = False
                cooldown_until = i + 24

    return trades

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    print(f"Testing Trend-Filtered Dip Buying (1h crash > 5% DURING 72h uptrend)...")
    
    all_trades = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        results = list(tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols)))
        for trades in results: all_trades.extend(trades)
            
    if not all_trades:
        print("No trades generated.")
        return
        
    trades_df = pd.DataFrame(all_trades)
    print("\n=== TREND FILTER RESULTS ===")
    print(f"Total Trades: {len(trades_df)}")
    print(f"Win Rate:     {(trades_df['pnl'] > 0).mean():.2%}")
    print(f"Average PnL:  {trades_df['pnl'].mean():.4%}")
    print(f"Total PnL:    {trades_df['pnl'].sum():.4%}")
    print(f"Median PnL:   {trades_df['pnl'].median():.4%}")
    print("\nExit Types:\n", trades_df['type'].value_counts())
    
    os.makedirs("/home/ubuntu/Projects/skytrade6/gemini-1", exist_ok=True)
    trades_df.to_csv("/home/ubuntu/Projects/skytrade6/gemini-1/trend_filter_trades.csv", index=False)

if __name__ == "__main__":
    main()
