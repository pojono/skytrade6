import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"

# Strategy Parameters
# Short Squeeze / Liquidation Cascade
# Look for negative funding + high OI + sharp momentum up
TP_PCT = 0.08
SL_PCT = -0.04
TIME_STOP_MINS = 60 * 12 # 12 hours

MAKER_FEE = 0.0004
TAKER_FEE = 0.0010

def load_symbol_data(symbol):
    files_kline = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_kline_1m.csv")
    files_funding = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_funding_rate.csv")
    
    if not files_kline or not files_funding: return None
        
    # Klines
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
    if not fund_list: return None
    
    df_fund = pd.concat(fund_list, ignore_index=True)
    if 'fundingRateTimestamp' in df_fund.columns:
        df_fund['timestamp'] = pd.to_datetime(df_fund['fundingRateTimestamp'], unit='ms')
    elif 'timestamp' in df_fund.columns and df_fund['timestamp'].dtype != 'datetime64[ns]':
        df_fund['timestamp'] = pd.to_datetime(df_fund['timestamp'], unit='ms')
    
    df_fund = df_fund.drop_duplicates(subset=['timestamp']).sort_values('timestamp').set_index('timestamp')
    df_fund = df_fund.resample('5min').ffill()
    
    # Merge
    df = df.join(df_fund[['fundingRate']], how='left').ffill()
    
    # Filter 180 days
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
        
    if df.empty or len(df) < 1000:
        return None
        
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    return backtest_squeeze(df, symbol)

def backtest_squeeze(df, symbol):
    # Features
    # Very negative funding (shorts paying longs) -> High short interest
    # Annualized funding < -50% (approx < -0.045% per 8h)
    df['is_neg_funding'] = df['fundingRate'] < -0.00045
    
    # Momentum: 4-hour return > 5%
    df['ret_4h'] = df['close'] / df['close'].shift(48) - 1 # 48 * 5m = 4h
    df['is_momentum'] = df['ret_4h'] > 0.05
    
    raw_signal = df['is_neg_funding'] & df['is_momentum']
    
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
                cooldown_until = i + 12 # wait 1 hour
                
            elif highs[i] >= target_price:
                exit_price = target_price
                net_pnl = ((exit_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE
                trades.append({'symbol': symbol, 'pnl': net_pnl, 'type': 'tp', 'hold_candles': i - entry_idx})
                in_position = False
                cooldown_until = i + 12
                
            elif (i - entry_idx) >= (TIME_STOP_MINS / 5):
                exit_price = closes[i]
                net_pnl = ((exit_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append({'symbol': symbol, 'pnl': net_pnl, 'type': 'time', 'hold_candles': i - entry_idx})
                in_position = False
                cooldown_until = i + 12

    return trades

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    print(f"Testing Squeeze Strategy (Neg Funding + Momentum)...")
    
    all_trades = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        results = list(tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols)))
        for trades in results: all_trades.extend(trades)
            
    if not all_trades:
        print("No trades generated.")
        return
        
    trades_df = pd.DataFrame(all_trades)
    print("\n=== SQUEEZE RESULTS ===")
    print(f"Total Trades: {len(trades_df)}")
    print(f"Win Rate:     {(trades_df['pnl'] > 0).mean():.2%}")
    print(f"Average PnL:  {trades_df['pnl'].mean():.4%}")
    print(f"Total PnL:    {trades_df['pnl'].sum():.4%}")
    print("\nExit Types:\n", trades_df['type'].value_counts())

if __name__ == "__main__":
    main()
