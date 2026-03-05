import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"

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
         
    # Fix the extreme bugs (which caused the massive means in the last test)
    # Filter out impossible prices
    df = df[(df['close'] > 0) & (df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0)]
    df = df[df['high'] >= df['low']]
    
    # Filter out flash crash artifacts (e.g. price drops 99% in 1m)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['ret'] = df['close'].pct_change()
    df = df[(df['ret'] > -0.5) & (df['ret'] < 1.0)] # Remove extreme 1m outliers
    
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
    return backtest_liquidation_edge(df, symbol)

def backtest_liquidation_edge(df, symbol):
    df['ret_4h'] = df['close'] / df['close'].shift(48) - 1
    if 'openInterest' not in df.columns: return []
    df['oi_4h'] = df['openInterest'] / df['openInterest'].shift(48) - 1
    
    # Clean the signals (no insane artifacts)
    # We want a flush: Price dumps AND OI dumps
    raw_signal = (df['ret_4h'] < -0.08) & (df['ret_4h'] > -0.50) & (df['oi_4h'] < -0.10)
    
    # Compute forward returns at various horizons
    df['fwd_1h'] = df['close'].shift(-12) / df['close'] - 1
    df['fwd_4h'] = df['close'].shift(-48) / df['close'] - 1
    df['fwd_12h'] = df['close'].shift(-144) / df['close'] - 1
    df['fwd_24h'] = df['close'].shift(-288) / df['close'] - 1
    df['fwd_48h'] = df['close'].shift(-576) / df['close'] - 1
    
    timestamps = df.index.values
    signals = raw_signal.values
    closes = df['close'].values
    
    trades = []
    cooldown_until = 0
    n = len(df)
    
    for i in range(48, n - 576):
        if signals[i] and i > cooldown_until:
            trades.append({
                'symbol': symbol,
                'entry_time': timestamps[i],
                'entry_price': closes[i],
                'fwd_1h': df['fwd_1h'].iloc[i],
                'fwd_4h': df['fwd_4h'].iloc[i],
                'fwd_12h': df['fwd_12h'].iloc[i],
                'fwd_24h': df['fwd_24h'].iloc[i],
                'fwd_48h': df['fwd_48h'].iloc[i]
            })
            cooldown_until = i + 48 # wait 4h before taking another event

    return trades

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    print(f"Testing Cleaned Liquidation flush Edge...")
    
    all_trades = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        results = list(tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols)))
        for trades in results: all_trades.extend(trades)
            
    if not all_trades:
        print("No events found.")
        return
        
    trades_df = pd.DataFrame(all_trades)
    print("\n=== CLEANED FORWARD RETURNS POST LIQUIDATION ===")
    print(f"Total Events: {len(trades_df)}")
    
    for horizon in ['fwd_1h', 'fwd_4h', 'fwd_12h', 'fwd_24h', 'fwd_48h']:
        print(f"\n{horizon.upper()}:")
        print(f"Mean: {trades_df[horizon].mean():.4%}")
        print(f"Median: {trades_df[horizon].median():.4%}")
        print(f"Win Rate: {(trades_df[horizon] > 0).mean():.2%}")
        # Assuming Maker/Taker round trip of ~0.14%
        edge_after_fees = trades_df[horizon].mean() - 0.0014
        print(f"Expected Net Edge: {edge_after_fees:.4%}")

if __name__ == "__main__":
    main()
