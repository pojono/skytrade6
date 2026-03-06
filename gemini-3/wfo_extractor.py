import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from multiprocessing import Pool
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def load_trades_around_event(symbol, event_time_str):
    event_time = pd.to_datetime(event_time_str)
    date_str = event_time.strftime('%Y-%m-%d')
    
    tick_file = DATALAKE / f"bybit/{symbol}/{date_str}_trades.csv.gz"
    if not tick_file.exists(): return None
        
    try:
        df = pd.read_csv(tick_file, usecols=['timestamp', 'side', 'size', 'price'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        start_time = event_time
        end_time = event_time + pd.Timedelta(hours=24)
        
        next_day = end_time.strftime('%Y-%m-%d')
        if next_day != date_str:
            next_file = DATALAKE / f"bybit/{symbol}/{next_day}_trades.csv.gz"
            if next_file.exists():
                df_next = pd.read_csv(next_file, usecols=['timestamp', 'side', 'size', 'price'])
                df_next['datetime'] = pd.to_datetime(df_next['timestamp'], unit='s')
                df_next.set_index('datetime', inplace=True)
                df_next.sort_index(inplace=True)
                df = pd.concat([df, df_next])
        
        window_df = df.loc[start_time:end_time].copy()
        return window_df if not window_df.empty else None
    except: return None

def process_event_for_wfo(row):
    symbol = row['symbol']
    event_time = row['timestamp']
    signal = row['signal']
    if signal != 1: return None # Only longs
    
    ticks = load_trades_around_event(symbol, event_time)
    if ticks is None or ticks.empty: return None
        
    ticks['signed_vol'] = np.where(ticks['side'] == 'Buy', ticks['size'], -ticks['size'])
    sec_df = ticks.resample('1s').agg({'price': 'last', 'size': 'sum', 'signed_vol': 'sum'}).ffill()
    
    sec_df['vol_imb_10s'] = sec_df['signed_vol'].rolling(10).sum() 
    sec_df['total_vol_10s'] = sec_df['size'].rolling(10).sum()
    sec_df['imb_ratio'] = sec_df['vol_imb_10s'] / (sec_df['total_vol_10s'] + 1e-8)
    
    entry_idx = None
    for idx, sec_row in sec_df.iterrows():
        if pd.isna(sec_row['imb_ratio']) or sec_row['total_vol_10s'] == 0: continue
        if sec_row['imb_ratio'] > 0.85:
            entry_idx = idx
            break
            
    if entry_idx is None: return None
        
    fill_time = entry_idx + pd.Timedelta(milliseconds=30)
    post_latency_ticks = ticks.loc[fill_time:]
    if post_latency_ticks.empty: return None
        
    actual_fill_price = post_latency_ticks.iloc[0]['price']
    
    # Extract 30 minutes of price action at 1-second resolution
    exit_time = fill_time + pd.Timedelta(minutes=30)
    post_trade = ticks.loc[fill_time:exit_time]
    
    if post_trade.empty: return None
    
    # Downsample to 1s array for fast grid search
    resampled_path = post_trade.resample('1s')['price'].last().ffill().values
    
    return {
        'symbol': symbol,
        'entry_time': fill_time,
        'fill_price': actual_fill_price,
        'price_path': resampled_path
    }

def run_extraction():
    events = pd.read_csv("keg_events_all_relaxed.csv")
    events = events[events['signal'] == 1].copy()
    rows = [row for _, row in events.iterrows()]
    
    results = []
    with Pool(min(16, os.cpu_count() or 4)) as p:
        for r in p.imap_unordered(process_event_for_wfo, rows):
            if r is not None:
                results.append(r)
                
    if results:
        with open("wfo_paths.pkl", "wb") as f:
            pickle.dump(results, f)
        print(f"Extracted {len(results)} execution paths for WFO.")

if __name__ == "__main__":
    run_extraction()
