import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from datetime import datetime, timedelta

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake"
EXCHANGE = "binance"
OUT_DIR = "/home/ubuntu/Projects/skytrade6/gemini-6/features"
os.makedirs(OUT_DIR, exist_ok=True)

# Read all symbols from datalake directory
with open("/tmp/all_binance_symbols.txt", "r") as f:
    ALL_SYMBOLS = [line.strip() for line in f if line.strip()]

# Exclude BTC/ETH as they are entirely different market regimes
ALL_SYMBOLS = [sym for sym in ALL_SYMBOLS if sym not in ["BTCUSDT", "ETHUSDT"]]

start_date = datetime(2025, 7, 1)
end_date = datetime(2026, 3, 4)
DATES = [(start_date + timedelta(days=x)).strftime('%Y-%m-%d') for x in range((end_date - start_date).days + 1)]

def get_symbol_volume_rank(symbol):
    """Quickly check the average file size for the symbol to estimate volume tier."""
    test_date = "2025-08-01"
    filepath = os.path.join(DATALAKE_DIR, EXCHANGE, symbol, f"{test_date}_trades.csv.gz")
    if os.path.exists(filepath):
        return os.path.getsize(filepath)
    return 0

def process_day(args):
    symbol, date = args
    fut_path = os.path.join(DATALAKE_DIR, EXCHANGE, symbol, f"{date}_trades.csv.gz")
    spot_path = os.path.join(DATALAKE_DIR, EXCHANGE, symbol, f"{date}_trades_spot.csv.gz")
    
    if not os.path.exists(fut_path) or not os.path.exists(spot_path):
        return None
        
    try:
        # --- FUTURES ---
        df_fut = pd.read_csv(fut_path, usecols=['price', 'quote_qty', 'time', 'is_buyer_maker'])
        if len(df_fut) == 0: return None
        
        fut_whale_thresh = df_fut['quote_qty'].quantile(0.98)
        fut_retail_thresh = df_fut['quote_qty'].quantile(0.20)
        
        df_fut['direction'] = np.where(df_fut['is_buyer_maker'], -1, 1)
        df_fut['signed_qty'] = df_fut['quote_qty'] * df_fut['direction']
        df_fut['time'] = pd.to_datetime(df_fut['time'], unit='ms')
        df_fut.set_index('time', inplace=True)
        
        fut_price = df_fut['price'].resample('1min').last().ffill()
        fut_whale_cvd = df_fut.loc[df_fut['quote_qty'] >= fut_whale_thresh, 'signed_qty'].resample('1min').sum().rename('fut_whale_cvd')
        fut_retail_cvd = df_fut.loc[df_fut['quote_qty'] <= fut_retail_thresh, 'signed_qty'].resample('1min').sum().rename('fut_retail_cvd')
        
        # --- SPOT ---
        try:
            df_spot = pd.read_csv(spot_path, names=['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker', 'is_best_match'])
        except:
            df_spot = pd.read_csv(spot_path)
            
        if len(df_spot) == 0 or 'time' not in df_spot.columns: return None
        
        if df_spot['time'].iloc[0] > 1e14:
            df_spot['time'] = pd.to_datetime(df_spot['time'], unit='us')
        else:
            df_spot['time'] = pd.to_datetime(df_spot['time'], unit='ms')
            
        spot_whale_thresh = df_spot['quote_qty'].quantile(0.98)
        spot_retail_thresh = df_spot['quote_qty'].quantile(0.20)
        
        df_spot['direction'] = np.where(df_spot['is_buyer_maker'], -1, 1)
        df_spot['signed_qty'] = df_spot['quote_qty'] * df_spot['direction']
        df_spot.set_index('time', inplace=True)
        
        spot_whale_cvd = df_spot.loc[df_spot['quote_qty'] >= spot_whale_thresh, 'signed_qty'].resample('1min').sum().rename('spot_whale_cvd')
        spot_retail_cvd = df_spot.loc[df_spot['quote_qty'] <= spot_retail_thresh, 'signed_qty'].resample('1min').sum().rename('spot_retail_cvd')
        
        # --- COMBINE ---
        res = pd.concat([fut_price.rename('price'), fut_whale_cvd, fut_retail_cvd, spot_whale_cvd, spot_retail_cvd], axis=1).fillna(0)
        res['price'] = res['price'].replace(0, np.nan).ffill()
        
        return res
    except Exception as e:
        return None

def process_symbol(symbol):
    out_file = os.path.join(OUT_DIR, f"{symbol}_1m.parquet")
    if os.path.exists(out_file):
        return True
        
    print(f"[{symbol}] Processing...")
    args_list = [(symbol, d) for d in DATES]
    
    daily_dfs = []
    # Use ThreadPool inside ProcessPool to prevent extreme overhead, or just do it linearly 
    # to avoid blowing up memory when running across 140 coins.
    # Actually, we will just use ProcessPool locally.
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_day, args_list))
        
    for res in results:
        if res is not None:
            daily_dfs.append(res)
            
    if not daily_dfs:
        print(f"[{symbol}] No valid dual-market data found.")
        return False
        
    df = pd.concat(daily_dfs)
    df.sort_index(inplace=True)
    df.to_parquet(out_file)
    print(f"[{symbol}] Saved {len(df)} rows to {out_file}")
    return True

if __name__ == "__main__":
    print(f"Scanning {len(ALL_SYMBOLS)} symbols to rank by approximate tick volume...")
    ranks = [(sym, get_symbol_volume_rank(sym)) for sym in ALL_SYMBOLS]
    ranks.sort(key=lambda x: x[1], reverse=True)
    
    # Filter out empty symbols
    valid_symbols = [r[0] for r in ranks if r[1] > 0]
    print(f"Found {len(valid_symbols)} valid altcoins with tick data.")
    
    # We already processed the top 20, let's just make sure we process the rest
    start_time = time.time()
    
    # Process sequentially at the macro level to avoid OOM, but multithreaded at the daily level
    for sym in valid_symbols:
        process_symbol(sym)
        
    print(f"\nUniversal Feature extraction completed in {(time.time() - start_time) / 60:.2f} mins.")
