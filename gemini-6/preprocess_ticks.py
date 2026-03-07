import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import time
from datetime import datetime, timedelta

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake"
EXCHANGE = "binance"
OUT_DIR = "/home/ubuntu/Projects/skytrade6/gemini-6/features"
os.makedirs(OUT_DIR, exist_ok=True)

TIER_1 = ["SOLUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "SUIUSDT", "LINKUSDT", "NEARUSDT", "APTUSDT", "ARBUSDT", "AAVEUSDT"]
TIER_2 = ["WLDUSDT", "TIAUSDT", "SEIUSDT", "ENAUSDT", "TAOUSDT", "TONUSDT", "ATOMUSDT", "ZROUSDT", "JTOUSDT"]
SYMBOLS = TIER_1 + TIER_2

# Full available dates from July 2025 to March 2026
start_date = datetime(2025, 7, 1)
end_date = datetime(2026, 3, 4)
DATES = [(start_date + timedelta(days=x)).strftime('%Y-%m-%d') for x in range((end_date - start_date).days + 1)]

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
        print(f"[{symbol}] Already processed.")
        return
        
    print(f"[{symbol}] Extracting 1m features from ticks...")
    args_list = [(symbol, d) for d in DATES]
    
    daily_dfs = []
    # 16 workers for high concurrency across daily files
    with ProcessPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(process_day, args_list))
        
    for res in results:
        if res is not None:
            daily_dfs.append(res)
            
    if not daily_dfs:
        print(f"[{symbol}] No data found.")
        return
        
    df = pd.concat(daily_dfs)
    df.sort_index(inplace=True)
    df.to_parquet(out_file)
    print(f"[{symbol}] Saved {len(df)} rows to {out_file}")

if __name__ == "__main__":
    start_time = time.time()
    for sym in SYMBOLS:
        process_symbol(sym)
    print(f"Feature extraction completed in {(time.time() - start_time) / 60:.2f} mins.")
