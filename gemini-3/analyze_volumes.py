import os
import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import time
from tqdm import tqdm
import gc

DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def get_common_symbols():
    bybit_dir = DATALAKE / "bybit"
    binance_dir = DATALAKE / "binance"
    if not bybit_dir.exists() or not binance_dir.exists():
        return []
    bybit_symbols = {d.name for d in bybit_dir.iterdir() if d.is_dir()}
    binance_symbols = {d.name for d in binance_dir.iterdir() if d.is_dir()}
    return sorted(list(bybit_symbols.intersection(binance_symbols)))

def extract_volume_for_symbol(symbol):
    try:
        # Binance format: open_time, open, high, low, close, volume, close_time, quote_volume...
        # Bybit format: startTime, open, high, low, close, volume, turnover
        
        # We want quote volume (turnover in USDT/USD) to compare apples to apples across all pairs.
        # If turnover/quote_volume is not available, we use volume * close price as an approximation.
        
        sources = {
            'bybit_futures': (DATALAKE / "bybit" / symbol, "*_kline_1m.csv", 'volume', 'turnover', 'close'),
            'bybit_spot': (DATALAKE / "bybit" / symbol, "*_kline_1m_spot.csv", 'volume', 'turnover', 'close'),
            'binance_futures': (DATALAKE / "binance" / symbol, "*_kline_1m.csv", 'volume', 'quote_volume', 'close'),
            'binance_spot': (DATALAKE / "binance" / symbol, "*_kline_1m_spot.csv", 5, 7, 4), # using indices for binance spot due to missing headers
        }
        
        results = {'symbol': symbol}
        
        for name, (path_dir, pattern, vol_col, quote_vol_col, close_col) in sources.items():
            files = sorted(list(path_dir.glob(pattern)))
            # exclude index/mark/premium
            if pattern == "*_kline_1m.csv":
                files = [f for f in files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name]
            
            # Use data from Jan 2025 onwards
            files = [f for f in files if f.name >= "2025-01-01"]
            
            if not files:
                results[name] = np.nan
                continue
                
            total_quote_vol = 0.0
            
            for f in files:
                try:
                    if name == 'binance_spot':
                        # Use indices
                        df = pd.read_csv(f, header=None, usecols=[vol_col, quote_vol_col, close_col], engine='c')
                        # Sometimes index 7 is quote volume. If missing, approximate
                        if quote_vol_col in df.columns:
                            daily_vol = pd.to_numeric(df[quote_vol_col], errors='coerce').sum()
                        else:
                            # fallback: volume * close
                            v = pd.to_numeric(df[vol_col], errors='coerce')
                            c = pd.to_numeric(df[close_col], errors='coerce')
                            daily_vol = (v * c).sum()
                    else:
                        df = pd.read_csv(f, usecols=[vol_col, quote_vol_col, close_col], engine='c')
                        if quote_vol_col in df.columns:
                            daily_vol = pd.to_numeric(df[quote_vol_col], errors='coerce').sum()
                        else:
                            v = pd.to_numeric(df[vol_col], errors='coerce')
                            c = pd.to_numeric(df[close_col], errors='coerce')
                            daily_vol = (v * c).sum()
                            
                    total_quote_vol += daily_vol
                except Exception:
                    pass
                    
            results[name] = total_quote_vol
            
        return results
    except Exception as e:
        return None

def main():
    symbols = get_common_symbols()
    print(f"Found {len(symbols)} common symbols")
    
    results = []
    
    print("Processing volume data...")
    num_processes = min(4, os.cpu_count() or 4)
    with Pool(num_processes) as pool:
        for res in tqdm(pool.imap_unordered(extract_volume_for_symbol, symbols), total=len(symbols)):
            if res is not None:
                results.append(res)
                
    df = pd.DataFrame(results)
    df.to_csv("volume_analysis.csv", index=False)
    
    # Analysis
    # Drop rows where all vols are NaN or 0
    cols = ['binance_futures', 'binance_spot', 'bybit_futures', 'bybit_spot']
    df = df.dropna(subset=cols, how='all')
    df[cols] = df[cols].fillna(0)
    
    # Calculate totals
    total_volumes = df[cols].sum()
    print("\n--- Total Volume (Since Jan 2025, in USDT) ---")
    for col in cols:
        print(f"{col:20s}: ${total_volumes[col]:,.2f}")
        
    df['total_vol'] = df[cols].sum(axis=1)
    df = df.sort_values('total_vol', ascending=False)
    
    # Market share per symbol
    for col in cols:
        df[f"{col}_share"] = df[col] / df['total_vol']
        
    print("\n--- Market Share Distribution (Averages across symbols) ---")
    for col in cols:
        mean_share = df[df['total_vol'] > 0][f"{col}_share"].mean()
        print(f"{col:20s}: {mean_share:.1%}")
        
    print("\n--- Top 10 by Volume ---")
    print(df[['symbol', 'total_vol', 'binance_futures_share', 'bybit_futures_share', 'binance_spot_share', 'bybit_spot_share']].head(10).to_string())

if __name__ == "__main__":
    main()
