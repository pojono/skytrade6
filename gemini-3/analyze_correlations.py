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
        print("Datalake directories not found!")
        return []
        
    bybit_symbols = {d.name for d in bybit_dir.iterdir() if d.is_dir()}
    binance_symbols = {d.name for d in binance_dir.iterdir() if d.is_dir()}
    
    return sorted(list(bybit_symbols.intersection(binance_symbols)))

def load_close_prices(symbol):
    try:
        # Define sources with specific time and close columns
        sources = {
            'bybit_futures': (DATALAKE / "bybit" / symbol, "*_kline_1m.csv", 'startTime', 'close'),
            'bybit_spot': (DATALAKE / "bybit" / symbol, "*_kline_1m_spot.csv", 'startTime', 'close'),
            'binance_futures': (DATALAKE / "binance" / symbol, "*_kline_1m.csv", 'open_time', 'close'),
            'binance_spot': (DATALAKE / "binance" / symbol, "*_kline_1m_spot.csv", 'open_time', 'close'),
        }
        
        dfs = []
        for name, (path_dir, pattern, time_col, close_col) in sources.items():
            if not path_dir.exists():
                return None
            
            files = sorted(list(path_dir.glob(pattern)))
            # exclude mark_price, index_price, premium_index from *_kline_1m.csv
            if pattern == "*_kline_1m.csv":
                files = [f for f in files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name]
            
            # Filter for 2025 onwards to reduce data size and ensure overlapping periods
            files = [f for f in files if f.name >= "2025-01-01"]
            
            if not files:
                return None
            
            # read and concat all daily files efficiently
            daily_dfs = []
            for f in files:
                try:
                    # Binance spot often doesn't have headers
                    if name == 'binance_spot':
                        df_day = pd.read_csv(f, header=None, usecols=[0, 4], engine='c')
                        df_day.columns = [time_col, close_col]
                    else:
                        df_day = pd.read_csv(f, usecols=[time_col, close_col], engine='c')
                    daily_dfs.append(df_day)
                except Exception as e:
                    # In case headers are missing or weird
                    pass
                    
            if not daily_dfs:
                return None
                
            df = pd.concat(daily_dfs, ignore_index=True)
            
            # rename for standard processing
            df = df.rename(columns={time_col: 'timestamp', close_col: name})
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            
            # handle ms vs s vs us
            # Standardize to ms (13 digits)
            # If s (10 digits) -> multiply by 1000
            # If us (16 digits) -> divide by 1000
            if df['timestamp'].max() < 1e11:  # likely seconds
                df['timestamp'] = df['timestamp'] * 1000
            elif df['timestamp'].max() > 1e14: # likely microseconds
                df['timestamp'] = df['timestamp'] // 1000
                
            df.set_index('timestamp', inplace=True)
            df = df[~df.index.duplicated(keep='last')]
            dfs.append(df)
            
        if len(dfs) != 4:
            return None
            
        # Join all 4 price series
        merged = pd.concat(dfs, axis=1, join='inner')
        
        # Free up memory
        del dfs
        gc.collect()
        
        if len(merged) < 10000: # Need at least ~7 days of data
            return None
            
        # Calculate returns
        returns = merged.pct_change().dropna()
        
        # Calculate correlation matrix
        corr = returns.corr()
        
        # Format output
        res = {
            'symbol': symbol,
            'corr_matrix': corr.to_dict(),
            'rows': len(merged)
        }
        
        del merged
        del returns
        gc.collect()
        
        return res
    except Exception as e:
        return None

def main():
    symbols = get_common_symbols()
    print(f"Found {len(symbols)} common symbols")
    
    results = []
    
    print("Processing symbols...")
    # Limit number of parallel processes to 4 to avoid OOM
    num_processes = min(4, os.cpu_count() or 4)
    with Pool(num_processes) as pool:
        for res in tqdm(pool.imap_unordered(load_close_prices, symbols), total=len(symbols)):
            if res is not None:
                results.append(res)
                
    print(f"Successfully processed {len(results)} symbols")
    
    if not results:
        print("No valid results!")
        return
        
    # Aggregate correlations
    pairs = [
        ('binance_futures', 'binance_spot'),
        ('bybit_futures', 'bybit_spot'),
        ('binance_futures', 'bybit_futures'),
        ('binance_spot', 'bybit_spot'),
        ('binance_futures', 'bybit_spot'),
        ('bybit_futures', 'binance_spot')
    ]
    
    aggregated = {pair: [] for pair in pairs}
    
    for res in results:
        corr = res['corr_matrix']
        for k1, k2 in pairs:
            # Check if keys exist in case of weird index
            if k1 in corr and k2 in corr[k1]:
                aggregated[(k1, k2)].append({
                    'symbol': res['symbol'],
                    'corr': corr[k1][k2]
                })
    
    print("\n--- Correlation Summary ---")
    summary = []
    for pair in pairs:
        corrs = [x['corr'] for x in aggregated[pair]]
        mean_corr = np.mean(corrs)
        median_corr = np.median(corrs)
        min_corr = np.min(corrs)
        max_corr = np.max(corrs)
        
        summary.append({
            'Pair': f"{pair[0]} <-> {pair[1]}",
            'Mean': mean_corr,
            'Median': median_corr,
            'Min': min_corr,
            'Max': max_corr
        })
        print(f"\n{pair[0]} <-> {pair[1]}:")
        print(f"  Mean:   {mean_corr:.4f}")
        print(f"  Median: {median_corr:.4f}")
        print(f"  Min:    {min_corr:.4f}")
        print(f"  Max:    {max_corr:.4f}")
        
        # Min/Max symbols
        sorted_symbols = sorted(aggregated[pair], key=lambda x: x['corr'])
        lowest = ', '.join([f"{x['symbol']} ({x['corr']:.4f})" for x in sorted_symbols[:3]])
        highest = ', '.join([f"{x['symbol']} ({x['corr']:.4f})" for x in sorted_symbols[-3:]])
        print(f"  Lowest corr symbols: {lowest}")
        print(f"  Highest corr symbols: {highest}")

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv("correlation_summary.csv", index=False)
    
if __name__ == "__main__":
    main()
