import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor
import time

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake"
EXCHANGE = "binance"
SYMBOLS = ["SOLUSDT", "SUIUSDT"]
DATES = ["2025-07-01", "2025-07-02", "2025-07-03"]

def process_file(file_path):
    """Read a single trade file and return basic statistics and a sample of trade sizes."""
    try:
        # We only need quote_qty to profile trade sizes in USD terms
        df = pd.read_csv(file_path, usecols=['quote_qty'])
        
        # Calculate stats
        total_volume = df['quote_qty'].sum()
        num_trades = len(df)
        
        # Extract percentiles for this file to aggregate later or just return all values if small enough
        # Since it's ~1-2 million trades per file, returning the array is fine for 3 days.
        return {
            'file': file_path,
            'values': df['quote_qty'].values,
            'total_volume': total_volume,
            'num_trades': num_trades
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    print("Starting trade size profiling...")
    start_time = time.time()
    
    results = {}
    
    for symbol in SYMBOLS:
        print(f"\nProfiling {symbol}...")
        files_to_process = []
        for date in DATES:
            filepath = os.path.join(DATALAKE_DIR, EXCHANGE, symbol, f"{date}_trades.csv.gz")
            if os.path.exists(filepath):
                files_to_process.append(filepath)
            else:
                print(f"File not found: {filepath}")
                
        if not files_to_process:
            continue
            
        all_values = []
        total_vol = 0
        total_trades = 0
        
        # Process files in parallel
        with ProcessPoolExecutor() as executor:
            file_results = list(executor.map(process_file, files_to_process))
            
        for res in file_results:
            if res:
                all_values.append(res['values'])
                total_vol += res['total_volume']
                total_trades += res['num_trades']
                
        if all_values:
            combined_values = np.concatenate(all_values)
            
            percentiles = [10, 25, 50, 75, 90, 95, 99, 99.9, 99.99]
            calc_percentiles = np.percentile(combined_values, percentiles)
            
            print(f"--- Statistics for {symbol} ---")
            print(f"Total Trades (3 days): {total_trades:,.0f}")
            print(f"Total Volume (3 days): ${total_vol:,.2f}")
            print(f"Mean Trade Size: ${total_vol/total_trades:,.2f}")
            print("\nPercentiles (USD):")
            for p, val in zip(percentiles, calc_percentiles):
                print(f"  {p}th Percentile: ${val:,.2f}")
                
            # Calculate volume contribution
            # How much total volume comes from trades > $10k, $50k, $100k?
            thresholds = [1000, 10000, 50000, 100000]
            print("\nVolume Contribution by Trade Size:")
            for t in thresholds:
                mask = combined_values >= t
                vol_above_t = combined_values[mask].sum()
                pct_vol = (vol_above_t / total_vol) * 100
                pct_trades = (mask.sum() / total_trades) * 100
                print(f"  > ${t:,}: {pct_vol:.1f}% of Volume ({pct_trades:.3f}% of Trades)")
                
            # Retail Contribution
            retail_mask = combined_values <= 1000
            retail_vol = combined_values[retail_mask].sum()
            retail_pct_vol = (retail_vol / total_vol) * 100
            retail_pct_trades = (retail_mask.sum() / total_trades) * 100
            print(f"  <= $1,000 (Retail): {retail_pct_vol:.1f}% of Volume ({retail_pct_trades:.1f}% of Trades)")

    print(f"\nCompleted in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
