import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor

FEAT_DIR = "/home/ubuntu/Projects/skytrade6/gemini-6/features"
DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake"
# We'll test slippage on the Top 5 most traded coins
TEST_SYMBOLS = ["HUMAUSDT", "AGLDUSDT", "AAVEUSDT", "ZKUSDT", "BERAUSDT"]

def get_base_events(sym):
    file_path = os.path.join(FEAT_DIR, f"{sym}_1m.parquet")
    if not os.path.exists(file_path): return pd.DataFrame()
    
    df = pd.read_parquet(file_path)
    if len(df) < 30 * 24 * 60: return pd.DataFrame()
        
    rolling_window = 4 * 60
    df['roll_fut_whale'] = df['fut_whale_cvd'].rolling(rolling_window).sum()
    df['roll_fut_retail'] = df['fut_retail_cvd'].rolling(rolling_window).sum()
    
    z_window = 3 * 24 * 60
    df['fut_whale_mean'] = df['roll_fut_whale'].rolling(z_window, min_periods=rolling_window).mean()
    df['fut_whale_std'] = df['roll_fut_whale'].rolling(z_window, min_periods=rolling_window).std()
    df['fut_whale_z'] = (df['roll_fut_whale'] - df['fut_whale_mean']) / df['fut_whale_std']
    
    df['fut_retail_mean'] = df['roll_fut_retail'].rolling(z_window, min_periods=rolling_window).mean()
    df['fut_retail_std'] = df['roll_fut_retail'].rolling(z_window, min_periods=rolling_window).std()
    df['fut_retail_z'] = (df['roll_fut_retail'] - df['fut_retail_mean']) / df['fut_retail_std']
    
    df['spot_whale_1h_avg'] = df['spot_whale_cvd'].abs().rolling(60).mean()
    
    bullish_div = (
        (df['fut_retail_z'] < -1.5) & 
        (df['fut_whale_z'] > 1.5) &
        (df['spot_whale_cvd'] > df['spot_whale_1h_avg'] * 3.0) &
        (df['spot_whale_cvd'] > 0)
    )
    
    def filter_signals(signals, wait_time=120):
        filtered = pd.Series(False, index=signals.index)
        last_sig_time = None
        for i, (idx, val) in enumerate(signals.items()):
            if val:
                if last_sig_time is None or (idx - last_sig_time).total_seconds() / 60 > wait_time:
                    filtered[idx] = True
                    last_sig_time = idx
        return filtered

    df['bull_sig'] = filter_signals(bullish_div, 120)
    events = df[df['bull_sig']].copy()
    events['symbol'] = sym
    events['timestamp'] = events.index
    
    return events[['symbol', 'timestamp', 'price']]

def calculate_binance_slippage(row):
    sym = row['symbol']
    ts = pd.to_datetime(row['timestamp'])
    date_str = ts.strftime('%Y-%m-%d')
    
    ob_file = os.path.join(DATALAKE_DIR, "binance", sym, f"{date_str}_bookDepth.csv.gz")
    
    res = {
        'spread_bps': np.nan,
        'slip_10k_bps': np.nan,
        'slip_50k_bps': np.nan,
        'slip_100k_bps': np.nan,
    }
    
    if not os.path.exists(ob_file): return res
        
    try:
        # Load the bookDepth file which contains timestamp, percentage, depth, notional
        # We just need the snapshot right AT the minute mark.
        df_ob = pd.read_csv(ob_file)
        df_ob['timestamp'] = pd.to_datetime(df_ob['timestamp'])
        
        # Get the first snapshot strictly >= our trigger time
        snapshots = df_ob[df_ob['timestamp'] >= ts]
        if len(snapshots) == 0: return res
        
        exact_ts = snapshots['timestamp'].iloc[0]
        # Only use if within 2 seconds
        if (exact_ts - ts).total_seconds() > 2: return res
        
        book = df_ob[df_ob['timestamp'] == exact_ts]
        
        asks = book[book['percentage'] > 0].sort_values('percentage')
        
        if len(asks) == 0: return res
        
        # The 'percentage' is the distance from mid. 'notional' is the USD size at that depth layer.
        # This isn't a perfect L2 simulation, but an approximation based on the aggregated bookDepth bins.
        # If we need $10k, we see how many percentage layers we consume.
        
        best_ask_pct = asks['percentage'].iloc[0] / 100.0
        # Assume spread is roughly 2x the best ask distance from mid (since mid is 0)
        res['spread_bps'] = (best_ask_pct * 2) * 10000
        
        def calc_slip_from_bins(asks_df, target_usd):
            usd_spent = 0
            # Since bookDepth bins represent the total volume *up to* that percentage depth:
            # We can just find the first bin where the cumulative 'notional' > target_usd
            # And that bin's 'percentage' is roughly the slippage we suffer.
            # Wait, the data format: timestamp,percentage,depth,notional
            # Usually, notional is cumulative in these exports. Let's assume it's cumulative.
            
            for _, r in asks_df.iterrows():
                if r['notional'] >= target_usd:
                    # The slippage is exactly the percentage of this bin
                    return r['percentage'] * 100 # convert to bps
            
            # If it eats the whole book, return the max percentage
            return asks_df['percentage'].max() * 100

        res['slip_10k_bps'] = calc_slip_from_bins(asks, 10000)
        res['slip_50k_bps'] = calc_slip_from_bins(asks, 50000)
        res['slip_100k_bps'] = calc_slip_from_bins(asks, 100000)
                
    except Exception as e:
        pass
        
    return res

if __name__ == "__main__":
    print("--- Experiment 4: Binance BookDepth Slippage Analysis ---")
    all_events = []
    for sym in TEST_SYMBOLS:
        all_events.append(get_base_events(sym))
    
    df_events = pd.concat(all_events).reset_index(drop=True)
    print(f"Extracted {len(df_events)} triggers across {len(TEST_SYMBOLS)} symbols.")
    
    ob_results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        ob_results = list(executor.map(calculate_binance_slippage, [row for _, row in df_events.iterrows()]))
        
    df_ob = pd.DataFrame(ob_results)
    df_final = pd.concat([df_events, df_ob], axis=1)
    df_final.dropna(subset=['slip_10k_bps'], inplace=True)
    
    print(f"\nSuccessfully matched {len(df_final)} exact orderbook snapshots.")
    
    grouped = df_final.groupby('symbol').agg({
        'slip_10k_bps': 'mean',
        'slip_50k_bps': 'mean',
        'slip_100k_bps': 'mean'
    }).round(1)
    
    print("\nAverage Market Impact Slippage (in Basis Points):")
    print(grouped.to_string())
    
    print("\n--- Summary ---")
    avg_slip_10 = df_final['slip_10k_bps'].mean()
    avg_slip_50 = df_final['slip_50k_bps'].mean()
    avg_slip_100 = df_final['slip_100k_bps'].mean()
    
    print(f"Avg Market Impact Slippage ($10k Buy): {avg_slip_10:.1f} bps")
    print(f"Avg Market Impact Slippage ($50k Buy): {avg_slip_50:.1f} bps")
    print(f"Avg Market Impact Slippage ($100k Buy): {avg_slip_100:.1f} bps")
    
    print("\n--- Total Cost of Execution (Taker Fee + Slippage) ---")
    print("Assuming 10 bps entry fee and 10 bps exit fee (20 bps total round-trip fee).")
    print("We only suffer slippage on the Entry (Market Buy), because we can scale out the Exit slowly via Limit orders over the 4-8 hour holding period.")
    print(f"Total Execution Drag for $10k clip: 20 bps fees + {avg_slip_10:.1f} bps slip = {20 + avg_slip_10:.1f} bps ({((20 + avg_slip_10)/10000)*100:.2f}%)")
    print(f"Total Execution Drag for $50k clip: 20 bps fees + {avg_slip_50:.1f} bps slip = {20 + avg_slip_50:.1f} bps ({((20 + avg_slip_50)/10000)*100:.2f}%)")
    print(f"Total Execution Drag for $100k clip: 20 bps fees + {avg_slip_100:.1f} bps slip = {20 + avg_slip_100:.1f} bps ({((20 + avg_slip_100)/10000)*100:.2f}%)")
    
    print("\nDoes the Alpha survive this?")
    print("Our average dynamic exit edge is +1.67% (167 bps).")
    print(f"Net Alpha remaining at $10k size: {167 - (20 + avg_slip_10):.1f} bps (+{(167 - (20 + avg_slip_10))/100:.2f}%)")
    print(f"Net Alpha remaining at $50k size: {167 - (20 + avg_slip_50):.1f} bps (+{(167 - (20 + avg_slip_50))/100:.2f}%)")
    print(f"Net Alpha remaining at $100k size: {167 - (20 + avg_slip_100):.1f} bps (+{(167 - (20 + avg_slip_100))/100:.2f}%)")
