import pandas as pd
import numpy as np
import os
import json
import gzip
from concurrent.futures import ProcessPoolExecutor

FEAT_DIR = "/home/ubuntu/Projects/skytrade6/gemini-6/features"
DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake"
TEST_SYMBOLS = ["HUMAUSDT", "AGLDUSDT", "AAVEUSDT", "ZKUSDT", "BERAUSDT"]

def simulate_market_buy(asks, usd_size):
    if not asks: return None, None
    best_ask = float(asks[0][0])
    remaining_usd = usd_size
    total_qty = 0
    total_usd_spent = 0
    
    for price_str, qty_str in asks:
        price = float(price_str)
        qty = float(qty_str)
        level_usd = price * qty
        
        if remaining_usd <= level_usd:
            fraction_qty = remaining_usd / price
            total_qty += fraction_qty
            total_usd_spent += remaining_usd
            break
        else:
            total_qty += qty
            total_usd_spent += level_usd
            remaining_usd -= level_usd
            
    if total_qty == 0: return None, None
    if remaining_usd > 0: 
        vwap = total_usd_spent / total_qty
        return vwap, (vwap / best_ask) - 1
        
    vwap = total_usd_spent / total_qty
    slippage_pct = (vwap / best_ask) - 1
    
    return vwap, slippage_pct

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
    
    # NOW filter by date, so we only look for Bybit orderbooks after they exist
    events = events[events.index >= '2025-08-21']
    
    if len(events) == 0: return pd.DataFrame()
    events['symbol'] = sym
    events['timestamp'] = events.index
    
    return events[['symbol', 'timestamp', 'price']]

def calculate_ob_slippage(row):
    sym = row['symbol']
    ts = pd.to_datetime(row['timestamp'])
    date_str = ts.strftime('%Y-%m-%d')
    
    ob_file = os.path.join(DATALAKE_DIR, "bybit", sym, f"{date_str}_orderbook.jsonl.gz")
    
    res = {
        'spread_bps': np.nan,
        'slip_10k_bps': np.nan,
        'slip_50k_bps': np.nan,
        'slip_100k_bps': np.nan,
    }
    
    if not os.path.exists(ob_file): return res
        
    try:
        target_ms = int(ts.timestamp() * 1000)
        closest_data = None
        min_diff = float('inf')
        
        with gzip.open(ob_file, 'rt', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                
                cts = data.get('cts', 0)
                diff = abs(cts - target_ms)
                
                if diff < 2000 and diff < min_diff:
                    min_diff = diff
                    closest_data = data
                
                if cts - target_ms > 5000:
                    break
                    
        if closest_data:
            bids = closest_data.get('b', [])
            asks = closest_data.get('a', [])
            
            if bids and asks:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                
                spread_bps = ((best_ask - best_bid) / best_bid) * 10000
                res['spread_bps'] = spread_bps
                
                vwap_10, slip_10 = simulate_market_buy(asks, 10000)
                vwap_50, slip_50 = simulate_market_buy(asks, 50000)
                vwap_100, slip_100 = simulate_market_buy(asks, 100000)
                
                if slip_10 is not None: res['slip_10k_bps'] = slip_10 * 10000
                if slip_50 is not None: res['slip_50k_bps'] = slip_50 * 10000
                if slip_100 is not None: res['slip_100k_bps'] = slip_100 * 10000
                
    except Exception as e:
        pass
        
    return res

if __name__ == "__main__":
    print("--- Experiment 4b: Bybit L2 Orderbook Slippage Analysis ---")
    all_events = []
    for sym in TEST_SYMBOLS:
        ev = get_base_events(sym)
        if len(ev) > 0:
            all_events.append(ev)
    
    if not all_events:
        print("No events found.")
        exit(0)
        
    df_events = pd.concat(all_events).reset_index(drop=True)
    print(f"Extracted {len(df_events)} triggers (post Aug 21, 2025) across {len(TEST_SYMBOLS)} symbols.")
    
    ob_results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        ob_results = list(executor.map(calculate_ob_slippage, [row for _, row in df_events.iterrows()]))
        
    df_ob = pd.DataFrame(ob_results)
    df_final = pd.concat([df_events, df_ob], axis=1)
    
    # Drop rows where spread_bps is nan
    df_final = df_final.dropna(subset=['spread_bps'])
    
    print(f"\nSuccessfully matched {len(df_final)} exact L2 orderbook snapshots on Bybit.")
    if len(df_final) == 0: exit(0)
    
    grouped = df_final.groupby('symbol').agg({
        'spread_bps': 'mean',
        'slip_10k_bps': 'mean',
        'slip_50k_bps': 'mean',
        'slip_100k_bps': 'mean'
    }).round(1)
    
    print("\nAverage Spread and Slippage on Bybit (in Basis Points):")
    print(grouped.to_string())
    
    print("\n--- Summary ---")
    avg_spread = df_final['spread_bps'].mean()
    avg_slip_10 = df_final['slip_10k_bps'].mean()
    avg_slip_50 = df_final['slip_50k_bps'].mean()
    avg_slip_100 = df_final['slip_100k_bps'].mean()
    
    print(f"Avg Bid-Ask Spread at Trigger: {avg_spread:.1f} bps")
    print(f"Avg Market Impact Slippage ($10k Buy): {avg_slip_10:.1f} bps")
    print(f"Avg Market Impact Slippage ($50k Buy): {avg_slip_50:.1f} bps")
    print(f"Avg Market Impact Slippage ($100k Buy): {avg_slip_100:.1f} bps")
    
    print("\n--- Total Cost of Execution (Taker Fee + Slippage) ---")
    print("Assuming Bybit VIP 0 fees: 5.5 bps entry and 5.5 bps exit (11 bps total round-trip).")
    print(f"Total Execution Drag for $10k clip: 11 bps fees + {avg_slip_10:.1f} bps slip = {11 + avg_slip_10:.1f} bps ({((11 + avg_slip_10)/10000)*100:.2f}%)")
    print(f"Total Execution Drag for $50k clip: 11 bps fees + {avg_slip_50:.1f} bps slip = {11 + avg_slip_50:.1f} bps ({((11 + avg_slip_50)/10000)*100:.2f}%)")
    print(f"Total Execution Drag for $100k clip: 11 bps fees + {avg_slip_100:.1f} bps slip = {11 + avg_slip_100:.1f} bps ({((11 + avg_slip_100)/10000)*100:.2f}%)")
    
    print("\nDoes the Alpha survive this?")
    print("Our average dynamic exit edge is +1.67% (167 bps).")
    print(f"Net Alpha remaining at $10k size: {167 - (11 + avg_slip_10):.1f} bps (+{(167 - (11 + avg_slip_10))/100:.2f}%)")
    print(f"Net Alpha remaining at $50k size: {167 - (11 + avg_slip_50):.1f} bps (+{(167 - (11 + avg_slip_50))/100:.2f}%)")
    print(f"Net Alpha remaining at $100k size: {167 - (11 + avg_slip_100):.1f} bps (+{(167 - (11 + avg_slip_100))/100:.2f}%)")
