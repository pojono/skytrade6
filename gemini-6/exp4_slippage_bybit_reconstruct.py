import pandas as pd
import numpy as np
import os
import json
import gzip
from concurrent.futures import ProcessPoolExecutor

FEAT_DIR = "/home/ubuntu/Projects/skytrade6/gemini-6/features"
DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake"

def simulate_market_buy(asks_dict, usd_size):
    if not asks_dict: return None, None
    
    # asks_dict is {price_float: qty_float}
    # Sort asks ascending
    sorted_asks = sorted(asks_dict.items(), key=lambda x: x[0])
    if not sorted_asks: return None, None
    
    best_ask = sorted_asks[0][0]
    remaining_usd = usd_size
    total_qty = 0
    total_usd_spent = 0
    
    for price, qty in sorted_asks:
        if qty <= 0: continue
        level_usd = price * qty
        
        if remaining_usd <= level_usd:
            fraction_qty = remaining_usd / price
            total_qty += fraction_qty
            total_usd_spent += remaining_usd
            remaining_usd = 0
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
        
        bids_book = {}
        asks_book = {}
        
        with gzip.open(ob_file, 'rt', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    msg = json.loads(line)
                    data = msg.get('data', {})
                    msg_type = msg.get('type')
                    cts = msg.get('cts', 0)
                    
                    if cts > target_ms:
                        break # We've reached our target state
                        
                    if msg_type == 'snapshot':
                        bids_book = {float(p): float(q) for p, q in data.get('b', [])}
                        asks_book = {float(p): float(q) for p, q in data.get('a', [])}
                    elif msg_type == 'delta':
                        for p, q in data.get('b', []):
                            pf, qf = float(p), float(q)
                            if qf == 0:
                                bids_book.pop(pf, None)
                            else:
                                bids_book[pf] = qf
                        for p, q in data.get('a', []):
                            pf, qf = float(p), float(q)
                            if qf == 0:
                                asks_book.pop(pf, None)
                            else:
                                asks_book[pf] = qf
                except:
                    continue
                    
        # Now we have the reconstructed book at target_ms
        if bids_book and asks_book:
            # Clean up 0s just in case
            bids_book = {p: q for p, q in bids_book.items() if q > 0}
            asks_book = {p: q for p, q in asks_book.items() if q > 0}
            
            if bids_book and asks_book:
                best_bid = max(bids_book.keys())
                best_ask = min(asks_book.keys())
                
                if best_bid > 0 and best_ask > best_bid:
                    spread_bps = ((best_ask - best_bid) / best_bid) * 10000
                    res['spread_bps'] = spread_bps
                    
                    vwap_10, slip_10 = simulate_market_buy(asks_book, 10000)
                    vwap_50, slip_50 = simulate_market_buy(asks_book, 50000)
                    vwap_100, slip_100 = simulate_market_buy(asks_book, 100000)
                    
                    if slip_10 is not None: res['slip_10k_bps'] = slip_10 * 10000
                    if slip_50 is not None: res['slip_50k_bps'] = slip_50 * 10000
                    if slip_100 is not None: res['slip_100k_bps'] = slip_100 * 10000
                
    except Exception as e:
        pass
        
    return res

if __name__ == "__main__":
    df_events = pd.read_csv("bybit_test_events.csv")
    print(f"Reconstructing Bybit Orderbooks for {len(df_events)} triggers (post Aug 21, 2025)...")
    
    ob_results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        ob_results = list(executor.map(calculate_ob_slippage, [row for _, row in df_events.iterrows()]))
        
    df_ob = pd.DataFrame(ob_results)
    df_final = pd.concat([df_events, df_ob], axis=1)
    
    df_final = df_final.dropna(subset=['spread_bps'])
    
    print(f"\nSuccessfully matched and reconstructed {len(df_final)} exact L2 orderbook snapshots on Bybit.")
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
    
    print("\n--- Total Cost of Execution on Bybit (Taker Fee + Slippage) ---")
    print("Assuming Bybit VIP 0 fees: 5.5 bps entry and 5.5 bps exit (11 bps total round-trip).")
    print(f"Total Execution Drag for $10k clip: 11 bps fees + {avg_slip_10:.1f} bps slip = {11 + avg_slip_10:.1f} bps ({((11 + avg_slip_10)/10000)*100:.2f}%)")
    print(f"Total Execution Drag for $50k clip: 11 bps fees + {avg_slip_50:.1f} bps slip = {11 + avg_slip_50:.1f} bps ({((11 + avg_slip_50)/10000)*100:.2f}%)")
    print(f"Total Execution Drag for $100k clip: 11 bps fees + {avg_slip_100:.1f} bps slip = {11 + avg_slip_100:.1f} bps ({((11 + avg_slip_100)/10000)*100:.2f}%)")
    
    print("\nDoes the Alpha survive this?")
    print("Our average dynamic exit edge is +1.67% (167 bps).")
    print(f"Net Alpha remaining at $10k size: {167 - (11 + avg_slip_10):.1f} bps (+{(167 - (11 + avg_slip_10))/100:.2f}%)")
    print(f"Net Alpha remaining at $50k size: {167 - (11 + avg_slip_50):.1f} bps (+{(167 - (11 + avg_slip_50))/100:.2f}%)")
    print(f"Net Alpha remaining at $100k size: {167 - (11 + avg_slip_100):.1f} bps (+{(167 - (11 + avg_slip_100))/100:.2f}%)")
