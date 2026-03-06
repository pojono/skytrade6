import pandas as pd
import numpy as np
import os
import gzip
import json
from datetime import datetime

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake"

def analyze_orderbook_absorption(symbol, target_date, target_time_str):
    """
    Reads the ob200 and tick data around a specific micro-event.
    """
    print(f"\n--- Analyzing Orderbook Absorption for {symbol} on {target_date} at {target_time_str} ---")
    
    # 1. Load the exact orderbook data around this minute
    target_time = pd.to_datetime(target_time_str)
    start_time = target_time - pd.Timedelta(seconds=30)
    end_time = target_time + pd.Timedelta(minutes=5)
    
    ob_file = os.path.join(DATALAKE_DIR, "bybit", symbol, f"{target_date}_orderbook.jsonl.gz")
    
    if not os.path.exists(ob_file):
        print(f"No orderbook file found at {ob_file}")
        return
        
    print("Parsing L2 Orderbook data...")
    ob_data = []
    
    try:
        with gzip.open(ob_file, 'rt', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                
                # 'cts' is creation timestamp in milliseconds
                ts = pd.to_datetime(data.get('cts', 0), unit='ms')
                
                if ts >= start_time and ts <= end_time:
                    # Bybit ob200 format typically has 'b' (bids) and 'a' (asks)
                    bids = data.get('b', [])
                    asks = data.get('a', [])
                    
                    # Calculate total volume in the top 20 levels
                    bid_vol = sum(float(qty) for price, qty in bids[:20]) if bids else 0
                    ask_vol = sum(float(qty) for price, qty in asks[:20]) if asks else 0
                    
                    best_bid = float(bids[0][0]) if bids else None
                    best_ask = float(asks[0][0]) if asks else None
                    
                    ob_data.append({
                        'time': ts,
                        'best_bid': best_bid,
                        'best_ask': best_ask,
                        'top20_bid_vol': bid_vol,
                        'top20_ask_vol': ask_vol,
                        'imbalance': (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0
                    })
                elif ts > end_time:
                    break
    except Exception as e:
        print(f"Error reading OB: {e}")
        
    df_ob = pd.DataFrame(ob_data)
    if len(df_ob) == 0:
        print("No orderbook data in the target window.")
        return
        
    df_ob.set_index('time', inplace=True)
    
    # Analyze the orderbook dynamic
    print("\nOrderbook Dynamics (30s before -> 5m after event):")
    
    # Let's track how the Ask Wall changes right at the event
    pre_event_asks = df_ob[df_ob.index < target_time]['top20_ask_vol'].mean()
    post_event_asks = df_ob[(df_ob.index >= target_time) & (df_ob.index < target_time + pd.Timedelta(seconds=30))]['top20_ask_vol'].mean()
    
    print(f"Pre-Event Top20 Ask Volume: {pre_event_asks:.2f} coins")
    print(f"Post-Event (30s) Top20 Ask Volume: {post_event_asks:.2f} coins")
    
    if post_event_asks < pre_event_asks * 0.7:
        print(">>> MASSIVE ASK ABSORPTION DETECTED (>30% drop). The market maker pulled liquidity.")
    elif post_event_asks > pre_event_asks * 1.3:
        print(">>> ASK WALL REPLENISHED (>30% jump). The market maker is suppressing the price.")
    else:
        print(">>> Orderbook stable. No immediate structural shift.")
        
    # Print a quick 10-second aggregate path of the imbalance
    df_ob['imbalance'].resample('10S').mean().dropna().apply(lambda x: print(f"Imbalance: {x:+.2f}")).values

if __name__ == "__main__":
    # We found an event on SOLUSDT on July 3, 2025 at 05:43:00 where it was a winner (+0.15%)
    # Let's inspect the orderbook at exactly that moment.
    analyze_orderbook_absorption("SOLUSDT", "2025-07-03", "2025-07-03 05:43:00")
    
    # And another event on July 6, 2025 at 17:18:00 where it failed (-0.22%)
    analyze_orderbook_absorption("SOLUSDT", "2025-07-06", "2025-07-06 17:18:00")
