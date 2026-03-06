import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
import gzip
from tqdm import tqdm

DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def analyze_orderbook_skew(symbol, target_date="2026-02-24"):
    print(f"Loading orderbook data for {symbol} on {target_date}...")
    
    # Binance Spot BookDepth is in binance/BTCUSDT/2026-02-24_bookDepth.csv.gz
    # Let's check Bybit futures JSONL first, it's more comprehensive.
    ob_file = DATALAKE / f"bybit/{symbol}/{target_date}_orderbook.jsonl.gz"
    if not ob_file.exists():
        print("No orderbook data.")
        return None
        
    results = []
    
    # Read line by line, it's massive. We sample every N lines.
    count = 0
    with gzip.open(ob_file, 'rt') as f:
        for line in f:
            count += 1
            if count % 10 != 0: continue # Sample every 10th snapshot
            
            data = json.loads(line)
            ts = int(data['ts']) # ms
            
            data_inner = data.get('data', {})
            bids = data_inner.get('b', [])
            asks = data_inner.get('a', [])
            
            if not bids or not asks: continue
            
            # Sum up top 10 levels of liquidity
            bid_vol = sum([float(b[0]) * float(b[1]) for b in bids[:10]])
            ask_vol = sum([float(a[0]) * float(a[1]) for a in asks[:10]])
            
            if bid_vol + ask_vol == 0: continue
            
            skew = (bid_vol - ask_vol) / (bid_vol + ask_vol) # -1 to 1
            
            # Approximate current price from mid price
            mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2
            
            results.append({
                'timestamp': ts,
                'skew': skew,
                'mid_price': mid_price
            })
            
            if len(results) > 20000: # Limit to 20k samples per day for speed
                break
                
    df = pd.DataFrame(results)
    if len(df) == 0: return None
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.floor('1min')
    
    # Aggregate to 1 minute
    df_1m = df.groupby('timestamp').agg({
        'skew': 'mean',
        'mid_price': 'last'
    })
    
    df_1m['fwd_ret_1m'] = df_1m['mid_price'].shift(-1) / df_1m['mid_price'] - 1
    
    df_1m = df_1m.dropna()
    
    # Test if extreme bid skew predicts upward movement
    extreme_bids = df_1m[df_1m['skew'] > 0.6] # Massive buy wall
    extreme_asks = df_1m[df_1m['skew'] < -0.8] # Massive sell wall
    
    return {
        'symbol': symbol,
        'extreme_bid_events': len(extreme_bids),
        'bid_skew_fwd_ret_bps': extreme_bids['fwd_ret_1m'].mean() * 10000 if len(extreme_bids) > 0 else 0,
        
        'extreme_ask_events': len(extreme_asks),
        'ask_skew_fwd_ret_bps': extreme_asks['fwd_ret_1m'].mean() * 10000 if len(extreme_asks) > 0 else 0
    }

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'SUIUSDT', 'WLDUSDT']
    
    res_list = []
    for sym in symbols:
        r = analyze_orderbook_skew(sym)
        if r: res_list.append(r)
        
    df = pd.DataFrame(res_list)
    print("\n--- L2 Orderbook Imbalance (Top 10 Levels) vs 1-Min Forward Return ---")
    print(df.to_string(index=False))

