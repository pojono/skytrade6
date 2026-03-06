import pandas as pd
import numpy as np
import os
from datetime import datetime

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake"

def analyze_binance_bookdepth(symbol, target_date, target_time_str):
    """
    Reads the _bookDepth.csv.gz from Binance.
    Format: timestamp,percentage,depth,notional
    We need to aggregate this to understand liquidity near the event.
    """
    print(f"\n--- Analyzing Orderbook Absorption for {symbol} on {target_date} at {target_time_str} ---")
    
    target_time = pd.to_datetime(target_time_str)
    start_time = target_time - pd.Timedelta(minutes=1)
    end_time = target_time + pd.Timedelta(minutes=5)
    
    ob_file = os.path.join(DATALAKE_DIR, "binance", symbol, f"{target_date}_bookDepth.csv.gz")
    
    if not os.path.exists(ob_file):
        print(f"No orderbook file found at {ob_file}")
        return
        
    print("Parsing Binance BookDepth data...")
    try:
        # Columns: timestamp,percentage,depth,notional
        # Percentage is the distance from the mid price: 
        # negative % = bids (below mid), positive % = asks (above mid)
        df_ob = pd.read_csv(ob_file)
        df_ob['timestamp'] = pd.to_datetime(df_ob['timestamp'])
        
        # Filter to our time window
        df_ob = df_ob[(df_ob['timestamp'] >= start_time) & (df_ob['timestamp'] <= end_time)]
        
        if len(df_ob) == 0:
            print("No orderbook data in the target window.")
            return
            
        # We want to measure the total Ask Notional (positive percentage) vs Bid Notional (negative percentage)
        # Let's sum the notional up to 5% depth
        
        asks = df_ob[df_ob['percentage'] > 0]
        bids = df_ob[df_ob['percentage'] < 0]
        
        ask_agg = asks.groupby('timestamp')['notional'].sum()
        bid_agg = bids.groupby('timestamp')['notional'].sum()
        
        df_agg = pd.DataFrame({'ask_vol': ask_agg, 'bid_vol': bid_agg}).fillna(0)
        
        pre_event_asks = df_agg[df_agg.index < target_time]['ask_vol'].mean()
        post_event_asks = df_agg[(df_agg.index >= target_time) & (df_agg.index < target_time + pd.Timedelta(minutes=1))]['ask_vol'].mean()
        
        print(f"Pre-Event Ask Volume (0-5%): ${pre_event_asks:,.0f}")
        print(f"Post-Event Ask Volume (0-5%): ${post_event_asks:,.0f}")
        
        if pd.isna(pre_event_asks) or pd.isna(post_event_asks) or pre_event_asks == 0:
            print("Not enough data to calculate absorption.")
            return

        change = (post_event_asks - pre_event_asks) / pre_event_asks
        print(f"Change in Ask Liquidity: {change*100:+.2f}%")
        
        if change < -0.15:
            print(">>> SIGNIFICANT ASK ABSORPTION DETECTED (>15% drop). Market maker pulled ask liquidity.")
        elif change > 0.15:
            print(">>> ASK WALL REPLENISHED (>15% jump). Market maker suppressing price.")
        else:
            print(">>> Orderbook stable.")
            
    except Exception as e:
        print(f"Error reading OB: {e}")

if __name__ == "__main__":
    # Test on the events we found in the dual spot-futures analysis
    # Event at 2025-07-03 05:43:00 (Winner)
    analyze_binance_bookdepth("SOLUSDT", "2025-07-03", "2025-07-03 05:43:00")
    
    # Event at 2025-07-06 17:18:00 (Loser)
    analyze_binance_bookdepth("SOLUSDT", "2025-07-06", "2025-07-06 17:18:00")
