import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake"

# Let's test on SOLUSDT for a specific day we know had an event (from our earlier tests, maybe July 2nd or 3rd)
# We will just scan July 1st to July 7th 2025 to find an exact divergence, then zoom into the tick data for Spot vs Futures.

SYMBOL = "SOLUSDT"
DATES = [f"2025-07-{str(i).zfill(2)}" for i in range(1, 8)]

def load_day_ticks(date, exchange, market_type="futures"):
    """
    Loads raw ticks for a specific market type.
    market_type: 'futures' or 'spot'
    """
    if exchange == "binance":
        if market_type == "futures":
            filepath = os.path.join(DATALAKE_DIR, exchange, SYMBOL, f"{date}_trades.csv.gz")
            if not os.path.exists(filepath): return None
            # id, price, qty, quote_qty, time, is_buyer_maker
            df = pd.read_csv(filepath)
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            
        elif market_type == "spot":
            filepath = os.path.join(DATALAKE_DIR, exchange, SYMBOL, f"{date}_trades_spot.csv.gz")
            if not os.path.exists(filepath): return None
            # Spot files don't have headers in the first line for Binance sometimes, let's check
            try:
                df = pd.read_csv(filepath, names=['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker', 'is_best_match'])
            except:
                df = pd.read_csv(filepath)
            
            # Spot time is sometimes in microseconds instead of milliseconds. Let's auto-detect.
            if df['time'].iloc[0] > 1e14: 
                # Probably microseconds
                df['time'] = pd.to_datetime(df['time'], unit='us')
            else:
                df['time'] = pd.to_datetime(df['time'], unit='ms')
                
        df['direction'] = np.where(df['is_buyer_maker'] == True, -1, 1) # True = Seller is taker (Sell Order)
        df['signed_qty'] = df['quote_qty'] * df['direction']
        
        # Add market type tag
        df['market'] = market_type
        
        return df

def analyze_spot_futures_flow():
    for date in DATES:
        print(f"\nLoading {date}...")
        df_fut = load_day_ticks(date, "binance", "futures")
        df_spot = load_day_ticks(date, "binance", "spot")
        
        if df_fut is None or df_spot is None:
            print(f"Missing data for {date}")
            continue
            
        # Get dynamic thresholds for futures (Whale = 98th percentile, Retail = 20th)
        whale_thresh = df_fut['quote_qty'].quantile(0.98)
        retail_thresh = df_fut['quote_qty'].quantile(0.20)
        
        print(f"Futures Whale Threshold: ${whale_thresh:,.2f} | Retail: ${retail_thresh:,.2f}")
        
        df_fut['is_whale'] = df_fut['quote_qty'] >= whale_thresh
        df_fut['is_retail'] = df_fut['quote_qty'] <= retail_thresh
        
        # Do the same for spot
        spot_whale_thresh = df_spot['quote_qty'].quantile(0.98)
        spot_retail_thresh = df_spot['quote_qty'].quantile(0.20)
        
        print(f"Spot Whale Threshold: ${spot_whale_thresh:,.2f} | Retail: ${spot_retail_thresh:,.2f}")
        
        df_spot['is_whale'] = df_spot['quote_qty'] >= spot_whale_thresh
        df_spot['is_retail'] = df_spot['quote_qty'] <= spot_retail_thresh
        
        # Resample to 1-minute bins to find divergence
        df_fut.set_index('time', inplace=True)
        df_spot.set_index('time', inplace=True)
        
        fut_whale_cvd = df_fut.loc[df_fut['is_whale'], 'signed_qty'].resample('1min').sum().fillna(0)
        fut_retail_cvd = df_fut.loc[df_fut['is_retail'], 'signed_qty'].resample('1min').sum().fillna(0)
        
        spot_whale_cvd = df_spot.loc[df_spot['is_whale'], 'signed_qty'].resample('1min').sum().fillna(0)
        spot_retail_cvd = df_spot.loc[df_spot['is_retail'], 'signed_qty'].resample('1min').sum().fillna(0)
        
        price = df_fut['price'].resample('1min').last().ffill()
        
        df_agg = pd.DataFrame({
            'price': price,
            'fut_whale': fut_whale_cvd,
            'fut_retail': fut_retail_cvd,
            'spot_whale': spot_whale_cvd,
            'spot_retail': spot_retail_cvd
        }).fillna(0)
        
        # Find exactly when Futures Retail is Selling AND Futures Whales are Buying (Bullish Div)
        rolling_window = 4 * 60
        df_agg['roll_fut_whale'] = df_agg['fut_whale'].rolling(rolling_window).sum()
        df_agg['roll_fut_retail'] = df_agg['fut_retail'].rolling(rolling_window).sum()
        
        # Find 1-minute spikes where Spot Whales step in aggressively during a Futures Divergence
        # This tests the Lead-Lag thesis: Futures diverges, but Spot triggers the actual reversal.
        
        # Calculate percentiles for the day to keep it simple for this diagnostic
        fut_whale_high = df_agg['roll_fut_whale'].quantile(0.90)
        fut_retail_low = df_agg['roll_fut_retail'].quantile(0.10)
        
        # Find 1-minute spot volume spikes
        spot_whale_spike = df_agg['spot_whale'].quantile(0.95)
        
        events = df_agg[
            (df_agg['roll_fut_retail'] < fut_retail_low) & 
            (df_agg['roll_fut_whale'] > fut_whale_high) &
            (df_agg['spot_whale'] > spot_whale_spike)
        ]
        
        if len(events) > 0:
            print(f"Found {len(events)} Dual Spot-Futures Triggers on {date}!")
            # Calculate forward return for these precise events
            for idx, row in events.iterrows():
                try:
                    fwd_price = df_agg.loc[idx + pd.Timedelta(minutes=60), 'price']
                    ret = (fwd_price / row['price']) - 1
                    print(f"  Event at {idx}: Price {row['price']:.2f}, 1h Fwd Ret: {ret*100:.2f}%")
                    print(f"    - Fut Whale Flow (4h): ${row['roll_fut_whale']:,.0f}")
                    print(f"    - Fut Retail Flow (4h): ${row['roll_fut_retail']:,.0f}")
                    print(f"    - SPOT Whale Buy (1m Spike): ${row['spot_whale']:,.0f}")
                except KeyError:
                    pass

if __name__ == "__main__":
    analyze_spot_futures_flow()
