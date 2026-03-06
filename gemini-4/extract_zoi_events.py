import pandas as pd
import numpy as np
import glob
import gc
import os
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings('ignore')

def load_data(symbol, exchange='bybit', timeframe='15min'):
    print(f"Loading data for {symbol}...")
    # Fix: Make sure to ONLY load the main klines, not mark price or premium index which end in the same suffix
    pattern = f"/home/ubuntu/Projects/skytrade6/datalake/{exchange}/{symbol}/202*_kline_1m.csv"
    files = glob.glob(pattern)
    
    # Filter out the mark price and premium index files explicitly
    files = [f for f in files if 'mark_price' not in f and 'premium' not in f]
    
    if not files:
        print(f"No files found for {symbol}")
        return None
    
    df_list = []
    # Only load 2025 onwards to match tick data
    for f in sorted(files):
        if "2025" in f or "2026" in f:
            try:
                df = pd.read_csv(f, usecols=['startTime', 'open', 'high', 'low', 'close'])
                df_list.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")
                pass
            
    if not df_list:
        print(f"No valid data loaded for {symbol}")
        return None
        
    df = pd.concat(df_list, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['startTime'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    return resampled

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def identify_swings(df, min_swing_pct=0.01):
    """
    Identifies market swings using a percentage-based ZigZag algorithm.
    """
    swings = []
    mode = 1 # 1 for uptrend, -1 for downtrend
    
    extreme_price = df['high'].iloc[0]
    extreme_idx = df.index[0]
    
    for idx, row in df.iterrows():
        if mode == 1:
            if row['high'] > extreme_price:
                extreme_price = row['high']
                extreme_idx = idx
            elif row['low'] < extreme_price * (1 - min_swing_pct):
                # Trend changed to down
                swings.append({'type': 'high', 'price': extreme_price, 'time': extreme_idx})
                mode = -1
                extreme_price = row['low']
                extreme_idx = idx
        else:
            if row['low'] < extreme_price:
                extreme_price = row['low']
                extreme_idx = idx
            elif row['high'] > extreme_price * (1 + min_swing_pct):
                # Trend changed to up
                swings.append({'type': 'low', 'price': extreme_price, 'time': extreme_idx})
                mode = 1
                extreme_price = row['high']
                extreme_idx = idx
                
    # Add final swing
    swings.append({'type': 'high' if mode == 1 else 'low', 'price': extreme_price, 'time': extreme_idx})
    return swings

def find_zoi_events(symbol, df, swings):
    zoi_events = []
    
    for i in range(len(swings) - 1):
        A = swings[i]
        B = swings[i+1]
        
        swing_size = abs(B['price'] - A['price'])
        if swing_size == 0 or swing_size / A['price'] < 0.01:
            continue
            
        trend = 'up' if A['type'] == 'low' else 'down'
        
        # Define the ZOI
        if trend == 'up':
            zoi_upper = B['price'] - (swing_size * 0.58)
            zoi_lower = B['price'] - (swing_size * 0.65)
            invalid_price = A['price'] # Went below origin
        else:
            zoi_lower = B['price'] + (swing_size * 0.58)
            zoi_upper = B['price'] + (swing_size * 0.65)
            invalid_price = A['price'] # Went above origin
            
        post_b_df = df.loc[B['time']:]
        
        entered_zoi = False
        zoi_time = None
        zoi_extreme = None
        
        for idx, row in post_b_df.iterrows():
            if idx == B['time']:
                continue
                
            # Check ZOI entry FIRST
            if not entered_zoi:
                if trend == 'up' and row['low'] <= zoi_upper:
                    entered_zoi = True
                    zoi_time = idx
                    zoi_extreme = row['low']
                elif trend == 'down' and row['high'] >= zoi_lower:
                    entered_zoi = True
                    zoi_time = idx
                    zoi_extreme = row['high']
                    
            # If we invalidate the setup before hitting ZOI, break
            if not entered_zoi:
                if trend == 'up' and (row['low'] <= invalid_price or row['high'] > B['price']):
                    break
                if trend == 'down' and (row['high'] >= invalid_price or row['low'] < B['price']):
                    break
            else:
                # Update extreme while in ZOI
                if trend == 'up':
                    zoi_extreme = min(zoi_extreme, row['low'])
                else:
                    zoi_extreme = max(zoi_extreme, row['high'])
                    
                # Determine success or failure after entering ZOI
                if trend == 'up':
                    if row['high'] > B['price']:
                        zoi_events.append({
                            'symbol': symbol, 'trend': trend, 'start_time': B['time'], 'zoi_time': zoi_time,
                            'swing_origin': A['price'], 'swing_extreme': B['price'], 'zoi_extreme': zoi_extreme,
                            'success': True, 'outcome_time': idx
                        })
                        break
                    elif row['low'] < invalid_price:
                        zoi_events.append({
                            'symbol': symbol, 'trend': trend, 'start_time': B['time'], 'zoi_time': zoi_time,
                            'swing_origin': A['price'], 'swing_extreme': B['price'], 'zoi_extreme': zoi_extreme,
                            'success': False, 'outcome_time': idx
                        })
                        break
                else:
                    if row['low'] < B['price']:
                        zoi_events.append({
                            'symbol': symbol, 'trend': trend, 'start_time': B['time'], 'zoi_time': zoi_time,
                            'swing_origin': A['price'], 'swing_extreme': B['price'], 'zoi_extreme': zoi_extreme,
                            'success': True, 'outcome_time': idx
                        })
                        break
                    elif row['high'] > invalid_price:
                        zoi_events.append({
                            'symbol': symbol, 'trend': trend, 'start_time': B['time'], 'zoi_time': zoi_time,
                            'swing_origin': A['price'], 'swing_extreme': B['price'], 'zoi_extreme': zoi_extreme,
                            'success': False, 'outcome_time': idx
                        })
                        break
                        
    return pd.DataFrame(zoi_events)

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
    all_events = []
    
    for sym in symbols:
        df = load_data(sym, timeframe='15min')
        if df is not None:
            swings = identify_swings(df, min_swing_pct=0.01) # 1% macro swings
            print(f"{sym}: Loaded df shape {df.shape}, identified {len(swings)} swings")
            events_df = find_zoi_events(sym, df, swings)
            if not events_df.empty:
                all_events.append(events_df)
                print(f"{sym}: Found {len(events_df)} macro swings entering 0.618 ZOI")
                print(f"  Success Rate (Reversed to new extreme): {events_df['success'].mean()*100:.1f}%\n")
                
    if all_events:
        final_df = pd.concat(all_events, ignore_index=True)
        final_df.to_csv('zoi_events.csv', index=False)
        print(f"Total ZOI events identified: {len(final_df)}")
        print(f"Total Structural Success Rate: {final_df['success'].mean()*100:.1f}%")
        print("We will now analyze tick-level CVD inside these windows to predict the successes.")
    else:
        print("No ZOI events found across all symbols.")
