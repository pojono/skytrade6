import pandas as pd
import numpy as np
import glob
import os
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def load_data(symbol, exchange='bybit', timeframe='1h'):
    """Load and resample kline data for a symbol"""
    print(f"Loading data for {symbol} on {exchange}...")
    pattern = f"/home/ubuntu/Projects/skytrade6/datalake/{exchange}/{symbol}/*_kline_1m.csv"
    files = glob.glob(pattern)
    if not files:
        print(f"No files found for {symbol}")
        return None
    
    df_list = []
    for f in sorted(files):
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            pass
            
    if not df_list:
        return None
        
    df = pd.concat(df_list, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['startTime'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    # Resample to timeframe
    resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return resampled

def identify_swings(df, order=10):
    """
    Identify swing highs and lows.
    order: how many points on each side to use for the comparison.
    """
    highs = df['high'].values
    lows = df['low'].values
    
    # Find local maxima and minima
    peak_indices, _ = find_peaks(highs, distance=order)
    valley_indices, _ = find_peaks(-lows, distance=order)
    
    # Combine and sort them chronologically
    swings = []
    for idx in peak_indices:
        swings.append({'type': 'high', 'idx': idx, 'price': highs[idx], 'time': df.index[idx]})
    for idx in valley_indices:
        swings.append({'type': 'low', 'idx': idx, 'price': lows[idx], 'time': df.index[idx]})
        
    swings = sorted(swings, key=lambda x: x['idx'])
    
    # Filter out consecutive highs or consecutive lows by keeping the extreme one
    filtered_swings = []
    for s in swings:
        if not filtered_swings:
            filtered_swings.append(s)
            continue
            
        last_s = filtered_swings[-1]
        if s['type'] == last_s['type']:
            if s['type'] == 'high':
                if s['price'] > last_s['price']:
                    filtered_swings[-1] = s # Replace with higher high
            else:
                if s['price'] < last_s['price']:
                    filtered_swings[-1] = s # Replace with lower low
        else:
            filtered_swings.append(s)
            
    return filtered_swings

def calculate_retracements(swings):
    """
    Calculate the retracement percentages for each swing.
    A complete movement requires 3 points: A -> B -> C
    If A is low, B is high, then we have an uptrend A->B, and C is the retracement low.
    Retracement = (B - C) / (B - A)
    """
    retracements = []
    for i in range(len(swings) - 2):
        A = swings[i]
        B = swings[i+1]
        C = swings[i+2]
        
        # We need alternating swings, which our filter guarantees
        swing_size = abs(B['price'] - A['price'])
        if swing_size == 0:
            continue
            
        retracement_size = abs(B['price'] - C['price'])
        ratio = retracement_size / swing_size
        
        # Only consider retracements between 0% and 150%
        if 0 < ratio <= 1.5:
            retracements.append({
                'trend': 'up' if A['type'] == 'low' else 'down',
                'swing_size': swing_size,
                'ratio': ratio,
                'A_time': A['time'],
                'B_time': B['time'],
                'C_time': C['time']
            })
            
    return pd.DataFrame(retracements)

import multiprocessing
from functools import partial

def analyze_symbol(symbol, timeframes=['5min', '15min', '30min', '1h', '4h', '1d']):
    all_ratios = []
    for tf in timeframes:
        df = load_data(symbol, timeframe=tf)
        if df is None:
            continue
            
        print(f"[{symbol}] Data shape for {tf}: {df.shape}")
        
        # Try different swing sizes (orders) depending on the timeframe to capture various fractal scales
        for order in [5, 10, 20]:
            swings = identify_swings(df, order=order)
            rets = calculate_retracements(swings)
            if not rets.empty:
                all_ratios.extend(rets['ratio'].values)
                
    return all_ratios

def process_symbol_wrapper(symbol):
    try:
        return analyze_symbol(symbol)
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return []

if __name__ == "__main__":
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT',
        'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT', 'MATICUSDT',
        'LTCUSDT', 'UNIUSDT', 'ATOMUSDT', 'BCHUSDT', 'NEARUSDT',
        'OPUSDT', 'ARBUSDT', 'APTUSDT', 'SUIUSDT', 'INJUSDT'
    ]
    
    all_ratios = []
    
    print(f"Starting analysis on {len(symbols)} symbols across 6 timeframes...")
    
    # Use multiprocessing to speed up
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_symbol_wrapper, symbols)
        
    for ratios in results:
        all_ratios.extend(ratios)
        
    all_ratios = np.array(all_ratios)
    
    np.save('fibonacci_ratios_expanded.npy', all_ratios)
    print(f"Total retracements calculated: {len(all_ratios)}")
