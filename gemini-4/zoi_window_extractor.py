import pandas as pd
import numpy as np
import glob
import gc
import os
import sys
import warnings

warnings.filterwarnings('ignore')

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def load_1h_data(symbol, exchange='bybit'):
    print(f"Loading 1h macro data for {symbol}...")
    pattern = f"/home/ubuntu/Projects/skytrade6/datalake/{exchange}/{symbol}/*_kline_1m.csv"
    files = glob.glob(pattern)
    if not files:
        return None
        
    df_list = []
    # Only load data from 2025 onwards to match tick data availability
    for f in sorted(files):
        if "2025" in f or "2026" in f:
            try:
                df = pd.read_csv(f, usecols=['startTime', 'open', 'high', 'low', 'close'])
                df_list.append(df)
            except Exception:
                pass
            
    if not df_list:
        return None
        
    df = pd.concat(df_list, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['startTime'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    resampled = df.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    return resampled

def find_zoi_windows(df_macro):
    """
    Scans the 1h macro chart to identify times when price enters the 0.618 Zone of Interest (ZOI).
    Returns a list of 'windows' (start_time, end_time, trade_dir, zoi_upper, zoi_lower, tp, sl)
    to be analyzed with high-resolution tick data.
    """
    df_macro['ATR'] = calculate_atr(df_macro, 14)
    df_macro['SMA_50'] = df_macro['close'].rolling(window=50).mean()
    df_macro = df_macro.dropna()
    
    FIB_LEVEL = 0.618
    ZOI_TOLERANCE = 0.04 # [0.578, 0.658]
    CONFIRMATION_ATR_MULT = 2.0
    MIN_SWING_PCT = 0.01 
    
    mode = 'search'
    swing_dir = 1 
    
    origin_price = df_macro['close'].iloc[0]
    extreme_price = origin_price
    
    zoi_upper = 0
    zoi_lower = 0
    trade_dir = 0
    
    zoi_windows = []
    active_window = None
    
    for i in range(1, len(df_macro)):
        current = df_macro.iloc[i]
        current_time = df_macro.index[i]
        
        if mode == 'search':
            if swing_dir == 1:
                if current['high'] > extreme_price:
                    extreme_price = current['high']
                elif extreme_price - current['low'] > current['ATR'] * CONFIRMATION_ATR_MULT:
                    if extreme_price > origin_price * (1 + MIN_SWING_PCT) and current['close'] > current['SMA_50']:
                        swing_size = extreme_price - origin_price
                        fib_price = extreme_price - (swing_size * FIB_LEVEL)
                        zoi_range = swing_size * ZOI_TOLERANCE
                        zoi_upper = fib_price + zoi_range
                        zoi_lower = fib_price - zoi_range
                        mode = 'armed'
                        trade_dir = 1
                        
                        # Immediately check if this confirmation candle is already in the ZOI
                        if current['low'] <= zoi_upper:
                            active_window = {
                                'start_time': current_time,
                                'dir': trade_dir,
                                'zoi_upper': zoi_upper,
                                'zoi_lower': zoi_lower,
                                'tp_price': extreme_price,
                                'invalid_price': origin_price + ((extreme_price - origin_price) * 0.2)
                            }
                    else:
                        swing_dir = -1
                        origin_price = extreme_price
                        extreme_price = current['low']
            elif swing_dir == -1:
                if current['low'] < extreme_price:
                    extreme_price = current['low']
                elif current['high'] - extreme_price > current['ATR'] * CONFIRMATION_ATR_MULT:
                    if extreme_price < origin_price * (1 - MIN_SWING_PCT) and current['close'] < current['SMA_50']:
                        swing_size = origin_price - extreme_price
                        fib_price = extreme_price + (swing_size * FIB_LEVEL)
                        zoi_range = swing_size * ZOI_TOLERANCE
                        zoi_upper = fib_price + zoi_range
                        zoi_lower = fib_price - zoi_range
                        mode = 'armed'
                        trade_dir = -1
                        
                        if current['high'] >= zoi_lower:
                            active_window = {
                                'start_time': current_time,
                                'dir': trade_dir,
                                'zoi_upper': zoi_upper,
                                'zoi_lower': zoi_lower,
                                'tp_price': extreme_price,
                                'invalid_price': origin_price - ((origin_price - extreme_price) * 0.2)
                            }
                    else:
                        swing_dir = 1
                        origin_price = extreme_price
                        extreme_price = current['high']
                        
        elif mode == 'armed':
            if trade_dir == 1:
                # Cancel if we slice completely through
                if current['close'] < origin_price + ((extreme_price - origin_price) * 0.2):
                    if active_window:
                        active_window['end_time'] = current_time
                        zoi_windows.append(active_window)
                        active_window = None
                    mode = 'search'
                    swing_dir = -1
                    origin_price = extreme_price
                    extreme_price = current['low']
                    continue
                    
                # We are actively in the ZOI
                if current['low'] <= zoi_upper:
                    if active_window is None:
                        active_window = {
                            'start_time': current_time,
                            'dir': trade_dir,
                            'zoi_upper': zoi_upper,
                            'zoi_lower': zoi_lower,
                            'tp_price': extreme_price,
                            'invalid_price': origin_price + ((extreme_price - origin_price) * 0.2)
                        }
                        
                # End window if we hit TP or new high
                if active_window and current['high'] >= extreme_price:
                    active_window['end_time'] = current_time
                    zoi_windows.append(active_window)
                    active_window = None
                    mode = 'search'
                    origin_price = current['low']
                    extreme_price = current['high']
                    
            elif trade_dir == -1:
                if current['close'] > origin_price - ((origin_price - extreme_price) * 0.2):
                    if active_window:
                        active_window['end_time'] = current_time
                        zoi_windows.append(active_window)
                        active_window = None
                    mode = 'search'
                    swing_dir = 1
                    origin_price = extreme_price
                    extreme_price = current['high']
                    continue
                    
                if current['high'] >= zoi_lower:
                    if active_window is None:
                        active_window = {
                            'start_time': current_time,
                            'dir': trade_dir,
                            'zoi_upper': zoi_upper,
                            'zoi_lower': zoi_lower,
                            'tp_price': extreme_price,
                            'invalid_price': origin_price - ((origin_price - extreme_price) * 0.2)
                        }
                        
                # End window if we hit TP or new low
                if active_window and current['low'] <= extreme_price:
                    active_window['end_time'] = current_time
                    zoi_windows.append(active_window)
                    active_window = None
                    mode = 'search'
                    origin_price = current['high']
                    extreme_price = current['low']
    
    if active_window:
        active_window['end_time'] = df_macro.index[-1]
        zoi_windows.append(active_window)
        
    return zoi_windows

if __name__ == "__main__":
    symbol = 'BTCUSDT'
    df_macro = load_1h_data(symbol)
    windows = find_zoi_windows(df_macro)
    print(f"Found {len(windows)} ZOI windows for {symbol} where tick data needs to be parsed.")
    for i, w in enumerate(windows[:5]):
        print(f"Window {i+1}: {w['start_time']} to {w['end_time']} (Dir: {w['dir']})")
