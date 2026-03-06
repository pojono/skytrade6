import pandas as pd
import numpy as np
import glob
import os
import gc
from datetime import timedelta
import gzip
import warnings

warnings.filterwarnings('ignore')

def get_tick_file_for_date(symbol, date_str, exchange='bybit'):
    # date_str format: 'YYYY-MM-DD'
    pattern = f"/home/ubuntu/Projects/skytrade6/datalake/{exchange}/{symbol}/{date_str}_trades.csv.gz"
    if os.path.exists(pattern):
        return pattern
    return None

def analyze_zoi_ticks():
    print("Loading ZOI events...")
    if not os.path.exists('zoi_events.csv'):
        print("zoi_events.csv not found. Run extract_zoi_events.py first.")
        return
        
    events = pd.read_csv('zoi_events.csv')
    events['start_time'] = pd.to_datetime(events['start_time'])
    events['zoi_time'] = pd.to_datetime(events['zoi_time'])
    events['outcome_time'] = pd.to_datetime(events['outcome_time'])
    
    # Sort by time so we can stream through tick files sequentially
    events = events.sort_values('zoi_time')
    
    print(f"Loaded {len(events)} events. Processing top 5000 events to analyze execution alpha...")
    events = events.head(5000) # Increased limit for statistical significance
    
    results = []
    
    # Group events by date to minimize file loading
    events['date_str'] = events['zoi_time'].dt.strftime('%Y-%m-%d')
    
    for date_str, date_events in events.groupby('date_str'):
        symbols = date_events['symbol'].unique()
        
        for symbol in symbols:
            sym_events = date_events[date_events['symbol'] == symbol]
            tick_file = get_tick_file_for_date(symbol, date_str)
            
            if not tick_file:
                continue
                
            print(f"Processing {len(sym_events)} events for {symbol} on {date_str}...")
            
            try:
                # Load the entire day's tick file
                # Columns: timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,foreignNotional,RPI
                ticks = pd.read_csv(tick_file, usecols=['timestamp', 'side', 'size', 'price'])
                # Bybit tick timestamps are seconds with fractional ms
                ticks['datetime'] = pd.to_datetime(ticks['timestamp'], unit='s')
                ticks.set_index('datetime', inplace=True)
                ticks.sort_index(inplace=True)
                
                # Calculate CVD for the whole day to allow fast slicing
                ticks['signed_volume'] = np.where(ticks['side'] == 'Buy', ticks['size'], -ticks['size'])
                
                for _, event in sym_events.iterrows():
                    trend = event['trend']
                    zoi_time = event['zoi_time']
                    
                    # We want to analyze the microstructure during the 30 minutes leading up to the absolute extreme inside the ZOI
                    start_scan = zoi_time - pd.Timedelta(minutes=30)
                    end_scan = zoi_time + pd.Timedelta(minutes=5) # Allow 5 minutes post-extreme for confirmation
                    
                    window_ticks = ticks.loc[start_scan:end_scan].copy()
                    if window_ticks.empty:
                        continue
                        
                    # Calculate local CVD within the window
                    window_ticks['cvd'] = window_ticks['signed_volume'].cumsum()
                    
                    # Microstructure Features at the bottom/top
                    zoi_extreme_price = event['zoi_extreme']
                    
                    # Find the exact tick where the extreme was hit
                    if trend == 'up':
                        # Look for absorption at the low
                        extreme_ticks = window_ticks[window_ticks['price'] <= zoi_extreme_price * 1.0005] # Within 5 bps of extreme
                        if not extreme_ticks.empty:
                            # How much volume traded at the absolute low?
                            vol_at_extreme = extreme_ticks['size'].sum()
                            buy_vol = extreme_ticks[extreme_ticks['side'] == 'Buy']['size'].sum()
                            sell_vol = extreme_ticks[extreme_ticks['side'] == 'Sell']['size'].sum()
                            delta_at_extreme = buy_vol - sell_vol
                            
                            # CVD Divergence: Did CVD make a higher low while price made a lower low?
                            # Simplify: Is the delta strongly positive precisely when price is crashing into the low?
                            absorption_ratio = buy_vol / (vol_at_extreme + 1e-8)
                            
                            results.append({
                                'success': event['success'],
                                'trend': trend,
                                'vol_at_extreme': vol_at_extreme,
                                'delta_at_extreme': delta_at_extreme,
                                'absorption_ratio': absorption_ratio
                            })
                    else:
                        # Look for absorption at the high
                        extreme_ticks = window_ticks[window_ticks['price'] >= zoi_extreme_price * 0.9995]
                        if not extreme_ticks.empty:
                            vol_at_extreme = extreme_ticks['size'].sum()
                            buy_vol = extreme_ticks[extreme_ticks['side'] == 'Buy']['size'].sum()
                            sell_vol = extreme_ticks[extreme_ticks['side'] == 'Sell']['size'].sum()
                            delta_at_extreme = buy_vol - sell_vol
                            
                            # For shorts, absorption means aggressive selling into the high
                            absorption_ratio = sell_vol / (vol_at_extreme + 1e-8)
                            
                            results.append({
                                'success': event['success'],
                                'trend': trend,
                                'vol_at_extreme': vol_at_extreme,
                                'delta_at_extreme': delta_at_extreme,
                                'absorption_ratio': absorption_ratio
                            })
                            
                del ticks
                gc.collect()
            except Exception as e:
                print(f"Error processing {tick_file}: {e}")
                
    if results:
        res_df = pd.DataFrame(results)
        print("\n=== Microstructure Execution Alpha Results ===")
        print(f"Analyzed {len(res_df)} ZOI extremes with tick data.")
        
        # Does high absorption predict success?
        # Let's say high absorption is absorption_ratio > 0.6
        high_absorption = res_df[res_df['absorption_ratio'] > 0.6]
        low_absorption = res_df[res_df['absorption_ratio'] <= 0.6]
        
        print(f"Baseline Success Rate: {res_df['success'].mean()*100:.1f}%")
        if not high_absorption.empty:
            print(f"Success Rate with HIGH Absorption (>60% opposing flow): {high_absorption['success'].mean()*100:.1f}% (N={len(high_absorption)})")
        if not low_absorption.empty:
            print(f"Success Rate with LOW Absorption (<=60% opposing flow): {low_absorption['success'].mean()*100:.1f}% (N={len(low_absorption)})")
            
        res_df.to_csv('zoi_tick_analysis.csv', index=False)
        print("Detailed results saved to zoi_tick_analysis.csv")

if __name__ == "__main__":
    analyze_zoi_ticks()
