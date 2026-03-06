import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from multiprocessing import Pool
from strat_3_powder_keg import load_data
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def generate_relaxed_signals(df):
    # Calculate rolling Z-scores
    df['oi_z'] = (df['oi_usd'] - df['oi_usd'].rolling(168).mean()) / df['oi_usd'].rolling(168).std()
    df['fr_z'] = (df['funding_rate'] - df['funding_rate'].rolling(168).mean()) / df['funding_rate'].rolling(168).std()
    
    df['signal'] = 0
    
    # Relaxed Threshold: Z > 1.5 instead of 2.0 (Top 6.6% instead of Top 2.2%)
    df.loc[(df['oi_z'] > 1.5) & (df['fr_z'] > 1.5), 'signal'] = -1
    df.loc[(df['oi_z'] > 1.5) & (df['fr_z'] < -1.5), 'signal'] = 1
    
    # Only take the FIRST signal in a sequence
    df['signal_shifted'] = df['signal'].shift(1).fillna(0)
    df.loc[df['signal'] == df['signal_shifted'], 'signal'] = 0
    
    return df

def extract_keg_events(symbol):
    df = load_data(symbol)
    if df is None: return None
    
    df = generate_relaxed_signals(df)
    events = df[df['signal'] != 0].copy()
    events = events.reset_index()
    events['symbol'] = symbol
    return events

if __name__ == "__main__":
    # Get all available symbols from Bybit datalake
    symbols = [d.name for d in (DATALAKE / "bybit").iterdir() if d.is_dir()]
    print(f"Scanning {len(symbols)} symbols for relaxed Powder Keg events...")
    
    all_events = []
    
    with Pool(min(12, os.cpu_count() or 4)) as p:
        results = p.map(extract_keg_events, symbols)
        
    for r in results:
        if r is not None and not r.empty:
            all_events.append(r)
            
    if all_events:
        events_df = pd.concat(all_events, ignore_index=True)
        events_df.to_csv("keg_events_all_relaxed.csv", index=False)
        print(f"Extracted {len(events_df)} Relaxed Powder Keg / Despair Pit events across {len(all_events)} coins.")
