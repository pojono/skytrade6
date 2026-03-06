import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from strat_3_powder_keg import load_data, generate_signals
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def extract_keg_events(symbol="BTCUSDT"):
    df = load_data(symbol)
    if df is None: return None
    
    df = generate_signals(df)
    
    # We only care about the exact hours where a Powder Keg (Short) or Despair Pit (Long) was triggered
    events = df[df['signal'] != 0].copy()
    events = events.reset_index()
    events['symbol'] = symbol
    return events

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    all_events = []
    
    for sym in symbols:
        events = extract_keg_events(sym)
        if events is not None and not events.empty:
            all_events.append(events)
            
    if all_events:
        events_df = pd.concat(all_events, ignore_index=True)
        events_df.to_csv("keg_events.csv", index=False)
        print(f"Extracted {len(events_df)} Powder Keg / Despair Pit events.")
        print(events_df[['timestamp', 'symbol', 'signal', 'oi_z', 'fr_z']].head(10))
