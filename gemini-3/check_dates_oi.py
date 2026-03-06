import pandas as pd
from pathlib import Path

DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")
cluster = ['BTCUSDT', 'SOLUSDT', 'LINKUSDT', 'AVAXUSDT', 'NEARUSDT']

for sym in cluster:
    print(f"--- {sym} ---")
    
    # check premium index
    files = list((DATALAKE / f"bybit/{sym}").glob("*_premium_index_1m.csv"))
    files.sort(key=lambda x: x.name)
    if files: print(f"Premium: {files[0].name[:10]} to {files[-1].name[:10]}")
    
    # check open interest
    files = list((DATALAKE / f"bybit/{sym}").glob("*_open_interest*.csv"))
    files.sort(key=lambda x: x.name)
    if files: print(f"OI: {files[0].name[:10]} to {files[-1].name[:10]}")
