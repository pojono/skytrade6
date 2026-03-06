import pandas as pd
from pathlib import Path

DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")
cluster = ['BTCUSDT', 'SOLUSDT', 'LINKUSDT', 'AVAXUSDT', 'NEARUSDT']

for sym in cluster:
    files = list((DATALAKE / f"bybit/{sym}").glob("*_funding_rate.csv"))
    files.sort(key=lambda x: x.name)
    if files: print(f"{sym} FR: {files[0].name[:10]} to {files[-1].name[:10]}")
