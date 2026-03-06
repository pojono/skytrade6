import pandas as pd
from pathlib import Path

DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")
cluster = ['BTCUSDT', 'SOLUSDT', 'LINKUSDT', 'AVAXUSDT', 'NEARUSDT', 'WLDUSDT']

for sym in cluster:
    kline_dir = DATALAKE / f"bybit/{sym}"
    if not kline_dir.exists():
        print(f"{sym} dir missing")
        continue
    files = list(kline_dir.glob("*_kline_1m.csv"))
    valid_files = [f.name for f in files if "mark" not in f.name and "index" not in f.name and "premium" not in f.name]
    valid_files.sort()
    if valid_files:
        print(f"{sym}: {valid_files[0][:10]} to {valid_files[-1][:10]}")
    else:
        print(f"{sym}: No valid kline files")
