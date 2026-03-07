import pandas as pd
import json
import gzip
import os

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake"
sym = "HUMAUSDT"
date_str = "2025-08-22"
ts_str = "2025-08-22 10:00:00"

ob_file = os.path.join(DATALAKE_DIR, "bybit", sym, f"{date_str}_orderbook.jsonl.gz")

target_ms = int(pd.to_datetime(ts_str).timestamp() * 1000)

with gzip.open(ob_file, 'rt', encoding='utf-8') as f:
    for i in range(5):
        line = f.readline()
        data = json.loads(line)
        print("First line CTS:", data.get('cts'), "Target MS:", target_ms)
        print("Diff:", abs(data.get('cts', 0) - target_ms))
