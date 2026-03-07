import os
DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake"
sym = "BERAUSDT"
path = os.path.join(DATALAKE_DIR, "binance", sym, f"{sym}_kline_1m.csv")
print(f"Exists: {os.path.exists(path)}")
if not os.path.exists(path):
    path = os.path.join(DATALAKE_DIR, "binance", sym, f"kline_1m.csv")
    print(f"kline_1m.csv Exists: {os.path.exists(path)}")
    path = os.path.join(DATALAKE_DIR, "binance", sym, f"2025-01-01_kline_1m.csv")
    print(f"Date kline_1m.csv Exists: {os.path.exists(path)}")
