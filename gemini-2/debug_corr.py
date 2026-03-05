import pandas as pd
import numpy as np
import os
import glob

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
START_DATE = "2025-01-01"
END_DATE = "2025-01-05"

def load_klines(symbol):
    pattern = f"{DATALAKE_DIR}/{symbol}/*_kline_1m.csv"
    files = glob.glob(pattern)
    filtered_files = [f for f in files if START_DATE <= os.path.basename(f).split('_')[0] <= END_DATE]
    dfs = []
    for f in filtered_files:
        df = pd.read_csv(f)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df['datetime'] = pd.to_datetime(df['startTime'], unit='ms')
    return df.sort_values('startTime').drop_duplicates('startTime').set_index('datetime')['close']

btc = load_klines("BTCUSDT")
eth = load_klines("ETHUSDT")

df = pd.DataFrame({'BTC': btc, 'ETH': eth})
print(df.head())
print(df.tail())
df = df.dropna()
ret = df.pct_change().dropna()

print("\n--- Correlation ---")
print(ret.corr())

