from ls_ratio_analysis import load_klines_and_ls, DATALAKE
import pandas as pd
import numpy as np

symbol = 'BTCUSDT'
start_date = '2025-01-01'

metrics_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_metrics.csv")))
metrics_files = [f for f in metrics_files if f.name >= start_date]
f = metrics_files[0]
df_m = pd.read_csv(f, usecols=['create_time', 'count_long_short_ratio'], engine='c')
df_m.rename(columns={'create_time': 'timestamp', 'count_long_short_ratio': 'long_short_ratio'}, inplace=True)
df_m['timestamp'] = pd.to_numeric(df_m['timestamp'])
if df_m['timestamp'].max() < 1e11: df_m['timestamp'] *= 1000
df_m.set_index('timestamp', inplace=True)
print(f"Metrics Min TS: {df_m.index.min()}, Max TS: {df_m.index.max()}")
print(df_m.head(2))

kline_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_kline_1m.csv")))
kline_files = [f for f in kline_files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name and f.name >= start_date]
f_k = kline_files[0]
df_k = pd.read_csv(f_k, usecols=['open_time', 'close'], engine='c')
df_k.rename(columns={'open_time': 'timestamp'}, inplace=True)
df_k['timestamp'] = pd.to_numeric(df_k['timestamp'])
if df_k['timestamp'].max() < 1e11: df_k['timestamp'] *= 1000
df_k.set_index('timestamp', inplace=True)
print(f"Kline Min TS: {df_k.index.min()}, Max TS: {df_k.index.max()}")
print(df_k.head(2))

merged = df_k.join(df_m, how='left')
print(f"Merged size before dropna: {len(merged)}")
merged['long_short_ratio'] = merged['long_short_ratio'].ffill()
merged = merged.dropna(subset=['close', 'long_short_ratio'])
print(f"Merged size after dropna: {len(merged)}")

