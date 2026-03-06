from ls_ratio_analysis import load_klines_and_ls, DATALAKE
import pandas as pd

symbol = 'BTCUSDT'
start_date = '2025-01-01'

metrics_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_metrics.csv")))
metrics_files = [f for f in metrics_files if f.name >= start_date]

print(f"Metrics files: {len(metrics_files)}")

kline_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_kline_1m.csv")))
kline_files = [f for f in kline_files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name and f.name >= start_date]

print(f"Kline files: {len(kline_files)}")
