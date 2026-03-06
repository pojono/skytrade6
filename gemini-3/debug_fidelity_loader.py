import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

symbol = 'SUIUSDT'
start_date = "2025-01-01"

kline_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_kline_1m.csv")))
kline_files = [f for f in kline_files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name and f.name >= start_date]
print(f"Klines: {len(kline_files)}")

metrics_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_metrics.csv")))
metrics_files = [f for f in metrics_files if f.name >= start_date]
print(f"Metrics: {len(metrics_files)}")

bb_fr_files = sorted(list((DATALAKE / f"bybit/{symbol}").glob("*_funding_rate.csv")))
bb_fr_files = [f for f in bb_fr_files if f.name >= start_date]
print(f"Funding: {len(bb_fr_files)}")

try:
    from full_universe_scan_fidelity import load_data
    h, m = load_data('SUIUSDT')
    print("Hourly size:", len(h) if h is not None else None)
except Exception as e:
    import traceback
    traceback.print_exc()

