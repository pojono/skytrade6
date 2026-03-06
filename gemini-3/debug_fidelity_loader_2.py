import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")
symbol = 'SUIUSDT'
start_date = "2025-01-01"

print("Loading Metrics...")
metrics_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_metrics.csv")))
metrics_files = [f for f in metrics_files if f.name >= start_date]
dfs = []
for f in metrics_files:
    try: dfs.append(pd.read_csv(f, usecols=['create_time', 'sum_open_interest_value'], engine='c'))
    except: pass
oi_df = pd.concat(dfs, ignore_index=True)
oi_df.rename(columns={'create_time': 'timestamp', 'sum_open_interest_value': 'oi_usd'}, inplace=True)
try: oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp']).astype(np.int64) // 10**6
except: oi_df['timestamp'] = pd.to_numeric(oi_df['timestamp'])
if oi_df['timestamp'].max() < 1e11: oi_df['timestamp'] *= 1000
oi_df.set_index('timestamp', inplace=True)
print("OI loaded:", len(oi_df))

print("Loading Funding...")
bb_fr_files = sorted(list((DATALAKE / f"bybit/{symbol}").glob("*_funding_rate.csv")))
bb_fr_files = [f for f in bb_fr_files if f.name >= start_date]
dfs = []
for f in bb_fr_files:
    try:
        df = pd.read_csv(f, engine='c')
        ts_col = 'fundingTime' if 'fundingTime' in df.columns else 'calcTime' if 'calcTime' in df.columns else df.columns[0]
        val_col = 'fundingRate' if 'fundingRate' in df.columns else df.columns[2]
        df = df[[ts_col, val_col]]
        df.columns = ['timestamp', 'funding_rate']
        dfs.append(df)
    except: pass
fr_df = pd.concat(dfs, ignore_index=True)
fr_df['timestamp'] = pd.to_numeric(fr_df['timestamp'])
if fr_df['timestamp'].max() < 1e11: fr_df['timestamp'] *= 1000
fr_df.set_index('timestamp', inplace=True)
fr_df = fr_df[~fr_df.index.duplicated(keep='last')]
print("Funding loaded:", len(fr_df))

print("Loading Klines...")
kline_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_kline_1m.csv")))
kline_files = [f for f in kline_files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name and f.name >= start_date]
dfs = []
for f in kline_files:
    try: dfs.append(pd.read_csv(f, usecols=['open_time', 'high', 'low', 'close'], engine='c'))
    except: pass
kline_df = pd.concat(dfs, ignore_index=True)
kline_df.rename(columns={'open_time': 'timestamp'}, inplace=True)
kline_df['timestamp'] = pd.to_numeric(kline_df['timestamp'])
if kline_df['timestamp'].max() < 1e11: kline_df['timestamp'] *= 1000
kline_df.set_index('timestamp', inplace=True)
kline_df.index = pd.to_datetime(kline_df.index, unit='ms')
print("Klines loaded:", len(kline_df))

try:
    print("Merging...")
    merged = kline_df.join(oi_df, how='left').join(fr_df, how='left')
    merged['oi_usd'] = merged['oi_usd'].ffill()
    merged['funding_rate'] = merged['funding_rate'].ffill()
    merged = merged.dropna(subset=['close', 'oi_usd', 'funding_rate'])
    merged = merged[~merged.index.duplicated(keep='last')]
    print("Merged size:", len(merged))
except Exception as e:
    import traceback
    traceback.print_exc()

