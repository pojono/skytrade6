import os
import pandas as pd

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
symbols = sorted([d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))])
print(f"Total Coins Tested: {len(symbols)}")
print("First 15 Coins:", ", ".join(symbols[:15]))
print("...")

# Let's get the exact date range from BTC to be precise
import glob
btc_files = glob.glob(f"{DATALAKE_DIR}/BTCUSDT/*_kline_1m.csv")
df = pd.concat([pd.read_csv(f) for f in btc_files], ignore_index=True)
df['timestamp'] = pd.to_datetime(df.get('startTime', df.get('timestamp', None)), unit='ms')
start = df['timestamp'].min()
end = df['timestamp'].max()
print(f"\nExact Data Date Range: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")

# Remember that the strategy requires 200 candles (4H) = 33 days of burn-in
# And we also specifically cut off the data at 180 days prior to the max date to focus on recent market structure
cutoff = end - pd.Timedelta(days=180)
print(f"Strategy Active Execution Range: {cutoff.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} (Last 6 Months)")
