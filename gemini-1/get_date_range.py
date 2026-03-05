import pandas as pd
import glob
import os

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
df_trades = pd.read_csv("/home/ubuntu/Projects/skytrade6/gemini-1/strategy_capitulation_bounce.csv")

print(f"Total Unique Coins Traded: {df_trades['symbol'].nunique()}")
print("Top 15 Traded Coins:")
print(df_trades['symbol'].value_counts().head(15))

min_date = df_trades['entry_time'].min()
max_date = df_trades['entry_time'].max()
print(f"\nDate Range of Trades:")
print(f"First Trade: {min_date}")
print(f"Last Trade:  {max_date}")

# Let's also check the actual dataset dates we loaded
files = glob.glob(f"{DATALAKE_DIR}/BTCUSDT/*_kline_1m.csv")
if files:
    df_btc = pd.concat([pd.read_csv(f) for f in files[:5] + files[-5:]])
    if 'startTime' in df_btc.columns:
        df_btc['timestamp'] = pd.to_datetime(df_btc['startTime'], unit='ms')
    elif 'timestamp' in df_btc.columns and df_btc['timestamp'].dtype != 'datetime64[ns]':
        df_btc['timestamp'] = pd.to_datetime(df_btc['timestamp'], unit='ms')
    print(f"\nRaw Data Example (BTCUSDT) ranges from: {df_btc['timestamp'].min()} to {df_btc['timestamp'].max()}")

