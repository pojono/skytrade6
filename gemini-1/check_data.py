import pandas as pd
import glob
files = glob.glob("/home/ubuntu/Projects/skytrade6/datalake/bybit/BEATUSDT/*_kline_1m.csv")
df_list = []
for f in files:
    df = pd.read_csv(f)
    df_list.append(df)
df = pd.concat(df_list)
print("Columns:", df.columns.tolist())
if 'startTime' in df.columns:
    df['timestamp'] = pd.to_datetime(df['startTime'], unit='ms')
df = df.sort_values('timestamp')
print(df[['timestamp', 'open', 'high', 'low', 'close']].head(20))
