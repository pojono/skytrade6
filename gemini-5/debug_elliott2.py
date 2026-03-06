import pandas as pd
import glob
from debug_elliott import identify_swings

path = "/home/ubuntu/Projects/skytrade6/datalake/bybit/BTCUSDT/*_kline_1m.csv"
files = sorted(glob.glob(path))[-180:]
df_list = [pd.read_csv(f) for f in files]
df = pd.concat(df_list, ignore_index=True)
df['startTime'] = pd.to_datetime(df['startTime'], unit='ms')
df = df.sort_values('startTime').set_index('startTime')

df_15m = df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()

for dev in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    swings = identify_swings(df_15m, dev)
    c_loose = 0
    c_strict = 0
    for i in range(len(swings)-4):
        p0, p1, p2, p3, p4 = swings[i], swings[i+1], swings[i+2], swings[i+3], swings[i+4]
        if p0[2] == 'Low' and p1[2] == 'High' and p2[2] == 'Low' and p3[2] == 'High' and p4[2] == 'Low':
            if p2[1] > p0[1] and p3[1] > p1[1] and p4[1] > p2[1]:
                c_loose += 1
                if p4[1] > p1[1]:
                    c_strict += 1
    print(f"Dev {dev}% -> Loose: {c_loose}, Strict: {c_strict}")

