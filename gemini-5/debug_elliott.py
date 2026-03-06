import pandas as pd
import glob

def identify_swings(df: pd.DataFrame, deviation_pct: float = 1.0):
    highs = df['high'].values
    lows = df['low'].values
    swings = []
    last_high_idx = 0
    last_low_idx = 0
    last_high = highs[0]
    last_low = lows[0]
    trend = 0
    
    for i in range(1, len(df)):
        if trend == 0:
            if highs[i] > last_high * (1 + deviation_pct / 100):
                trend = 1
                last_high = highs[i]
                last_high_idx = i
                swings.append((last_low_idx, last_low, 'Low'))
            elif lows[i] < last_low * (1 - deviation_pct / 100):
                trend = -1
                last_low = lows[i]
                last_low_idx = i
                swings.append((last_high_idx, last_high, 'High'))
            else:
                if highs[i] > last_high:
                    last_high = highs[i]
                    last_high_idx = i
                if lows[i] < last_low:
                    last_low = lows[i]
                    last_low_idx = i
        elif trend == 1:
            if highs[i] > last_high:
                last_high = highs[i]
                last_high_idx = i
            elif lows[i] < last_high * (1 - deviation_pct / 100):
                swings.append((last_high_idx, last_high, 'High'))
                trend = -1
                last_low = lows[i]
                last_low_idx = i
        elif trend == -1:
            if lows[i] < last_low:
                last_low = lows[i]
                last_low_idx = i
            elif highs[i] > last_low * (1 + deviation_pct / 100):
                swings.append((last_low_idx, last_low, 'Low'))
                trend = 1
                last_high = highs[i]
                last_high_idx = i
    if trend == 1: swings.append((last_high_idx, last_high, 'High'))
    elif trend == -1: swings.append((last_low_idx, last_low, 'Low'))
    return swings

path = "/home/ubuntu/Projects/skytrade6/datalake/bybit/BTCUSDT/*_kline_1m.csv"
files = sorted(glob.glob(path))[-180:]
df_list = [pd.read_csv(f) for f in files]
df = pd.concat(df_list, ignore_index=True)
df['startTime'] = pd.to_datetime(df['startTime'], unit='ms')
df = df.sort_values('startTime').set_index('startTime')

df_15m = df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
swings = identify_swings(df_15m, 1.5)

c0 = 0
c1 = 0
c2 = 0
c3 = 0
c4 = 0

for i in range(len(swings)-4):
    p0, p1, p2, p3, p4 = swings[i], swings[i+1], swings[i+2], swings[i+3], swings[i+4]
    if p0[2] == 'Low' and p1[2] == 'High' and p2[2] == 'Low' and p3[2] == 'High' and p4[2] == 'Low':
        c0 += 1
        if p2[1] > p0[1]: # W2 > W0
            c1 += 1
            if p3[1] > p1[1]: # W3 > W1
                c2 += 1
                if p4[1] > p2[1]: # W4 > W2
                    c3 += 1
                    if p4[1] > p1[1]: # W4 > W1 (Strict overlap)
                        c4 += 1

print(f"Raw bull sequences: {c0}")
print(f"W2 > W0: {c1}")
print(f"W2 > W0 AND W3 > W1: {c2}")
print(f"W2 > W0 AND W3 > W1 AND W4 > W2 (Loose setup): {c3}")
print(f"W2 > W0 AND W3 > W1 AND W4 > W1 (Strict setup): {c4}")
