import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"

def load_btc_data():
    files = glob.glob(f"{DATALAKE_DIR}/BTCUSDT/*_kline_1m.csv")
    if not files: return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['timestamp'] = pd.to_datetime(df.get('startTime', df.get('timestamp', None)), unit='ms')
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).set_index('timestamp')
    df = df.resample('1d').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
    df.dropna(inplace=True)
    return df

df_btc = load_btc_data()

# Look at Oct to Jan vs Feb to Mar
print("=== BTC Macro Analysis ===")
oct_jan = df_btc[(df_btc.index >= '2025-10-01') & (df_btc.index < '2026-02-01')]
feb_mar = df_btc[(df_btc.index >= '2026-02-01') & (df_btc.index <= '2026-03-31')]

print(f"Oct-Jan BTC Return: {(oct_jan['close'].iloc[-1] / oct_jan['open'].iloc[0]) - 1:.2%}")
print(f"Feb-Mar BTC Return: {(feb_mar['close'].iloc[-1] / feb_mar['open'].iloc[0]) - 1:.2%}")

# Let's calculate the "chop factor" (Average True Range divided by net trend)
def chop_factor(df):
    tr = df['high'] - df['low']
    atr = tr.mean()
    net_move = abs(df['close'].iloc[-1] - df['open'].iloc[0])
    return (atr * len(df)) / net_move if net_move > 0 else np.inf

print(f"\nOct-Jan Chop Factor: {chop_factor(oct_jan):.2f}")
print(f"Feb-Mar Chop Factor: {chop_factor(feb_mar):.2f}")

print("\n(A higher chop factor means price is oscillating wildly without making forward progress. Trend following strategies bleed out in high-chop environments because breakouts immediately reverse).")
