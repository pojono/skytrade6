import pandas as pd
import numpy as np
import glob
import os

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"

def check_alignment(symbol="BTCUSDT"):
    print(f"Auditing Data Alignment for {symbol}...")
    
    files_kline = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_kline_1m.csv")
    files_oi = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_open_interest_5min.csv")
    
    df_kline = pd.concat([pd.read_csv(f) for f in files_kline[:5]])
    df_kline['timestamp'] = pd.to_datetime(df_kline['startTime'], unit='ms')
    df_kline = df_kline.sort_values('timestamp').head(10)
    
    df_oi = pd.concat([pd.read_csv(f) for f in files_oi[:5]])
    df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'], unit='ms')
    df_oi = df_oi.sort_values('timestamp').head(10)
    
    print("\n--- Klines Head ---")
    print(df_kline[['timestamp', 'open', 'close']])
    
    print("\n--- OI Head ---")
    print(df_oi[['timestamp', 'openInterest']])

check_alignment("SOLUSDT")

