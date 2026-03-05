import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
START_DATE = "2025-01-01"
END_DATE = "2025-06-30" # Split into in-sample

def load_klines(symbol, is_spot=False):
    pattern = f"{DATALAKE_DIR}/{symbol}/*_kline_1m{'_spot' if is_spot else ''}.csv"
    files = glob.glob(pattern)
    
    filtered_files = []
    for f in files:
        basename = os.path.basename(f)
        date_str = basename.split('_')[0]
        if START_DATE <= date_str <= END_DATE:
            filtered_files.append(f)
            
    if not filtered_files:
        return pd.DataFrame()
    
    dfs = []
    for f in filtered_files:
        try:
            df = pd.read_csv(f)
            for col in ['startTime', 'open', 'high', 'low', 'close', 'volume', 'turnover']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            dfs.append(df)
        except Exception as e:
            pass
            
    if not dfs:
        return pd.DataFrame()
        
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=['startTime', 'close'])
    df['datetime'] = pd.to_datetime(df['startTime'], unit='ms')
    df = df.sort_values('startTime').drop_duplicates('startTime').set_index('datetime')
    return df

for sym in SYMBOLS:
    print(f"\n--- {sym} ---")
    fut_df = load_klines(sym, is_spot=False)
    spot_df = load_klines(sym, is_spot=True)
    
    if fut_df.empty or spot_df.empty:
        print("Missing data")
        continue
        
    aligned = pd.DataFrame({
        'fut_close': fut_df['close'],
        'spot_close': spot_df['close'],
        'fut_vol': fut_df['volume'],
        'spot_vol': spot_df['volume']
    }).dropna()
    
    aligned['fut_ret'] = aligned['fut_close'].pct_change()
    aligned['spot_ret'] = aligned['spot_close'].pct_change()
    aligned['basis_bps'] = (aligned['fut_close'] - aligned['spot_close']) / aligned['spot_close'] * 10000
    aligned['basis_diff'] = aligned['basis_bps'].diff()
    
    aligned.replace([np.inf, -np.inf], np.nan, inplace=True)
    aligned.dropna(inplace=True)
    
    print(f"Days: {len(aligned)/1440:.1f}")
    print(f"1m Return Correlation: {aligned['fut_ret'].corr(aligned['spot_ret']):.4f}")
    
    # Analyze basis reversions
    print(f"Mean Basis: {aligned['basis_bps'].mean():.2f} bps, Std: {aligned['basis_bps'].std():.2f} bps")
    print(f"Max Basis: {aligned['basis_bps'].max():.2f} bps, Min: {aligned['basis_bps'].min():.2f} bps")
    
    # Autocorrelation of basis
    print(f"Basis Autocorrelation (lag 1m): {aligned['basis_bps'].autocorr(1):.4f}")
    
    # Lead-lag
    for lag in [1, 5, 10]:
        fut_lead_spot = aligned['fut_ret'].shift(lag).corr(aligned['spot_ret'])
        spot_lead_fut = aligned['spot_ret'].shift(lag).corr(aligned['fut_ret'])
        print(f"Spot(t-{lag}) -> Fut(t): {spot_lead_fut:.4f} | Fut(t-{lag}) -> Spot(t): {fut_lead_spot:.4f}")
        
    # Volatility Ratio
    print(f"Fut Volatility (annualized): {aligned['fut_ret'].std() * np.sqrt(525600) * 100:.2f}%")
    print(f"Spot Volatility (annualized): {aligned['spot_ret'].std() * np.sqrt(525600) * 100:.2f}%")

