import pandas as pd
import numpy as np
import os
import glob

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]

def load_klines(symbol, is_spot=False):
    pattern = f"{DATALAKE_DIR}/{symbol}/*_kline_1m{'_spot' if is_spot else ''}.csv"
    files = glob.glob(pattern)
    if not files:
        return pd.DataFrame()
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Ensure columns are numeric
            for col in ['startTime', 'open', 'high', 'low', 'close', 'volume', 'turnover']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    if not dfs:
        return pd.DataFrame()
        
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=['startTime', 'close'])
    df['startTime'] = pd.to_numeric(df['startTime'])
    df['datetime'] = pd.to_datetime(df['startTime'], unit='ms')
    df = df.sort_values('startTime').drop_duplicates('startTime').set_index('datetime')
    return df

all_data = {}
for sym in SYMBOLS:
    print(f"Loading {sym}...")
    fut_df = load_klines(sym, is_spot=False)
    spot_df = load_klines(sym, is_spot=True)
    
    if not fut_df.empty and not spot_df.empty:
        # Calculate 1m returns
        fut_ret = fut_df['close'].pct_change()
        spot_ret = spot_df['close'].pct_change()
        
        # Align
        aligned = pd.DataFrame({
            'fut_ret': fut_ret,
            'spot_ret': spot_ret,
            'fut_close': fut_df['close'],
            'spot_close': spot_df['close'],
            'fut_vol': fut_df['volume'],
            'spot_vol': spot_df['volume']
        }).dropna()
        
        # Replace inf with nan
        aligned.replace([np.inf, -np.inf], np.nan, inplace=True)
        aligned.dropna(inplace=True)
        
        all_data[sym] = aligned
        
        # Check correlations
        corr = aligned['fut_ret'].corr(aligned['spot_ret'])
        print(f"  {sym} 1m Return Correlation: {corr:.4f}")
        
        # Check lead-lag (1m shift)
        spot_lead_fut = aligned['spot_ret'].shift(1).corr(aligned['fut_ret'])
        fut_lead_spot = aligned['fut_ret'].shift(1).corr(aligned['spot_ret'])
        print(f"  Spot(t-1) -> Fut(t): {spot_lead_fut:.4f}")
        print(f"  Fut(t-1) -> Spot(t): {fut_lead_spot:.4f}")
        
        # Calculate basis
        aligned['basis_bps'] = (aligned['fut_close'] - aligned['spot_close']) / aligned['spot_close'] * 10000
        print(f"  Mean Basis: {aligned['basis_bps'].mean():.2f} bps, Std: {aligned['basis_bps'].std():.2f} bps")

