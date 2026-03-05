import pandas as pd
import glob

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
symbol = "BTCUSDT"

def load_klines(symbol, is_spot=False):
    pattern = f"{DATALAKE_DIR}/{symbol}/*_kline_1m{'_spot' if is_spot else ''}.csv"
    files = glob.glob(pattern)
    files.sort()
    
    dfs = []
    for f in files[:2]: # just load first 2 files for debugging
        df = pd.read_csv(f)
        for col in ['startTime', 'open', 'high', 'low', 'close', 'volume', 'turnover']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        dfs.append(df)
        
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=['startTime', 'close'])
    df['datetime'] = pd.to_datetime(df['startTime'], unit='ms')
    df = df.sort_values('startTime').drop_duplicates('startTime')
    return df

fut_df = load_klines(symbol, is_spot=False)
spot_df = load_klines(symbol, is_spot=True)

print("Futures:")
print(fut_df[['startTime', 'datetime', 'close']].head())
print("\nSpot:")
print(spot_df[['startTime', 'datetime', 'close']].head())

fut_df = fut_df.set_index('datetime')
spot_df = spot_df.set_index('datetime')

aligned = pd.DataFrame({
    'fut_close': fut_df['close'],
    'spot_close': spot_df['close'],
}).dropna()

print("\nAligned:")
print(aligned.head())

