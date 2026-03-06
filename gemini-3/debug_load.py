from analyze_correlations import DATALAKE
import pandas as pd
from pathlib import Path
import gc

symbol = 'BTCUSDT'
sources = {
    'bybit_futures': (DATALAKE / "bybit" / symbol, "*_kline_1m.csv", 'startTime', 'close'),
    'bybit_spot': (DATALAKE / "bybit" / symbol, "*_kline_1m_spot.csv", 'startTime', 'close'),
    'binance_futures': (DATALAKE / "binance" / symbol, "*_kline_1m.csv", 'open_time', 'close'),
    'binance_spot': (DATALAKE / "binance" / symbol, "*_kline_1m_spot.csv", 'open_time', 'close'),
}

dfs = []
for name, (path_dir, pattern, time_col, close_col) in sources.items():
    print(f"Checking {name}...")
    if not path_dir.exists():
        print(f"  Dir does not exist")
        continue
    
    files = sorted(list(path_dir.glob(pattern)))
    if pattern == "*_kline_1m.csv":
        files = [f for f in files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name]
    
    files = [f for f in files if f.name >= "2025-01-01"]
    
    if not files:
        print(f"  No files found")
        continue
    
    daily_dfs = []
    # Test just first 5 files to speed up debug
    for f in files[:5]:
        try:
            if name == 'binance_spot':
                df_day = pd.read_csv(f, header=None, usecols=[0, 4], engine='c')
                df_day.columns = [time_col, close_col]
            else:
                df_day = pd.read_csv(f, usecols=[time_col, close_col], engine='c')
            daily_dfs.append(df_day)
        except Exception as e:
            print(f"  Error reading {f}: {e}")
            pass
            
    if not daily_dfs:
        print(f"  Failed to read any files")
        continue
        
    df = pd.concat(daily_dfs, ignore_index=True)
    df = df.rename(columns={time_col: 'timestamp', close_col: name})
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    
    if df['timestamp'].max() < 1e11:
        df['timestamp'] = df['timestamp'] * 1000
        
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='last')]
    dfs.append((name, df))
    print(f"  Loaded {name} with shape {df.shape}")

print(f"Loaded {len(dfs)} sources")
if len(dfs) == 4:
    merged = pd.concat([d[1] for d in dfs], axis=1, join='inner')
    print(f"Merged shape: {merged.shape}")
