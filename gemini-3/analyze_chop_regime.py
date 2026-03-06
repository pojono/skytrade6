import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def extract_macro_features(symbol="BTCUSDT"):
    print(f"Extracting Macro Features for {symbol} (2024)...")
    
    kline_files = sorted(list((DATALAKE / f"bybit/{symbol}").glob("*_kline_1m.csv")))
    kline_files = [f for f in kline_files if "2024-01-01" <= f.name[:10] <= "2024-12-31" and "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name]
    
    if not kline_files: return None
    
    dfs = []
    for f in kline_files:
        try: dfs.append(pd.read_csv(f, usecols=['startTime', 'high', 'low', 'close'], engine='c'))
        except: pass
    
    kline_df = pd.concat(dfs, ignore_index=True)
    kline_df.rename(columns={'startTime': 'timestamp'}, inplace=True)
    kline_df['timestamp'] = pd.to_numeric(kline_df['timestamp'])
    if kline_df['timestamp'].max() < 1e11: kline_df['timestamp'] *= 1000
    kline_df.set_index('timestamp', inplace=True)
    kline_df.index = pd.to_datetime(kline_df.index, unit='ms')
    
    # Resample to 1H to calculate macro features
    hourly = kline_df.resample('1h').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    # Feature 1: 14-day Rolling Volatility (ATR %)
    # We will use hourly data but calculate it over 336 periods (14 days)
    hourly['prev_close'] = hourly['close'].shift(1)
    tr1 = hourly['high'] - hourly['low']
    tr2 = (hourly['high'] - hourly['prev_close']).abs()
    tr3 = (hourly['low'] - hourly['prev_close']).abs()
    hourly['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 14-day ATR (336 hours)
    hourly['atr_14d'] = hourly['tr'].rolling(336).mean() * 24 # Annualize to daily equivalent
    hourly['atr_pct_14d'] = (hourly['atr_14d'] / hourly['close']) * 100
    
    # Feature 2: Volatility Percentile (Is the market unusually quiet?)
    # Compare current 14d ATR to the rolling 90-day (2160 hours) ATR history
    hourly['atr_pct_rank'] = hourly['atr_pct_14d'].rolling(2160).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Feature 3: Macro Trend (ADX equivalent - Directional Movement)
    # 14-day momentum
    hourly['mom_14d_pct'] = hourly['close'].pct_change(336) * 100
    
    return hourly

if __name__ == "__main__":
    hourly_btc = extract_macro_features("BTCUSDT")
    
    if hourly_btc is not None:
        # Check August - October
        print("\n--- August to October 2024 Chop Analysis ---")
        chop_period = hourly_btc.loc['2024-08-01':'2024-10-31']
        trend_period = hourly_btc.loc['2024-02-01':'2024-04-30']
        
        print("Volatility (ATR %) Comparison:")
        print(f"Q1 Bull Market (Feb-Apr) Avg ATR: {trend_period['atr_pct_14d'].mean():.2f}%")
        print(f"Q3 Chop Market (Aug-Oct) Avg ATR: {chop_period['atr_pct_14d'].mean():.2f}%")
        
        print("\nVolatility Percentile Rank Comparison:")
        print(f"Q1 Bull Market (Feb-Apr) Avg Rank: {trend_period['atr_pct_rank'].mean():.2f}")
        print(f"Q3 Chop Market (Aug-Oct) Avg Rank: {chop_period['atr_pct_rank'].mean():.2f}")
        
        # Let's see if there's a clear threshold
        # If rank < 0.20, it's extreme low vol (chop)
