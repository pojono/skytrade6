import pandas as pd
import numpy as np
from pathlib import Path
import gzip

DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def load_trades_for_hour(symbol, target_date, exchange, market_type, hour=None):
    """Load trades and optionally filter by a specific hour to save memory"""
    if exchange == 'binance' and market_type == 'futures':
        file_path = DATALAKE / f"binance/{symbol}/{target_date}_trades.csv.gz"
        if not file_path.exists(): return None
        df = pd.read_csv(file_path, usecols=['time', 'price', 'quote_qty', 'is_buyer_maker'], engine='c')
        df.rename(columns={'time': 'timestamp', 'quote_qty': 'vol'}, inplace=True)
        df['is_taker_buy'] = ~df['is_buyer_maker']
        
    elif exchange == 'binance' and market_type == 'spot':
        file_path = DATALAKE / f"binance/{symbol}/{target_date}_trades_spot.csv.gz"
        if not file_path.exists(): return None
        df = pd.read_csv(file_path, header=None, usecols=[1, 3, 4, 5], engine='c')
        df.columns = ['price', 'vol', 'timestamp', 'is_buyer_maker']
        df['is_taker_buy'] = ~df['is_buyer_maker']
        
    elif exchange == 'bybit' and market_type == 'futures':
        file_path = DATALAKE / f"bybit/{symbol}/{target_date}_trades.csv.gz"
        if not file_path.exists(): return None
        df = pd.read_csv(file_path, usecols=['timestamp', 'price', 'foreignNotional', 'side'], engine='c')
        df.rename(columns={'foreignNotional': 'vol'}, inplace=True)
        df['is_taker_buy'] = df['side'] == 'Buy'
        # Bybit timestamp is seconds as float, convert to ms
        df['timestamp'] = (df['timestamp'] * 1000).astype(np.int64)
        
    elif exchange == 'bybit' and market_type == 'spot':
        file_path = DATALAKE / f"bybit/{symbol}/{target_date}_trades_spot.csv.gz"
        if not file_path.exists(): return None
        df = pd.read_csv(file_path, usecols=['timestamp', 'price', 'volume', 'side'], engine='c')
        df['vol'] = df['price'] * df['volume']
        df['is_taker_buy'] = df['side'].str.lower() == 'buy'

    else:
        return None

    if len(df) == 0:
        return None

    # Handle weird timestamp formats (Binance spot can be us, Binance futures is ms)
    if df['timestamp'].max() > 1e14:
        df['timestamp'] = df['timestamp'] // 1000
        
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    if hour is not None:
        # Filter for specific hour (0-23)
        # Find start of day timestamp
        start_of_day = df['timestamp'].min()
        # Find the start of the day in UTC
        dt_sod = pd.to_datetime(start_of_day, unit='ms').floor('D')
        ms_sod = int(dt_sod.timestamp() * 1000)
        
        start_ts = ms_sod + (hour * 3600 * 1000)
        end_ts = start_ts + (3600 * 1000)
        df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] < end_ts)]
        
    return df

def analyze_trade_sizes(symbol, target_date):
    print(f"\n--- Trade Size Analysis: {symbol} on {target_date} (Binance Futures) ---")
    df = load_trades_for_hour(symbol, target_date, 'binance', 'futures')
    if df is None or len(df) == 0:
        print("No data.")
        return

    # Price change over the day
    open_px = df['price'].iloc[0]
    close_px = df['price'].iloc[-1]
    daily_ret = (close_px - open_px) / open_px
    print(f"Daily Return: {daily_ret*100:.2f}%")

    # Define buckets
    bins = [0, 1000, 10000, 100000, 1000000, np.inf]
    labels = ['< $1k (Shrimps)', '$1k-$10k (Fish)', '$10k-$100k (Dolphins)', '$100k-$1M (Whales)', '> $1M (Leviathans)']
    df['bucket'] = pd.cut(df['vol'], bins=bins, labels=labels)
    
    # Calculate net flow per bucket
    df['signed_vol'] = np.where(df['is_taker_buy'], df['vol'], -df['vol'])
    
    grouped = df.groupby('bucket').agg(
        trade_count=('vol', 'count'),
        total_vol=('vol', 'sum'),
        net_flow=('signed_vol', 'sum')
    )
    
    grouped['flow_imbalance_%'] = (grouped['net_flow'] / grouped['total_vol']) * 100
    grouped['vol_share_%'] = (grouped['total_vol'] / grouped['total_vol'].sum()) * 100
    
    print(grouped[['trade_count', 'vol_share_%', 'net_flow', 'flow_imbalance_%']].to_string())
    
    # Analyze if the smart money (Whales) were right about the daily direction
    whale_flow = grouped.loc['$100k-$1M (Whales)', 'net_flow'] + grouped.loc['> $1M (Leviathans)', 'net_flow']
    retail_flow = grouped.loc['< $1k (Shrimps)', 'net_flow']
    
    print(f"\nNet Whale Flow: ${whale_flow:,.0f}")
    print(f"Net Retail Flow: ${retail_flow:,.0f}")
    
    del df

def analyze_lead_lag(symbol, target_date, hour=14):
    print(f"\n--- Lead-Lag Analysis (1-second resolution): {symbol} on {target_date}, Hour {hour} UTC ---")
    
    # Load 1 hour of data to keep memory usage low, but resolution high
    df_bn_fut = load_trades_for_hour(symbol, target_date, 'binance', 'futures', hour)
    df_bb_fut = load_trades_for_hour(symbol, target_date, 'bybit', 'futures', hour)
    df_bn_spot = load_trades_for_hour(symbol, target_date, 'binance', 'spot', hour)
    
    if df_bn_fut is None or df_bb_fut is None or len(df_bn_fut) == 0 or len(df_bb_fut) == 0:
        print("Missing data for lead-lag.")
        return
        
    def resample_to_1s(df, name):
        # Convert ms to seconds
        df['sec'] = df['timestamp'] // 1000
        # Get last price per second
        res = df.groupby('sec')['price'].last().rename(name)
        return res
        
    s_bn_fut = resample_to_1s(df_bn_fut, 'bn_fut')
    s_bb_fut = resample_to_1s(df_bb_fut, 'bb_fut')
    
    if df_bn_spot is not None:
        s_bn_spot = resample_to_1s(df_bn_spot, 'bn_spot')
        merged = pd.concat([s_bn_fut, s_bb_fut, s_bn_spot], axis=1).ffill().dropna()
    else:
        merged = pd.concat([s_bn_fut, s_bb_fut], axis=1).ffill().dropna()
        
    # Calculate 1-second returns
    rets = merged.pct_change().dropna()
    
    # Calculate cross-correlation (Lead-Lag)
    # Does Binance Futures at T-1 predict Bybit Futures at T?
    
    lags = range(-5, 6) # -5s to +5s
    print(f"Cross-Correlation: Binance Futures vs Bybit Futures")
    print(" Negative lag = Binance leads Bybit")
    print(" Positive lag = Bybit leads Binance")
    print(" Lag 0 = Coincident")
    
    for lag in lags:
        # shifted > 0 means we shift Bybit forward in time, comparing Bybit[T] with Binance[T-lag]
        # Equivalently: corr(Binance, Bybit.shift(-lag))
        # If lag is -1, we correlate Binance(T) with Bybit(T+1). If high, Binance leads Bybit.
        c = rets['bn_fut'].corr(rets['bb_fut'].shift(-lag))
        marker = "<-- Binance Leads" if lag < 0 and c > 0.1 else "--> Bybit Leads" if lag > 0 and c > 0.1 else ""
        print(f"  Lag {lag:2d}s: {c:.4f} {marker}")

    if 'bn_spot' in rets.columns:
        print(f"\nCross-Correlation: Binance Futures vs Binance Spot")
        for lag in lags:
            c = rets['bn_fut'].corr(rets['bn_spot'].shift(-lag))
            marker = "<-- Futures Lead" if lag < 0 and c > 0.1 else "--> Spot Leads" if lag > 0 and c > 0.1 else ""
            print(f"  Lag {lag:2d}s: {c:.4f} {marker}")

if __name__ == "__main__":
    target_date = "2026-02-24"
    
    # 1. Trade Size Analysis (Who was right about the daily move?)
    analyze_trade_sizes('BTCUSDT', target_date)
    analyze_trade_sizes('ETHUSDT', target_date)
    analyze_trade_sizes('DOGEUSDT', target_date)
    
    # 2. Lead-Lag Analysis (Who moves first at the 1-second level during a busy hour?)
    # 14:00 UTC is usually US market open / active period
    analyze_lead_lag('BTCUSDT', target_date, hour=14)
    analyze_lead_lag('DOGEUSDT', target_date, hour=14)
