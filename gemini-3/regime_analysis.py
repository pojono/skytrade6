import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def load_tick_metrics(symbol, target_date="2026-02-24"):
    try:
        # Load Trades to calculate tick-level metrics (volatility, spread approx, frequency)
        file_path = DATALAKE / f"binance/{symbol}/{target_date}_trades.csv.gz"
        if not file_path.exists(): return None
        
        df = pd.read_csv(file_path, usecols=['time', 'price', 'quote_qty', 'is_buyer_maker'], engine='c')
        df.rename(columns={'time': 'timestamp', 'quote_qty': 'vol'}, inplace=True)
        
        # Aggregate to 1-minute bins
        df['minute'] = pd.to_datetime(df['timestamp'], unit='ms').dt.floor('1min')
        
        agg = df.groupby('minute').agg(
            trade_count=('price', 'count'),
            volume=('vol', 'sum'),
            high=('price', 'max'),
            low=('price', 'min'),
            close=('price', 'last'),
            taker_buy_vol=('vol', lambda x: x[~df.loc[x.index, 'is_buyer_maker']].sum())
        )
        
        agg['range_bps'] = (agg['high'] - agg['low']) / agg['low'] * 10000
        agg['avg_trade_size'] = agg['volume'] / agg['trade_count']
        agg['taker_buy_ratio'] = agg['taker_buy_vol'] / agg['volume']
        
        # Calculate forward 5-minute absolute return to measure "Breakout Potential"
        agg['fwd_ret_5m'] = agg['close'].shift(-5) / agg['close'] - 1
        agg['fwd_abs_ret_5m_bps'] = agg['fwd_ret_5m'].abs() * 10000
        
        agg = agg.dropna()
        return agg
    except:
        return None

def analyze_regimes(symbol):
    df = load_tick_metrics(symbol)
    if df is None or len(df) < 500: return None
    
    # We want to cluster the market into 3 regimes based on:
    # 1. Trade Frequency (Activity)
    # 2. Average Trade Size (Whale presence)
    # 3. 1-minute Range (Volatility)
    
    features = ['trade_count', 'avg_trade_size', 'range_bps']
    
    # Normalize
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    
    # Cluster into 3 regimes
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['regime'] = kmeans.fit_predict(scaled_features)
    
    # Analyze the regimes
    results = []
    for regime in range(3):
        mask = df['regime'] == regime
        subset = df[mask]
        
        res = {
            'symbol': symbol,
            'regime_id': regime,
            'samples': len(subset),
            'avg_trade_count': subset['trade_count'].mean(),
            'avg_trade_size': subset['avg_trade_size'].mean(),
            'avg_1m_range_bps': subset['range_bps'].mean(),
            'fwd_abs_move_5m_bps': subset['fwd_abs_ret_5m_bps'].mean() # How much does it move in next 5m?
        }
        results.append(res)
        
    return results

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'SUIUSDT', 'WLDUSDT']
    
    print("--- Market Microstructure Regimes & Breakout Predictability ---")
    all_res = []
    for sym in symbols:
        res = analyze_regimes(sym)
        if res: all_res.extend(res)
        
    if all_res:
        df = pd.DataFrame(all_res)
        # Sort by forward absolute move to identify the "Breakout Regime"
        df = df.sort_values(['symbol', 'fwd_abs_move_5m_bps'])
        
        # Label regimes
        def label_regime(row):
            if row['avg_1m_range_bps'] > 15 and row['avg_trade_size'] > df[df['symbol']==row['symbol']]['avg_trade_size'].median():
                return "Volatile/Whale"
            elif row['avg_trade_count'] > df[df['symbol']==row['symbol']]['avg_trade_count'].median() * 1.5:
                return "High Frequency Retail"
            else:
                return "Quiet/Chop"
                
        df['regime_type'] = df.apply(label_regime, axis=1)
        
        cols = ['symbol', 'regime_type', 'samples', 'avg_trade_count', 'avg_trade_size', 'avg_1m_range_bps', 'fwd_abs_move_5m_bps']
        pd.set_option('display.float_format', lambda x: '%.1f' % x)
        print(df[cols].to_string(index=False))

