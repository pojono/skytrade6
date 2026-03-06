import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def load_kline_with_cvd(symbol, target_date="2026-02-24"):
    try:
        # We need raw tick data to calculate pure CVD (Cumulative Volume Delta)
        # We'll use Bybit Futures because we have high quality 'side' tags
        file_path = DATALAKE / f"bybit/{symbol}/{target_date}_trades.csv.gz"
        if not file_path.exists(): return None
        
        df = pd.read_csv(file_path, usecols=['timestamp', 'price', 'foreignNotional', 'side'], engine='c')
        df.rename(columns={'foreignNotional': 'vol'}, inplace=True)
        
        # Convert timestamp to ms
        df['timestamp'] = (df['timestamp'] * 1000).astype(np.int64)
        df['minute'] = pd.to_datetime(df['timestamp'], unit='ms').dt.floor('1min')
        
        # Calculate Delta: Buy Vol - Sell Vol
        df['delta'] = np.where(df['side'] == 'Buy', df['vol'], -df['vol'])
        
        agg = df.groupby('minute').agg(
            close=('price', 'last'),
            delta=('delta', 'sum')
        )
        
        # Cumulative Volume Delta (resets daily for this analysis)
        agg['cvd'] = agg['delta'].cumsum()
        
        # Forward 60m return
        agg['fwd_ret_60m_bps'] = (agg['close'].shift(-60) / agg['close'] - 1) * 10000
        
        return agg.dropna()
    except:
        return None

def analyze_cvd_divergence(symbol):
    # Hypothesis: CVD represents the cumulative net flow of market orders.
    # If Price makes a New High, but CVD makes a Lower High (Bearish Divergence),
    # it means the price is being driven up by limit order pulling (spoofing) or low liquidity, 
    # not actual market buying. The move will fail.
    
    # We will test on a single recent day to prove the concept
    df = load_kline_with_cvd(symbol)
    if df is None or len(df) < 60: return None
    
    # Calculate rolling 60-min max/min
    df['price_max_60m'] = df['close'].rolling(60).max()
    df['cvd_max_60m'] = df['cvd'].rolling(60).max()
    
    df['price_min_60m'] = df['close'].rolling(60).min()
    df['cvd_min_60m'] = df['cvd'].rolling(60).min()
    
    # Bearish Divergence: Price is at or near 60m high, but CVD is significantly below its 60m high
    bear_div = df[(df['close'] >= df['price_max_60m'] * 0.9995) & 
                  (df['cvd'] < df['cvd_max_60m'] * 0.8)] # CVD is at least 20% off its local high
                  
    # Bullish Divergence: Price is at 60m low, but CVD is significantly above its 60m low
    # CVD can be negative, so we use absolute differences or just simple relative checks
    cvd_range = df['cvd_max_60m'] - df['cvd_min_60m']
    cvd_range = cvd_range.replace(0, 1) # Avoid division by zero
    
    # Refined Bearish Div: Price at 60m high, CVD is in the bottom 50% of its 60m range
    bear_div = df[(df['close'] >= df['price_max_60m']) & 
                  ((df['cvd'] - df['cvd_min_60m']) / cvd_range < 0.5)]
                  
    # Refined Bullish Div: Price at 60m low, CVD is in the top 50% of its 60m range
    bull_div = df[(df['close'] <= df['price_min_60m']) & 
                  ((df['cvd'] - df['cvd_min_60m']) / cvd_range > 0.5)]
                  
    res = {'symbol': symbol}
    
    if len(bear_div) > 0:
        res['bear_div_events'] = len(bear_div)
        res['bear_fwd_ret_60m_bps'] = bear_div['fwd_ret_60m_bps'].mean()
        res['bear_short_wr'] = (bear_div['fwd_ret_60m_bps'] < 0).mean()
        
    if len(bull_div) > 0:
        res['bull_div_events'] = len(bull_div)
        res['bull_fwd_ret_60m_bps'] = bull_div['fwd_ret_60m_bps'].mean()
        res['bull_long_wr'] = (bull_div['fwd_ret_60m_bps'] > 0).mean()
        
    return res if (len(bear_div) > 0 or len(bull_div) > 0) else None

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'SUIUSDT', 'WLDUSDT']
    
    print("--- Cumulative Volume Delta (CVD) Divergences (Feb 24, 2026) ---")
    results = []
    with Pool(min(4, os.cpu_count() or 4)) as p:
        for res in p.imap_unordered(analyze_cvd_divergence, symbols):
            if res: results.append(res)
            
    if results:
        df = pd.DataFrame(results).fillna(0)
        cols = ['symbol', 'bear_div_events', 'bear_fwd_ret_60m_bps', 'bear_short_wr', 
                'bull_div_events', 'bull_fwd_ret_60m_bps', 'bull_long_wr']
        cols = [c for c in cols if c in df.columns]
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        print(df[cols].to_string(index=False))

