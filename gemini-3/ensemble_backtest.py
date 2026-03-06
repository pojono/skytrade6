import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings
from backtest_core_v2 import run_backtest_advanced

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

# Filter to Top Liquid Coins to avoid low-cap chop and manipulation
TOP_COINS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 
    'SUIUSDT', 'WLDUSDT', 'PEPEUSDT', 'DYDXUSDT', 'ENAUSDT',
    'AVAXUSDT', 'NEARUSDT', 'LINKUSDT', 'WIFUSDT', 'ARBUSDT',
    'OPUSDT', 'APTUSDT', 'INJUSDT', 'RENDERUSDT', 'TIAUSDT'
]

def load_ensemble_data(symbol, start_date="2025-01-01"):
    try:
        # 1. Klines
        kline_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_kline_1m.csv")))
        kline_files = [f for f in kline_files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name and f.name >= start_date]
        if not kline_files: return None
        dfs = []
        for f in kline_files:
            try: dfs.append(pd.read_csv(f, usecols=['open_time', 'high', 'low', 'close'], engine='c'))
            except: pass
        if not dfs: return None
        kline_df = pd.concat(dfs, ignore_index=True)
        kline_df.rename(columns={'open_time': 'timestamp'}, inplace=True)
        kline_df['timestamp'] = pd.to_numeric(kline_df['timestamp'])
        if kline_df['timestamp'].max() < 1e11: kline_df['timestamp'] *= 1000
        kline_df.set_index('timestamp', inplace=True)
        
        # 2. Binance Metrics (OI & L/S)
        metrics_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_metrics.csv")))
        metrics_files = [f for f in metrics_files if f.name >= start_date]
        if not metrics_files: return None
        dfs = []
        for f in metrics_files:
            try: dfs.append(pd.read_csv(f, usecols=[
                'create_time', 'sum_open_interest_value', 
                'count_toptrader_long_short_ratio', 'sum_toptrader_long_short_ratio'
            ], engine='c'))
            except: pass
        if not dfs: return None
        m_df = pd.concat(dfs, ignore_index=True)
        m_df.rename(columns={
            'create_time': 'timestamp', 
            'sum_open_interest_value': 'oi_usd',
            'count_toptrader_long_short_ratio': 'count_ls',
            'sum_toptrader_long_short_ratio': 'vol_ls'
        }, inplace=True)
        try: m_df['timestamp'] = pd.to_datetime(m_df['timestamp']).astype(np.int64) // 10**6
        except: m_df['timestamp'] = pd.to_numeric(m_df['timestamp'])
        if m_df['timestamp'].max() < 1e11: m_df['timestamp'] *= 1000
        m_df.set_index('timestamp', inplace=True)
        
        # 3. Bybit Funding Rate
        bb_fr_files = sorted(list((DATALAKE / f"bybit/{symbol}").glob("*_funding_rate.csv")))
        bb_fr_files = [f for f in bb_fr_files if f.name >= start_date]
        if not bb_fr_files: return None
        dfs = []
        for f in bb_fr_files:
            try:
                df = pd.read_csv(f, engine='c')
                ts_col = 'fundingTime' if 'fundingTime' in df.columns else 'calcTime' if 'calcTime' in df.columns else df.columns[0]
                val_col = 'fundingRate' if 'fundingRate' in df.columns else df.columns[2]
                df = df[[ts_col, val_col]]
                df.columns = ['timestamp', 'funding_rate']
                dfs.append(df)
            except: pass
        if not dfs: return None
        fr_df = pd.concat(dfs, ignore_index=True)
        fr_df['timestamp'] = pd.to_numeric(fr_df['timestamp'])
        if fr_df['timestamp'].max() < 1e11: fr_df['timestamp'] *= 1000
        fr_df.set_index('timestamp', inplace=True)
        fr_df = fr_df[~fr_df.index.duplicated(keep='last')]
        
        # Merge
        merged = kline_df.join(m_df, how='left').join(fr_df, how='left')
        merged = merged.ffill()
        merged = merged.dropna(subset=['close', 'oi_usd', 'count_ls', 'vol_ls', 'funding_rate'])
        merged = merged[~merged.index.duplicated(keep='last')]
        
        merged.index = pd.to_datetime(merged.index, unit='ms')
        
        # ATR logic on 1m then sum to 1h
        prev_close = merged['close'].shift(1)
        tr1 = merged['high'] - merged['low']
        tr2 = (merged['high'] - prev_close).abs()
        tr3 = (merged['low'] - prev_close).abs()
        merged['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        hourly = merged.resample('1h').agg({
            'close': 'last',
            'oi_usd': 'last',
            'count_ls': 'last',
            'vol_ls': 'last',
            'funding_rate': 'last',
            'tr': 'sum'
        }).dropna()
        
        hourly['atr_pct'] = (hourly['tr'].rolling(24).mean() * 24) / hourly['close']
        
        return hourly
    except: return None

def analyze_ensemble(symbol):
    df = load_ensemble_data(symbol)
    if df is None or len(df) < 500: return None
    
    # ML-guided Features
    # Leverage (7-day = 168h)
    df['oi_z'] = (df['oi_usd'] - df['oi_usd'].rolling(168).mean()) / df['oi_usd'].rolling(168).std()
    df['fr_z'] = (df['funding_rate'] - df['funding_rate'].rolling(168).mean()) / df['funding_rate'].rolling(168).std()
    
    # Sentiment (1-day = 24h)
    df['count_z'] = (df['count_ls'] - df['count_ls'].rolling(24).mean()) / df['count_ls'].rolling(24).std()
    df['vol_z'] = (df['vol_ls'] - df['vol_ls'].rolling(24).mean()) / df['vol_ls'].rolling(24).std()
    
    # Momentum (24h)
    df['mom_24h'] = df['close'] / df['close'].shift(24) - 1
    
    df['signal'] = 0
    
    # The "God Signal" (Short)
    # 1. Market is over-leveraged long (Powder Keg)
    # 2. Retail is max long, Whales are max short (Smart Money Divergence)
    # 3. Momentum is NOT exploding upward (> 10% in a day usually means true squeeze, avoid shorting)
    df.loc[(df['oi_z'] > 1.5) & 
           (df['fr_z'] > 1.0) & 
           (df['count_z'] > 1.5) & 
           (df['vol_z'] < -1.5) & 
           (df['mom_24h'] < 0.10), 'signal'] = -1
           
    # The "God Signal" (Long)
    # 1. Market is max short leveraged (Despair Pit)
    # 2. Retail is max short, Whales are max long
    # 3. Momentum is NOT falling off a cliff
    df.loc[(df['oi_z'] > 1.5) & 
           (df['fr_z'] < -1.0) & 
           (df['count_z'] < -1.5) & 
           (df['vol_z'] > 1.5) & 
           (df['mom_24h'] > -0.10), 'signal'] = 1
           
    df['signal_shifted'] = df['signal'].shift(1).fillna(0)
    df.loc[df['signal'] == df['signal_shifted'], 'signal'] = 0
    
    # Backtest with optimized params
    # Max hold 24h, Trailing Stop 4%, Take Profit 12%, Fee 5 bps, Vol Sizing ON
    results = run_backtest_advanced(df, max_hold_periods=24, trailing_stop_bps=400.0, take_profit_bps=1200.0, fee_bps=5.0, use_volatility_sizing=True)
    
    if results['events'] > 0:
        results['symbol'] = symbol
        return results
    return None

if __name__ == "__main__":
    print(f"--- Running The 'God Signal' Ensemble on Top {len(TOP_COINS)} Liquid Coins ---")
    
    res_list = []
    with Pool(min(4, os.cpu_count() or 4)) as p:
        for r in p.imap_unordered(analyze_ensemble, TOP_COINS):
            if r: res_list.append(r)
            
    if res_list:
        res_df = pd.DataFrame(res_list)
        res_df = res_df.sort_values('total_net_ret_%', ascending=False)
        
        cols = ['symbol', 'events', 'win_rate', 'total_net_ret_%', 'avg_net_ret_bps', 'max_drawdown_%', 'sharpe']
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        print(res_df[cols].to_string(index=False))
        print("\n--- Aggregate Portfolio Performance ---")
        print(f"Total Trades: {res_df['events'].sum()}")
        print(f"Avg Win Rate: {res_df['win_rate'].mean():.2%}")
        print(f"Total Net Return: {res_df['total_net_ret_%'].sum():.2f}%")
        print(f"Avg Sharpe: {res_df['sharpe'].mean():.2f}")
