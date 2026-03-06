import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def load_funding_and_klines(symbol, exchange='bybit', start_date="2025-01-01"):
    try:
        # Load Klines
        kline_files = sorted(list((DATALAKE / f"{exchange}/{symbol}").glob("*_kline_1m.csv")))
        kline_files = [f for f in kline_files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name and f.name >= start_date]
        
        if not kline_files: return (None, None)
        
        dfs = []
        time_col = 'startTime' if exchange == 'bybit' else 'open_time'
        for f in kline_files:
            try: dfs.append(pd.read_csv(f, usecols=[time_col, 'close'], engine='c'))
            except: pass
            
        kline_df = pd.concat(dfs, ignore_index=True)
        kline_df.rename(columns={time_col: 'timestamp'}, inplace=True)
        kline_df['timestamp'] = pd.to_numeric(kline_df['timestamp'])
        if kline_df['timestamp'].max() < 1e11: kline_df['timestamp'] *= 1000
        kline_df.set_index('timestamp', inplace=True)
        
        # Load Funding Rate
        fr_files = sorted(list((DATALAKE / f"{exchange}/{symbol}").glob("*_funding_rate.csv")))
        fr_files = [f for f in fr_files if f.name >= start_date]
        
        if not fr_files: return (None, None)
        
        dfs = []
        for f in fr_files:
            try:
                # usually has timestamp, fundingRate
                df = pd.read_csv(f, engine='c')
                # get right columns
                ts_col = 'fundingTime' if 'fundingTime' in df.columns else 'calcTime' if 'calcTime' in df.columns else df.columns[0]
                val_col = 'fundingRate' if 'fundingRate' in df.columns else df.columns[2]
                df = df[[ts_col, val_col]]
                df.columns = ['timestamp', 'funding_rate']
                dfs.append(df)
            except: pass
            
        if not dfs: return (None, None)
        
        fr_df = pd.concat(dfs, ignore_index=True)
        fr_df['timestamp'] = pd.to_numeric(fr_df['timestamp'])
        if fr_df['timestamp'].max() < 1e11: fr_df['timestamp'] *= 1000
        fr_df.set_index('timestamp', inplace=True)
        fr_df = fr_df[~fr_df.index.duplicated(keep='last')]
        
        # We only want to look at the exact minutes where funding was paid.
        merged = fr_df.join(kline_df, how='inner')
        merged['fr_bps'] = merged['funding_rate'] * 10000
        
        # We also need future prices to calculate returns, so let's keep the full kline
        # and look forward.
        return fr_df, kline_df
    except Exception as e:
        return (None, None)

def analyze_funding_decay(symbol):
    res = load_funding_and_klines(symbol)
    if not isinstance(res, tuple) or res[0] is None or res[1] is None: return None
    fr_df, kline_df = res
    
    if len(fr_df) < 50: return None
    
    results = []
    # Identify extreme funding events (e.g. > 5 bps per 8 hours)
    ext_long_fr = fr_df[fr_df['funding_rate'] > 0.0005] # 5 bps
    ext_short_fr = fr_df[fr_df['funding_rate'] < -0.0005]
    
    def get_fwd_returns(events_df, is_long_fr):
        trade_rets = []
        for ts, row in events_df.iterrows():
            # ts is the exact minute funding is paid. 
            # Often price dumps *immediately* after funding is paid as arb traders close out.
            # Let's look at the return from T to T+60m
            try:
                entry_px = kline_df.loc[ts, 'close']
                exit_ts = ts + (60 * 60 * 1000) # +60 mins
                # find closest kline to exit_ts
                idx = kline_df.index.get_indexer([exit_ts], method='nearest')[0]
                exit_px = kline_df.iloc[idx]['close']
                
                ret = (exit_px - entry_px) / entry_px
                # If FR is highly positive, longs pay shorts. 
                # Arb traders are short. After funding pays, they close shorts (buy back). 
                # Does price go UP after high positive funding?
                
                trade_rets.append(ret)
            except: pass
        return trade_rets
        
    pos_rets = get_fwd_returns(ext_long_fr, True)
    neg_rets = get_fwd_returns(ext_short_fr, False)
    
    res_dict = {'symbol': symbol}
    
    if len(pos_rets) > 5:
        res_dict['pos_fr_events'] = len(pos_rets)
        res_dict['fwd_60m_after_pos_fr_bps'] = np.mean(pos_rets) * 10000
        
    if len(neg_rets) > 5:
        res_dict['neg_fr_events'] = len(neg_rets)
        res_dict['fwd_60m_after_neg_fr_bps'] = np.mean(neg_rets) * 10000
        
    return res_dict

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'SUIUSDT', 'WLDUSDT', 'PEPEUSDT', 'DYDXUSDT', 'ENAUSDT']
    
    print("--- Analyzing Post-Funding Rate Mean Reversion ---")
    results = []
    with Pool(min(4, os.cpu_count() or 4)) as p:
        for res in p.imap_unordered(analyze_funding_decay, symbols):
            if res: results.append(res)
            
    if results:
        df = pd.DataFrame(results).fillna(0)
        # Reorder columns
        cols = ['symbol', 'pos_fr_events', 'fwd_60m_after_pos_fr_bps', 'neg_fr_events', 'fwd_60m_after_neg_fr_bps']
        cols = [c for c in cols if c in df.columns]
        print(df[cols].to_string(index=False))

