import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def load_data(symbol, start_date="2025-01-01"):
    try:
        # Load Binance Funding Rate (from metrics if available, or direct if available)
        bn_fr_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_funding_rate.csv")))
        bn_fr_files = [f for f in bn_fr_files if f.name >= start_date]
        
        if not bn_fr_files: return None
        
        dfs = []
        for f in bn_fr_files:
            try:
                df = pd.read_csv(f, engine='c')
                ts_col = 'fundingTime' if 'fundingTime' in df.columns else 'calcTime' if 'calcTime' in df.columns else df.columns[0]
                val_col = 'fundingRate' if 'fundingRate' in df.columns else df.columns[2]
                df = df[[ts_col, val_col]]
                df.columns = ['timestamp', 'bn_fr']
                dfs.append(df)
            except: pass
            
        if not dfs: return None
        bn_fr = pd.concat(dfs, ignore_index=True)
        bn_fr['timestamp'] = pd.to_numeric(bn_fr['timestamp'])
        if bn_fr['timestamp'].max() < 1e11: bn_fr['timestamp'] *= 1000
        bn_fr.set_index('timestamp', inplace=True)
        bn_fr = bn_fr[~bn_fr.index.duplicated(keep='last')]
        
        # Load Bybit Funding Rate
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
                df.columns = ['timestamp', 'bb_fr']
                dfs.append(df)
            except: pass
            
        if not dfs: return None
        bb_fr = pd.concat(dfs, ignore_index=True)
        bb_fr['timestamp'] = pd.to_numeric(bb_fr['timestamp'])
        if bb_fr['timestamp'].max() < 1e11: bb_fr['timestamp'] *= 1000
        bb_fr.set_index('timestamp', inplace=True)
        bb_fr = bb_fr[~bb_fr.index.duplicated(keep='last')]
        
        # Load Bybit Futures Prices
        kline_files = sorted(list((DATALAKE / f"bybit/{symbol}").glob("*_kline_1m.csv")))
        kline_files = [f for f in kline_files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name and f.name >= start_date]
        if not kline_files: return None
        
        dfs = []
        for f in kline_files:
            try: dfs.append(pd.read_csv(f, usecols=['startTime', 'close'], engine='c'))
            except: pass
        if not dfs: return None
        bb_px = pd.concat(dfs, ignore_index=True)
        bb_px.rename(columns={'startTime': 'timestamp', 'close': 'bb_price'}, inplace=True)
        bb_px['timestamp'] = pd.to_numeric(bb_px['timestamp'])
        if bb_px['timestamp'].max() < 1e11: bb_px['timestamp'] *= 1000
        bb_px.set_index('timestamp', inplace=True)
        bb_px = bb_px[~bb_px.index.duplicated(keep='last')]
        
        merged = bn_fr.join(bb_fr, how='inner')
        return merged, bb_px
    except:
        return None, None

def analyze_funding_arb(symbol):
    # Hypothesis: If Binance funding rate is vastly different from Bybit funding rate, 
    # it creates a cross-exchange arbitrage opportunity. 
    # Do prices on Bybit adjust to close the cross-exchange funding gap?
    res = load_data(symbol)
    if not isinstance(res, tuple) or res[0] is None or res[1] is None: return None
    fr_df, px_df = res
    
    fr_df['fr_diff_bps'] = (fr_df['bn_fr'] - fr_df['bb_fr']) * 10000
    fr_df['fr_diff_z'] = (fr_df['fr_diff_bps'] - fr_df['fr_diff_bps'].rolling(30).mean()) / fr_df['fr_diff_bps'].rolling(30).std()
    
    fr_df = fr_df.dropna()
    
    results = []
    
    # Binance FR is much higher than Bybit FR
    # This means Binance is overly long compared to Bybit.
    # Arb traders will Short Binance, Long Bybit. 
    # This should push Bybit price UP relative to where it started.
    bn_high = fr_df[fr_df['fr_diff_z'] > 2.0]
    
    # Binance FR is much lower than Bybit FR
    # Arb traders will Long Binance, Short Bybit.
    # This should push Bybit price DOWN.
    bn_low = fr_df[fr_df['fr_diff_z'] < -2.0]
    
    def get_fwd_returns(events_df, is_bn_high):
        trade_rets = []
        for ts, row in events_df.iterrows():
            try:
                entry_px = px_df.loc[ts, 'bb_price']
                exit_ts = ts + (4 * 60 * 60 * 1000) # 4 hours forward
                idx = px_df.index.get_indexer([exit_ts], method='nearest')[0]
                exit_px = px_df.iloc[idx]['bb_price']
                
                ret = (exit_px - entry_px) / entry_px
                # If BN is high, we expect Bybit to go UP (positive return)
                # If BN is low, we expect Bybit to go DOWN (negative return)
                if is_bn_high:
                    trade_rets.append(ret) # Long Bybit
                else:
                    trade_rets.append(-ret) # Short Bybit
            except: pass
        return trade_rets
        
    rets_high = get_fwd_returns(bn_high, True)
    rets_low = get_fwd_returns(bn_low, False)
    
    res_dict = {'symbol': symbol}
    
    if len(rets_high) > 5:
        res_dict['bn_high_events'] = len(rets_high)
        res_dict['bn_high_fwd_4h_bps'] = np.mean(rets_high) * 10000
        res_dict['bn_high_wr'] = np.mean(np.array(rets_high) > 0)
        
    if len(rets_low) > 5:
        res_dict['bn_low_events'] = len(rets_low)
        res_dict['bn_low_fwd_4h_bps'] = np.mean(rets_low) * 10000
        res_dict['bn_low_wr'] = np.mean(np.array(rets_low) > 0)
        
    return res_dict

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'SUIUSDT', 'WLDUSDT', 'PEPEUSDT', 'DYDXUSDT', 'ENAUSDT']
    
    print("--- Cross-Exchange Funding Rate Arbitrage (Binance vs Bybit) ---")
    results = []
    with Pool(min(4, os.cpu_count() or 4)) as p:
        for res in p.imap_unordered(analyze_funding_arb, symbols):
            if res: results.append(res)
            
    if results:
        df = pd.DataFrame(results).fillna(0)
        cols = ['symbol', 'bn_high_events', 'bn_high_fwd_4h_bps', 'bn_high_wr', 'bn_low_events', 'bn_low_fwd_4h_bps', 'bn_low_wr']
        cols = [c for c in cols if c in df.columns]
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        print(df[cols].to_string(index=False))
