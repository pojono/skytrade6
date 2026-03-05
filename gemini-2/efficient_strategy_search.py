import pandas as pd
import numpy as np
import os
import glob
import gc
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
# Let's use last 6 months for solid backtest: 2025-09-01 to 2026-03-04 (or just process everything we can load quickly)
# To avoid memory issues, we just process all available data for each coin.

def get_all_symbols():
    return sorted([d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d)) and d.endswith('USDT')])

def process_symbol(symbol):
    try:
        # Load klines
        files_kline = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_kline_1m.csv")
        if not files_kline: return None
        
        dfs_k = []
        for f in files_kline:
            try:
                df = pd.read_csv(f, usecols=['startTime', 'close', 'volume'])
                dfs_k.append(df)
            except: pass
        if not dfs_k: return None
        kline = pd.concat(dfs_k, ignore_index=True).dropna()
        kline['startTime'] = pd.to_numeric(kline['startTime'], errors='coerce')
        kline['close'] = pd.to_numeric(kline['close'], errors='coerce')
        kline['volume'] = pd.to_numeric(kline['volume'], errors='coerce')
        kline = kline[kline['close'] > 0]
        kline['datetime'] = pd.to_datetime(kline['startTime'], unit='ms')
        kline = kline.set_index('datetime').resample('5min').agg({'close': 'last', 'volume': 'sum'}).ffill()
        
        # Load OI
        files_oi = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_open_interest_5min.csv")
        dfs_o = []
        for f in files_oi:
            try:
                df = pd.read_csv(f)
                ts_col = 'timestamp' if 'timestamp' in df.columns else 'startTime' if 'startTime' in df.columns else None
                oi_col = 'openInterest' if 'openInterest' in df.columns else 'OpenInterest' if 'OpenInterest' in df.columns else None
                if not ts_col or not oi_col: continue
                df = df[[ts_col, oi_col]].dropna()
                df.columns = ['startTime', 'oi']
                dfs_o.append(df)
            except: pass
        
        if dfs_o:
            oi_df = pd.concat(dfs_o, ignore_index=True)
            oi_df['startTime'] = pd.to_numeric(oi_df['startTime'], errors='coerce')
            oi_df['oi'] = pd.to_numeric(oi_df['oi'], errors='coerce')
            oi_df['datetime'] = pd.to_datetime(oi_df['startTime'], unit='ms')
            oi_df = oi_df.set_index('datetime').resample('5min').last().ffill()
            df = pd.concat([kline, oi_df['oi']], axis=1).ffill().dropna()
        else:
            df = kline
            df['oi'] = np.nan
            df = df.dropna(subset=['close'])
            
        # Feature Engineering (5m timeframe)
        # Price Returns
        df['ret_15m'] = df['close'].pct_change(3)
        df['ret_30m'] = df['close'].pct_change(6)
        df['ret_1h'] = df['close'].pct_change(12)
        
        # Volatility
        df['volatility_1h'] = df['close'].pct_change().rolling(12).std()
        
        # Volume spikes
        df['vol_sma_4h'] = df['volume'].rolling(48).mean()
        df['vol_spike_15m'] = df['volume'].rolling(3).sum() / (df['vol_sma_4h'] * 3 + 1e-9)
        
        # OI changes
        if not df['oi'].isna().all():
            df['oi_chg_15m'] = df['oi'].pct_change(3)
            df['oi_chg_30m'] = df['oi'].pct_change(6)
            df['oi_chg_1h'] = df['oi'].pct_change(12)
        else:
            for c in ['oi_chg_15m', 'oi_chg_30m', 'oi_chg_1h']: df[c] = np.nan

        # Forward Returns (targets)
        # Holding periods: 15m (3 bars), 30m (6), 1h (12), 4h (48)
        targets = {
            'fwd_15m': df['close'].shift(-3) / df['close'] - 1,
            'fwd_30m': df['close'].shift(-6) / df['close'] - 1,
            'fwd_1h': df['close'].shift(-12) / df['close'] - 1,
            'fwd_4h': df['close'].shift(-48) / df['close'] - 1,
        }
        for k, v in targets.items(): df[k] = v
        
        df = df.dropna(subset=['fwd_1h', 'ret_1h']) # Ensure we have valid targets
        
        # Rules evaluation
        results = []
        
        # Rule 1: OI Breakout Long (OI up, Price up)
        mask = (df['oi_chg_30m'] > 0.05) & (df['ret_30m'] > 0.03)
        for t_name in targets.keys():
            rets = df.loc[mask, t_name]
            results.append({'rule': 'OI_Breakout_Long_30m_5pct_3pct', 'target': t_name, 'sum_ret': rets.sum(), 'count': len(rets), 'sum_sq': (rets**2).sum()})
            
        # Rule 2: OI Breakdown Short (OI up, Price down)
        mask = (df['oi_chg_30m'] > 0.05) & (df['ret_30m'] < -0.03)
        for t_name in targets.keys():
            rets = -df.loc[mask, t_name] # Short return
            results.append({'rule': 'OI_Breakdown_Short_30m_5pct_3pct', 'target': t_name, 'sum_ret': rets.sum(), 'count': len(rets), 'sum_sq': (rets**2).sum()})
            
        # Rule 3: Volume Spike Momentum Long
        mask = (df['vol_spike_15m'] > 3.0) & (df['ret_15m'] > 0.03)
        for t_name in targets.keys():
            rets = df.loc[mask, t_name]
            results.append({'rule': 'Vol_Spike_Mom_Long_15m_3x_3pct', 'target': t_name, 'sum_ret': rets.sum(), 'count': len(rets), 'sum_sq': (rets**2).sum()})
            
        # Rule 4: Volume Spike Momentum Short
        mask = (df['vol_spike_15m'] > 3.0) & (df['ret_15m'] < -0.03)
        for t_name in targets.keys():
            rets = -df.loc[mask, t_name]
            results.append({'rule': 'Vol_Spike_Mom_Short_15m_3x_3pct', 'target': t_name, 'sum_ret': rets.sum(), 'count': len(rets), 'sum_sq': (rets**2).sum()})
            
        # Rule 5: Reversal from extreme move (Price drops 5% in 15m with high vol, expect bounce)
        mask = (df['vol_spike_15m'] > 2.0) & (df['ret_15m'] < -0.05)
        for t_name in targets.keys():
            rets = df.loc[mask, t_name] # Long
            results.append({'rule': 'Reversal_Long_15m_drop_5pct', 'target': t_name, 'sum_ret': rets.sum(), 'count': len(rets), 'sum_sq': (rets**2).sum()})
            
        # Rule 6: Reversal from extreme pump
        mask = (df['vol_spike_15m'] > 2.0) & (df['ret_15m'] > 0.05)
        for t_name in targets.keys():
            rets = -df.loc[mask, t_name] # Short
            results.append({'rule': 'Reversal_Short_15m_pump_5pct', 'target': t_name, 'sum_ret': rets.sum(), 'count': len(rets), 'sum_sq': (rets**2).sum()})
            
        # Rule 7: OI flush Long (OI drops > 5%, Price drops > 3%) -> implies liquidations, expect bounce
        mask = (df['oi_chg_15m'] < -0.05) & (df['ret_15m'] < -0.03)
        for t_name in targets.keys():
            rets = df.loc[mask, t_name] # Long
            results.append({'rule': 'OI_Flush_Long_15m', 'target': t_name, 'sum_ret': rets.sum(), 'count': len(rets), 'sum_sq': (rets**2).sum()})
            
        # Rule 8: OI flush Short (OI drops > 5%, Price pumps > 3%) -> short squeeze over, expect drop
        mask = (df['oi_chg_15m'] < -0.05) & (df['ret_15m'] > 0.03)
        for t_name in targets.keys():
            rets = -df.loc[mask, t_name] # Short
            results.append({'rule': 'OI_Flush_Short_15m', 'target': t_name, 'sum_ret': rets.sum(), 'count': len(rets), 'sum_sq': (rets**2).sum()})

        return results
    except Exception as e:
        return None

if __name__ == '__main__':
    symbols = get_all_symbols()
    print(f"Starting analysis on {len(symbols)} symbols...")
    
    with Pool(processes=12) as pool:
        all_results = pool.map(process_symbol, symbols)
        
    # Aggregate results
    agg = {}
    for res_list in all_results:
        if not res_list: continue
        for r in res_list:
            key = (r['rule'], r['target'])
            if key not in agg:
                agg[key] = {'sum_ret': 0, 'count': 0, 'sum_sq': 0}
            agg[key]['sum_ret'] += r['sum_ret']
            agg[key]['count'] += r['count']
            agg[key]['sum_sq'] += r['sum_sq']
            
    # Print summary
    print("\n--- Strategy Search Results (Gross Returns, no fees yet) ---")
    FEE_BPS = 20 # 0.2%
    
    summary = []
    for (rule, target), data in agg.items():
        if data['count'] < 50: continue # Ignore low sample size
        mean_ret = data['sum_ret'] / data['count']
        # std_ret = np.sqrt(data['sum_sq'] / data['count'] - mean_ret**2)
        net_bps = (mean_ret - (FEE_BPS / 10000)) * 10000
        
        summary.append({
            'Rule': rule,
            'Target': target,
            'Trades': data['count'],
            'Gross_bps': mean_ret * 10000,
            'Net_bps': net_bps
        })
        
    df_sum = pd.DataFrame(summary).sort_values('Net_bps', ascending=False)
    print(df_sum.to_string(index=False))

