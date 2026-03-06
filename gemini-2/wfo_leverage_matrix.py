import pandas as pd
import numpy as np

# Re-run WFO strictly saving the raw OOS trades to get the exact matrix with the worst-case 50 bps execution penalty.

import os
import glob
from multiprocessing import Pool
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"

def process_symbol(symbol):
    try:
        files_kline = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_kline_1m.csv")
        if not files_kline: return None
        
        dfs_k = []
        for f in files_kline:
            try:
                df = pd.read_csv(f, usecols=['startTime', 'open', 'high', 'low', 'close', 'volume'])
                dfs_k.append(df)
            except: pass
        if not dfs_k: return None
        kline = pd.concat(dfs_k, ignore_index=True).dropna()
        kline['startTime'] = pd.to_numeric(kline['startTime'], errors='coerce')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            kline[col] = pd.to_numeric(kline[col], errors='coerce')
        kline = kline[kline['close'] > 0]
        kline['datetime'] = pd.to_datetime(kline['startTime'], unit='ms')
        kline = kline.set_index('datetime').sort_index()
        kline = kline[~kline.index.duplicated(keep='first')]
        kline_5m = kline.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).ffill()
        
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
            oi_df = oi_df.set_index('datetime').sort_index()
            oi_df = oi_df[~oi_df.index.duplicated(keep='first')]
            oi_df = oi_df.resample('5min').last().ffill()
            df = pd.concat([kline_5m, oi_df['oi']], axis=1).ffill().dropna()
        else:
            return None
            
        df['ret_15m'] = df['close'].pct_change(3)
        df['oi_chg_15m'] = df['oi'].pct_change(3)
        
        # Broad filter
        df['sig_broad'] = ((df['oi_chg_15m'] <= -0.05) & (df['ret_15m'] <= -0.04)).astype(int)
        
        trades = []
        indices = np.where(df['sig_broad'] == 1)[0]
        HOLD = 72
        
        for i in indices:
            if i + HOLD >= len(df): continue 
            entry_time = df.index[i]
            entry_price = df['close'].iloc[i]
            fwd_window = df.iloc[i+1 : i+1+HOLD]
            trades.append({
                'symbol': symbol,
                'entry_time': entry_time,
                'entry_price': entry_price,
                'oi_drop': df['oi_chg_15m'].iloc[i],
                'px_drop': df['ret_15m'].iloc[i],
                'path_high': fwd_window['high'].values.tolist(),
                'path_low': fwd_window['low'].values.tolist(),
                'path_close': fwd_window['close'].values.tolist(),
                'path_times': fwd_window.index.tolist()
            })
        return trades
    except Exception as e:
        return None

if __name__ == '__main__':
    symbols = sorted([d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d)) and d.endswith('USDT')])
    
    # Check if we already have the raw trades from the previous run to save time
    # Actually, it's safer to just re-extract fast
    trades = []
    with Pool(processes=12) as pool:
        for res_list in pool.imap_unordered(process_symbol, symbols):
            if res_list: trades.extend(res_list)
                
    df_trades = pd.DataFrame(trades).sort_values('entry_time')
    df_trades['month'] = df_trades['entry_time'].dt.to_period('M')
    months = sorted(df_trades['month'].unique())
    
    TOTAL_PENALTY = 0.0050 # 50 bps!
    
    oi_threshs = [-0.05, -0.08, -0.10, -0.12]
    px_threshs = [-0.04, -0.05, -0.06]
    tps = [0.20, 0.30, 0.40]
    sls = [-0.10, -0.15, -0.20]
    
    final_oos_trades = []
    
    for i in range(len(months) - 3):
        is_months = months[i:i+3]
        oos_month = months[i+3]
        
        df_is = df_trades[df_trades['month'].isin(is_months)]
        df_oos = df_trades[df_trades['month'] == oos_month]
        
        if len(df_is) < 10 or len(df_oos) == 0: continue
            
        best_pnl = -np.inf
        best_params = None
        
        for oi_th in oi_threshs:
            for px_th in px_threshs:
                mask_is = (df_is['oi_drop'] <= oi_th) & (df_is['px_drop'] <= px_th)
                train_subset = df_is[mask_is]
                if len(train_subset) < 3: continue
                
                for tp in tps:
                    for sl in sls:
                        train_pnls = []
                        for _, t in train_subset.iterrows():
                            ep = t['entry_price']
                            highs = t['path_high']
                            lows = t['path_low']
                            closes = t['path_close']
                            exit_price = closes[-1]
                            for j in range(len(highs)):
                                if (lows[j] - ep)/ep <= sl:
                                    exit_price = ep * (1+sl)
                                    break
                                elif (highs[j] - ep)/ep >= tp:
                                    exit_price = ep * (1+tp)
                                    break
                            train_pnls.append((exit_price - ep)/ep - TOTAL_PENALTY)
                        tot_pnl = sum(train_pnls)
                        if tot_pnl > best_pnl:
                            best_pnl = tot_pnl
                            best_params = (oi_th, px_th, tp, sl)
                            
        if best_params is None: continue
        oi_th, px_th, tp, sl = best_params
        
        mask_oos = (df_oos['oi_drop'] <= oi_th) & (df_oos['px_drop'] <= px_th)
        test_subset = df_oos[mask_oos]
        
        for idx, t in test_subset.iterrows():
            ep = t['entry_price']
            highs = t['path_high']
            lows = t['path_low']
            closes = t['path_close']
            times = t['path_times']
            exit_price = closes[-1]
            exit_time = times[-1]
            for j in range(len(highs)):
                if (lows[j] - ep)/ep <= sl:
                    exit_price = ep * (1+sl)
                    exit_time = times[j]
                    break
                elif (highs[j] - ep)/ep >= tp:
                    exit_price = ep * (1+tp)
                    exit_time = times[j]
                    break
            net_ret = (exit_price - ep)/ep - TOTAL_PENALTY
            final_oos_trades.append({
                'symbol': t['symbol'],
                'entry_time': t['entry_time'],
                'exit_time': exit_time,
                'net_ret': net_ret
            })

    df_exec = pd.DataFrame(final_oos_trades).sort_values('entry_time').reset_index(drop=True)
    
    print("\n--- MATRIX WITH WFO (NO LOOKAHEAD) + 50 BPS FEES ---")
    
    events = []
    for idx, t in df_exec.iterrows():
        events.append({'time': t['entry_time'], 'type': 'enter', 'trade_idx': idx})
        events.append({'time': t['exit_time'], 'type': 'exit', 'trade_idx': idx})
    events.sort(key=lambda x: x['time'])

    alloc_pcts = [0.05, 0.10, 0.15, 0.20]
    leverages = [1, 2, 3, 5]
    results = []

    for alloc in alloc_pcts:
        for lev in leverages:
            INITIAL_CAPITAL = 10000
            capital = INITIAL_CAPITAL
            MAX_POSITIONS = 5
            active_trades = {}
            bankrupt = False
            
            for ev in events:
                if bankrupt: break
                tid = ev['trade_idx']
                t = df_exec.loc[tid]
                
                if ev['type'] == 'enter':
                    if len(active_trades) < MAX_POSITIONS:
                        symbols_active = [df_exec.loc[at_id, 'symbol'] for at_id in active_trades.keys()]
                        if t['symbol'] not in symbols_active:
                            pos_size = capital * alloc * lev
                            active_trades[tid] = pos_size
                elif ev['type'] == 'exit':
                    if tid in active_trades:
                        pos_size = active_trades.pop(tid)
                        pnl = pos_size * t['net_ret']
                        capital += pnl
                        if capital <= 0:
                            capital = 0
                            bankrupt = True
                            break
                            
            ret_pct = (capital / INITIAL_CAPITAL - 1) * 100
            results.append({
                'Alloc %': f"{alloc*100:.0f}%",
                'Leverage': f"{lev}x",
                'Final Capital': f"${capital:,.0f}",
                'Return %': f"{ret_pct:,.1f}%",
                'Bankrupt': bankrupt
            })

    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
