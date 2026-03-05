import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"

def get_all_symbols():
    return sorted([d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d)) and d.endswith('USDT')])

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
        
        # Flush signal
        # Let's tighten criteria back slightly to avoid trash trades
        df['signal'] = ((df['oi_chg_15m'] < -0.10) & (df['ret_15m'] < -0.05)).astype(int)
        
        trades = []
        signal_indices = np.where(df['signal'] == 1)[0]
        
        for i in signal_indices:
            if i + 48 >= len(df): continue 
            
            entry_time = df.index[i]
            entry_price = df['close'].iloc[i]
            
            fwd_window = df.iloc[i+1 : i+49]
            
            trade_info = {
                'symbol': symbol,
                'entry_time': entry_time,
                'entry_price': entry_price,
                'path_high': fwd_window['high'].values.tolist(),
                'path_low': fwd_window['low'].values.tolist(),
                'path_close': fwd_window['close'].values.tolist()
            }
            trades.append(trade_info)

        return trades
    except Exception as e:
        return None

if __name__ == '__main__':
    symbols = get_all_symbols()
    print(f"Running full data extraction on {len(symbols)} symbols...")
    
    with Pool(processes=12) as pool:
        all_results = pool.map(process_symbol, symbols)
        
    trades = []
    for res_list in all_results:
        if res_list:
            trades.extend(res_list)
            
    if not trades:
        print("No trades found.")
        exit(0)
        
    print(f"Found {len(trades)} extreme trades. Simulating parameter grid...")
    
    FEE_BPS = 20 # 0.2% total
    
    # Notice we removed TP=None and SL=None to force fixed risk management
    tps = [0.05, 0.10, 0.15, 0.20, None] 
    sls = [-0.03, -0.05, -0.08, -0.10, -0.15] 
    holds = [12, 24, 36, 48] # 1h, 2h, 3h, 4h
    
    results_grid = []
    n_trades = len(trades)
    
    for tp in tps:
        for sl in sls:
            for hold in holds:
                total_pnl = 0
                wins = 0
                trade_pnls = []
                
                for t in trades:
                    ep = t['entry_price']
                    highs = t['path_high'][:hold]
                    lows = t['path_low'][:hold]
                    closes = t['path_close'][:hold]
                    
                    exit_price = closes[-1]
                    
                    for step in range(hold):
                        h = highs[step]
                        l = lows[step]
                        
                        ret_h = (h - ep) / ep
                        ret_l = (l - ep) / ep
                        
                        hit_tp = False
                        hit_sl = False
                        
                        if tp is not None and ret_h >= tp: hit_tp = True
                        if sl is not None and ret_l <= sl: hit_sl = True
                        
                        # Within the same 5m candle, if both hit, assume SL hit first (conservative)
                        if hit_sl:
                            exit_price = ep * (1 + sl)
                            break
                        elif hit_tp:
                            exit_price = ep * (1 + tp)
                            break
                            
                    net_ret = (exit_price - ep) / ep - (FEE_BPS/10000)
                    trade_pnls.append(net_ret)
                    if net_ret > 0: wins += 1
                
                trade_pnls = np.array(trade_pnls)
                mean_pnl = trade_pnls.mean() * 10000
                wr = wins / n_trades * 100
                sharpe = trade_pnls.mean() / (trade_pnls.std() + 1e-9) * np.sqrt(n_trades) # pseudo-sharpe
                
                results_grid.append({
                    'TP': tp if tp else 'None',
                    'SL': sl,
                    'Hold_h': hold/12,
                    'Net_bps': mean_pnl,
                    'WR': wr,
                    'Sharpe': sharpe
                })

    df_grid = pd.DataFrame(results_grid)
    print("\nTop 15 Configurations (sorted by Sharpe, minimum WR 40%):")
    valid_grid = df_grid[df_grid['WR'] >= 40]
    print(valid_grid.sort_values('Sharpe', ascending=False).head(15).to_string(index=False))

