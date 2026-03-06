import pandas as pd
import numpy as np
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
        
        # Base Long Entry
        df['sig_long'] = ((df['oi_chg_15m'] <= -0.10) & (df['ret_15m'] <= -0.05)).astype(int)
        
        trades = []
        indices = np.where(df['sig_long'] == 1)[0]
        
        HOLD = 72 # 6 hours
        
        for i in indices:
            if i + HOLD >= len(df): continue 
            
            entry_time = df.index[i]
            entry_price = df['close'].iloc[i]
            
            fwd_window = df.iloc[i+1 : i+1+HOLD]
            highs = fwd_window['high'].values
            lows = fwd_window['low'].values
            closes = fwd_window['close'].values
            
            trades.append({
                'symbol': symbol,
                'entry_time': entry_time,
                'entry_price': entry_price,
                'path_high': highs,
                'path_low': lows,
                'path_close': closes
            })

        return trades
    except Exception as e:
        return None

if __name__ == '__main__':
    symbols = sorted([d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d)) and d.endswith('USDT')])
    print(f"Extracting signals for slippage modeling...")
    
    trades = []
    with Pool(processes=12) as pool:
        for res_list in tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols)):
            if res_list: trades.extend(res_list)
                
    df_trades = pd.DataFrame(trades)
    print(f"Extracted {len(df_trades)} base trades.")
    
    # We test different levels of slippage.
    # Slippage applies to entry price (worse entry) AND exit price (worse exit)
    # Total roundtrip fee = 2 * taker (0.10%) + entry slippage + exit slippage
    TAKER_FEE = 0.0010
    TP = 0.40
    SL = -0.20
    HOLD = 72
    
    slippages = [0.0000, 0.0010, 0.0020, 0.0030, 0.0050, 0.0100] # 0 bps, 10 bps, 20 bps, 30 bps, 50 bps, 100 bps
    
    print("\n--- Slippage Sensitivity Analysis ---")
    print("Assuming 0.10% Taker Fee per leg. Baseline 0 bps slippage = 20 bps total roundtrip.")
    
    results = []
    for slip in slippages:
        # Penalize entry
        penalty_entry = TAKER_FEE + slip
        # Penalize exit
        penalty_exit = TAKER_FEE + slip
        
        total_ret = []
        wins = 0
        for _, t in df_trades.iterrows():
            # Actual entry price is higher than signal close due to slip
            ep_real = t['entry_price'] * (1 + penalty_entry)
            
            highs = t['path_high']
            lows = t['path_low']
            closes = t['path_close']
            
            exit_price = closes[-1]
            for j in range(len(highs)):
                # Evaluate TP/SL based on real entry
                ret_h = (highs[j] - ep_real) / ep_real
                ret_l = (lows[j] - ep_real) / ep_real
                
                if ret_l <= SL:
                    exit_price = ep_real * (1 + SL)
                    break
                elif ret_h >= TP:
                    exit_price = ep_real * (1 + TP)
                    break
                    
            # Actual exit price is lower due to slip
            actual_exit = exit_price * (1 - penalty_exit)
            
            net_ret = (actual_exit - ep_real) / ep_real
            total_ret.append(net_ret)
            if net_ret > 0: wins += 1
            
        mean_bps = np.mean(total_ret) * 10000
        wr = wins / len(total_ret) * 100
        total_pnl = sum(total_ret) * 100
        
        print(f"Slippage: {slip*10000:3.0f} bps/leg | Net Roundtrip Cost: {(penalty_entry + penalty_exit)*10000:3.0f} bps | Mean Ret: {mean_bps:7.2f} bps | WR: {wr:.1f}% | Total unlev PnL: {total_pnl:.1f}%")

