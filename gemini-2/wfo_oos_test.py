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
        
        # Broad filter to extract possible signals
        df['sig_broad'] = ((df['oi_chg_15m'] < -0.04) & (df['ret_15m'] < -0.03)).astype(int)
        
        trades = []
        indices = np.where(df['sig_broad'] == 1)[0]
        
        HOLD = 72 # max hold 6h
        
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
                'oi_drop': df['oi_chg_15m'].iloc[i],
                'px_drop': df['ret_15m'].iloc[i],
                'path_high': highs,
                'path_low': lows,
                'path_close': closes
            })

        return trades
    except Exception as e:
        return None

if __name__ == '__main__':
    symbols = sorted([d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d)) and d.endswith('USDT')])
    print("Extracting base trades for Walk-Forward Optimization...")
    
    trades = []
    with Pool(processes=12) as pool:
        for res_list in tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols)):
            if res_list: trades.extend(res_list)
                
    df_trades = pd.DataFrame(trades).sort_values('entry_time')
    df_trades['month'] = df_trades['entry_time'].dt.to_period('M')
    
    print(f"Extracted {len(df_trades)} base trades.")
    
    # Simple Walk-Forward Optimization
    # We will use 3 months of In-Sample (IS) data to pick the best params,
    # then apply them to the following 1 month of Out-of-Sample (OOS) data.
    
    months = sorted(df_trades['month'].unique())
    
    # Parameter grid
    oi_threshs = [-0.05, -0.08, -0.10, -0.12]
    px_threshs = [-0.04, -0.05, -0.06, -0.08]
    tps = [0.20, 0.30, 0.40]
    sls = [-0.10, -0.15, -0.20]
    
    FEE_BPS = 20 # 20 bps base fees
    SLIPPAGE_BPS = 30 # 30 bps roundtrip slippage penalty
    TOTAL_PENALTY = (FEE_BPS + SLIPPAGE_BPS) / 10000
    
    oos_results = []
    
    for i in range(len(months) - 3):
        is_months = months[i:i+3]
        oos_month = months[i+3]
        
        df_is = df_trades[df_trades['month'].isin(is_months)]
        df_oos = df_trades[df_trades['month'] == oos_month]
        
        if len(df_is) < 10 or len(df_oos) == 0:
            continue
            
        best_pnl = -np.inf
        best_params = None
        
        # Train
        for oi_th in oi_threshs:
            for px_th in px_threshs:
                mask_is = (df_is['oi_drop'] <= oi_th) & (df_is['px_drop'] <= px_th)
                train_subset = df_is[mask_is]
                
                if len(train_subset) < 5: continue
                
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
                            
                        total_train_pnl = sum(train_pnls)
                        if total_train_pnl > best_pnl:
                            best_pnl = total_train_pnl
                            best_params = (oi_th, px_th, tp, sl)
                            
        # Test (OOS)
        if best_params is None:
            continue
            
        oi_th, px_th, tp, sl = best_params
        mask_oos = (df_oos['oi_drop'] <= oi_th) & (df_oos['px_drop'] <= px_th)
        test_subset = df_oos[mask_oos]
        
        test_pnls = []
        for _, t in test_subset.iterrows():
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
                    
            test_pnls.append((exit_price - ep)/ep - TOTAL_PENALTY)
            
        oos_results.append({
            'OOS_Month': str(oos_month),
            'Trades': len(test_subset),
            'WinRate': (np.array(test_pnls) > 0).mean() * 100 if test_pnls else 0,
            'Net_PnL_bps': sum(test_pnls) * 10000 if test_pnls else 0,
            'Opt_OI': oi_th,
            'Opt_Px': px_th,
            'Opt_TP': tp,
            'Opt_SL': sl
        })
        
    df_res = pd.DataFrame(oos_results)
    print("\n=== WALK-FORWARD OPTIMIZATION (HONEST OOS) ===")
    print("Includes 50 bps roundtrip penalty (20 bps fee + 30 bps slippage)")
    print(df_res.to_string(index=False))
    
    total_trades = df_res['Trades'].sum()
    total_pnl = df_res['Net_PnL_bps'].sum()
    print(f"\nTotal OOS Trades: {total_trades}")
    print(f"Total OOS Unleveraged Net PnL: {total_pnl/10000 * 100:.2f}%")
    if total_trades > 0:
        print(f"Average PnL per trade: {total_pnl / total_trades:.2f} bps")
        
