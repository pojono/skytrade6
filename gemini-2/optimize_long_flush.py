import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool
import warnings
from tqdm import tqdm
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
        
        df['sig_long_base'] = ((df['oi_chg_15m'] < -0.05) & (df['ret_15m'] < -0.04)).astype(int)
        
        trades = []
        indices = np.where(df['sig_long_base'] == 1)[0]
        
        for i in indices:
            if i + 48 >= len(df): continue 
            
            entry_time = df.index[i]
            entry_price = df['close'].iloc[i]
            
            fwd_window = df.iloc[i+1 : i+49]
            
            trades.append({
                'symbol': symbol,
                'entry_time': entry_time,
                'entry_price': entry_price,
                'ret_15m': df['ret_15m'].iloc[i],
                'oi_chg_15m': df['oi_chg_15m'].iloc[i],
                'path_high': fwd_window['high'].values.tolist(),
                'path_low': fwd_window['low'].values.tolist(),
                'path_close': fwd_window['close'].values.tolist()
            })

        return trades
    except Exception as e:
        return None

if __name__ == '__main__':
    symbols = get_all_symbols()
    print(f"Extracting baseline long flush trades...")
    
    trades = []
    with Pool(processes=12) as pool:
        for res_list in tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols)):
            if res_list:
                trades.extend(res_list)
                
    if not trades:
        print("No trades found.")
        exit(0)
        
    df_trades = pd.DataFrame(trades)
    print(f"Extracted {len(df_trades)} base trades.")
    
    FEE_BPS = 20
    
    configs = [
        {'name': 'Moderate', 'oi_drop': -0.08, 'px_drop': -0.05, 'tp': 0.15, 'sl': -0.08, 'hold': 24},
        {'name': 'Strict', 'oi_drop': -0.10, 'px_drop': -0.06, 'tp': 0.20, 'sl': -0.10, 'hold': 36},
        {'name': 'Ultra_Strict', 'oi_drop': -0.15, 'px_drop': -0.08, 'tp': 0.30, 'sl': -0.10, 'hold': 24}
    ]
    
    for cfg in configs:
        mask = (df_trades['oi_chg_15m'] < cfg['oi_drop']) & (df_trades['ret_15m'] < cfg['px_drop'])
        filtered_trades = df_trades[mask].sort_values('entry_time')
        
        INITIAL_CAPITAL = 10000
        capital = INITIAL_CAPITAL
        MAX_POSITIONS = 5
        pos_size = INITIAL_CAPITAL / MAX_POSITIONS
        
        events = []
        executed_count = 0
        active_trades = set()
        eq_curve = []
        
        for idx, t in filtered_trades.iterrows():
            events.append({'time': t['entry_time'], 'type': 'enter', 'trade': t, 'id': idx})
            
        events.sort(key=lambda x: x['time'])
        
        # Chronological simulation
        # Need to pre-calculate exits to know when they free up slots
        exit_events = []
        
        # It's easier to simulate step by step
        sim_trades = []
        for idx, t in filtered_trades.iterrows():
            ep = t['entry_price']
            hold = cfg['hold']
            highs = t['path_high'][:hold]
            lows = t['path_low'][:hold]
            closes = t['path_close'][:hold]
            
            exit_price = closes[-1]
            exit_time = t['entry_time'] + pd.Timedelta(minutes=5*hold)
            
            for j in range(len(highs)):
                ret_h = (highs[j] - ep) / ep
                ret_l = (lows[j] - ep) / ep
                if ret_l <= cfg['sl']:
                    exit_price = ep * (1 + cfg['sl'])
                    exit_time = t['entry_time'] + pd.Timedelta(minutes=5*(j+1))
                    break
                elif ret_h >= cfg['tp']:
                    exit_price = ep * (1 + cfg['tp'])
                    exit_time = t['entry_time'] + pd.Timedelta(minutes=5*(j+1))
                    break
                    
            net_ret = (exit_price - ep) / ep - (FEE_BPS/10000)
            
            sim_trades.append({
                'id': idx,
                'symbol': t['symbol'],
                'entry_time': t['entry_time'],
                'exit_time': exit_time,
                'net_ret': net_ret
            })
            
        df_sim = pd.DataFrame(sim_trades).sort_values('entry_time')
        
        # Portfolio queue
        port_events = []
        for _, st in df_sim.iterrows():
            port_events.append({'time': st['entry_time'], 'type': 'enter', 'trade': st})
            port_events.append({'time': st['exit_time'], 'type': 'exit', 'trade': st})
            
        port_events.sort(key=lambda x: x['time'])
        
        active_ids = set()
        portfolio_log = []
        
        for ev in port_events:
            tr = ev['trade']
            tid = tr['id']
            if ev['type'] == 'enter':
                if len(active_ids) < MAX_POSITIONS:
                    symbols_active = [df_sim[df_sim['id'] == atid]['symbol'].iloc[0] for atid in active_ids]
                    if tr['symbol'] not in symbols_active:
                        active_ids.add(tid)
            elif ev['type'] == 'exit':
                if tid in active_ids:
                    active_ids.remove(tid)
                    pnl = pos_size * tr['net_ret']
                    capital += pnl
                    eq_curve.append({'time': ev['time'], 'capital': capital})
                    portfolio_log.append(tr)
                    
        df_port = pd.DataFrame(portfolio_log)
        df_eq = pd.DataFrame(eq_curve)
        
        ret_pct = (capital / INITIAL_CAPITAL - 1) * 100
        
        print(f"\n--- Config: {cfg['name']} ---")
        print(f"Condition: OI < {cfg['oi_drop']*100}%, Px < {cfg['px_drop']*100}% | TP {cfg['tp']*100}%, SL {cfg['sl']*100}%, Hold {cfg['hold']/12}h")
        if len(df_port) > 0:
            max_dd = ((df_eq['capital'].cummax() - df_eq['capital']) / df_eq['capital'].cummax()).max() * 100
            print(f"Trades Taken: {len(df_port)}")
            print(f"Win Rate: {(df_port['net_ret'] > 0).mean() * 100:.1f}%")
            print(f"Mean Net Return/Trade: {df_port['net_ret'].mean() * 10000:.2f} bps")
            print(f"Total Portfolio Return: {ret_pct:.2f}%")
            print(f"Max Drawdown: {max_dd:.2f}%")
            print(f"Return / Max DD: {ret_pct / max_dd if max_dd > 0 else 0:.2f}")

