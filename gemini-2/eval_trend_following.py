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
        
        # 1-hour candles for trend analysis
        kline_1h = kline.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).ffill()
        
        df_sig = kline_1h.dropna()
        if len(df_sig) < 200: return None
        
        # Moving averages
        df_sig['sma_20'] = df_sig['close'].rolling(20).mean()
        df_sig['sma_50'] = df_sig['close'].rolling(50).mean()
        
        # Trend condition: 20 SMA > 50 SMA, and price above 20 SMA
        df_sig['trend_up'] = (df_sig['sma_20'] > df_sig['sma_50']) & (df_sig['close'] > df_sig['sma_20'])
        df_sig['trend_down'] = (df_sig['sma_20'] < df_sig['sma_50']) & (df_sig['close'] < df_sig['sma_20'])
        
        # Entering: crossover
        df_sig['signal_long'] = (df_sig['trend_up'] & ~df_sig['trend_up'].shift(1).fillna(False)).astype(int)
        df_sig['signal_short'] = (df_sig['trend_down'] & ~df_sig['trend_down'].shift(1).fillna(False)).astype(int)
        
        trades = []
        for d in ['long', 'short']:
            signal_times = df_sig[df_sig[f'signal_{d}'] == 1].index
            
            for t_sig in signal_times:
                entry_price = df_sig.loc[t_sig, 'close']
                
                # Check path over next 48 hours (48 * 60 = 2880 mins)
                end_time = t_sig + pd.Timedelta(hours=48)
                path_df = kline.loc[t_sig + pd.Timedelta(minutes=1) : end_time]
                
                if len(path_df) == 0: continue
                
                trades.append({
                    'symbol': symbol,
                    'direction': d,
                    'entry_time': t_sig,
                    'entry_price': entry_price,
                    'path_high': path_df['high'].values.tolist()[:2880], 
                    'path_low': path_df['low'].values.tolist()[:2880],
                    'path_close': path_df['close'].values.tolist()[:2880],
                })

        return trades
    except Exception as e:
        return None

if __name__ == '__main__':
    symbols = get_all_symbols()
    print(f"Extracting Trend Following data on {len(symbols)} symbols...")
    
    trades = []
    with Pool(processes=12) as pool:
        for res_list in tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols), desc="Extracting"):
            if res_list:
                trades.extend(res_list)
                
    if not trades:
        print("No trades found.")
        exit(0)
        
    df_trades = pd.DataFrame(trades)
    print(f"\nFound {len(df_trades)} trend crossovers.")
    
    FEE_BPS = 20
    
    for direction in ['long', 'short']:
        print(f"\n=== {direction.upper()} TREND CROSSOVERS ===")
        dir_trades = [t for t in trades if t['direction'] == direction]
        
        if len(dir_trades) == 0: continue
        
        n_trades = len(dir_trades)
        paths_h = np.array([t['path_high'] + [t['path_high'][-1]] * (2880 - len(t['path_high'])) for t in dir_trades])
        paths_l = np.array([t['path_low'] + [t['path_low'][-1]] * (2880 - len(t['path_low'])) for t in dir_trades])
        paths_c = np.array([t['path_close'] + [t['path_close'][-1]] * (2880 - len(t['path_close'])) for t in dir_trades])
        entries = np.array([t['entry_price'] for t in dir_trades]).reshape(-1, 1)
        
        if direction == 'long':
            ret_c = (paths_c - entries) / entries
        else:
            ret_c = (entries - paths_c) / entries
            
        holds_hours = [4, 12, 24, 48]
        for hold_h in holds_hours:
            hold_m = hold_h * 60
            
            final_ret = ret_c[:, hold_m - 1] - (FEE_BPS/10000)
            
            mean_bps = final_ret.mean() * 10000
            wr = (final_ret > 0).mean() * 100
            
            print(f"Hold {hold_h:2d}h: Net = {mean_bps:6.2f} bps | WR = {wr:.1f}%")

