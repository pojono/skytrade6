import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
MAKER_FEE = 0.0004
TAKER_FEE = 0.0010

def load_symbol_data(symbol):
    files = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_kline_1m.csv")
    files_oi = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_open_interest_5min.csv")
    if not files or not files_oi: return None
        
    df_list = []
    for f in files:
        try: df_list.append(pd.read_csv(f))
        except: pass
    if not df_list: return None
        
    df = pd.concat(df_list, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df.get('startTime', df.get('timestamp', None)), unit='ms')
         
    df = df[(df['close'] > 0) & (df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0)]
    df = df[df['high'] >= df['low']]
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['ret'] = df['close'].pct_change()
    df = df[(df['ret'] > -0.5) & (df['ret'] < 1.0)]
    
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp')
    df = df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
    df.dropna(inplace=True)
    
    oi_list = []
    for f in files_oi:
        try: oi_list.append(pd.read_csv(f))
        except: pass
    if oi_list:
        df_oi = pd.concat(oi_list, ignore_index=True)
        df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'], unit='ms')
        df_oi = df_oi.drop_duplicates(subset=['timestamp']).sort_values('timestamp').set_index('timestamp')
        df_oi = df_oi.resample('15min').ffill()
        df = df.join(df_oi[['openInterest']], how='left').ffill()
    else: return None
    
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
    if df.empty or len(df) < (7 * 24 * 4): return None
    return symbol, df

def process_symbol(args):
    symbol, params = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    
    # Unpack params
    p_drop = params['drop']
    p_oi = params['oi_drop']
    p_tp = params['tp']
    p_sl = params['sl']
    
    df['ret_8h'] = df['close'] / df['close'].shift(32) - 1 
    if 'openInterest' not in df.columns: return []
    df['oi_8h'] = df['openInterest'] / df['openInterest'].shift(32) - 1
    
    df['flush_zone'] = (df['ret_8h'] < p_drop) & (df['ret_8h'] > -0.50) & (df['oi_8h'] < p_oi)
    df['flush_recent'] = df['flush_zone'].rolling(16).max().fillna(0).astype(bool) 
    
    df['ret_1h'] = df['close'] / df['close'].shift(4) - 1
    df['bounce_conf'] = df['ret_1h'] > 0.03
    
    signals = (df['flush_recent'] & df['bounce_conf']).values
    opens = df['open'].values # Use NEXT candle open for entry to be strictly realistic
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    trades = []
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    cooldown_until = 0
    n = len(df)
    
    for i in range(32, n - 1):
        if not in_position:
            if signals[i] and i > cooldown_until:
                in_position = True
                entry_price = opens[i+1] # Entering on the OPEN of the NEXT candle (no lookahead)
                entry_idx = i + 1
        else:
            target_price = entry_price * (1 + p_tp)
            stop_price = entry_price * (1 + p_sl)
            
            # Since we check 15m candles, pessimistic execution: SL hits before TP if both in same candle
            if lows[i] <= stop_price:
                net_pnl = ((stop_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append(net_pnl)
                in_position = False
                cooldown_until = i + 32 
                
            elif highs[i] >= target_price:
                net_pnl = ((target_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE
                trades.append(net_pnl)
                in_position = False
                cooldown_until = i + 32
                
            elif (i - entry_idx) >= 96 * 2: # 48 hours
                net_pnl = ((closes[i] - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append(net_pnl)
                in_position = False
                cooldown_until = i + 32

    return [t for t in trades]

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    # Subset to 30 symbols to run matrix quickly
    test_symbols = symbols[:30]
    
    # Sensitivity matrix
    param_sets = [
        {'name': 'Base', 'drop': -0.12, 'oi_drop': -0.15, 'tp': 0.18, 'sl': -0.12},
        {'name': 'Tighter Entry', 'drop': -0.10, 'oi_drop': -0.10, 'tp': 0.18, 'sl': -0.12},
        {'name': 'Extreme Entry', 'drop': -0.15, 'oi_drop': -0.20, 'tp': 0.18, 'sl': -0.12},
        {'name': 'Tighter SL', 'drop': -0.12, 'oi_drop': -0.15, 'tp': 0.18, 'sl': -0.08},
        {'name': 'Wider TP', 'drop': -0.12, 'oi_drop': -0.15, 'tp': 0.25, 'sl': -0.12},
    ]
    
    print(f"Running Parameter Sensitivity Audit on 30 symbols (with Strict Execution Reality)...")
    for params in param_sets:
        tasks = [(sym, params) for sym in test_symbols]
        all_pnls = []
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
            for pnl_list in pool.imap_unordered(process_symbol, tasks):
                all_pnls.extend(pnl_list)
                
        if len(all_pnls) > 0:
            win_rate = sum(1 for x in all_pnls if x > 0) / len(all_pnls)
            avg_pnl = np.mean(all_pnls)
            print(f"[{params['name']}] Trades: {len(all_pnls):<4} | Win Rate: {win_rate:.2%} | Avg PnL: {avg_pnl:.4%}")
        else:
            print(f"[{params['name']}] No trades generated.")

if __name__ == "__main__":
    main()
