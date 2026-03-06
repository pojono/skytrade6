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
    if not files: return None
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
    df = df[(df['ret'] > -0.15) & (df['ret'] < 0.15)]
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp')
    df.dropna(inplace=True)
    if df.empty or len(df) < (60 * 24 * 7): return None
    return symbol, df

def calc_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def calc_adx(df, period=14):
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff(-1) * -1
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    
    atr = calc_atr(df, period)
    plus_di = 100 * (plus_dm.rolling(period).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(period).sum() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return adx

def calc_choppiness(df, period=14):
    atr = calc_atr(df, 1)
    atr_sum = atr.rolling(period).sum()
    max_high = df['high'].rolling(period).max()
    min_low = df['low'].rolling(period).min()
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(period)
    return chop

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    
    df_4h = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df_4h.dropna(inplace=True)
    
    # Existing KER
    period = 21
    change = abs(df_4h['close'] - df_4h['close'].shift(period))
    volatility = abs(df_4h['close'] - df_4h['close'].shift(1)).rolling(period).sum()
    df_4h['ker'] = change / volatility
    
    # ADX
    df_4h['adx'] = calc_adx(df_4h, 14)
    
    # Choppiness Index
    df_4h['chop'] = calc_choppiness(df_4h, 14)
    
    # Basic Breakout Logic
    df_4h['ema_200'] = df_4h['close'].ewm(span=200, adjust=False).mean()
    df_4h['macro_bull'] = df_4h['close'] > df_4h['ema_200']
    df_4h['high_20'] = df_4h['high'].rolling(20).max().shift(1)
    df_4h['low_20'] = df_4h['low'].rolling(20).min().shift(1)
    df_4h['vol_ma'] = df_4h['volume'].rolling(20).mean().shift(1)
    df_4h['vol_spike'] = df_4h['volume'] > (df_4h['vol_ma'] * 2.0)
    
    base_long = (df_4h['close'] > df_4h['high_20']) & df_4h['macro_bull'] & df_4h['vol_spike']
    base_short = (df_4h['close'] < df_4h['low_20']) & (~df_4h['macro_bull']) & df_4h['vol_spike']
    
    configs = [
        {'name': 'No_Filter', 'filter': pd.Series(True, index=df_4h.index)},
        {'name': 'KER_Standard (>0.15)', 'filter': df_4h['ker'] >= 0.15},
        {'name': 'KER_Strict (>0.25)', 'filter': df_4h['ker'] >= 0.25},
        {'name': 'ADX_Standard (>20)', 'filter': df_4h['adx'] >= 20},
        {'name': 'ADX_Strict (>25)', 'filter': df_4h['adx'] >= 25},
        {'name': 'Chop_Index (<61.8)', 'filter': df_4h['chop'] <= 61.8},
        {'name': 'Chop_Strict (<50)', 'filter': df_4h['chop'] <= 50},
        {'name': 'KER + Chop', 'filter': (df_4h['ker'] >= 0.15) & (df_4h['chop'] <= 61.8)}
    ]
    
    results = []
    TP_PCT = 0.20
    SL_PCT = 0.10
    TIME_STOP_CANDLES = 6 * 14
    
    opens = df_4h['open'].values
    highs = df_4h['high'].values
    lows = df_4h['low'].values
    closes = df_4h['close'].values
    
    for cfg in configs:
        f_long = (base_long & cfg['filter']).values
        f_short = (base_short & cfg['filter']).values
        
        trades = []
        in_position = False
        entry_price = 0.0
        entry_idx = 0
        cooldown_until = 0
        pos_type = 0
        
        for i in range(200, len(df_4h) - 1):
            if not in_position:
                if i > cooldown_until:
                    if f_long[i]:
                        in_position = True
                        pos_type = 1
                        entry_price = opens[i+1]
                        entry_idx = i + 1
                    elif f_short[i]:
                        in_position = True
                        pos_type = -1
                        entry_price = opens[i+1]
                        entry_idx = i + 1
            else:
                if pos_type == 1:
                    target_price = entry_price * (1 + TP_PCT)
                    stop_price = entry_price * (1 - SL_PCT)
                    if lows[i] <= stop_price:
                        net_pnl = ((stop_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                        trades.append({'symbol': symbol, 'config': cfg['name'], 'pnl': net_pnl})
                        in_position = False
                        cooldown_until = i + 6
                    elif highs[i] >= target_price:
                        net_pnl = ((target_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE
                        trades.append({'symbol': symbol, 'config': cfg['name'], 'pnl': net_pnl})
                        in_position = False
                        cooldown_until = i + 6
                    elif (i - entry_idx) >= TIME_STOP_CANDLES:
                        net_pnl = ((closes[i] - entry_price) / entry_price) - (TAKER_FEE * 2)
                        trades.append({'symbol': symbol, 'config': cfg['name'], 'pnl': net_pnl})
                        in_position = False
                        cooldown_until = i + 6
                elif pos_type == -1:
                    target_price = entry_price * (1 - TP_PCT)
                    stop_price = entry_price * (1 + SL_PCT)
                    if highs[i] >= stop_price:
                        net_pnl = ((entry_price - stop_price) / entry_price) - (TAKER_FEE * 2)
                        trades.append({'symbol': symbol, 'config': cfg['name'], 'pnl': net_pnl})
                        in_position = False
                        cooldown_until = i + 6
                    elif lows[i] <= target_price:
                        net_pnl = ((entry_price - target_price) / entry_price) - TAKER_FEE - MAKER_FEE
                        trades.append({'symbol': symbol, 'config': cfg['name'], 'pnl': net_pnl})
                        in_position = False
                        cooldown_until = i + 6
                    elif (i - entry_idx) >= TIME_STOP_CANDLES:
                        net_pnl = ((entry_price - closes[i]) / entry_price) - (TAKER_FEE * 2)
                        trades.append({'symbol': symbol, 'config': cfg['name'], 'pnl': net_pnl})
                        in_position = False
                        cooldown_until = i + 6
        results.extend(trades)
    return results

def main():
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT', 'AVAXUSDT', 'LINKUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    print("Testing Various Chop Filters...")
    
    all_trades = []
    pool = multiprocessing.Pool(processes=min(8, multiprocessing.cpu_count()))
    
    for trades in tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols)):
        all_trades.extend(trades)
            
    pool.close()
    pool.join()
                
    if not all_trades: return
    df_trades = pd.DataFrame(all_trades)
    
    print("\n" + "="*80)
    print("--- CHOP FILTER EXPERIMENTS ---")
    print("="*80)
    print(f"{'Configuration':<20} | {'Trades':<8} | {'Win Rate':<10} | {'EV/Trade':<10} | {'Total Ret (2% risk)':<20}")
    print("-" * 80)
    
    for config, group in df_trades.groupby('config'):
        trades = len(group)
        wins = sum(group['pnl'] > 0)
        win_rate = wins / trades
        ev = group['pnl'].mean()
        equity_return = group['pnl'].sum() * 0.20
        print(f"{config:<20} | {trades:<8} | {win_rate:<10.2%} | {ev:<10.2%} | {equity_return:<20.2%}")

if __name__ == "__main__":
    main()
