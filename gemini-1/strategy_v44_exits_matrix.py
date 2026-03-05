import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
MAKER_FEE = 0.0004
TAKER_FEE = 0.0010

def get_btc_regime():
    files = glob.glob(f"{DATALAKE_DIR}/BTCUSDT/*_kline_1m.csv")
    if not files: return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['timestamp'] = pd.to_datetime(df.get('startTime', df.get('timestamp', None)), unit='ms')
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).set_index('timestamp')
    df_4h = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
    df_4h.dropna(inplace=True)
    period = 21
    change = abs(df_4h['close'] - df_4h['close'].shift(period))
    volatility = abs(df_4h['close'] - df_4h['close'].shift(1)).rolling(period).sum()
    df_4h['btc_ker'] = change / volatility
    df_4h['btc_is_trending'] = df_4h['btc_ker'] >= 0.20
    return df_4h[['btc_is_trending']]

BTC_REGIME = get_btc_regime()

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
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
    if df.empty or len(df) < (60 * 24 * 7): return None
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    
    df_4h = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df_4h.dropna(inplace=True)
    
    if BTC_REGIME is not None:
        df_4h = df_4h.join(BTC_REGIME, how='left').ffill()
    else:
        df_4h['btc_is_trending'] = True
        
    period = 21
    change = abs(df_4h['close'] - df_4h['close'].shift(period))
    volatility = abs(df_4h['close'] - df_4h['close'].shift(1)).rolling(period).sum()
    df_4h['local_ker'] = change / volatility
    df_4h['local_is_trending'] = df_4h['local_ker'] >= 0.15
    
    df_4h['ema_200'] = df_4h['close'].ewm(span=200, adjust=False).mean()
    df_4h['macro_bull'] = df_4h['close'] > df_4h['ema_200']
    
    df_4h['high_20'] = df_4h['high'].rolling(20).max().shift(1)
    df_4h['low_20'] = df_4h['low'].rolling(20).min().shift(1)
    
    df_4h['vol_ma'] = df_4h['volume'].rolling(20).mean().shift(1)
    df_4h['vol_spike'] = df_4h['volume'] > (df_4h['vol_ma'] * 2.0)
    
    df_4h['regime_ok'] = df_4h['btc_is_trending'] | df_4h['local_is_trending']
    df_4h['long_breakout'] = (df_4h['close'] > df_4h['high_20']) & df_4h['macro_bull'] & df_4h['vol_spike'] & df_4h['regime_ok']
    df_4h['short_breakout'] = (df_4h['close'] < df_4h['low_20']) & (~df_4h['macro_bull']) & df_4h['vol_spike'] & df_4h['regime_ok']
    
    signals_long = df_4h['long_breakout'].values
    signals_short = df_4h['short_breakout'].values
    opens = df_4h['open'].values
    highs = df_4h['high'].values
    lows = df_4h['low'].values
    closes = df_4h['close'].values
    
    # Define execution matrices
    # Exits: [ (TP, SL, Name) ]
    matrices = [
        (0.10, 0.05, "TP 10% / SL 5%"),
        (0.15, 0.05, "TP 15% / SL 5% (Base)"),
        (0.20, 0.05, "TP 20% / SL 5%"),
        (0.25, 0.05, "TP 25% / SL 5%"),
        (0.15, 0.03, "TP 15% / SL 3%"),
        (0.15, 0.07, "TP 15% / SL 7%"),
        (0.20, 0.10, "TP 20% / SL 10%")
    ]
    
    TIME_STOP_CANDLES = 6 * 14
    n = len(df_4h)
    
    # Find all entries (to ensure fair comparison, all strategies take the exact same entries)
    # We use a simple state machine to find valid entries given a cooldown
    entries = []
    cooldown_until = 0
    for i in range(200, n - 1):
        if i > cooldown_until:
            if signals_long[i]:
                entries.append({'idx': i+1, 'type': 1, 'price': opens[i+1], 'time': df_4h.index[i+1]})
                cooldown_until = i + 6
            elif signals_short[i]:
                entries.append({'idx': i+1, 'type': -1, 'price': opens[i+1], 'time': df_4h.index[i+1]})
                cooldown_until = i + 6
                
    results = []
    
    for entry in entries:
        idx = entry['idx']
        pos_type = entry['type']
        entry_price = entry['price']
        
        # Test Standard Matrices
        for tp_pct, sl_pct, name in matrices:
            trade_pnl = 0
            for j in range(idx, min(idx + TIME_STOP_CANDLES, n)):
                if pos_type == 1:
                    tp_price = entry_price * (1 + tp_pct)
                    sl_price = entry_price * (1 - sl_pct)
                    if lows[j] <= sl_price:
                        trade_pnl = ((sl_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                        break
                    elif highs[j] >= tp_price:
                        trade_pnl = ((tp_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE
                        break
                else:
                    tp_price = entry_price * (1 - tp_pct)
                    sl_price = entry_price * (1 + sl_pct)
                    if highs[j] >= sl_price:
                        trade_pnl = ((entry_price - sl_price) / entry_price) - (TAKER_FEE * 2)
                        break
                    elif lows[j] <= tp_price:
                        trade_pnl = ((entry_price - tp_price) / entry_price) - TAKER_FEE - MAKER_FEE
                        break
            else:
                # Time stop
                if pos_type == 1: trade_pnl = ((closes[min(idx + TIME_STOP_CANDLES - 1, n-1)] - entry_price) / entry_price) - (TAKER_FEE * 2)
                else: trade_pnl = ((entry_price - closes[min(idx + TIME_STOP_CANDLES - 1, n-1)]) / entry_price) - (TAKER_FEE * 2)
                
            results.append({'matrix': name, 'pnl': trade_pnl})
            
        # Test Partial TP (Sell 50% at 10%, move SL to BE, Sell 50% at 20%)
        p_tp1 = 0.10
        p_tp2 = 0.20
        p_sl = 0.05
        state = 'full'
        trade_pnl = 0
        current_sl = p_sl
        for j in range(idx, min(idx + TIME_STOP_CANDLES, n)):
            if pos_type == 1:
                tp1_price = entry_price * (1 + p_tp1)
                tp2_price = entry_price * (1 + p_tp2)
                sl_price = entry_price * (1 - current_sl)
                
                if lows[j] <= sl_price:
                    if state == 'full':
                        trade_pnl = ((sl_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                    else:
                        # 50% already sold at TP1. Remaining 50% stopped at BE (0%)
                        trade_pnl += 0.5 * (0 - TAKER_FEE - TAKER_FEE)
                    break
                elif highs[j] >= tp1_price and state == 'full':
                    # Hit TP1
                    trade_pnl += 0.5 * (((tp1_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE)
                    state = 'half'
                    current_sl = -0.002 # Move SL to BE (cover fees)
                    # Check if it also hit TP2 in the same candle
                    if highs[j] >= tp2_price:
                        trade_pnl += 0.5 * (((tp2_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE)
                        break
                elif highs[j] >= tp2_price and state == 'half':
                    trade_pnl += 0.5 * (((tp2_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE)
                    break
            else:
                tp1_price = entry_price * (1 - p_tp1)
                tp2_price = entry_price * (1 - p_tp2)
                sl_price = entry_price * (1 + current_sl)
                
                if highs[j] >= sl_price:
                    if state == 'full':
                        trade_pnl = ((entry_price - sl_price) / entry_price) - (TAKER_FEE * 2)
                    else:
                        trade_pnl += 0.5 * (0 - TAKER_FEE - TAKER_FEE)
                    break
                elif lows[j] <= tp1_price and state == 'full':
                    trade_pnl += 0.5 * (((entry_price - tp1_price) / entry_price) - TAKER_FEE - MAKER_FEE)
                    state = 'half'
                    current_sl = -0.002
                    if lows[j] <= tp2_price:
                        trade_pnl += 0.5 * (((entry_price - tp2_price) / entry_price) - TAKER_FEE - MAKER_FEE)
                        break
                elif lows[j] <= tp2_price and state == 'half':
                    trade_pnl += 0.5 * (((entry_price - tp2_price) / entry_price) - TAKER_FEE - MAKER_FEE)
                    break
        else:
            if state == 'full':
                if pos_type == 1: trade_pnl = ((closes[min(idx + TIME_STOP_CANDLES - 1, n-1)] - entry_price) / entry_price) - (TAKER_FEE * 2)
                else: trade_pnl = ((entry_price - closes[min(idx + TIME_STOP_CANDLES - 1, n-1)]) / entry_price) - (TAKER_FEE * 2)
            else:
                if pos_type == 1: trade_pnl += 0.5 * (((closes[min(idx + TIME_STOP_CANDLES - 1, n-1)] - entry_price) / entry_price) - (TAKER_FEE * 2))
                else: trade_pnl += 0.5 * (((entry_price - closes[min(idx + TIME_STOP_CANDLES - 1, n-1)]) / entry_price) - (TAKER_FEE * 2))
                
        results.append({'matrix': 'Partial TP (10/20) + BE', 'pnl': trade_pnl})
        
        # Test Trailing SL (5% Trailing from highest watermark)
        trail_pct = 0.07 # 7% trailing stop
        high_watermark = entry_price
        low_watermark = entry_price
        trade_pnl = 0
        for j in range(idx, min(idx + TIME_STOP_CANDLES, n)):
            if pos_type == 1:
                high_watermark = max(high_watermark, highs[j])
                trail_sl = high_watermark * (1 - trail_pct)
                # Hard floor at original 5% SL
                trail_sl = max(trail_sl, entry_price * (1 - 0.05))
                
                if lows[j] <= trail_sl:
                    trade_pnl = ((trail_sl - entry_price) / entry_price) - (TAKER_FEE * 2)
                    break
            else:
                low_watermark = min(low_watermark, lows[j])
                trail_sl = low_watermark * (1 + trail_pct)
                trail_sl = min(trail_sl, entry_price * (1 + 0.05))
                
                if highs[j] >= trail_sl:
                    trade_pnl = ((entry_price - trail_sl) / entry_price) - (TAKER_FEE * 2)
                    break
        else:
            if pos_type == 1: trade_pnl = ((closes[min(idx + TIME_STOP_CANDLES - 1, n-1)] - entry_price) / entry_price) - (TAKER_FEE * 2)
            else: trade_pnl = ((entry_price - closes[min(idx + TIME_STOP_CANDLES - 1, n-1)]) / entry_price) - (TAKER_FEE * 2)
            
        results.append({'matrix': 'Trailing SL 7%', 'pnl': trade_pnl})

    return results

def main():
    # Subset of 30 coins for faster optimization processing
    all_symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    symbols = all_symbols[:40]
    print(f"Running Optimization Matrix on {len(symbols)} Coins...")
    
    all_results = []
    pool = multiprocessing.Pool(processes=min(8, multiprocessing.cpu_count()))
    
    for results in tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols)):
        all_results.extend(results)
            
    pool.close()
    pool.join()
                
    if not all_results:
        print("No trades generated.")
        return
        
    df = pd.DataFrame(all_results)
    
    summary = df.groupby('matrix').agg(
        trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).mean()),
        avg_pnl=('pnl', 'mean'),
        total_pnl=('pnl', 'sum')
    ).reset_index()
    
    # Calculate EV
    summary['EV'] = summary['avg_pnl']
    
    print("\n" + "="*80)
    print("--- TP / SL OPTIMIZATION RESULTS ---")
    print("="*80)
    summary = summary.sort_values('EV', ascending=False)
    print(f"{'Exit Strategy':<25} | {'Win Rate':<10} | {'Avg PnL (EV)':<15} | {'Total PnL':<15}")
    print("-" * 75)
    
    for _, row in summary.iterrows():
        print(f"{row['matrix']:<25} | {row['win_rate']:<10.2%} | {row['avg_pnl']:<15.4%} | {row['total_pnl']:<15.4%}")
        
if __name__ == "__main__":
    main()
