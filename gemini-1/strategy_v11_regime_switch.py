import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
MAKER_FEE = 0.0004
TAKER_FEE = 0.0010

# To achieve consistency across months, we must adapt to market regimes.
# Strategy: 
# 1. Detect Macro Regime (Bitcoin Trend & Volatility)
# 2. If Bull Market -> Trade Trend Pullbacks
# 3. If Bear/Crab Market -> FADE extreme deviations (Mean Reversion / Capitulation Bounce)

def load_btc_regime():
    files = glob.glob(f"{DATALAKE_DIR}/BTCUSDT/*_kline_1m.csv")
    if not files: return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['timestamp'] = pd.to_datetime(df.get('startTime', df.get('timestamp', None)), unit='ms')
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).set_index('timestamp')
    df = df.resample('1d').agg({'close': 'last'})
    df.dropna(inplace=True)
    
    # 30-Day SMA
    df['btc_sma_30'] = df['close'].rolling(30).mean()
    # 30-Day Volatility
    df['btc_ret'] = df['close'].pct_change()
    df['btc_vol_30'] = df['btc_ret'].rolling(30).std() * np.sqrt(365)
    
    # Forward fill to 1H resolution so we can join it to altcoins easily
    df_1h = df.resample('1h').ffill()
    return df_1h[['btc_sma_30', 'btc_vol_30', 'close']].rename(columns={'close': 'btc_close'})

BTC_REGIME = load_btc_regime()

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
    df = df[(df['ret'] > -0.5) & (df['ret'] < 1.0)]
    
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp')
    
    df = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df.dropna(inplace=True)
    
    # Join BTC Regime
    if BTC_REGIME is not None:
        df = df.join(BTC_REGIME, how='left').ffill()
    
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
    
    if df.empty or len(df) < (6 * 30): return None
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    
    # Regimes
    df['is_bull'] = df['btc_close'] > df['btc_sma_30']
    
    # Sub-Strategy A: Bull Market Pullback (Trend Continuation)
    df['ema_fast'] = df['close'].ewm(span=6*7, adjust=False).mean() # 7-day EMA
    df['trend_pullback'] = (df['low'] < df['ema_fast']) & (df['close'] > df['ema_fast']) & (df['close'] > df['close'].shift(6*7))
    
    # Sub-Strategy B: Bear Market Capitulation (Mean Reversion)
    df['ret_24h'] = df['close'] / df['close'].shift(6) - 1
    df['capitulation'] = df['ret_24h'] < -0.15 # 15% drop in 24h
    
    # Master Signal Map
    df['long_signal'] = np.where(df['is_bull'], df['trend_pullback'], df['capitulation'])
    
    signals = df['long_signal'].values
    is_bull_arr = df['is_bull'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    trades = []
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    cooldown_until = 0
    trade_regime = ""
    n = len(df)
    
    for i in range(42, n - 1):
        if not in_position:
            if signals[i] and i > cooldown_until:
                in_position = True
                entry_price = opens[i+1] # Entering Long
                entry_idx = i + 1
                trade_regime = "BULL" if is_bull_arr[i] else "BEAR"
        else:
            # Dynamic targets based on regime
            if trade_regime == "BULL":
                target_price = entry_price * 1.15
                stop_price = entry_price * 0.90
                time_stop = 6 * 7 # 7 days
            else:
                target_price = entry_price * 1.15
                stop_price = entry_price * 0.85 # Wider stop for volatile bear bounces
                time_stop = 6 * 3 # 3 days (quicker exit in bear markets)
                
            if lows[i] <= stop_price:
                net_pnl = ((stop_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'sl', 'regime': trade_regime})
                in_position = False
                cooldown_until = i + 6
                
            elif highs[i] >= target_price:
                net_pnl = ((target_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE
                trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'tp', 'regime': trade_regime})
                in_position = False
                cooldown_until = i + 6
                
            elif (i - entry_idx) >= time_stop:
                net_pnl = ((closes[i] - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'time', 'regime': trade_regime})
                in_position = False
                cooldown_until = i + 6

    return trades

def main():
    if BTC_REGIME is None:
        print("Could not load BTC data for regime tracking.")
        return
        
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    print(f"Testing Regime Switching Strategy...")
    
    all_trades = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        for pnl_list in tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols)):
            all_trades.extend(pnl_list)
                
    if not all_trades:
        print("No trades generated.")
        return
        
    df_trades = pd.DataFrame(all_trades)
    df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
    df_time = df_trades.set_index('exit_time')
    
    monthly_group = df_time.resample('1ME').agg(
        trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).mean() if len(x) > 0 else 0),
        total_pnl=('pnl', 'sum'),
        avg_pnl=('pnl', 'mean')
    )
    
    print("\n--- Strategy Results ---")
    print(f"Total Trades: {len(df_trades)}")
    print(f"Win Rate: {(df_trades['pnl'] > 0).mean():.2%}")
    print(f"Avg PnL: {df_trades['pnl'].mean():.4%}")
    print(f"Total PnL (1x Lev): {df_trades['pnl'].sum():.4%}")
    
    print("\n--- By Regime ---")
    print(df_trades.groupby('regime').agg(
        trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).mean()),
        avg_pnl=('pnl', 'mean')
    ))
    
    print("\n--- Monthly Distribution ---")
    print(monthly_group)

if __name__ == "__main__":
    main()
