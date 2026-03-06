import os
import glob
import pandas as pd
import numpy as np
import gc

DATALAKE_BYBIT = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
DATALAKE_BINANCE = "/home/ubuntu/Projects/skytrade6/datalake/binance"
MAKER_FEE = 0.0004
TAKER_FEE = 0.0010

def load_and_resample_kline(files, is_spot=False):
    df_list = []
    for f in files:
        try:
            df = pd.read_csv(f)
            time_col = 'open_time' if is_spot and 'open_time' in df.columns else 'startTime'
            if time_col not in df.columns:
                time_col = 'timestamp'
            df['timestamp'] = pd.to_datetime(df[time_col], unit='ms')
            df.set_index('timestamp', inplace=True)
            df_4h = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
            df_list.append(df_4h)
        except Exception as e:
            pass
    if not df_list: return None
    return pd.concat(df_list).sort_index()

def load_and_resample_oi(files):
    df_list = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df_4h = df.resample('4h').agg({'openInterest': 'last'})
            df_list.append(df_4h)
        except:
            pass
    if not df_list: return None
    return pd.concat(df_list).sort_index()

def get_btc_regime():
    files = glob.glob(f"{DATALAKE_BYBIT}/BTCUSDT/2025*_kline_1m.csv") + glob.glob(f"{DATALAKE_BYBIT}/BTCUSDT/2026*_kline_1m.csv")
    df_4h = load_and_resample_kline(files)
    if df_4h is None: return None
    df_4h.dropna(inplace=True)
    period = 21
    change = abs(df_4h['close'] - df_4h['close'].shift(period))
    volatility = abs(df_4h['close'] - df_4h['close'].shift(1)).rolling(period).sum()
    df_4h['btc_ker'] = change / volatility
    df_4h['btc_is_trending'] = df_4h['btc_ker'] >= 0.20
    return df_4h[['btc_is_trending']]

def process_symbol(symbol, btc_regime):
    print(f"Processing {symbol}...")
    
    # Only use 2025 and 2026 data since that's what we have for Binance Spot
    bybit_files = glob.glob(f"{DATALAKE_BYBIT}/{symbol}/2025*_kline_1m.csv") + glob.glob(f"{DATALAKE_BYBIT}/{symbol}/2026*_kline_1m.csv")
    if not bybit_files: return []
    
    df_bybit = load_and_resample_kline(bybit_files)
    if df_bybit is None: return []

    oi_files = glob.glob(f"{DATALAKE_BYBIT}/{symbol}/2025*_open_interest_5min.csv") + glob.glob(f"{DATALAKE_BYBIT}/{symbol}/2026*_open_interest_5min.csv")
    if oi_files:
        df_oi = load_and_resample_oi(oi_files)
        if df_oi is not None:
            df_bybit['oi'] = df_oi['openInterest']
        else:
            df_bybit['oi'] = np.nan
    else:
        df_bybit['oi'] = np.nan
        
    binance_files = glob.glob(f"{DATALAKE_BINANCE}/{symbol}/2025*_kline_1m_spot.csv") + glob.glob(f"{DATALAKE_BINANCE}/{symbol}/2026*_kline_1m_spot.csv")
    if binance_files:
        df_binance = load_and_resample_kline(binance_files, is_spot=True)
        if df_binance is not None:
            df_bybit['spot_volume'] = df_binance['volume']
        else:
            df_bybit['spot_volume'] = np.nan
    else:
        df_bybit['spot_volume'] = np.nan

    df_4h = df_bybit.dropna(subset=['close']).copy()
    if df_4h.empty: return []
    
    if btc_regime is not None:
        # Avoid the future warning by using infer_objects
        df_4h = df_4h.join(btc_regime, how='left')
        df_4h['btc_is_trending'] = df_4h['btc_is_trending'].ffill().infer_objects(copy=False)
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
    df_4h['perp_vol_spike'] = df_4h['volume'] > (df_4h['vol_ma'] * 2.0)
    
    df_4h['has_spot'] = ~df_4h['spot_volume'].isna()
    if df_4h['has_spot'].any():
        df_4h['spot_vol_ma'] = df_4h['spot_volume'].rolling(20).mean().shift(1)
        df_4h['spot_vol_spike'] = df_4h['spot_volume'] > (df_4h['spot_vol_ma'] * 2.0)
        df_4h['spot_vol_spike'] = df_4h['spot_vol_spike'].fillna(False)
    else:
        df_4h['spot_vol_spike'] = False
        
    df_4h['has_oi'] = ~df_4h['oi'].isna()
    if df_4h['has_oi'].any():
        df_4h['oi_rising'] = df_4h['oi'] > df_4h['oi'].shift(4)
        df_4h['oi_rising'] = df_4h['oi_rising'].fillna(False)
    else:
        df_4h['oi_rising'] = False

    df_4h['regime_ok'] = df_4h['btc_is_trending'] | df_4h['local_is_trending']
    
    df_4h['long_breakout'] = (df_4h['close'] > df_4h['high_20']) & df_4h['macro_bull'] & df_4h['perp_vol_spike'] & df_4h['regime_ok']
    df_4h['short_breakout'] = (df_4h['close'] < df_4h['low_20']) & (~df_4h['macro_bull']) & df_4h['perp_vol_spike'] & df_4h['regime_ok']
    
    signals_long = df_4h['long_breakout'].values
    signals_short = df_4h['short_breakout'].values
    spot_spike = df_4h['spot_vol_spike'].values if isinstance(df_4h['spot_vol_spike'], pd.Series) else np.full(len(df_4h), False)
    oi_rising = df_4h['oi_rising'].values if isinstance(df_4h['oi_rising'], pd.Series) else np.full(len(df_4h), False)
    has_spot = df_4h['has_spot'].values if isinstance(df_4h['has_spot'], pd.Series) else np.full(len(df_4h), False)
    has_oi = df_4h['has_oi'].values if isinstance(df_4h['has_oi'], pd.Series) else np.full(len(df_4h), False)
    
    opens = df_4h['open'].values
    highs = df_4h['high'].values
    lows = df_4h['low'].values
    closes = df_4h['close'].values
    
    configs = [
        {'name': 'Baseline (Bybit Only)', 'filter_func': lambda i: True},
        {'name': 'Spot Confirmed', 'filter_func': lambda i: spot_spike[i] if has_spot[i] else True},
        {'name': 'OI Rising Confirmed', 'filter_func': lambda i: oi_rising[i] if has_oi[i] else True},
        {'name': 'Spot + OI Combo', 'filter_func': lambda i: (spot_spike[i] and oi_rising[i]) if (has_spot[i] and has_oi[i]) else True}
    ]
    
    results = []
    
    for cfg in configs:
        trades = []
        in_position = False
        entry_price = 0.0
        entry_idx = 0
        cooldown_until = 0
        pos_type = 0
        
        TP_PCT = 0.20
        SL_PCT = 0.10
        TIME_STOP_CANDLES = 6 * 14
        
        for i in range(200, len(df_4h) - 1):
            if not in_position:
                if i > cooldown_until:
                    if signals_long[i] and cfg['filter_func'](i):
                        in_position = True
                        pos_type = 1
                        entry_price = opens[i+1]
                        entry_idx = i + 1
                        entry_time = df_4h.index[i+1]
                    elif signals_short[i] and cfg['filter_func'](i):
                        in_position = True
                        pos_type = -1
                        entry_price = opens[i+1]
                        entry_idx = i + 1
                        entry_time = df_4h.index[i+1]
            else:
                if pos_type == 1:
                    target_price = entry_price * (1 + TP_PCT)
                    stop_price = entry_price * (1 - SL_PCT)
                    
                    if lows[i] <= stop_price:
                        net_pnl = ((stop_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                        trades.append({'symbol': symbol, 'config': cfg['name'], 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'pnl': net_pnl})
                        in_position = False
                        cooldown_until = i + 6
                    elif highs[i] >= target_price:
                        net_pnl = ((target_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE
                        trades.append({'symbol': symbol, 'config': cfg['name'], 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'pnl': net_pnl})
                        in_position = False
                        cooldown_until = i + 6
                    elif (i - entry_idx) >= TIME_STOP_CANDLES:
                        net_pnl = ((closes[i] - entry_price) / entry_price) - (TAKER_FEE * 2)
                        trades.append({'symbol': symbol, 'config': cfg['name'], 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'pnl': net_pnl})
                        in_position = False
                        cooldown_until = i + 6
                        
                elif pos_type == -1:
                    target_price = entry_price * (1 - TP_PCT)
                    stop_price = entry_price * (1 + SL_PCT)
                    
                    if highs[i] >= stop_price:
                        net_pnl = ((entry_price - stop_price) / entry_price) - (TAKER_FEE * 2)
                        trades.append({'symbol': symbol, 'config': cfg['name'], 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'pnl': net_pnl})
                        in_position = False
                        cooldown_until = i + 6
                    elif lows[i] <= target_price:
                        net_pnl = ((entry_price - target_price) / entry_price) - TAKER_FEE - MAKER_FEE
                        trades.append({'symbol': symbol, 'config': cfg['name'], 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'pnl': net_pnl})
                        in_position = False
                        cooldown_until = i + 6
                    elif (i - entry_idx) >= TIME_STOP_CANDLES:
                        net_pnl = ((entry_price - closes[i]) / entry_price) - (TAKER_FEE * 2)
                        trades.append({'symbol': symbol, 'config': cfg['name'], 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'pnl': net_pnl})
                        in_position = False
                        cooldown_until = i + 6

        results.extend(trades)
    
    del df_4h, df_bybit
    gc.collect()
    
    return results

def main():
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT', 'AVAXUSDT', 'LINKUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    
    print("Pre-loading BTC Regime...")
    btc_regime = get_btc_regime()
    
    print("\nTesting Cross-Exchange Filters on Binance Spot / Bybit Perps (2025-2026 Data)...")
    print("Running sequentially to save RAM.")
    
    all_trades = []
    
    for symbol in symbols:
        trades = process_symbol(symbol, btc_regime)
        all_trades.extend(trades)
                
    if not all_trades:
        print("No trades generated.")
        return
        
    df_trades = pd.DataFrame(all_trades)
    
    print("\n" + "="*90)
    print("--- CROSS-EXCHANGE FILTER RESULTS (Jan 2025 - Mar 2026) ---")
    print("="*90)
    print(f"{'Configuration':<25} | {'Trades':<8} | {'Win Rate':<10} | {'EV/Trade':<10} | {'Equity Ret (2% Risk)':<20}")
    print("-" * 90)
    
    for config, group in df_trades.groupby('config'):
        trades = len(group)
        if trades == 0: continue
        wins = sum(group['pnl'] > 0)
        win_rate = wins / trades
        ev = group['pnl'].mean()
        equity_return = group['pnl'].sum() * 0.20
        
        print(f"{config:<25} | {trades:<8} | {win_rate:<10.2%} | {ev:<10.2%} | {equity_return:<20.2%}")
        
if __name__ == "__main__":
    main()
