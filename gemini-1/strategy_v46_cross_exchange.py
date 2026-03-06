import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm

DATALAKE_BYBIT = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
DATALAKE_BINANCE = "/home/ubuntu/Projects/skytrade6/datalake/binance"
MAKER_FEE = 0.0004
TAKER_FEE = 0.0010

def get_btc_regime():
    files = glob.glob(f"{DATALAKE_BYBIT}/BTCUSDT/*_kline_1m.csv")
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
    # Load Bybit Kline
    bybit_files = glob.glob(f"{DATALAKE_BYBIT}/{symbol}/*_kline_1m.csv")
    if not bybit_files: return None
    
    df_bybit = pd.concat([pd.read_csv(f) for f in bybit_files], ignore_index=True)
    df_bybit['timestamp'] = pd.to_datetime(df_bybit.get('startTime', df_bybit.get('timestamp', None)), unit='ms')
    df_bybit = df_bybit.drop_duplicates(subset=['timestamp']).set_index('timestamp')
    df_bybit = df_bybit[(df_bybit['close'] > 0)]
    df_bybit = df_bybit.sort_index()

    # Load Bybit OI
    oi_files = glob.glob(f"{DATALAKE_BYBIT}/{symbol}/*_open_interest_5min.csv")
    if oi_files:
        df_oi = pd.concat([pd.read_csv(f) for f in oi_files], ignore_index=True)
        df_oi['timestamp'] = pd.to_datetime(df_oi.get('timestamp', None), unit='ms')
        df_oi = df_oi.drop_duplicates(subset=['timestamp']).set_index('timestamp')
        df_oi = df_oi.sort_index()
        # Resample OI to 1m by forward filling 
        df_bybit['oi'] = df_oi['openInterest'].reindex(df_bybit.index, method='ffill')
    else:
        df_bybit['oi'] = np.nan
        
    # Load Binance Spot
    binance_files = glob.glob(f"{DATALAKE_BINANCE}/{symbol}/*_kline_1m_spot.csv")
    if binance_files:
        df_binance = pd.concat([pd.read_csv(f) for f in binance_files], ignore_index=True)
        df_binance['timestamp'] = pd.to_datetime(df_binance.get('open_time', df_binance.get('timestamp', None)), unit='ms')
        df_binance = df_binance.drop_duplicates(subset=['timestamp']).set_index('timestamp')
        df_binance = df_binance.sort_index()
        
        # Merge Spot volume to Bybit df
        df_bybit['spot_volume'] = df_binance['volume']
    else:
        df_bybit['spot_volume'] = np.nan

    # Ensure overlapping time availability
    df_bybit = df_bybit.dropna(subset=['close'])
    if df_bybit.empty: return None

    return symbol, df_bybit

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    
    # 4-Hour Resampling
    agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    if 'spot_volume' in df.columns: agg_dict['spot_volume'] = 'sum'
    if 'oi' in df.columns: agg_dict['oi'] = 'last'
        
    df_4h = df.resample('4h').agg(agg_dict)
    df_4h.dropna(subset=['close'], inplace=True)
    
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
    
    # Standard Perp Volume Breakout
    df_4h['vol_ma'] = df_4h['volume'].rolling(20).mean().shift(1)
    df_4h['perp_vol_spike'] = df_4h['volume'] > (df_4h['vol_ma'] * 2.0)
    
    # Cross-Exchange Filters
    df_4h['has_spot'] = ~df_4h['spot_volume'].isna()
    if df_4h['has_spot'].any():
        df_4h['spot_vol_ma'] = df_4h['spot_volume'].rolling(20).mean().shift(1)
        # Check if Binance spot volume is ALSO spiking
        df_4h['spot_vol_spike'] = df_4h['spot_volume'] > (df_4h['spot_vol_ma'] * 2.0)
    else:
        df_4h['spot_vol_spike'] = False # If no data, fail conservative
        
    df_4h['has_oi'] = ~df_4h['oi'].isna()
    if df_4h['has_oi'].any():
        # OI must be higher than 4 periods ago (16 hours) indicating accumulation, not just shorts closing
        df_4h['oi_rising'] = df_4h['oi'] > df_4h['oi'].shift(4)
    else:
        df_4h['oi_rising'] = False

    df_4h['regime_ok'] = df_4h['btc_is_trending'] | df_4h['local_is_trending']
    
    # Baseline Signals
    df_4h['long_breakout'] = (df_4h['close'] > df_4h['high_20']) & df_4h['macro_bull'] & df_4h['perp_vol_spike'] & df_4h['regime_ok']
    df_4h['short_breakout'] = (df_4h['close'] < df_4h['low_20']) & (~df_4h['macro_bull']) & df_4h['perp_vol_spike'] & df_4h['regime_ok']
    
    signals_long = df_4h['long_breakout'].values
    signals_short = df_4h['short_breakout'].values
    spot_spike = df_4h['spot_vol_spike'].values
    oi_rising = df_4h['oi_rising'].values
    has_spot = df_4h['has_spot'].values
    has_oi = df_4h['has_oi'].values
    
    opens = df_4h['open'].values
    highs = df_4h['high'].values
    lows = df_4h['low'].values
    closes = df_4h['close'].values
    
    # Test multiple configurations to see cross-exchange impact
    # Config 1: Baseline (V45)
    # Config 2: Spot Confirmed (Baseline + Binance Spot Volume Spike)
    # Config 3: OI Confirmed (Baseline + OI Rising)
    
    configs = [
        {'name': 'Baseline', 'filter_func': lambda i: True},
        {'name': 'Spot_Confirmed', 'filter_func': lambda i: spot_spike[i] if has_spot[i] else True},
        {'name': 'OI_Rising', 'filter_func': lambda i: oi_rising[i] if has_oi[i] else True},
        {'name': 'Spot_AND_OI', 'filter_func': lambda i: (spot_spike[i] and oi_rising[i]) if (has_spot[i] and has_oi[i]) else True}
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
    return results

def main():
    # Only test on the 10 coins we have deep history for
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT', 'AVAXUSDT', 'LINKUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    
    print(f"Testing Cross-Exchange Filters on Binance Spot / Bybit Perps...")
    
    all_trades = []
    pool = multiprocessing.Pool(processes=min(8, multiprocessing.cpu_count()))
    
    for trades in tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols), desc="Backtesting"):
        all_trades.extend(trades)
            
    pool.close()
    pool.join()
                
    if not all_trades:
        print("No trades generated.")
        return
        
    df_trades = pd.DataFrame(all_trades)
    
    print("\n" + "="*80)
    print("--- CROSS-EXCHANGE FILTER RESULTS (2025/2026 Data Overlap) ---")
    print("="*80)
    print(f"{'Configuration':<20} | {'Trades':<8} | {'Win Rate':<10} | {'EV/Trade':<10} | {'Equity Ret (2% Risk)':<20}")
    print("-" * 80)
    
    # Group by configuration
    for config, group in df_trades.groupby('config'):
        # Filter to only trades executed in 2025/2026 (since Binance data starts Jan 1 2025)
        # For baseline, we only want to compare identical time periods
        group_recent = group[group['entry_time'] >= '2025-01-01']
        
        trades = len(group_recent)
        if trades == 0: continue
        wins = sum(group_recent['pnl'] > 0)
        win_rate = wins / trades
        ev = group_recent['pnl'].mean()
        equity_return = group_recent['pnl'].sum() * 0.20
        
        print(f"{config:<20} | {trades:<8} | {win_rate:<10.2%} | {ev:<10.2%} | {equity_return:<20.2%}")
        
if __name__ == "__main__":
    main()
