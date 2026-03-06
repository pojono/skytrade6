import pandas as pd
import numpy as np
import glob
import warnings

warnings.filterwarnings('ignore')

def load_1m_data(symbol, exchange='bybit'):
    print(f"Loading 1m data for {symbol}...")
    pattern = f"/home/ubuntu/Projects/skytrade6/datalake/{exchange}/{symbol}/*_kline_1m.csv"
    files = glob.glob(pattern)
    if not files:
        return None
        
    df_list = []
    for f in sorted(files):
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception:
            pass
            
    if not df_list:
        return None
        
    df = pd.concat(df_list, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['startTime'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df

def resample_to_macro(df_1m, timeframe='1h'):
    resampled = df_1m.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return resampled

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def calculate_volume_ma(df, period=50):
    return df['volume'].rolling(period).mean()

def backtest_confluence_strategy(df_1m, df_macro, symbol):
    df_macro['ATR'] = calculate_atr(df_macro, 14)
    df_macro['SMA_50'] = df_macro['close'].rolling(window=50).mean()
    df_macro = df_macro.dropna()
    
    df_1m['vol_ma'] = calculate_volume_ma(df_1m, 50)
    df_1m = df_1m.dropna()
    
    # Strategy Parameters
    FIB_LEVEL = 0.618
    ZOI_TOLERANCE = 0.04 # [0.578, 0.658]
    CONFIRMATION_ATR_MULT = 2.0
    MIN_SWING_PCT = 0.01 
    FEE_RATE = 0.001 
    
    # State variables
    mode = 'search'  # 'search', 'armed' (in ZOI, looking for 1m trigger), 'in_position'
    swing_dir = 1 
    
    origin_price = df_macro['close'].iloc[0]
    extreme_price = origin_price
    
    zoi_upper = 0
    zoi_lower = 0
    
    entry_price = 0
    tp_price = 0
    sl_price = 0
    position_size = 0
    trade_dir = 0
    
    trades = []
    equity = [10000]
    
    # We step through 1m data, but update our macro view based on time
    macro_idx = 0
    current_macro = df_macro.iloc[macro_idx]
    
    for i in range(50, len(df_1m)): # Start after 1m MA is warmed up
        current_1m = df_1m.iloc[i]
        current_time = df_1m.index[i]
        
        # Advance macro state if we've crossed a macro boundary
        if macro_idx + 1 < len(df_macro) and current_time >= df_macro.index[macro_idx + 1]:
            macro_idx += 1
            current_macro = df_macro.iloc[macro_idx]
            
            # 1. Update Macro Swings
            if mode in ['search', 'armed']:
                if swing_dir == 1:
                    if current_macro['high'] > extreme_price:
                        extreme_price = current_macro['high']
                        mode = 'search' 
                    elif extreme_price - current_macro['low'] > current_macro['ATR'] * CONFIRMATION_ATR_MULT:
                        if extreme_price > origin_price * (1 + MIN_SWING_PCT) and current_macro['close'] > current_macro['SMA_50']:
                            if mode == 'search':
                                swing_size = extreme_price - origin_price
                                fib_price = extreme_price - (swing_size * FIB_LEVEL)
                                zoi_range = swing_size * ZOI_TOLERANCE
                                zoi_upper = fib_price + zoi_range
                                zoi_lower = fib_price - zoi_range
                                mode = 'armed'
                                trade_dir = 1
                        else:
                            swing_dir = -1
                            origin_price = extreme_price
                            extreme_price = current_macro['low']
                elif swing_dir == -1:
                    if current_macro['low'] < extreme_price:
                        extreme_price = current_macro['low']
                        mode = 'search'
                    elif current_macro['high'] - extreme_price > current_macro['ATR'] * CONFIRMATION_ATR_MULT:
                        if extreme_price < origin_price * (1 - MIN_SWING_PCT) and current_macro['close'] < current_macro['SMA_50']:
                            if mode == 'search':
                                swing_size = origin_price - extreme_price
                                fib_price = extreme_price + (swing_size * FIB_LEVEL)
                                zoi_range = swing_size * ZOI_TOLERANCE
                                zoi_upper = fib_price + zoi_range
                                zoi_lower = fib_price - zoi_range
                                mode = 'armed'
                                trade_dir = -1
                        else:
                            swing_dir = 1
                            origin_price = extreme_price
                            extreme_price = current_macro['high']
                            
        # 2. Check 1m Trigger if Armed
        if mode == 'armed':
            triggered = False
            
            # LONG setup
            if trade_dir == 1:
                # Cancel if we slice completely through the ZOI (below 0.786 equiv)
                if current_1m['close'] < origin_price + ((extreme_price - origin_price) * 0.2):
                    mode = 'search'
                    continue
                    
                # Check if in ZOI
                if zoi_lower <= current_1m['low'] <= zoi_upper or zoi_lower <= current_1m['close'] <= zoi_upper:
                    # Kinematic Rejection: Vol Climax + Pin Bar
                    if current_1m['volume'] > current_1m['vol_ma'] * 2.5:
                        candle_range = current_1m['high'] - current_1m['low']
                        if candle_range > 0:
                            close_pct = (current_1m['close'] - current_1m['low']) / candle_range
                            # Close in upper 40% of the candle
                            if close_pct > 0.6:
                                triggered = True
                                
            # SHORT setup
            elif trade_dir == -1:
                if current_1m['close'] > origin_price - ((origin_price - extreme_price) * 0.2):
                    mode = 'search'
                    continue
                    
                if zoi_lower <= current_1m['high'] <= zoi_upper or zoi_lower <= current_1m['close'] <= zoi_upper:
                    if current_1m['volume'] > current_1m['vol_ma'] * 2.5:
                        candle_range = current_1m['high'] - current_1m['low']
                        if candle_range > 0:
                            close_pct = (current_1m['close'] - current_1m['low']) / candle_range
                            # Close in lower 40% of the candle
                            if close_pct < 0.4:
                                triggered = True
                                
            if triggered:
                mode = 'in_position'
                entry_price = current_1m['close']
                
                # Volatility-adjusted stop (1.5x 1m ATR roughly, or 0.5% fixed to avoid noise)
                min_stop_pct = 0.005 
                
                if trade_dir == 1:
                    sl_price = min(current_1m['low'] * 0.998, entry_price * (1 - min_stop_pct))
                    tp_price = extreme_price # Target macro high
                else:
                    sl_price = max(current_1m['high'] * 1.002, entry_price * (1 + min_stop_pct))
                    tp_price = extreme_price
                    
                risk_amount = equity[-1] * 0.02
                risk_per_unit = abs(entry_price - sl_price)
                if risk_per_unit == 0: risk_per_unit = 0.0001
                
                position_size = risk_amount / risk_per_unit
                equity[-1] -= (position_size * entry_price * FEE_RATE)
                
        # 3. Manage Open Position (1m resolution)
        elif mode == 'in_position':
            closed = False
            exit_price = 0
            cause = ""
            
            if trade_dir == 1:
                if current_1m['low'] <= sl_price:
                    exit_price = sl_price
                    closed = True
                    cause = "SL"
                elif current_1m['high'] >= tp_price:
                    exit_price = tp_price
                    closed = True
                    cause = "TP"
            else:
                if current_1m['high'] >= sl_price:
                    exit_price = sl_price
                    closed = True
                    cause = "SL"
                elif current_1m['low'] <= tp_price:
                    exit_price = tp_price
                    closed = True
                    cause = "TP"
                    
            if closed:
                pnl = (exit_price - entry_price) * position_size * trade_dir
                fee = position_size * exit_price * FEE_RATE
                net_pnl = pnl - fee
                
                trades.append({
                    'symbol': symbol,
                    'dir': 'LONG' if trade_dir == 1 else 'SHORT',
                    'entry_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'cause': cause,
                    'return_pct': (net_pnl / equity[-1]) * 100
                })
                
                equity[-1] = equity[-1] + net_pnl
                
                mode = 'search'
                origin_price = exit_price
                extreme_price = exit_price
                
        equity.append(equity[-1])
        
    return trades, equity

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    all_trades = []
    
    for sym in symbols:
        df_1m = load_1m_data(sym)
        if df_1m is not None:
            df_macro = resample_to_macro(df_1m, '1h')
            print(f"Running confluence backtest for {sym}...")
            trades, equity = backtest_confluence_strategy(df_1m, df_macro, sym)
            all_trades.extend(trades)
            
            pnl_series = [t['pnl'] for t in trades if t['symbol'] == sym]
            win_rate = len([p for p in pnl_series if p > 0]) / len(pnl_series) if pnl_series else 0
            
            print(f"--- {sym} Results ---")
            print(f"Total Trades: {len(trades)}")
            print(f"Win Rate: {win_rate*100:.1f}%")
            print(f"Final Equity: ${equity[-1]:.2f}")
            print(f"Net Profit: ${(equity[-1] - 10000):.2f}")
            print("--------------------\n")
            
    if all_trades:
        all_pnl = [t['pnl'] for t in all_trades]
        win_rate = len([p for p in all_pnl if p > 0]) / len(all_pnl)
        
        print(f"=== CONFLUENCE PORTFOLIO RESULTS ===")
        print(f"Total Trades: {len(all_trades)}")
        print(f"Win Rate: {win_rate*100:.1f}%")
        print(f"Avg Trade Net PnL: ${np.mean(all_pnl):.2f}")
        
        returns = pd.Series([t['return_pct']/100 for t in all_trades])
        sharpe = (returns.mean() / returns.std()) * np.sqrt(len(returns))
        print(f"Trade-Level Sharpe: {sharpe:.2f}")
