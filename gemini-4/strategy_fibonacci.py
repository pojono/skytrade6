import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def load_data(symbol, exchange='bybit', timeframe='15min'):
    print(f"Loading data for {symbol} on {exchange} ({timeframe})...")
    pattern = f"/home/ubuntu/Projects/skytrade6/datalake/{exchange}/{symbol}/*_kline_1m.csv"
    files = glob.glob(pattern)
    if not files:
        print(f"No files found for {symbol}")
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
    
    resampled = df.resample(timeframe).agg({
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
    atr = true_range.rolling(period).mean()
    return atr

def backtest_fibonacci(df, symbol):
    df['ATR'] = calculate_atr(df, 14)
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df = df.dropna()
    
    # Strategy Parameters
    FIB_LEVEL = 0.618  # The Golden Ratio
    CONFIRMATION_ATR_MULT = 2.0  # Slightly looser confirmation
    FEE_RATE = 0.001  # 10 bps taker fee
    MIN_SWING_PCT = 0.01 # 1% minimum swing size (filter noise)
    
    # Risk Management
    RR_RATIO = 2.0 # Target 2:1 Reward to Risk
    SL_BUFFER_ATR = 1.0 # Stop loss 1 ATR beyond origin
    
    # State variables
    mode = 'search'  # 'search' for swing, 'armed' (order placed), 'in_position'
    swing_dir = 1 # Start assuming uptrend
    
    origin_price = df['close'].iloc[0]
    extreme_price = origin_price
    
    entry_price = 0
    tp_price = 0
    sl_price = 0
    position_size = 0
    trade_dir = 0
    
    trades = []
    equity = [10000] # Start with $10,000
    
    for i in range(1, len(df)):
        current = df.iloc[i]
        
        # 1. Update Swings if searching or armed
        if mode in ['search', 'armed']:
            # Assume we are tracing an uptrend
            if swing_dir == 1:
                if current['high'] > extreme_price:
                    extreme_price = current['high']
                    mode = 'search' # Reset to search, invalidating any armed orders
                elif extreme_price - current['low'] > current['ATR'] * CONFIRMATION_ATR_MULT:
                    # Pullback detected! Check if swing was large enough and aligns with macro trend
                    if extreme_price > origin_price * (1 + MIN_SWING_PCT) and current['close'] > current['SMA_50']:
                        if mode == 'search':
                            # Arm a LONG trade at the Fib level
                            swing_size = extreme_price - origin_price
                            entry_price = extreme_price - (swing_size * FIB_LEVEL)
                            
                            # Dynamic SL based on structure and volatility
                            sl_price = origin_price - (current['ATR'] * SL_BUFFER_ATR) 
                            
                            # Fixed RR Target
                            risk = entry_price - sl_price
                            tp_price = entry_price + (risk * RR_RATIO)
                            
                            # Only arm if current price hasn't already smashed through the entry
                            if current['close'] > entry_price:
                                mode = 'armed'
                                trade_dir = 1
                    else:
                        # Swing too small or counter-trend, flip trend assumption
                        swing_dir = -1
                        origin_price = extreme_price
                        extreme_price = current['low']
                            
            # Assume we are tracing a downtrend
            elif swing_dir == -1:
                if current['low'] < extreme_price:
                    extreme_price = current['low']
                    mode = 'search'
                elif current['high'] - extreme_price > current['ATR'] * CONFIRMATION_ATR_MULT:
                    # Bounce detected!
                    if extreme_price < origin_price * (1 - MIN_SWING_PCT) and current['close'] < current['SMA_50']:
                        if mode == 'search':
                            # Arm a SHORT trade at the Fib level
                            swing_size = origin_price - extreme_price
                            entry_price = extreme_price + (swing_size * FIB_LEVEL)
                            
                            sl_price = origin_price + (current['ATR'] * SL_BUFFER_ATR)
                            risk = sl_price - entry_price
                            tp_price = entry_price - (risk * RR_RATIO)
                            
                            if current['close'] < entry_price:
                                mode = 'armed'
                                trade_dir = -1
                    else:
                        # Swing too small or counter-trend, flip trend assumption
                        swing_dir = 1
                        origin_price = extreme_price
                        extreme_price = current['high']
                            
        # 2. Trigger Armed Orders
        if mode == 'armed':
            triggered = False
            # Check if current bar intersects entry
            if trade_dir == 1 and current['low'] <= entry_price:
                triggered = True
            elif trade_dir == -1 and current['high'] >= entry_price:
                triggered = True
                
            if triggered:
                mode = 'in_position'
                # Position sizing: Risk 2% of equity
                risk_amount = equity[-1] * 0.02
                risk_per_unit = abs(entry_price - sl_price)
                if risk_per_unit == 0: risk_per_unit = 0.0001
                
                position_size = risk_amount / risk_per_unit
                
                # Deduct entry fee
                equity[-1] -= (position_size * entry_price * FEE_RATE)
                
        # 3. Manage Open Position
        elif mode == 'in_position':
            closed = False
            exit_price = 0
            cause = ""
            
            if trade_dir == 1:
                if current['low'] <= sl_price:
                    exit_price = sl_price
                    closed = True
                    cause = "SL"
                elif current['high'] >= tp_price:
                    exit_price = tp_price
                    closed = True
                    cause = "TP"
            else:
                if current['high'] >= sl_price:
                    exit_price = sl_price
                    closed = True
                    cause = "SL"
                elif current['low'] <= tp_price:
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
                    'entry_time': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'cause': cause,
                    'return_pct': (net_pnl / equity[-1]) * 100
                })
                
                equity[-1] = equity[-1] + net_pnl
                
                # Reset state to search for next swing
                mode = 'search'
                origin_price = exit_price
                extreme_price = exit_price
                
        # Always propagate equity to next step (handled by updating [-1] in loop and appending at end)
        equity.append(equity[-1])
        
    return trades, equity

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    all_trades = []
    
    for sym in symbols:
        df = load_data(sym, timeframe='1h')
        if df is not None:
            trades, equity = backtest_fibonacci(df, sym)
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
        
        print(f"=== PORTFOLIO RESULTS ===")
        print(f"Total Trades: {len(all_trades)}")
        print(f"Win Rate: {win_rate*100:.1f}%")
        print(f"Avg Trade Net PnL: ${np.mean(all_pnl):.2f}")
        
        # Calculate Sharpe
        returns = pd.Series([t['return_pct']/100 for t in all_trades])
        sharpe = (returns.mean() / returns.std()) * np.sqrt(len(returns)) # Approx trade-level Sharpe
        print(f"Trade-Level Sharpe: {sharpe:.2f}")
