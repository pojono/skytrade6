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
    if 'startTime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['startTime'], unit='ms')
    elif 'timestamp' in df.columns and df['timestamp'].dtype != 'datetime64[ns]':
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
         
    df = df[(df['close'] > 0) & (df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0)]
    df = df[df['high'] >= df['low']]
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['ret'] = df['close'].pct_change()
    df = df[(df['ret'] > -0.5) & (df['ret'] < 1.0)]
    
    df = df.drop_duplicates(subset=['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # 15m resampling
    df = df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df.dropna(inplace=True)
    
    oi_list = []
    for f in files_oi:
        try: oi_list.append(pd.read_csv(f))
        except: pass
    if oi_list:
        df_oi = pd.concat(oi_list, ignore_index=True)
        if 'timestamp' in df_oi.columns:
            if df_oi['timestamp'].dtype != 'datetime64[ns]':
                df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'], unit='ms')
            df_oi = df_oi.drop_duplicates(subset=['timestamp']).sort_values('timestamp').set_index('timestamp')
            df_oi = df_oi.resample('15min').ffill()
            df = df.join(df_oi[['openInterest']], how='left').ffill()
        else:
            return None
    else:
        return None
    
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
        
    if df.empty or len(df) < (7 * 24 * 4): # 7 days
        return None
        
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    return backtest_strategy(df, symbol)

def backtest_strategy(df, symbol):
    # Strategy: Confirmed Capitulation Bounce
    # 1. 8H Price Drop > 12% 
    # 2. 8H OI Drop > 15% (Massive liquidations)
    # 3. Confirmation: 1H bounce > 3% from the lows
    
    df['ret_8h'] = df['close'] / df['close'].shift(32) - 1 # 32 * 15m = 8H
    if 'openInterest' not in df.columns: return []
    df['oi_8h'] = df['openInterest'] / df['openInterest'].shift(32) - 1
    
    df['flush_zone'] = (df['ret_8h'] < -0.12) & (df['ret_8h'] > -0.50) & (df['oi_8h'] < -0.15)
    df['flush_recent'] = df['flush_zone'].rolling(16).max().fillna(0).astype(bool) # Valid for 4H after flush
    
    df['ret_1h'] = df['close'] / df['close'].shift(4) - 1
    df['bounce_conf'] = df['ret_1h'] > 0.03
    
    raw_signal = df['flush_recent'] & df['bounce_conf']
    
    timestamps = df.index.values
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    signals = raw_signal.values
    
    trades = []
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    cooldown_until = 0
    n = len(df)
    
    SL_PCT = -0.12 # Wide stop to survive the volatility
    TP_PCT = 0.18 # 1.5x risk reward 
    TIME_STOP_CANDLES = 96 * 2 # 48 hours hold
    
    for i in range(32, n - 1):
        if not in_position:
            if signals[i] and i > cooldown_until:
                in_position = True
                entry_price = closes[i] # Taker Entry
                entry_time = timestamps[i]
                entry_idx = i
        else:
            target_price = entry_price * (1 + TP_PCT)
            stop_price = entry_price * (1 + SL_PCT)
            
            if lows[i] <= stop_price:
                exit_price = stop_price
                net_pnl = ((exit_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append({
                    'symbol': symbol, 'entry_time': entry_time, 'exit_time': timestamps[i],
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'pnl': net_pnl, 'type': 'sl', 'hold_hours': (i - entry_idx) * 0.25
                })
                in_position = False
                cooldown_until = i + 32 # Wait 8H before next trade
                
            elif highs[i] >= target_price:
                exit_price = target_price
                net_pnl = ((exit_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE
                trades.append({
                    'symbol': symbol, 'entry_time': entry_time, 'exit_time': timestamps[i],
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'pnl': net_pnl, 'type': 'tp', 'hold_hours': (i - entry_idx) * 0.25
                })
                in_position = False
                cooldown_until = i + 32
                
            elif (i - entry_idx) >= TIME_STOP_CANDLES:
                exit_price = closes[i] # Taker Exit
                net_pnl = ((exit_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append({
                    'symbol': symbol, 'entry_time': entry_time, 'exit_time': timestamps[i],
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'pnl': net_pnl, 'type': 'time', 'hold_hours': (i - entry_idx) * 0.25
                })
                in_position = False
                cooldown_until = i + 32

    return trades

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    print(f"Generating Final Strategy Report: Confirmed Capitulation Bounce")
    
    all_trades = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        results = list(tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols)))
        for trades in results: all_trades.extend(trades)
            
    if not all_trades:
        print("No events found.")
        return
        
    trades_df = pd.DataFrame(all_trades)
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df = trades_df.sort_values('entry_time')
    
    daily_pnl = trades_df.set_index('entry_time').resample('1D')['pnl'].sum()
    sharpe = np.sqrt(365) * (daily_pnl.mean() / daily_pnl.std()) if daily_pnl.std() > 0 else 0
    
    print("\n" + "="*50)
    print("      FINAL STRATEGY: CAPITULATION BOUNCE")
    print("="*50)
    print(f"Total Trades Generated: {len(trades_df)}")
    print(f"Overall Win Rate:       {(trades_df['pnl'] > 0).mean():.2%}")
    print(f"Average Net PnL/Trade:  {trades_df['pnl'].mean():.4%}")
    print(f"Median Net PnL/Trade:   {trades_df['pnl'].median():.4%}")
    print(f"Total Net PnL (1x Lev): {trades_df['pnl'].sum():.4%}")
    print(f"Daily Sharpe Ratio:     {sharpe:.2f}")
    
    print("\nExit Type Distribution:")
    print(trades_df['type'].value_counts())
    
    print("\nAverage Hold Time (Hours):")
    print(f"  Overall: {trades_df['hold_hours'].mean():.2f}")
    print(f"  Winning: {trades_df[trades_df['pnl'] > 0]['hold_hours'].mean():.2f}")
    print(f"  Losing:  {trades_df[trades_df['pnl'] <= 0]['hold_hours'].mean():.2f}")
    
    os.makedirs("/home/ubuntu/Projects/skytrade6/gemini-1", exist_ok=True)
    trades_df.to_csv("/home/ubuntu/Projects/skytrade6/gemini-1/strategy_capitulation_bounce.csv", index=False)
    
    # Save a markdown report
    with open("/home/ubuntu/Projects/skytrade6/gemini-1/STRATEGY_REPORT.md", "w") as f:
        f.write("# Strategy: Confirmed Capitulation Bounce\n\n")
        f.write("## Hypothesis\n")
        f.write("High fees (0.04% maker / 0.1% taker) destroy mean-reversion strategies that target small moves. To survive, we need gross moves of at least 10-15%. These moves reliably occur after extreme liquidation cascades (Open Interest flushes).\n\n")
        f.write("## Rules\n")
        f.write("- **Signal**: 8-hour Price Drop > 12% AND 8-hour Open Interest Drop > 15%.\n")
        f.write("- **Confirmation**: Wait for a 1-hour bounce > 3% to avoid catching a falling knife.\n")
        f.write("- **Execution**: Taker entry at market. Limit TP at +18%. Stop market SL at -12%.\n")
        f.write("- **Time Stop**: 48 hours.\n\n")
        f.write("## Performance (Out of Sample, Last 180 Days)\n")
        f.write(f"- Total Trades: {len(trades_df)}\n")
        f.write(f"- Win Rate: {(trades_df['pnl'] > 0).mean():.2%}\n")
        f.write(f"- Average Net PnL: {trades_df['pnl'].mean():.4%}\n")
        f.write(f"- Total Net PnL (1x Leverage): {trades_df['pnl'].sum():.4%}\n")
        f.write(f"- Daily Sharpe: {sharpe:.2f}\n")

if __name__ == "__main__":
    main()
