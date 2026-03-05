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
    df = df[(df['ret'] > -0.5) & (df['ret'] < 1.0)]
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp')
    
    # 4H candles for trend following (faster than 1D, avoids the single massive trade bug from v27)
    df = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df.dropna(inplace=True)
    
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
    
    if df.empty or len(df) < (6 * 20): return None
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    
    # Trend Following (4H Donchian Channels)
    # Entry: 20-period breakout (approx 3 days)
    # Exit: 10-period trailing low (approx 1.5 days)
    # Filter: Must be in macro uptrend (Price > 200 EMA)
    
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['macro_bull'] = df['close'] > df['ema_200']
    
    df['high_20'] = df['high'].rolling(20).max().shift(1)
    df['low_10'] = df['low'].rolling(10).min().shift(1)
    
    # Only long trades to avoid getting squeezed in bear market bounces
    df['long_signal'] = (df['close'] > df['high_20']) & df['macro_bull']
    
    signals = df['long_signal'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    low_10 = df['low_10'].values
    
    trades = []
    in_position = False
    entry_price = 0.0
    
    for i in range(200, len(df) - 1):
        if not in_position:
            if signals[i]:
                in_position = True
                entry_price = opens[i+1] # Taker Entry
        else:
            # Trailing Stop
            exit_price = low_10[i]
            
            # Prevent infinite loop or bug if exit price is 0
            if exit_price <= 0: exit_price = entry_price * 0.5
            
            if lows[i] < exit_price:
                net_pnl = ((exit_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                
                # Sanity check to prevent -3000% bug from previous script
                if net_pnl > -1.0 and net_pnl < 10.0:
                    trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'trailing_stop'})
                
                in_position = False

    return trades

def main():
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT', 'AVAXUSDT', 'LINKUSDT', 'MATICUSDT']
    
    print(f"Testing 4H Trend Following (Donchian + EMA Filter) on Top 8 Coins...")
    
    all_trades = []
    pool = multiprocessing.Pool(processes=min(4, multiprocessing.cpu_count()))
    
    for trades in tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols), desc="Backtesting"):
        all_trades.extend(trades)
            
    pool.close()
    pool.join()
                
    if not all_trades:
        print("No trades generated.")
        return
        
    df_trades = pd.DataFrame(all_trades)
    df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
    df_time = df_trades.set_index('exit_time')
    df_time.sort_index(inplace=True)
    df_time['cumulative'] = df_time['pnl'].cumsum()
    
    monthly_group = df_time.resample('1ME').agg(
        trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).mean() if len(x)>0 else 0),
        total_pnl=('pnl', 'sum'),
        avg_pnl=('pnl', 'mean')
    )
    
    print("\n" + "="*50)
    print("--- Strategy Results (4H Trend Following) ---")
    print("="*50)
    print(f"Total Trades: {len(df_trades)}")
    print(f"Win Rate: {(df_trades['pnl'] > 0).mean():.2%}")
    print(f"Avg Net PnL per Trade: {df_trades['pnl'].mean():.4%}")
    print(f"Total Cumulative PnL: {df_time['cumulative'].iloc[-1]:.4%}")
    
    print("\n--- Monthly Distribution ---")
    print(monthly_group)

if __name__ == "__main__":
    main()
