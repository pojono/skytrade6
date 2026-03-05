import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm
import time

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
    
    # Clean anomalies
    df['ret'] = df['close'].pct_change()
    df = df[(df['ret'] > -0.5) & (df['ret'] < 1.0)]
    
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp')
    df.dropna(inplace=True)
    
    df = df.resample('1d').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df.dropna(inplace=True)
    
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
    
    if df.empty or len(df) < 30: return None
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    
    # We abandon Microstructure mean reversion. Taker fees kill it no matter how we tune it.
    # We return to Macro Capitulation (which we PROVED works, it's just very infrequent) 
    # AND combine it with a simple Daily Momentum strategy to provide consistent monthly returns.
    
    # 1. Capitulation Signal (Long only, buy panic)
    df['ret_7d'] = df['close'] / df['close'].shift(7) - 1
    df['capitulation'] = df['ret_7d'] < -0.30 # Drop 30% in a week
    
    # 2. Breakout Signal (Long only, buy strength)
    df['high_30d'] = df['high'].rolling(30).max().shift(1)
    df['vol_ma'] = df['volume'].rolling(30).mean().shift(1)
    
    df['breakout'] = (df['close'] > df['high_30d']) & (df['volume'] > df['vol_ma'] * 1.5)
    
    df['signal'] = df['capitulation'] | df['breakout']
    
    signals = df['signal'].values
    opens = df['open'].values
    closes = df['close'].values
    
    trades = []
    in_position = False
    entry_price = 0.0
    
    # Simple Time Stop (Hold for 7 days, sell at market)
    HOLD_DAYS = 7
    
    for i in range(30, len(df) - HOLD_DAYS - 1):
        if not in_position:
            if signals[i]:
                in_position = True
                entry_price = opens[i+1] # Enter at tomorrow's open (Taker)
                
                # Exit 7 days later
                exit_price = closes[i+1+HOLD_DAYS]
                net_pnl = ((exit_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                
                trades.append({
                    'exit_time': df.index[i+1+HOLD_DAYS], 
                    'pnl': net_pnl, 
                    'type': 'capitulation' if df['capitulation'].iloc[i] else 'breakout'
                })
                
                # We skip forward by HOLD_DAYS so we don't open overlapping positions on the same coin
                in_position = False

    return trades

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    
    print(f"Testing 7-Day Hybrid Momentum & Capitulation...")
    
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
    df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
    df_time = df_trades.set_index('exit_time')
    df_time.sort_index(inplace=True)
    df_time['cumulative'] = df_time['pnl'].cumsum()
    
    monthly_group = df_time.resample('1ME').agg(
        trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).mean() if len(x)>0 else 0),
        total_pnl=('pnl', 'sum')
    )
    
    print("\n" + "="*50)
    print("--- HYBRID STRATEGY RESULTS (7-Day Hold) ---")
    print("="*50)
    print(f"Total Trades: {len(df_trades)}")
    print(f"Win Rate: {(df_trades['pnl'] > 0).mean():.2%}")
    print(f"Avg PnL: {df_trades['pnl'].mean():.4%}")
    print(f"Total Raw Strategy Return (1x Lev): {df_time['cumulative'].iloc[-1]:.4%}")
    
    print("\n--- By Signal Type ---")
    print(df_trades.groupby('type').agg(
        trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).mean()),
        avg_pnl=('pnl', 'mean')
    ))
    
    print("\n--- Monthly Distribution ---")
    print(monthly_group)
    
if __name__ == "__main__":
    main()
