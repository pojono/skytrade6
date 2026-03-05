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
    
    # Very strict filter to ensure NO DATA BUGS ruin the backtest
    df['ret'] = df['close'].pct_change()
    df = df[(df['ret'] > -0.15) & (df['ret'] < 0.15)]
    
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp')
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
    
    df['ret_3d'] = df['close'] / df['close'].shift(3) - 1
    df['setup'] = df['ret_3d'] < -0.20
    df['setup_active'] = df['setup'].rolling(3).max().shift(1).fillna(0).astype(bool)
    
    df['confirmation'] = df['close'] > df['open']
    
    df['vol_ma'] = df['volume'].rolling(30).mean().shift(1)
    df['vol_spike'] = df['volume'] > (df['vol_ma'] * 1.5)
    
    # We want to buy TOMORROW'S OPEN if TODAY has the signal.
    df['signal'] = df['setup_active'] & df['confirmation'] & df['vol_spike']
    
    signals = df['signal'].values
    opens = df['open'].values
    closes = df['close'].values
    
    trades = []
    in_position = False
    
    HOLD_DAYS = 5
    
    for i in range(30, len(df) - HOLD_DAYS - 1):
        if not in_position:
            if signals[i]:
                in_position = True
                entry_price = opens[i+1] # Next day's open
                
                # Exit at the close of entry_day + HOLD_DAYS
                exit_price = closes[i+1+HOLD_DAYS]
                
                # Extreme outlier protection (ignore bugged data spikes > 500%)
                if exit_price > 0 and entry_price > 0:
                    net_pnl = ((exit_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                    if net_pnl < 3.0: 
                        trades.append({
                            'exit_time': df.index[i+1+HOLD_DAYS], 
                            'pnl': net_pnl
                        })
                
                in_position = False

    return trades

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    
    print(f"Testing Pure Capitulation (Zero Lookahead, Data Bug Filtered)...")
    
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
    
    df_time['equity_pnl'] = df_time['pnl'] * 0.05
    df_time['equity'] = 1.0 + df_time['equity_pnl'].cumsum()
    
    monthly_group = df_time.resample('1ME').agg(
        trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).mean() if len(x)>0 else 0),
        avg_pnl=('pnl', 'mean'),
        equity_return=('equity_pnl', 'sum')
    )
    
    print("\n" + "="*50)
    print("--- REALISTIC CAPITULATION RESULTS ---")
    print("="*50)
    print(f"Total Trades: {len(df_trades)}")
    print(f"Win Rate: {(df_trades['pnl'] > 0).mean():.2%}")
    print(f"Avg PnL (Gross): {df_trades['pnl'].mean():.4%}")
    print(f"Total Portfolio Return (5% Risk/Trade): {(df_time['equity'].iloc[-1] - 1):.2%}")
    
    print("\n--- Monthly Portfolio Returns ---")
    print(monthly_group[['trades', 'win_rate', 'avg_pnl', 'equity_return']])
    
if __name__ == "__main__":
    main()
