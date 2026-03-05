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
    
    # We found a bug in the funding lookahead.
    # Let's completely drop Funding. 
    # Let's strictly run the Pure Capitulation Bounce (v37) but guarantee absolutely zero lookahead
    # by shifting ALL signals forward before taking the entry price.
    
    df['ret_3d'] = df['close'] / df['close'].shift(3) - 1
    df['setup'] = df['ret_3d'] < -0.20
    df['setup_active'] = df['setup'].rolling(3).max().shift(1).fillna(0).astype(bool)
    
    df['confirmation'] = df['close'] > df['open']
    
    df['vol_ma'] = df['volume'].rolling(30).mean().shift(1)
    df['vol_spike'] = df['volume'].shift(1) > (df['vol_ma'] * 1.5)
    
    df['signal'] = df['setup_active'] & df['confirmation'] & df['vol_spike']
    
    # Ensure signal is from YESTERDAY when we buy TODAY'S open
    df['execute_today'] = df['signal'].shift(1).fillna(False)
    
    signals = df['execute_today'].values
    opens = df['open'].values
    closes = df['close'].values
    
    trades = []
    in_position = False
    entry_price = 0.0
    
    HOLD_DAYS = 5
    
    for i in range(30, len(df) - HOLD_DAYS - 1):
        if not in_position:
            # We are currently at day `i`. 
            # `execute_today[i]` is True if `signal[i-1]` was True.
            # This means by the open of day `i`, we already have the confirmed close of `i-1`.
            # Zero Lookahead Guarantee.
            if signals[i]:
                in_position = True
                entry_price = opens[i] # Taker Entry on the Open of Day `i`
                
                # We hold for `HOLD_DAYS` and exit on the close.
                # So we enter at `opens[i]`, and exit at `closes[i + HOLD_DAYS - 1]`
                exit_price = closes[i + HOLD_DAYS - 1]
                net_pnl = ((exit_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                
                trades.append({
                    'exit_time': df.index[i + HOLD_DAYS - 1], 
                    'pnl': net_pnl
                })
                
                in_position = False

    return trades

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    
    print(f"Testing Pure Capitulation Bounce (Zero Lookahead)...")
    
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
    
    # Portfolio equity simulation (risking 5% per trade)
    df_time['equity_pnl'] = df_time['pnl'] * 0.05
    df_time['equity'] = 1.0 + df_time['equity_pnl'].cumsum()
    
    monthly_group = df_time.resample('1ME').agg(
        trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).mean() if len(x)>0 else 0),
        avg_pnl=('pnl', 'mean'),
        equity_return=('equity_pnl', 'sum')
    )
    
    print("\n" + "="*50)
    print("--- ZERO LOOKAHEAD CAPITULATION RESULTS ---")
    print("="*50)
    print(f"Total Trades: {len(df_trades)}")
    print(f"Win Rate: {(df_trades['pnl'] > 0).mean():.2%}")
    print(f"Avg PnL (Gross): {df_trades['pnl'].mean():.4%}")
    print(f"Total Portfolio Return (5% Risk/Trade): {(df_time['equity'].iloc[-1] - 1):.2%}")
    
    print("\n--- Monthly Portfolio Returns ---")
    print(monthly_group[['trades', 'win_rate', 'avg_pnl', 'equity_return']])
    
if __name__ == "__main__":
    main()
