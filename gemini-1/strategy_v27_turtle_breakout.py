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
    
    # 1D candles
    df = df.resample('1d').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df.dropna(inplace=True)
    
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
    
    if df.empty or len(df) < 40: return None
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    
    # Classic Turtle Rules:
    # Enter Long on 20-Day Breakout. Exit on 10-Day Low.
    # Enter Short on 20-Day Breakdown. Exit on 10-Day High.
    
    df['high_20d'] = df['high'].rolling(20).max().shift(1)
    df['low_20d'] = df['low'].rolling(20).min().shift(1)
    
    df['high_10d'] = df['high'].rolling(10).max().shift(1)
    df['low_10d'] = df['low'].rolling(10).min().shift(1)
    
    # ATR for position sizing/volatility filter
    df['tr'] = df['high'] - df['low']
    df['atr'] = df['tr'].rolling(20).mean()
    
    df['long_signal'] = df['close'] > df['high_20d']
    df['short_signal'] = df['close'] < df['low_20d']
    
    signals_long = df['long_signal'].values
    signals_short = df['short_signal'].values
    
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    low_10d = df['low_10d'].values
    high_10d = df['high_10d'].values
    atrs = df['atr'].values
    
    trades = []
    in_position = False
    entry_price = 0.0
    pos_type = 0
    
    for i in range(20, len(df) - 1):
        if not in_position:
            if signals_long[i]:
                in_position = True
                pos_type = 1
                entry_price = opens[i+1] # Taker
            elif signals_short[i]:
                in_position = True
                pos_type = -1
                entry_price = opens[i+1] # Taker
        else:
            if pos_type == 1:
                exit_price = low_10d[i]
                stop_loss = entry_price - (2 * atrs[i]) # 2 ATR hard stop
                
                actual_exit = max(exit_price, stop_loss)
                
                if lows[i] < actual_exit:
                    net_pnl = ((actual_exit - entry_price) / entry_price) - (TAKER_FEE * 2)
                    trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'exit_long'})
                    in_position = False
                    
            elif pos_type == -1:
                exit_price = high_10d[i]
                stop_loss = entry_price + (2 * atrs[i])
                
                actual_exit = min(exit_price, stop_loss)
                
                if highs[i] > actual_exit:
                    net_pnl = ((entry_price - actual_exit) / entry_price) - (TAKER_FEE * 2)
                    trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'exit_short'})
                    in_position = False

    return trades

def main():
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'MATICUSDT', 'NEARUSDT']
    
    print(f"Testing 20-Day Turtle Breakout on Top 10 Coins...")
    
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
    print("--- Strategy Results (Turtle Breakout) ---")
    print("="*50)
    print(f"Total Trades: {len(df_trades)}")
    print(f"Win Rate: {(df_trades['pnl'] > 0).mean():.2%}")
    print(f"Avg Net PnL per Trade: {df_trades['pnl'].mean():.4%}")
    print(f"Total Cumulative PnL: {df_time['cumulative'].iloc[-1]:.4%}")
    
    print("\n--- Monthly Distribution ---")
    print(monthly_group)

if __name__ == "__main__":
    main()
