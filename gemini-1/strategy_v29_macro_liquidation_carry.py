import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
MAKER_FEE = 0.0004
TAKER_FEE = 0.0010

# We need a robust strategy that doesn't bleed.
# Let's combine the macro capitulation strategy that worked perfectly in October (v8/v9)
# with a strict regime filter so it doesn't bleed during sideways chop.
#
# Thesis: Buy forced liquidations ONLY when the macro regime is an established uptrend, 
# because in a bull market, liquidations are purely structural (overleverage) rather than 
# fundamental (people dumping the coin).

def load_btc_regime():
    files = glob.glob(f"{DATALAKE_DIR}/BTCUSDT/*_kline_1m.csv")
    if not files: return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['timestamp'] = pd.to_datetime(df.get('startTime', df.get('timestamp', None)), unit='ms')
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).set_index('timestamp')
    df = df.resample('1d').agg({'close': 'last'})
    df.dropna(inplace=True)
    
    # Macro Uptrend: BTC above 30-day and 60-day moving averages
    df['btc_sma_30'] = df['close'].rolling(30).mean()
    df['btc_sma_60'] = df['close'].rolling(60).mean()
    df['macro_bull'] = (df['close'] > df['btc_sma_30']) & (df['btc_sma_30'] > df['btc_sma_60'])
    
    df_4h = df.resample('4h').ffill()
    return df_4h[['macro_bull']]

BTC_REGIME = load_btc_regime()

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
    df['timestamp'] = pd.to_datetime(df.get('startTime', df.get('timestamp', None)), unit='ms')
         
    df = df[(df['close'] > 0) & (df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0)]
    df = df[df['high'] >= df['low']]
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['ret'] = df['close'].pct_change()
    df = df[(df['ret'] > -0.5) & (df['ret'] < 1.0)]
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp')
    
    df = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df.dropna(inplace=True)
    
    oi_list = []
    for f in files_oi:
        try: oi_list.append(pd.read_csv(f))
        except: pass
    if oi_list:
        df_oi = pd.concat(oi_list, ignore_index=True)
        df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'], unit='ms')
        df_oi = df_oi.drop_duplicates(subset=['timestamp']).sort_values('timestamp').set_index('timestamp')
        df_oi = df_oi.resample('4h').ffill()
        df = df.join(df_oi[['openInterest']], how='left').ffill()
    else: return None
    
    if BTC_REGIME is not None:
        df = df.join(BTC_REGIME, how='left').ffill()
        
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
    
    if df.empty or len(df) < (6 * 30): return None
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    
    if 'openInterest' not in df.columns or 'macro_bull' not in df.columns: return []
    
    # 24H Price Drop > 15% AND 24H OI Drop > 15%
    df['ret_24h'] = df['close'] / df['close'].shift(6) - 1
    df['oi_24h'] = df['openInterest'] / df['openInterest'].shift(6) - 1
    
    df['flush_zone'] = (df['ret_24h'] < -0.15) & (df['oi_24h'] < -0.15)
    
    # Only buy the flush if we are in a MACRO BULL market
    df['long_signal'] = df['flush_zone'] & df['macro_bull']
    
    signals = df['long_signal'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    trades = []
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    cooldown_until = 0
    n = len(df)
    
    # Massive RR parameters to survive fees and volatility
    TP_PCT = 0.25 # 25% target
    SL_PCT = 0.15 # 15% stop loss
    TIME_STOP_CANDLES = 6 * 5 # 5 Days
    
    for i in range(6, n - 1):
        if not in_position:
            if signals[i] and i > cooldown_until:
                in_position = True
                entry_price = opens[i+1] # Taker
                entry_idx = i + 1
        else:
            target_price = entry_price * (1 + TP_PCT)
            stop_price = entry_price * (1 - SL_PCT)
            
            if lows[i] <= stop_price:
                net_pnl = ((stop_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'sl'})
                in_position = False
                cooldown_until = i + 6
                
            elif highs[i] >= target_price:
                net_pnl = ((target_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE
                trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'tp'})
                in_position = False
                cooldown_until = i + 6
                
            elif (i - entry_idx) >= TIME_STOP_CANDLES:
                net_pnl = ((closes[i] - entry_price) / entry_price) - (TAKER_FEE * 2)
                trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'time'})
                in_position = False
                cooldown_until = i + 6

    return trades

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    
    print(f"Testing Macro-Filtered Liquidation Buy Strategy...")
    
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
        total_pnl=('pnl', 'sum'),
        avg_pnl=('pnl', 'mean')
    )
    
    print("\n" + "="*50)
    print("--- Strategy Results (Macro Filtered Liquidation) ---")
    print("="*50)
    print(f"Total Trades: {len(df_trades)}")
    print(f"Win Rate: {(df_trades['pnl'] > 0).mean():.2%}")
    print(f"Avg Net PnL per Trade: {df_trades['pnl'].mean():.4%}")
    print(f"Total Cumulative PnL: {df_time['cumulative'].iloc[-1]:.4%}")
    
    print("\n--- Exit Types ---")
    print(df_trades['type'].value_counts())
    
    print("\n--- Monthly Distribution ---")
    print(monthly_group)
    
    # Add a portfolio equity curve simulation
    # Risk 2% per trade
    df_time['equity_pnl'] = df_time['pnl'] * 0.02
    df_time['equity'] = 1.0 + df_time['equity_pnl'].cumsum()
    
    monthly_equity = df_time.resample('1ME')['equity_pnl'].sum()
    print("\n--- Simulated Monthly Equity Returns (2% Risk/Trade) ---")
    for date, ret in monthly_equity.items():
        print(f"{date.strftime('%Y-%b')}: {ret:.2%}")

if __name__ == "__main__":
    main()
