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
    files_funding = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_funding_rate.csv")
    
    if not files or not files_funding: return None
        
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
    
    # Since funding is every 8 hours, let's look at 1-hour candles
    df = df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
    df.dropna(inplace=True)
    
    fund_list = []
    for f in files_funding:
        try: fund_list.append(pd.read_csv(f))
        except: pass
        
    if fund_list:
        df_fund = pd.concat(fund_list, ignore_index=True)
        if 'fundingRateTimestamp' in df_fund.columns:
            df_fund['timestamp'] = pd.to_datetime(df_fund['fundingRateTimestamp'], unit='ms')
        elif 'timestamp' in df_fund.columns and df_fund['timestamp'].dtype != 'datetime64[ns]':
            df_fund['timestamp'] = pd.to_datetime(df_fund['timestamp'], unit='ms')
        df_fund = df_fund.drop_duplicates(subset=['timestamp']).sort_values('timestamp').set_index('timestamp')
        df_fund = df_fund.resample('1h').ffill()
        df = df.join(df_fund[['fundingRate']], how='left').ffill()
    else: return None
    
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
    
    if df.empty or len(df) < (24 * 7): return None
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    
    # We want consistently positive funding where shorts are heavily paid.
    # The thesis is that highly crowded longs slowly bleed out to pay funding.
    if 'fundingRate' not in df.columns: return []
    
    # 3-day average funding rate
    df['funding_ma_3d'] = df['fundingRate'].rolling(72).mean()
    
    # Extreme sustained positive funding (> 0.1% per 8h)
    df['extreme_long_crowding'] = df['funding_ma_3d'] > 0.001
    
    # Price action starting to crack (lower highs)
    df['hh_24h'] = df['high'].rolling(24).max()
    df['cracking'] = df['close'] < (df['hh_24h'] * 0.90) # down 10% from local top
    
    # SHORT signal
    signals = (df['extreme_long_crowding'] & df['cracking']).values
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
    
    TP_PCT = -0.15 # Take profit when price drops 15%
    SL_PCT = 0.10  # Stop loss if price rises 10%
    TIME_STOP_CANDLES = 24 * 5 # 5 days
    
    for i in range(72, n - 1):
        if not in_position:
            if signals[i] and i > cooldown_until:
                in_position = True
                entry_price = opens[i+1] # Entering Short
                entry_idx = i + 1
        else:
            # Short logic
            target_price = entry_price * (1 + TP_PCT)
            stop_price = entry_price * (1 + SL_PCT)
            
            # Pessimistic: assume SL hit before TP
            if highs[i] >= stop_price:
                net_pnl = ((entry_price - stop_price) / entry_price) - (TAKER_FEE * 2)
                trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'sl'})
                in_position = False
                cooldown_until = i + 24
                
            elif lows[i] <= target_price:
                net_pnl = ((entry_price - target_price) / entry_price) - TAKER_FEE - MAKER_FEE
                trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'tp'})
                in_position = False
                cooldown_until = i + 24
                
            elif (i - entry_idx) >= TIME_STOP_CANDLES:
                net_pnl = ((entry_price - closes[i]) / entry_price) - (TAKER_FEE * 2)
                trades.append({'exit_time': df.index[i], 'pnl': net_pnl, 'type': 'time'})
                in_position = False
                cooldown_until = i + 24

    return trades

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    print(f"Testing Structural Funding Arbitrage (Shorting Crowded Longs)...")
    
    all_trades = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        for pnl_list in tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols)):
            all_trades.extend(pnl_list)
                
    if not all_trades:
        print("No trades generated.")
        return
        
    df_trades = pd.DataFrame(all_trades)
    df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
    df_time = df_trades.set_index('exit_time')
    
    monthly_group = df_time.resample('1ME').agg(
        trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).mean() if len(x) > 0 else 0),
        total_pnl=('pnl', 'sum'),
        avg_pnl=('pnl', 'mean')
    )
    
    print("\n--- Strategy Results ---")
    print(f"Total Trades: {len(df_trades)}")
    print(f"Win Rate: {(df_trades['pnl'] > 0).mean():.2%}")
    print(f"Avg PnL: {df_trades['pnl'].mean():.4%}")
    print(f"Total PnL (1x Lev): {df_trades['pnl'].sum():.4%}")
    
    print("\n--- Exit Types ---")
    print(df_trades['type'].value_counts())
    
    print("\n--- Monthly Distribution ---")
    print(monthly_group)

if __name__ == "__main__":
    main()
