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
    
    # Clean anomalies
    df['ret'] = df['close'].pct_change()
    df = df[(df['ret'] > -0.15) & (df['ret'] < 0.15)]
    
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp')
    df.dropna(inplace=True)
    
    if not df.empty:
        cutoff = df.index.max() - pd.Timedelta(days=180)
        df = df[df.index >= cutoff].copy()
    
    if df.empty or len(df) < (60 * 24 * 7): return None
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    
    # 4H Multi-Timeframe Trend
    df_4h = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df_4h.dropna(inplace=True)
    
    # Trend: 200 EMA
    df_4h['ema_200'] = df_4h['close'].ewm(span=200, adjust=False).mean()
    df_4h['macro_bull'] = df_4h['close'] > df_4h['ema_200']
    
    # Volatility / Range Expansion (Donchian 20)
    df_4h['high_20'] = df_4h['high'].rolling(20).max().shift(1)
    df_4h['low_20'] = df_4h['low'].rolling(20).min().shift(1)
    
    # Volume Expansion
    df_4h['vol_ma'] = df_4h['volume'].rolling(20).mean().shift(1)
    df_4h['vol_spike'] = df_4h['volume'] > (df_4h['vol_ma'] * 2.0)
    
    # Signals
    df_4h['long_breakout'] = (df_4h['close'] > df_4h['high_20']) & df_4h['macro_bull'] & df_4h['vol_spike']
    df_4h['short_breakout'] = (df_4h['close'] < df_4h['low_20']) & (~df_4h['macro_bull']) & df_4h['vol_spike']
    
    signals_long = df_4h['long_breakout'].values
    signals_short = df_4h['short_breakout'].values
    
    opens = df_4h['open'].values
    highs = df_4h['high'].values
    lows = df_4h['low'].values
    closes = df_4h['close'].values
    
    trades = []
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    cooldown_until = 0
    n = len(df_4h)
    pos_type = 0
    
    TP_PCT = 0.15
    SL_PCT = 0.05
    TIME_STOP_CANDLES = 6 * 14 # 14 days maximum hold
    
    for i in range(200, n - 1):
        if not in_position:
            if i > cooldown_until:
                if signals_long[i]:
                    in_position = True
                    pos_type = 1
                    entry_price = opens[i+1] # Taker Entry next 4H open
                    entry_idx = i + 1
                    
                    # We log the entry time to track when the trade actually started
                    entry_time = df_4h.index[i+1]
                elif signals_short[i]:
                    in_position = True
                    pos_type = -1
                    entry_price = opens[i+1]
                    entry_idx = i + 1
                    entry_time = df_4h.index[i+1]
        else:
            if pos_type == 1:
                target_price = entry_price * (1 + TP_PCT)
                stop_price = entry_price * (1 - SL_PCT)
                
                if lows[i] <= stop_price:
                    net_pnl = ((stop_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                    trades.append({'entry_time': entry_time, 'exit_time': df_4h.index[i], 'pnl': net_pnl, 'type': 'sl'})
                    in_position = False
                    cooldown_until = i + 6
                elif highs[i] >= target_price:
                    net_pnl = ((target_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE
                    trades.append({'entry_time': entry_time, 'exit_time': df_4h.index[i], 'pnl': net_pnl, 'type': 'tp'})
                    in_position = False
                    cooldown_until = i + 6
                elif (i - entry_idx) >= TIME_STOP_CANDLES:
                    net_pnl = ((closes[i] - entry_price) / entry_price) - (TAKER_FEE * 2)
                    trades.append({'entry_time': entry_time, 'exit_time': df_4h.index[i], 'pnl': net_pnl, 'type': 'time'})
                    in_position = False
                    cooldown_until = i + 6
                    
            elif pos_type == -1:
                target_price = entry_price * (1 - TP_PCT)
                stop_price = entry_price * (1 + SL_PCT)
                
                if highs[i] >= stop_price:
                    net_pnl = ((entry_price - stop_price) / entry_price) - (TAKER_FEE * 2)
                    trades.append({'entry_time': entry_time, 'exit_time': df_4h.index[i], 'pnl': net_pnl, 'type': 'sl'})
                    in_position = False
                    cooldown_until = i + 6
                elif lows[i] <= target_price:
                    net_pnl = ((entry_price - target_price) / entry_price) - TAKER_FEE - MAKER_FEE
                    trades.append({'entry_time': entry_time, 'exit_time': df_4h.index[i], 'pnl': net_pnl, 'type': 'tp'})
                    in_position = False
                    cooldown_until = i + 6
                elif (i - entry_idx) >= TIME_STOP_CANDLES:
                    net_pnl = ((entry_price - closes[i]) / entry_price) - (TAKER_FEE * 2)
                    trades.append({'entry_time': entry_time, 'exit_time': df_4h.index[i], 'pnl': net_pnl, 'type': 'time'})
                    in_position = False
                    cooldown_until = i + 6

    return trades

def main():
    symbols = [d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d))]
    print(f"Generating Detailed Monthly Breakdown for v42 on {len(symbols)} Coins...")
    
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
    df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
    df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
    
    # We will aggregate by the EXit Month to match realized PnL logic.
    df_trades['month'] = df_trades['exit_time'].dt.to_period('M')
    
    # Calculate portfolio equity PnL
    # Since Stop Loss is 5%, to risk 2% of the account, we allocate 40% of the account per trade.
    # Therefore, Portfolio PnL = Raw PnL * 0.40
    # Wait, let's keep it simple: if a trade loses 5%, and we want that to equal 2% of our portfolio...
    # (5% loss) * Position Size = 2% loss -> Position Size = 0.02 / 0.05 = 0.40 (40% allocation).
    
    df_trades['equity_pnl'] = df_trades['pnl'] * 0.40
    
    # Group by month
    monthly_stats = []
    for month, group in df_trades.groupby('month'):
        trades = len(group)
        wins = sum(group['pnl'] > 0)
        win_rate = wins / trades if trades > 0 else 0
        raw_pnl = group['pnl'].sum()
        equity_pnl = group['equity_pnl'].sum()
        
        # Calculate EV for this month
        avg_win = group[group['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
        avg_loss = group[group['pnl'] <= 0]['pnl'].mean() if trades - wins > 0 else 0
        ev = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        monthly_stats.append({
            'Month': str(month),
            'Trades': trades,
            'Win_Rate': win_rate,
            'Avg_Win': avg_win,
            'Avg_Loss': avg_loss,
            'EV_Per_Trade': ev,
            'Equity_Return': equity_pnl
        })
        
    df_monthly = pd.DataFrame(monthly_stats)
    
    print("\n" + "="*80)
    print("--- V42 ASYMMETRIC TREND: DETAILED MONTHLY BREAKDOWN ---")
    print("="*80)
    print("Assumptions:")
    print(" - Take Profit: +15.0%")
    print(" - Stop Loss: -5.0%")
    print(" - Risk Per Trade: 2.0% of Portfolio (requires 40% position sizing)")
    print(" - Fees: Explicitly deducts 0.20% Taker fee round-trip.")
    print("-" * 80)
    
    # Format for printing
    print(f"{'Month':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Avg Win':<10} | {'Avg Loss':<10} | {'EV/Trade':<10} | {'Equity Return (2% Risk)':<20}")
    print("-" * 80)
    
    for _, row in df_monthly.iterrows():
        print(f"{row['Month']:<10} | {row['Trades']:<8} | {row['Win_Rate']:<10.2%} | {row['Avg_Win']:<10.2%} | {row['Avg_Loss']:<10.2%} | {row['EV_Per_Trade']:<10.2%} | {row['Equity_Return']:<20.2%}")
        
    print("-" * 80)
    total_equity_return = df_monthly['Equity_Return'].sum()
    print(f"TOTAL PORTFOLIO RETURN (6 Months): {total_equity_return:.2%}")
    print("="*80)

if __name__ == "__main__":
    main()
