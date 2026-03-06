import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
MAKER_FEE = 0.0004
TAKER_FEE = 0.0010

def get_btc_regime():
    files = glob.glob(f"{DATALAKE_DIR}/BTCUSDT/*_kline_1m.csv")
    if not files: return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['timestamp'] = pd.to_datetime(df.get('startTime', df.get('timestamp', None)), unit='ms')
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).set_index('timestamp')
    df_4h = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
    df_4h.dropna(inplace=True)
    period = 21
    change = abs(df_4h['close'] - df_4h['close'].shift(period))
    volatility = abs(df_4h['close'] - df_4h['close'].shift(1)).rolling(period).sum()
    df_4h['btc_ker'] = change / volatility
    df_4h['btc_is_trending'] = df_4h['btc_ker'] >= 0.20
    return df_4h[['btc_is_trending']]

BTC_REGIME = get_btc_regime()

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
    if df.empty or len(df) < (60 * 24 * 7): return None
    return symbol, df

def process_symbol(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    
    df_4h = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df_4h.dropna(inplace=True)
    
    if BTC_REGIME is not None:
        df_4h = df_4h.join(BTC_REGIME, how='left').ffill().infer_objects(copy=False)
    else:
        df_4h['btc_is_trending'] = True
        
    period = 21
    change = abs(df_4h['close'] - df_4h['close'].shift(period))
    volatility = abs(df_4h['close'] - df_4h['close'].shift(1)).rolling(period).sum()
    df_4h['local_ker'] = change / volatility
    df_4h['local_is_trending'] = df_4h['local_ker'] >= 0.15
    
    df_4h['ema_200'] = df_4h['close'].ewm(span=200, adjust=False).mean()
    df_4h['macro_bull'] = df_4h['close'] > df_4h['ema_200']
    
    df_4h['high_20'] = df_4h['high'].rolling(20).max().shift(1)
    df_4h['low_20'] = df_4h['low'].rolling(20).min().shift(1)
    
    df_4h['vol_ma'] = df_4h['volume'].rolling(20).mean().shift(1)
    df_4h['vol_spike'] = df_4h['volume'] > (df_4h['vol_ma'] * 2.0)
    
    df_4h['regime_ok'] = df_4h['btc_is_trending'] | df_4h['local_is_trending']
    df_4h['long_breakout'] = (df_4h['close'] > df_4h['high_20']) & df_4h['macro_bull'] & df_4h['vol_spike'] & df_4h['regime_ok']
    df_4h['short_breakout'] = (df_4h['close'] < df_4h['low_20']) & (~df_4h['macro_bull']) & df_4h['vol_spike'] & df_4h['regime_ok']
    
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
    
    TP_PCT = 0.20
    SL_PCT = 0.10
    TIME_STOP_CANDLES = 6 * 14
    
    for i in range(200, n - 1):
        if not in_position:
            if i > cooldown_until:
                if signals_long[i]:
                    in_position = True
                    pos_type = 1
                    entry_price = opens[i+1]
                    entry_idx = i + 1
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
                    trades.append({'symbol': symbol, 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'raw_pnl': net_pnl})
                    in_position = False
                    cooldown_until = i + 6
                elif highs[i] >= target_price:
                    net_pnl = ((target_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE
                    trades.append({'symbol': symbol, 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'raw_pnl': net_pnl})
                    in_position = False
                    cooldown_until = i + 6
                elif (i - entry_idx) >= TIME_STOP_CANDLES:
                    net_pnl = ((closes[i] - entry_price) / entry_price) - (TAKER_FEE * 2)
                    trades.append({'symbol': symbol, 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'raw_pnl': net_pnl})
                    in_position = False
                    cooldown_until = i + 6
                    
            elif pos_type == -1:
                target_price = entry_price * (1 - TP_PCT)
                stop_price = entry_price * (1 + SL_PCT)
                
                if highs[i] >= stop_price:
                    net_pnl = ((entry_price - stop_price) / entry_price) - (TAKER_FEE * 2)
                    trades.append({'symbol': symbol, 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'raw_pnl': net_pnl})
                    in_position = False
                    cooldown_until = i + 6
                elif lows[i] <= target_price:
                    net_pnl = ((entry_price - target_price) / entry_price) - TAKER_FEE - MAKER_FEE
                    trades.append({'symbol': symbol, 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'raw_pnl': net_pnl})
                    in_position = False
                    cooldown_until = i + 6
                elif (i - entry_idx) >= TIME_STOP_CANDLES:
                    net_pnl = ((entry_price - closes[i]) / entry_price) - (TAKER_FEE * 2)
                    trades.append({'symbol': symbol, 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'raw_pnl': net_pnl})
                    in_position = False
                    cooldown_until = i + 6

    return trades

def main():
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT', 'AVAXUSDT', 'LINKUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    
    print(f"Generating Portfolio Simulation ($1000 Start) on Deep History...")
    
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
    
    # Sort trades chronologically by exit time for compounding portfolio math
    df_trades = df_trades.sort_values('exit_time').reset_index(drop=True)
    
    INITIAL_CAPITAL = 1000.0
    RISK_PER_TRADE = 0.02 # 2% risk per trade
    STOP_LOSS_PCT = 0.10 # 10% SL
    POSITION_SIZE = RISK_PER_TRADE / STOP_LOSS_PCT # 20% of capital per trade
    
    capital = INITIAL_CAPITAL
    equity_curve = []
    dates = []
    
    df_trades['equity'] = INITIAL_CAPITAL
    df_trades['trade_pnl_usd'] = 0.0
    
    for idx, row in df_trades.iterrows():
        # Trade outcome in percent (e.g. +0.198 or -0.102)
        raw_pnl = row['raw_pnl']
        
        # Position amount in USD
        pos_usd = capital * POSITION_SIZE
        
        # PnL in USD
        trade_usd_pnl = pos_usd * raw_pnl
        capital += trade_usd_pnl
        
        df_trades.at[idx, 'trade_pnl_usd'] = trade_usd_pnl
        df_trades.at[idx, 'equity'] = capital
        
        equity_curve.append(capital)
        dates.append(row['exit_time'])
        
    # Plot Equity Curve
    plt.figure(figsize=(12, 6))
    plt.plot(dates, equity_curve, label='Portfolio Equity', color='blue')
    plt.axhline(INITIAL_CAPITAL, color='red', linestyle='--', label='Starting Capital ($1000)')
    plt.title(f'Asymmetric Macro Trend Portfolio (Starting $1000, 2% Risk/Trade)')
    plt.xlabel('Date')
    plt.ylabel('Account Balance ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    chart_path = '/home/ubuntu/Projects/skytrade6/gemini-1/portfolio_equity_curve.png'
    plt.savefig(chart_path)
    print(f"\nEquity curve saved to {chart_path}")
    
    # Monthly Breakdown
    df_trades['month'] = df_trades['exit_time'].dt.to_period('M')
    
    monthly_stats = []
    for month, group in df_trades.groupby('month'):
        trades = len(group)
        wins = sum(group['raw_pnl'] > 0)
        win_rate = wins / trades if trades > 0 else 0
        
        # Monthly return is (End Equity - Start Equity) / Start Equity
        # We find the equity at the start of the month by looking at the last trade of the previous month
        start_equity = group.iloc[0]['equity'] - group.iloc[0]['trade_pnl_usd']
        end_equity = group.iloc[-1]['equity']
        monthly_return = (end_equity - start_equity) / start_equity
        
        profit_usd = end_equity - start_equity
        
        monthly_stats.append({
            'Month': str(month),
            'Trades': trades,
            'Win Rate': win_rate,
            'Return %': monthly_return,
            'Profit USD': profit_usd,
            'End Balance': end_equity
        })
        
    df_monthly = pd.DataFrame(monthly_stats)
    
    print("\n" + "="*85)
    print(f"--- MONTHLY PORTFOLIO BREAKDOWN (Compounding 2% Risk) ---")
    print("="*85)
    print(f"{'Month':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Return %':<12} | {'Profit USD':<12} | {'End Balance':<15}")
    print("-" * 85)
    
    for _, row in df_monthly.iterrows():
        print(f"{row['Month']:<10} | {row['Trades']:<8} | {row['Win Rate']:<10.2%} | {row['Return %']:<12.2%} | ${row['Profit USD']:<11.2f} | ${row['End Balance']:<14.2f}")
        
    print("-" * 85)
    final_balance = df_trades.iloc[-1]['equity']
    total_return = (final_balance - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    # Calculate Max Drawdown
    peak = df_trades['equity'].expanding(min_periods=1).max()
    drawdown = (df_trades['equity'] - peak) / peak
    max_dd = drawdown.min()
    
    print(f"STARTING BALANCE: ${INITIAL_CAPITAL:.2f}")
    print(f"FINAL BALANCE:    ${final_balance:.2f}")
    print(f"TOTAL RETURN:     {total_return:.2%}")
    print(f"MAX DRAWDOWN:     {max_dd:.2%}")
    print(f"TOTAL TRADES:     {len(df_trades)}")

if __name__ == "__main__":
    main()
