import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm
from dateutil.relativedelta import relativedelta

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
    df = df[(df['ret'] > -0.15) & (df['ret'] < 0.15)]
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp')
    df.dropna(inplace=True)
    if df.empty or len(df) < (60 * 24 * 7): return None
    return symbol, df

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
    return df_4h[['btc_ker']]

BTC_REGIME = get_btc_regime()

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
        df_4h['btc_ker'] = 1.0
        
    period = 21
    change = abs(df_4h['close'] - df_4h['close'].shift(period))
    volatility = abs(df_4h['close'] - df_4h['close'].shift(1)).rolling(period).sum()
    df_4h['local_ker'] = change / volatility
    
    df_4h['ema_200'] = df_4h['close'].ewm(span=200, adjust=False).mean()
    df_4h['macro_bull'] = df_4h['close'] > df_4h['ema_200']
    
    df_4h['high_20'] = df_4h['high'].rolling(20).max().shift(1)
    df_4h['low_20'] = df_4h['low'].rolling(20).min().shift(1)
    
    df_4h['vol_ma'] = df_4h['volume'].rolling(20).mean().shift(1)
    df_4h['vol_spike'] = df_4h['volume'] > (df_4h['vol_ma'] * 2.0)
    
    # Best KER threshold found
    KER_THRESH = 0.15
    df_4h['regime_ok'] = (df_4h['btc_ker'] >= KER_THRESH) | (df_4h['local_ker'] >= KER_THRESH)
    
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
    pos_type = 0
    
    TP_PCT = 0.20
    SL_PCT = 0.10
    TIME_STOP_CANDLES = 6 * 14
    
    for i in range(200, len(df_4h) - 1):
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
                    trades.append({'symbol': symbol, 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'raw_pnl': net_pnl, 'type': 'SL'})
                    in_position = False
                    cooldown_until = i + 6
                elif highs[i] >= target_price:
                    net_pnl = ((target_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE
                    trades.append({'symbol': symbol, 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'raw_pnl': net_pnl, 'type': 'TP'})
                    in_position = False
                    cooldown_until = i + 6
                elif (i - entry_idx) >= TIME_STOP_CANDLES:
                    net_pnl = ((closes[i] - entry_price) / entry_price) - (TAKER_FEE * 2)
                    trades.append({'symbol': symbol, 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'raw_pnl': net_pnl, 'type': 'TIME'})
                    in_position = False
                    cooldown_until = i + 6
                    
            elif pos_type == -1:
                target_price = entry_price * (1 - TP_PCT)
                stop_price = entry_price * (1 + SL_PCT)
                
                if highs[i] >= stop_price:
                    net_pnl = ((entry_price - stop_price) / entry_price) - (TAKER_FEE * 2)
                    trades.append({'symbol': symbol, 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'raw_pnl': net_pnl, 'type': 'SL'})
                    in_position = False
                    cooldown_until = i + 6
                elif lows[i] <= target_price:
                    net_pnl = ((entry_price - target_price) / entry_price) - TAKER_FEE - MAKER_FEE
                    trades.append({'symbol': symbol, 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'raw_pnl': net_pnl, 'type': 'TP'})
                    in_position = False
                    cooldown_until = i + 6
                elif (i - entry_idx) >= TIME_STOP_CANDLES:
                    net_pnl = ((entry_price - closes[i]) / entry_price) - (TAKER_FEE * 2)
                    trades.append({'symbol': symbol, 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'raw_pnl': net_pnl, 'type': 'TIME'})
                    in_position = False
                    cooldown_until = i + 6

    return trades

def simulate_portfolio(df_trades, initial_capital, alloc_pct, leverage):
    """
    alloc_pct: base % of total capital allocated per trade (margin)
    leverage: leverage multiplier applied to that margin
    
    Example: alloc_pct=5%, leverage=2x
    We put 5% of account balance as margin. Total position size = 10% of account.
    If raw_pnl is +20%, we make 10% * 20% = +2% on total account.
    """
    capital = float(initial_capital)
    df_trades = df_trades.sort_values('exit_time').reset_index(drop=True)
    
    # We need to track actual equity over time correctly
    # If the loss exceeds 100%, the account is liquidated
    
    max_capital = capital
    max_dd = 0.0
    
    for idx, row in df_trades.iterrows():
        if capital <= 0:
            return 0.0, -1.0 # Liquidated
            
        # Position size in USD based on Leverage
        pos_usd = capital * alloc_pct * leverage
        
        # PnL in USD
        trade_usd_pnl = pos_usd * row['raw_pnl']
        
        # If we lose more than the allocated margin, we get liquidated on that specific trade
        max_loss = capital * alloc_pct
        if trade_usd_pnl < -max_loss:
            trade_usd_pnl = -max_loss # We only lose our isolated margin
            
        capital += trade_usd_pnl
        
        if capital > max_capital:
            max_capital = capital
            
        current_dd = (capital - max_capital) / max_capital
        if current_dd < max_dd:
            max_dd = current_dd
            
    return capital, max_dd

def main():
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT', 'AVAXUSDT', 'LINKUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    print(f"Extracting all trades for final KER strategy...")
    
    all_trades = []
    pool = multiprocessing.Pool(processes=min(8, multiprocessing.cpu_count()))
    for trades in tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols)):
        all_trades.extend(trades)
    pool.close()
    pool.join()
                
    if not all_trades:
        return
        
    df_trades = pd.DataFrame(all_trades)
    df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
    df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
    
    # Only evaluate recent 4 years to be relevant
    df_trades = df_trades[df_trades['entry_time'] >= '2022-01-01'].copy()
    
    print("\n" + "="*80)
    print("--- RISK MANAGEMENT MATRIX (Jan 2022 - Mar 2026) ---")
    print("="*80)
    print(f"{'Alloc %':<10} | {'Leverage':<10} | {'Risk/Trade':<12} | {'Final Bal ($)':<15} | {'Max DD':<10}")
    print("-" * 80)
    
    allocations = [0.05, 0.10, 0.15, 0.20, 0.25]
    leverages = [1, 2, 3, 5]
    
    results_matrix = []
    
    for alloc in allocations:
        for lev in leverages:
            final_bal, max_dd = simulate_portfolio(df_trades, 1000.0, alloc, lev)
            
            # Risk per trade is simply the Position Size (alloc * lev) multiplied by our 10% Stop Loss
            risk_per_trade = (alloc * lev) * 0.10
            
            status = "LIQUIDATED" if final_bal <= 0 else f"${final_bal:.2f}"
            dd_str = f"{max_dd:.2%}" if final_bal > 0 else "-100.00%"
            
            print(f"{alloc*100:>2.0f}%      | {lev}x        | {risk_per_trade*100:>4.1f}%       | {status:<15} | {dd_str:<10}")
            results_matrix.append({'alloc': alloc, 'lev': lev, 'final': final_bal, 'dd': max_dd, 'risk': risk_per_trade})

    # Pick an optimal one for Monthly Breakdown: e.g., 20% Alloc, 1x Lev (2% Risk) or 10% Alloc, 3x Lev (3% Risk)
    optimal_alloc = 0.10
    optimal_lev = 3
    
    print("\n" + "="*80)
    print(f"--- MONTHLY PNL BREAKDOWN (Optimal: {optimal_alloc*100:.0f}% Alloc, {optimal_lev}x Lev = {optimal_alloc*optimal_lev*0.10*100:.1f}% Risk/Trade) ---")
    print("="*80)
    
    df_trades = df_trades.sort_values('exit_time').reset_index(drop=True)
    df_trades['month'] = df_trades['exit_time'].dt.to_period('M')
    
    capital = 1000.0
    monthly_stats = []
    
    for month, group in df_trades.groupby('month'):
        start_capital = capital
        trades_count = len(group)
        wins = sum(group['raw_pnl'] > 0)
        
        for _, row in group.iterrows():
            pos_usd = capital * optimal_alloc * optimal_lev
            pnl_usd = pos_usd * row['raw_pnl']
            
            max_loss = capital * optimal_alloc
            if pnl_usd < -max_loss: pnl_usd = -max_loss
                
            capital += pnl_usd
            
        monthly_return = (capital - start_capital) / start_capital
        monthly_stats.append({
            'Month': str(month),
            'Trades': trades_count,
            'Win Rate': wins / trades_count if trades_count > 0 else 0,
            'Return %': monthly_return,
            'End Balance': capital
        })
        
    df_monthly = pd.DataFrame(monthly_stats)
    print(f"{'Month':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Return %':<12} | {'End Balance':<15}")
    print("-" * 80)
    
    for _, row in df_monthly.iterrows():
        print(f"{row['Month']:<10} | {row['Trades']:<8} | {row['Win Rate']:<10.2%} | {row['Return %']:<12.2%} | ${row['End Balance']:<14.2f}")

if __name__ == "__main__":
    main()
