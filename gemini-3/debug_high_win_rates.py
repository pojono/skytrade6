import pandas as pd
from full_universe_scan_fidelity import analyze_symbol, load_data
from backtest_core_high_fidelity import run_high_fidelity_backtest
import warnings

warnings.filterwarnings('ignore')

def get_detailed_trades(symbol):
    hourly, m1_df = load_data(symbol)
    if hourly is None or m1_df is None or len(hourly) < 500: return
    
    hourly['oi_z'] = (hourly['oi_usd'] - hourly['oi_usd'].rolling(168).mean()) / hourly['oi_usd'].rolling(168).std()
    hourly['fr_z'] = (hourly['funding_rate'] - hourly['funding_rate'].rolling(168).mean()) / hourly['funding_rate'].rolling(168).std()
    
    hourly['signal'] = 0
    hourly.loc[(hourly['oi_z'] > 2.0) & (hourly['fr_z'] > 2.0), 'signal'] = -1
    hourly.loc[(hourly['oi_z'] > 2.0) & (hourly['fr_z'] < -2.0), 'signal'] = 1
    
    hourly['signal_shifted'] = hourly['signal'].shift(1).fillna(0)
    hourly.loc[hourly['signal'] == hourly['signal_shifted'], 'signal'] = 0
    
    trades = hourly[hourly['signal'] != 0].copy()
    
    print(f"\n--- Detailed Trades for {symbol} ---")
    print(f"Total Signals Fired: {len(trades)}")
    
    win_count = 0
    
    for entry_ts, row in trades.iterrows():
        entry_price = row['close']
        signal = row['signal']
        
        start_time = entry_ts + pd.Timedelta(hours=1)
        end_time = start_time + pd.Timedelta(hours=24)
        
        path = m1_df.loc[start_time:end_time]
        if len(path) == 0:
            continue
            
        exit_price = path.iloc[-1]['close'] 
        exit_reason = "Time Stop (24h)"
        best_price = entry_price
        
        for ts, m1_row in path.iterrows():
            high = m1_row['high']
            low = m1_row['low']
            
            if signal == 1:
                if high > best_price: best_price = high
                if ((high - entry_price) / entry_price) * 10000 >= 1000.0:
                    exit_price = entry_price * 1.10
                    exit_reason = "Take Profit (+10%)"
                    break
            elif signal == -1:
                if low < best_price: best_price = low
                if ((entry_price - low) / entry_price) * 10000 >= 1000.0:
                    exit_price = entry_price * 0.90
                    exit_reason = "Take Profit (+10%)"
                    break
                    
        gross_ret = (exit_price - entry_price) / entry_price * signal
        net_ret = gross_ret - (10.0 / 10000) # 5bps x 2
        
        is_win = net_ret > 0
        if is_win: win_count += 1
        
        side = "LONG" if signal == 1 else "SHORT"
        result_str = "WIN" if is_win else "LOSS"
        
        print(f"{entry_ts} | {side} | Entry: {entry_price:.4f} | Exit: {exit_price:.4f} | Net Ret: {net_ret*100:.2f}% | Reason: {exit_reason} | {result_str}")
        
    print(f"Win Rate: {win_count / len(trades) * 100:.1f}%")

for sym in ['BTCUSDT', 'SOLUSDT', 'LINKUSDT']:
    get_detailed_trades(sym)

