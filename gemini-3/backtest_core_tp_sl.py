import pandas as pd
import numpy as np

def run_backtest_tp_sl(df, tp_pct, sl_pct, fee_bps=5.0):
    if 'signal' not in df.columns or 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'signal' and 'close' columns.")
        
    trades = df[df['signal'] != 0].copy()
    if len(trades) == 0:
        return {'events': 0}
        
    df = df.reset_index()
    time_col = df.columns[0]
    trades = trades.reset_index()
    
    trade_results = []
    
    for idx, row in trades.iterrows():
        entry_time = row[time_col]
        entry_price = row['close']
        signal = row['signal']
        
        # Look forward
        post_entry = df[df[time_col] > entry_time]
        
        if post_entry.empty:
            continue
            
        if signal == 1:
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
            
            # Find first hit
            hit_tp = post_entry[post_entry['high'] >= tp_price]
            hit_sl = post_entry[post_entry['low'] <= sl_price]
            
        elif signal == -1:
            tp_price = entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 + sl_pct)
            
            hit_tp = post_entry[post_entry['low'] <= tp_price]
            hit_sl = post_entry[post_entry['high'] >= sl_price]
            
        tp_idx = hit_tp.index[0] if not hit_tp.empty else float('inf')
        sl_idx = hit_sl.index[0] if not hit_sl.empty else float('inf')
        
        if tp_idx == float('inf') and sl_idx == float('inf'):
            # Close at end of data
            exit_price = post_entry.iloc[-1]['close']
        elif tp_idx < sl_idx:
            exit_price = tp_price
        else:
            exit_price = sl_price
            
        gross_ret = (exit_price - entry_price) / entry_price * signal
        net_ret = gross_ret - (fee_bps * 2 / 10000)
        
        trade_results.append({
            'net_ret': net_ret
        })
        
    if not trade_results:
        return {'events': 0}
        
    res_df = pd.DataFrame(trade_results)
    res_df['equity'] = res_df['net_ret'].cumsum()
    
    events = len(res_df)
    win_rate = (res_df['net_ret'] > 0).mean()
    total_net_ret_pct = res_df['net_ret'].sum() * 100
    avg_net_ret_bps = res_df['net_ret'].mean() * 10000
    
    cum_max = res_df['equity'].cummax()
    drawdown = res_df['equity'] - cum_max
    max_drawdown_pct = drawdown.min() * 100
    
    trade_std = res_df['net_ret'].std()
    sharpe = np.sqrt(events) * (res_df['net_ret'].mean() / trade_std) if trade_std > 0 else 0
    
    return {
        'events': events,
        'win_rate': win_rate,
        'total_net_ret_%': total_net_ret_pct,
        'avg_net_ret_bps': avg_net_ret_bps,
        'max_drawdown_%': max_drawdown_pct,
        'sharpe': sharpe
    }
