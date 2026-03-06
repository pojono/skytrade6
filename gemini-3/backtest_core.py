import pandas as pd
import numpy as np

def run_backtest(df, hold_periods, fee_bps=5.0):
    """
    Generic vectorized backtest engine.
    df must have:
      - 'close': the execution price
      - 'signal': 1 (Long), -1 (Short), 0 (Neutral)
    hold_periods: integer number of periods to hold the trade before closing.
    fee_bps: Trading fee in basis points per side (entry and exit).
    """
    if 'signal' not in df.columns or 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'signal' and 'close' columns.")
        
    # We only care about rows where a signal is generated
    trades = df[df['signal'] != 0].copy()
    if len(trades) == 0:
        return {'events': 0}
        
    # Calculate exit price by looking forward 'hold_periods' rows in the original dataframe
    # We use index matching to find the exact exit row
    
    # Create a positional mapping for fast lookup
    df = df.reset_index()
    # The time column name depends on what it was, usually 'timestamp'
    time_col = df.columns[0]
    
    trades = trades.reset_index()
    
    # We need to find the index of the exit row.
    # We will do this by shifting the original dataframe's close price backward by hold_periods.
    df['exit_price'] = df['close'].shift(-hold_periods)
    df['exit_time'] = df[time_col].shift(-hold_periods)
    
    # Now merge this back into trades
    trades = trades.merge(df[[time_col, 'exit_price', 'exit_time']], on=time_col, how='left')
    
    # Drop trades that don't have an exit (end of dataset)
    trades = trades.dropna(subset=['exit_price'])
    
    if len(trades) == 0:
        return {'events': 0}
        
    # Calculate Gross Return
    trades['gross_ret'] = (trades['exit_price'] - trades['close']) / trades['close']
    trades['trade_ret'] = trades['gross_ret'] * trades['signal']
    
    # Apply Fees (Entry + Exit = 2 * fee_bps)
    total_fee_pct = (fee_bps * 2) / 10000
    trades['net_ret'] = trades['trade_ret'] - total_fee_pct
    
    # Calculate Equity Curve (simple cumulative sum of returns, assuming 1x leverage, non-compounding)
    trades['equity'] = trades['net_ret'].cumsum()
    
    # Metrics
    events = len(trades)
    win_rate = (trades['net_ret'] > 0).mean()
    total_net_ret_pct = trades['net_ret'].sum() * 100
    avg_net_ret_bps = trades['net_ret'].mean() * 10000
    
    # Max Drawdown
    cum_max = trades['equity'].cummax()
    drawdown = trades['equity'] - cum_max
    max_drawdown_pct = drawdown.min() * 100
    
    # Sharpe Ratio (Annualized, assuming periods depends on frequency, we'll just do raw Sharpe of trades)
    # This is "Sharpe per trade"
    trade_std = trades['net_ret'].std()
    sharpe = np.sqrt(events) * (trades['net_ret'].mean() / trade_std) if trade_std > 0 else 0
    
    return {
        'events': events,
        'win_rate': win_rate,
        'total_net_ret_%': total_net_ret_pct,
        'avg_net_ret_bps': avg_net_ret_bps,
        'max_drawdown_%': max_drawdown_pct,
        'sharpe': sharpe
    }
