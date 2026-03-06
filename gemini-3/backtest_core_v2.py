import pandas as pd
import numpy as np

def run_backtest_advanced(df, max_hold_periods, trailing_stop_bps, take_profit_bps, fee_bps=5.0, use_volatility_sizing=False):
    """
    Advanced vectorized backtest engine.
    df must have:
      - 'close': the execution price
      - 'signal': 1 (Long), -1 (Short), 0 (Neutral)
      - 'atr_pct' (Optional): if use_volatility_sizing=True, this represents Average True Range as a %.
    """
    if 'signal' not in df.columns or 'close' not in df.columns:
        return {'events': 0}
        
    trades = df[df['signal'] != 0].copy()
    if len(trades) == 0:
        return {'events': 0}
        
    df = df.reset_index()
    time_col = df.columns[0]
    trades = trades.reset_index()
    
    results = []
    
    # Target risk per trade is 1% of portfolio. 
    # If a coin moves 5% in a day (ATR), we size down. If it moves 1%, we size up.
    TARGET_RISK = 0.01 
    
    for i, row in trades.iterrows():
        entry_idx = df[df[time_col] == row[time_col]].index[0]
        entry_price = row['close']
        signal = row['signal']
        
        # Sizing
        size_multiplier = 1.0
        if use_volatility_sizing and 'atr_pct' in row and row['atr_pct'] > 0:
            # If ATR is 0.05 (5%), multiplier = 0.01 / 0.05 = 0.2x
            size_multiplier = TARGET_RISK / row['atr_pct']
            # Cap maximum leverage per trade to 3x
            size_multiplier = min(size_multiplier, 3.0)
            
        # Look forward up to max_hold_periods
        end_idx = min(entry_idx + max_hold_periods, len(df) - 1)
        if end_idx <= entry_idx: continue
        
        path = df.iloc[entry_idx+1:end_idx+1]['close'].values
        
        best_price = entry_price
        exit_price = path[-1] # Default to time stop
        
        for px in path:
            ret_bps = ((px - entry_price) / entry_price) * signal * 10000
            
            # Update best price for trailing stop
            if signal == 1 and px > best_price: best_price = px
            if signal == -1 and px < best_price: best_price = px
            
            drawdown_from_peak_bps = ((px - best_price) / best_price) * signal * 10000
            
            # Hit Take Profit
            if ret_bps >= take_profit_bps:
                exit_price = px
                break
                
            # Hit Trailing Stop
            if drawdown_from_peak_bps <= -trailing_stop_bps:
                exit_price = px
                break
                
        gross_ret = (exit_price - entry_price) / entry_price * signal
        net_ret = (gross_ret - (fee_bps * 2 / 10000)) * size_multiplier
        
        results.append(net_ret)
        
    if not results:
        return {'events': 0}
        
    res_series = pd.Series(results)
    
    return {
        'events': len(res_series),
        'win_rate': (res_series > 0).mean(),
        'total_net_ret_%': res_series.sum() * 100,
        'avg_net_ret_bps': res_series.mean() * 10000,
        'max_drawdown_%': (res_series.cumsum() - res_series.cumsum().cummax()).min() * 100,
        'sharpe': np.sqrt(len(res_series)) * (res_series.mean() / res_series.std()) if res_series.std() > 0 else 0
    }
