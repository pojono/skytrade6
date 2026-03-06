import pandas as pd
import numpy as np

def run_high_fidelity_backtest(signals_df, m1_df, max_hold_hours, trailing_stop_bps, take_profit_bps, fee_bps=5.0, use_volatility_sizing=False):
    """
    High-fidelity vectorized backtest engine.
    signals_df: Hourly dataframe containing 'signal', 'close' (entry price), and 'atr_pct'
    m1_df: 1-minute dataframe containing 'high', 'low', 'close' for intra-hour high-resolution execution
    max_hold_hours: Max hours to hold trade before time stop.
    """
    if 'signal' not in signals_df.columns or len(signals_df[signals_df['signal'] != 0]) == 0:
        return {'events': 0}
        
    trades = signals_df[signals_df['signal'] != 0].copy()
    
    # We will iterate over each trade and use the 1-minute data for exact execution path
    results = []
    
    TARGET_RISK = 0.01 
    
    for entry_ts, row in trades.iterrows():
        entry_price = row['close']
        signal = row['signal']
        
        # Sizing
        size_multiplier = 1.0
        if use_volatility_sizing and 'atr_pct' in row and row['atr_pct'] > 0:
            size_multiplier = TARGET_RISK / row['atr_pct']
            size_multiplier = min(size_multiplier, 3.0)
            
        # Define the execution window in 1-minute data
        start_time = entry_ts + pd.Timedelta(hours=1) # The signal is generated at the end of the hourly candle
        end_time = start_time + pd.Timedelta(hours=max_hold_hours)
        
        # Get the 1-minute path
        path = m1_df.loc[start_time:end_time]
        if len(path) == 0:
            continue
            
        best_price = entry_price
        exit_price = path.iloc[-1]['close'] # Default to time stop
        
        for ts, m1_row in path.iterrows():
            high = m1_row['high']
            low = m1_row['low']
            
            # Update best price for trailing stop
            if signal == 1:
                if high > best_price: best_price = high
                
                # Check Take Profit (using high for long)
                if ((high - entry_price) / entry_price) * 10000 >= take_profit_bps:
                    exit_price = entry_price * (1 + take_profit_bps / 10000)
                    break
                    
                # Check Trailing Stop (using low for long)
                drawdown_bps = ((low - best_price) / best_price) * 10000
                if drawdown_bps <= -trailing_stop_bps:
                    exit_price = best_price * (1 - trailing_stop_bps / 10000)
                    break
                    
            elif signal == -1:
                if low < best_price: best_price = low
                
                # Check Take Profit (using low for short)
                if ((entry_price - low) / entry_price) * 10000 >= take_profit_bps:
                    exit_price = entry_price * (1 - take_profit_bps / 10000)
                    break
                    
                # Check Trailing Stop (using high for short)
                drawdown_bps = ((best_price - high) / best_price) * 10000
                if drawdown_bps <= -trailing_stop_bps:
                    exit_price = best_price * (1 + trailing_stop_bps / 10000)
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
