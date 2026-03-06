import pandas as pd
import numpy as np
from full_universe_scan_fidelity import analyze_symbol
import matplotlib.pyplot as plt

# We want to check if the Z-Score > 2.0 is curve-fitted.
# What happens if we relax the parameter to Z > 1.5, or tighten to Z > 2.5?

def analyze_parameter_sensitivity(symbol, z_threshold):
    from full_universe_scan_fidelity import load_data
    from backtest_core_high_fidelity import run_high_fidelity_backtest
    
    hourly, m1_df = load_data(symbol)
    if hourly is None or m1_df is None or len(hourly) < 500: return None
    
    hourly['oi_z'] = (hourly['oi_usd'] - hourly['oi_usd'].rolling(168).mean()) / hourly['oi_usd'].rolling(168).std()
    hourly['fr_z'] = (hourly['funding_rate'] - hourly['funding_rate'].rolling(168).mean()) / hourly['funding_rate'].rolling(168).std()
    
    hourly['signal'] = 0
    hourly.loc[(hourly['oi_z'] > z_threshold) & (hourly['fr_z'] > z_threshold), 'signal'] = -1
    hourly.loc[(hourly['oi_z'] > z_threshold) & (hourly['fr_z'] < -z_threshold), 'signal'] = 1
    
    hourly['signal_shifted'] = hourly['signal'].shift(1).fillna(0)
    hourly.loc[hourly['signal'] == hourly['signal_shifted'], 'signal'] = 0
    
    results = run_high_fidelity_backtest(
        hourly, 
        m1_df, 
        max_hold_hours=24, 
        trailing_stop_bps=10000.0, 
        take_profit_bps=1000.0, 
        fee_bps=5.0, 
        use_volatility_sizing=True
    )
    
    if results['events'] > 0:
        return results['sharpe'], results['events'], results['win_rate']
    return None, 0, 0

print("--- Overfitting Sensitivity Analysis (BTCUSDT) ---")
print("Z-Score | Events | Win Rate | Sharpe")
for z in [1.0, 1.5, 2.0, 2.5, 3.0]:
    res = analyze_parameter_sensitivity("BTCUSDT", z)
    if res and res[1] > 0:
        print(f"Z > {z:.1f} |   {res[1]:4d} |   {res[2]*100:.1f}%   | {res[0]:.2f}")

