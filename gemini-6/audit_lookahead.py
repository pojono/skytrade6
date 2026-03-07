import pandas as pd
import numpy as np

print("--- AUDITING 1-MINUTE LOOKAHEAD BIAS ---")
# 1. How did we calculate 'price' in the features DataFrame?
# If the index is '2025-01-01 10:00:00', does 'price' represent the CLOSE of the 10:00-10:01 minute?
# If so, and we execute our trade at '2025-01-01 10:00:00', we are looking ahead by 1 minute!

with open('calc_cvd_divergence.py', 'r') as f:
    code = f.read()
    
if "resample('1min')" in code or "resample('1Min')" in code:
    print("WARNING: Using pandas resample. Need to check 'label' and 'closed' parameters.")
    # Pandas resample('1min') defaults to label='left', closed='left'.
    # This means ticks from 10:00:00 to 10:00:59.999 are aggregated into the index '10:00:00'.
    # Therefore, the 'price' (last tick) at index '10:00:00' actually occurs at 10:00:59.
    # If we pull the Bybit Orderbook at '10:00:00', we are simulating a fill BEFORE the minute happened!

print("--- AUDITING Z-SCORE LOOKAHEAD BIAS ---")
print("We used df.rolling(window).mean() and std().")
print("Pandas rolling is strictly backward looking (closed='right' by default). It includes the current row and looks backwards.")
print("This is safe, EXCEPT if the current row itself contains data from the future (as per the 1m aggregation above).")

print("--- AUDITING DAILY UNIVERSE SELECTION ---")
# In exp6:
# lookback_start = current_date - timedelta(days=lookback_days)
# mask = (df_metrics.index >= lookback_start) & (df_metrics.index < current_date)
# We used strict '< current_date' so we don't include the current day in the lookback. This is SAFE.

