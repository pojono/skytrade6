import pandas as pd
import numpy as np

# We need to deeply audit `strategy_final_capitulation_bounce.py` for lookahead bias.
# The critical merge is:
# df_oi.resample('15min').ffill()
# df = df.join(df_oi[['openInterest']], how='left').ffill()
#
# If 'openInterest' for 00:15:00 represents the state AT 00:15:00, joining it to the 
# 00:00:00 - 00:15:00 kline (which has a timestamp of 00:00:00 usually, but in our resample, what is it?)

# Let's write a quick script to see EXACTLY what happens during the pandas resample and merge.

df_kline = pd.DataFrame({
    'timestamp': pd.to_datetime(['2025-01-01 00:00:00', '2025-01-01 00:01:00', '2025-01-01 00:14:00', '2025-01-01 00:15:00']),
    'close': [100, 101, 105, 110]
}).set_index('timestamp')

df_oi = pd.DataFrame({
    'timestamp': pd.to_datetime(['2025-01-01 00:00:00', '2025-01-01 00:05:00', '2025-01-01 00:10:00', '2025-01-01 00:15:00']),
    'openInterest': [1000, 1050, 1100, 2000] # Massive spike at 00:15
}).set_index('timestamp')

# Imitate script exactly
df_kline_resampled = df_kline.resample('15min').agg({'close': 'last'})
df_oi_resampled = df_oi.resample('15min').ffill()
merged = df_kline_resampled.join(df_oi_resampled[['openInterest']], how='left').ffill()

print("Klines Resampled (Label is the START of the period):")
print(df_kline_resampled)

print("\nOI Resampled:")
print(df_oi_resampled)

print("\nMerged (Is there lookahead?):")
print(merged)
