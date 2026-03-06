import pandas as pd
import numpy as np

# Create dummy 1m data
times = pd.date_range('2025-01-01 07:00:00', '2025-01-01 08:05:00', freq='1min')
df = pd.DataFrame({'close': np.random.randn(len(times))}, index=times)

# Let's see what resample('1h').agg('last') actually does
resampled = df.resample('1h').agg({'close': 'last'})

print("Raw 1m Data near 08:00:")
print(df.loc['2025-01-01 07:58:00':'2025-01-01 08:02:00'])

print("\nResampled Data:")
print(resampled.loc['2025-01-01 07:00:00':'2025-01-01 08:00:00'])

# To avoid lookahead, an 08:00:00 signal should ONLY use data up to 07:59:00 (since 08:00:00 hasn't fully closed yet if we execute at 08:00:00, or we execute at 08:01:00).
# Actually, if the signal is generated at 08:00:00, the last known close is 07:59:00 (which closed at 08:00:00 in Binance standard).
