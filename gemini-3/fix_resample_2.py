import pandas as pd
import numpy as np

times = pd.date_range('2025-01-01 07:00:00', '2025-01-01 08:05:00', freq='1min')
df = pd.DataFrame({'close': np.arange(len(times))}, index=times)

print("Raw 1m Data near 08:00:")
print(df.loc['2025-01-01 07:58:00':'2025-01-01 08:02:00'])

# Default resample: label='left', closed='left'
resampled1 = df.resample('1h').agg({'close': 'last'})
print("\nDefault Resample (label=left, closed=left):")
print(resampled1.loc['2025-01-01 07:00:00':'2025-01-01 08:00:00'])

# Safe resample: label='right', closed='right'
# This means the 08:00:00 row will contain data from 07:01:00 up to and including 08:00:00.
# BUT wait! We want the signal at 08:00:00 to use data UP TO 07:59:00, because the 07:59:00 candle closes EXACTLY at 08:00:00.000.
# The 08:00:00 candle doesn't close until 08:01:00.000.
# Binance 'open_time' is the START of the minute. So '07:59:00' represents 07:59:00 to 07:59:59.999.

resampled2 = df.resample('1h', label='right', closed='left').agg({'close': 'last'})
print("\nSafe Resample (label=right, closed=left):")
print(resampled2.loc['2025-01-01 07:00:00':'2025-01-01 08:00:00'])

