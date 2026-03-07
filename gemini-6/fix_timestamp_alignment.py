import pandas as pd

# The issue is exactly here:
# df['price'].resample('1min').ohlc()
# Pandas defaults to label='left', closed='left'.
# This means ticks from 10:00:00 to 10:00:59 are assigned to '10:00:00'.
# The 'close' price at index '10:00:00' is the price at '10:00:59'.
# If our signal triggers at index '10:00:00' and we pull the Bybit Orderbook at '10:00:00',
# we are pulling the orderbook 59 seconds BEFORE the final tick that confirmed the signal!
# THIS IS A 59-SECOND LOOKAHEAD BIAS.

print("Lookahead Bias Found: 59 Seconds.")
print("A signal timestamped '10:00:00' actually finalized at '10:00:59'.")
print("We pulled the ob200 snapshot at '10:00:00'.")
print("To fix this, we must shift the execution time to '10:01:00' (the start of the NEXT minute).")
