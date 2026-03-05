import pandas as pd
import numpy as np

# Wait, the sensitivity test just returned Negative PnL for the 5-day hold that was previously massively positive.
# Let's verify why. 
# In the original v38, I used:
# df['fwd_funding_5d'] = df['fundingRate'].shift(-1).rolling(5).sum()
# In pandas, if you shift(-1), the row at index `i` gets the value of `i+1`.
# Then rolling(5).sum() takes the values from `i-3` to `i+1`.
#
# BUT, if I wanted forward returns over the NEXT 5 days, I should have used:
# df['fundingRate'].rolling(5).sum().shift(-5)
# which takes `i+1` to `i+5`.
#
# Let's prove that the original v38 was accidentally peeking backwards!

df = pd.DataFrame({
    'idx': [0, 1, 2, 3, 4, 5, 6],
    'fund': [10, 20, 30, 40, 50, 60, 70]
})

df['v38_buggy'] = df['fund'].shift(-1).rolling(5).sum()
df['v38_correct'] = df['fund'].rolling(5).sum().shift(-5)

print(df)
