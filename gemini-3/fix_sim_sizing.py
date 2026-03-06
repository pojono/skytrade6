import pandas as pd
from simulate_portfolio import get_symbol_trades, GOLDEN_CLUSTER

all_trades = []
for sym in GOLDEN_CLUSTER:
    t = get_symbol_trades(sym)
    if t: all_trades.extend(t)

df = pd.DataFrame(all_trades)

# The ATR is currently a percentage (e.g. 0.05 for 5%)
# Our sizing logic was: size_multiplier = 0.01 / 0.05 = 0.2x leverage
# Wait, if a coin moves 5% in a day, we want a 5% move to equal 1% of equity.
# If we use 0.2x leverage, a 5% move = 0.2 * 5% = 1% move. This is correct.

# BUT if net_ret is 0.10 (10% TP), then 10% move * 0.2x leverage = 2% profit.
# The calculation in my simulation was: PnL = equity * size_multiplier * net_ret
# PnL = 1000 * 0.2 * 0.10 = $20 (which is +2% on $1000)

# Let's check the ATR values. Ah! 
# In `simulate_portfolio.py` I calculated: 
# atr_absolute = tr.mean() * 60 * 24 
# hourly['atr_pct'] = (hourly['tr'].rolling(24).mean() * 24) / hourly['close']
# Wait, `tr` is ALREADY the absolute True Range.
# The sum of 1-minute TRs over 24 hours is NOT the 24-hour TR.
# The sum of 1-minute TRs over 24 hours is a massive number (it counts every intra-minute wiggle).
# For example, BTC might move $50 every minute. Summing that over 1440 minutes = $72,000 ATR!
# So ATR % was calculated as like 300% instead of 5%. 
# Therefore size_multiplier was microscopic!

print(df[['symbol', 'entry_time', 'atr_pct']].head(10))
