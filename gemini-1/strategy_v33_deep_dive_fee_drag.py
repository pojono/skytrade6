import pandas as pd

# The paradox: Win rate is 96.36%, but total PnL is negative.
# We need to mathematically prove why this is happening.
# Let's recreate a few trades explicitly.

MAKER_FEE = 0.0004
TAKER_FEE = 0.0010

# Config 2 logic
ENTRY_OFFSET = 0.0015
TP_PCT = 0.0012

# WINNING TRADE: Limit Entry, Limit Exit
# Gross move = +0.12%
# Fees = 0.04% + 0.04% = 0.08%
win_net = ENTRY_OFFSET + TP_PCT - (MAKER_FEE * 2)
# Actually, the net PnL formulation in the script was:
# net_pnl = ((target_price - entry_price) / entry_price) - (MAKER_FEE * 2)
# target_price = entry_price * (1 + 0.0012)
# So gross pnl = 0.0012
# net_pnl = 0.0012 - 0.0008 = +0.0004 (+0.04%)

print(f"Net PnL of a Winning Trade: {0.0012 - (MAKER_FEE * 2):.4%}")

# LOSING TRADE: Time Stop (60 min)
# Let's say it moves against us by 1% in 60 minutes
gross_loss = -0.0100
loss_net = gross_loss - MAKER_FEE - TAKER_FEE
print(f"Net PnL of a 1% Losing Trade: {loss_net:.4%}")

# What is the average loss of the 3.64% of trades that lose?
# If the strategy executed 6953 trades, and lost 253 of them.
win_pnl = 6700 * 0.0004
loss_pnl = -1.6704 - win_pnl
avg_loss = loss_pnl / 253

print(f"Calculated average loss required to bleed account: {avg_loss:.4%}")
print("\nConclusion:")
print("Config 2 is mechanically flawless in win rate (96%+).")
print("HOWEVER, the take profit (+0.04% net) is so small that a single catastrophic tail-risk time stop (-10% to -20% drop during a cascade) completely wipes out 250+ winning trades.")
print("The edge relies on the underlying dataset artificially capping max drawdown per trade, OR we must implement a catestrophic stop loss.")

