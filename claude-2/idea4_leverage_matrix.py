import pandas as pd
import numpy as np

# In Idea 4 (BTC Pump), the existing audit showed that removing Oct 2025 kills the edge.
# We need to simulate the portfolio for Idea 4 using daily net returns.
# But idea4_audit_raw doesn't have the individual trade times, it has aggregated rows.
# To do a real leverage matrix, we need individual trade dates and returns.
# Let's extract them quickly based on the logic in PRE_PRODUCTION_AUDIT.md.
import sys, os
from pathlib import Path

BYBIT = Path("/home/ubuntu/Projects/skytrade6/datalake/bybit")
BINANCE = Path("/home/ubuntu/Projects/skytrade6/datalake/binance")

def load_btc():
    f = BYBIT / "BTCUSDT/combined_kline_1m.csv"
    if not f.exists(): return None
    df = pd.read_csv(f, usecols=['startTime','close']).dropna()
    df['datetime'] = pd.to_datetime(pd.to_numeric(df['startTime']), unit='ms')
    df = df.set_index('datetime').sort_index()
    return df['close']

print("Loading BTC...")
btc = load_btc()
btc_3m = btc.pct_change(3)

# Filter out Oct 2025 to see the "honest" view, but we can do the matrix on the full data and mark Oct 2025.
# Let's do the matrix on the full data since that's what was traded, but we will print both.

# Actually, the user asked to do a monthly breakdown on profits based on allocated % capital per trade and different leverages.
# This means we need the actual trades.
# The PRE_PRODUCTION_AUDIT.md mentions "150 bps in 3 minutes".
# Let's find those events.
sig_times = btc_3m[btc_3m > 0.015].index

# Decluster 30 mins
clean_times = []
last_t = pd.Timestamp('2000-01-01')
for t in sig_times:
    if (t - last_t).total_seconds() >= 1800:
        clean_times.append(t)
        last_t = t

print(f"Found {len(clean_times)} signal events.")

# We don't have the alt data loaded, so we can just approximate the edge from the audit.
# The audit says: 
# 2024: 15 days, +213 bps avg
# 2025 (no oct): 12 days, -3 bps avg
# 2026: 5 days, -7 bps avg
# Oct 2025: 2 days, +6026 bps

# If we don't have the actual alt coin paths, we can't do an exact portfolio sim.
# Let me re-run a quick extraction just for top 5 alts to get real trades.

