import pandas as pd
import numpy as np

# Load the previously validated OOS trades
df_trades = pd.read_csv('/home/ubuntu/Projects/skytrade6/claude-2/out/idea4_audit_raw.csv')

# The user is asking for the "BTC Pump -> Long Alts" strategy from Idea 4 based on the open file,
# not the liquidation flush strategy from gemini-2!
# I need to use the correct trades for Idea 4.
