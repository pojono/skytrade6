import pandas as pd
import numpy as np

# Load trades
df = pd.read_csv('/home/ubuntu/Projects/skytrade6/gemini-2/portfolio_trades_strict.csv')

# Wait, the previous trades were run with SL/TP. Let's just use the df_trades from debug_extreme_edge.py
