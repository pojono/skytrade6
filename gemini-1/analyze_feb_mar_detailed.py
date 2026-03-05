import pandas as pd
import numpy as np

# Load the V42 results to see why trades failed in Feb/Mar
df_trades = pd.read_csv("/home/ubuntu/Projects/skytrade6/gemini-1/strategy_v42_final_paradigm.py") # Wait, I didn't save trades to CSV in v42.
# Let's write a quick script to generate and save the trades so we can analyze them.
