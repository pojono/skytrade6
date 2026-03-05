import pandas as pd
import numpy as np

trades_file = "/home/ubuntu/Projects/skytrade6/gemini-1/strategy_capitulation_bounce.csv"
df = pd.read_csv(trades_file)
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time'] = pd.to_datetime(df['exit_time'])

october_trades = df[(df['exit_time'] >= '2025-10-01') & (df['exit_time'] < '2025-11-01')]
print("October 2025 Trades Breakdown:")
print(f"Total October Trades: {len(october_trades)}")
print("\nTrades grouped by Day:")
print(october_trades.groupby(october_trades['entry_time'].dt.date).size())

