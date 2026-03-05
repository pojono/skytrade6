import pandas as pd

df = pd.read_csv('/home/ubuntu/Projects/skytrade6/gemini-2/flush_trades.csv')
print("--- All Trades ---")
print(df['net_ret'].describe())
print(f"Total Net PnL (unleveraged sum): {df['net_ret'].sum()*100:.2f}%")

print("\n--- Trades Grouped by Month ---")
df['month'] = pd.to_datetime(df['entry_time']).dt.to_period('M')
print(df.groupby('month')['net_ret'].agg(['count', 'mean', 'sum', lambda x: (x>0).mean()]).rename(columns={'<lambda_0>': 'win_rate'}))

