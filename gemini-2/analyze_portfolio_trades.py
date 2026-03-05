import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load trades
df = pd.read_csv('/home/ubuntu/Projects/skytrade6/gemini-2/portfolio_trades.csv')
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time'] = pd.to_datetime(df['exit_time'])

print(f"Total Trades Taken: {len(df)}")
print(f"Date Range: {df['entry_time'].min()} to {df['entry_time'].max()}")

# Unique coins
coins = df['symbol'].unique()
print(f"\nNumber of unique coins traded: {len(coins)}")
print(f"Top 10 most traded coins:")
print(df['symbol'].value_counts().head(10).to_string())

# Monthly breakdown
df['month'] = df['entry_time'].dt.to_period('M')

# Assuming $2000 fixed size per trade as used in the portfolio sim
POS_SIZE = 2000
df['pnl_usd'] = df['net_ret'] * POS_SIZE

monthly = df.groupby('month').agg(
    trades=('pnl_usd', 'count'),
    win_rate=('net_ret', lambda x: (x > 0).mean() * 100),
    net_pnl_usd=('pnl_usd', 'sum')
).reset_index()

print("\n--- Monthly Breakdown ---")
print(monthly.to_string(index=False))

# Equity Curve
INITIAL_CAPITAL = 10000

# To properly plot the equity curve, we need a chronological series of events.
events = []
for idx, row in df.iterrows():
    events.append({'time': row['exit_time'], 'pnl': row['pnl_usd']})

df_events = pd.DataFrame(events).sort_values('time')
df_events['equity'] = INITIAL_CAPITAL + df_events['pnl'].cumsum()

print("\n--- Equity Summary ---")
final_equity = df_events['equity'].iloc[-1] if not df_events.empty else INITIAL_CAPITAL
print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
print(f"Final Capital: ${final_equity:,.2f}")
print(f"Total Return: {((final_equity / INITIAL_CAPITAL) - 1) * 100:.2f}%")

if not df_events.empty:
    peak = df_events['equity'].cummax()
    drawdown = (peak - df_events['equity']) / peak
    max_dd = drawdown.max() * 100
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Sharpe Ratio (Trade-based approx): {df['net_ret'].mean() / df['net_ret'].std() * np.sqrt(len(df)):.2f}")

