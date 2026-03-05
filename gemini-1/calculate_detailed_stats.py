import pandas as pd
import numpy as np
import os

# Load trades from our finalized strategy
trades_file = "/home/ubuntu/Projects/skytrade6/gemini-1/strategy_capitulation_bounce.csv"
if not os.path.exists(trades_file):
    print("Trades file not found. Ensure the final strategy has been run.")
    exit(1)

df = pd.read_csv(trades_file)
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time'] = pd.to_datetime(df['exit_time'])

# Sort chronologically by exit time for cumulative PnL calculation
df = df.sort_values('exit_time').reset_index(drop=True)

# 1. Total Trades & Time Range
total_trades = len(df)
first_trade = df['entry_time'].min()
last_trade = df['entry_time'].max()
total_days = (last_trade - first_trade).days
if total_days == 0: total_days = 1

# 2. Trade Frequencies
trades_per_day = total_trades / total_days
trades_per_week = trades_per_day * 7
trades_per_month = trades_per_day * 30.44

# 3. Win/Loss Stats
winners = df[df['pnl'] > 0]
losers = df[df['pnl'] <= 0]
win_rate = len(winners) / total_trades if total_trades > 0 else 0

avg_winner = winners['pnl'].mean() if not winners.empty else 0
avg_loser = losers['pnl'].mean() if not losers.empty else 0
max_winner = df['pnl'].max() if not df.empty else 0
max_loser = df['pnl'].min() if not df.empty else 0
profit_factor = abs(avg_winner / avg_loser) if avg_loser != 0 else np.inf

# 4. Cumulative PnL and Max Drawdown (assuming 1x leverage, non-compounded)
# We treat PnL as absolute % change to equity for simplicity of standard DD calculation
df['cumulative_pnl'] = df['pnl'].cumsum()
df['peak'] = df['cumulative_pnl'].cummax()
df['drawdown'] = df['cumulative_pnl'] - df['peak']
max_dd = df['drawdown'].min()

# 5. Returns over Time
# Re-index by exit time to compute calendar returns
df_time = df.set_index('exit_time')
daily_returns = df_time.resample('1D')['pnl'].sum()
weekly_returns = df_time.resample('1W')['pnl'].sum()
monthly_returns = df_time.resample('1ME')['pnl'].sum()

print("\n" + "="*50)
print("      DETAILED STRATEGY STATISTICS")
print("="*50)
print(f"Date Range: {first_trade.date()} to {last_trade.date()} ({total_days} days)")
print(f"Total Trades: {total_trades}")
print(f"\n--- Trade Frequency ---")
print(f"Avg Trades per Day:   {trades_per_day:.2f}")
print(f"Avg Trades per Week:  {trades_per_week:.2f}")
print(f"Avg Trades per Month: {trades_per_month:.2f}")

print(f"\n--- Performance Metrics ---")
print(f"Win Rate:           {win_rate:.2%}")
print(f"Average Winner:     {avg_winner:.2%}")
print(f"Average Loser:      {avg_loser:.2%}")
print(f"Max Winner:         {max_winner:.2%}")
print(f"Max Loser:          {max_loser:.2%}")
print(f"Profit Factor:      {profit_factor:.2f}")

print(f"\n--- Risk Metrics (1x Leverage) ---")
print(f"Total Cumulative PnL: {df['cumulative_pnl'].iloc[-1]:.2%}")
print(f"Maximum Drawdown:     {max_dd:.2%}")
return_over_dd = df['cumulative_pnl'].iloc[-1] / abs(max_dd) if max_dd != 0 else np.inf
print(f"Return / Max DD:      {return_over_dd:.2f}")

print(f"\n--- Monthly Returns ---")
for month, ret in monthly_returns.items():
    print(f"{month.strftime('%Y-%b')}: {ret:.2%}")

print("="*50)
