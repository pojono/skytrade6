import pandas as pd
import numpy as np

trades_file = "/home/ubuntu/Projects/skytrade6/gemini-1/strategy_capitulation_bounce.csv"
df = pd.read_csv(trades_file)
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time'] = pd.to_datetime(df['exit_time'])
df = df.sort_values('exit_time').reset_index(drop=True)

# Because of the October 10th anomaly (where 93 coins triggered simultaneously), 
# raw statistics overstate the daily win rate. 
# We need to simulate realistic portfolio allocation: 
# e.g., allocating a max of 2% account equity per trade.
ALLOCATION_PER_TRADE = 0.05 # 5% max position sizing

df['equity_pnl'] = df['pnl'] * ALLOCATION_PER_TRADE
df['cumulative_equity'] = 1.0 + df['equity_pnl'].cumsum()
df['peak_equity'] = df['cumulative_equity'].cummax()
df['drawdown'] = (df['cumulative_equity'] - df['peak_equity']) / df['peak_equity']
max_dd = df['drawdown'].min()

# Daily Returns Series (Equity)
df_time = df.set_index('exit_time')
daily_equity_returns = df_time.resample('1D')['equity_pnl'].sum()
monthly_equity_returns = df_time.resample('1ME')['equity_pnl'].sum()

print("="*50)
print("      PORTFOLIO STATISTICS (5% ALLOCATION / TRADE)")
print("="*50)
print(f"Initial Equity:    1.000")
print(f"Final Equity:      {df['cumulative_equity'].iloc[-1]:.3f} ({(df['cumulative_equity'].iloc[-1]-1):.2%} Return)")
print(f"Max Drawdown:      {max_dd:.2%}")
print(f"Return/MaxDD:      {(df['cumulative_equity'].iloc[-1]-1) / abs(max_dd):.2f}")

print("\n--- Trade Level Stats ---")
print(f"Total Trades: {len(df)}")
print(f"Win Rate: {(df['pnl'] > 0).mean():.2%}")
print(f"Avg Winner (Gross Move): {df[df['pnl'] > 0]['pnl'].mean():.2%}")
print(f"Avg Loser (Gross Move):  {df[df['pnl'] <= 0]['pnl'].mean():.2%}")
print(f"Profit Factor: {abs(df[df['pnl'] > 0]['pnl'].sum() / df[df['pnl'] <= 0]['pnl'].sum()):.2f}")

print("\n--- Monthly Returns (Equity %) ---")
for month, ret in monthly_equity_returns.items():
    print(f"{month.strftime('%Y-%b')}: {ret:.2%}")

print("="*50)
