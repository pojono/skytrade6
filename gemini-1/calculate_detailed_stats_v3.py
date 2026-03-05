import pandas as pd
import numpy as np
import os

trades_file = "/home/ubuntu/Projects/skytrade6/gemini-1/strategy_capitulation_bounce.csv"
df = pd.read_csv(trades_file)
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time'] = pd.to_datetime(df['exit_time'])
df = df.sort_values('exit_time').reset_index(drop=True)

# Important adjustment: we need to group simultaneous trades on the same day as 1 portfolio event
# to avoid skewing the average trade size and win rate.
df['trade_date'] = df['entry_time'].dt.date
portfolio_daily = df.groupby('trade_date').agg(
    concurrent_trades=('symbol', 'count'),
    net_pnl=('pnl', 'sum'),
    avg_pnl=('pnl', 'mean')
)

total_days_in_market = len(portfolio_daily)
win_days = portfolio_daily[portfolio_daily['net_pnl'] > 0]
loss_days = portfolio_daily[portfolio_daily['net_pnl'] <= 0]

print("="*50)
print("      PORTFOLIO-ADJUSTED STATISTICS")
print("="*50)
print(f"Total Market Event Days: {total_days_in_market}")
print(f"Portfolio Win Rate (by day): {len(win_days)/total_days_in_market:.2%}")
print(f"Average PnL per Event Day: {portfolio_daily['net_pnl'].mean():.2%}")
print(f"Max PnL on a single day: {portfolio_daily['net_pnl'].max():.2%}")
print(f"Max Loss on a single day: {portfolio_daily['net_pnl'].min():.2%}")

# Let's see the distribution of coins traded per day
print("\nDistribution of Concurrent Trades per Day:")
print(portfolio_daily['concurrent_trades'].value_counts().sort_index())
