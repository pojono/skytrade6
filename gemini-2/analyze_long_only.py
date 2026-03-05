import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load trades
df = pd.read_csv('/home/ubuntu/Projects/skytrade6/gemini-2/portfolio_trades.csv')
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time'] = pd.to_datetime(df['exit_time'])

# Filter out the terrible short trades
df_long = df[df['direction'] == 'long'].copy()

print(f"Total Long Trades: {len(df_long)}")

# Monthly breakdown
df_long['month'] = df_long['entry_time'].dt.to_period('M')
POS_SIZE = 2000
df_long['pnl_usd'] = df_long['net_ret'] * POS_SIZE

monthly = df_long.groupby('month').agg(
    trades=('pnl_usd', 'count'),
    win_rate=('net_ret', lambda x: (x > 0).mean() * 100),
    net_pnl_usd=('pnl_usd', 'sum')
).reset_index()

print("\n--- Monthly Breakdown (Long Only) ---")
print(monthly.to_string(index=False))

# Equity Curve
INITIAL_CAPITAL = 10000

events = []
for idx, row in df_long.iterrows():
    events.append({'time': row['exit_time'], 'pnl': row['pnl_usd']})

df_events = pd.DataFrame(events).sort_values('time')
df_events['equity'] = INITIAL_CAPITAL + df_events['pnl'].cumsum()

print("\n--- Equity Summary (Long Only) ---")
final_equity = df_events['equity'].iloc[-1] if not df_events.empty else INITIAL_CAPITAL
print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
print(f"Final Capital: ${final_equity:,.2f}")
print(f"Total Return: {((final_equity / INITIAL_CAPITAL) - 1) * 100:.2f}%")

if not df_events.empty:
    peak = df_events['equity'].cummax()
    drawdown = (peak - df_events['equity']) / peak
    max_dd = drawdown.max() * 100
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Sharpe Ratio (Trade-based approx): {df_long['net_ret'].mean() / df_long['net_ret'].std() * np.sqrt(len(df_long)):.2f}")

plt.figure(figsize=(12, 6))
plt.plot(df_events['time'], df_events['equity'], label='Equity', color='green')
plt.axhline(y=INITIAL_CAPITAL, color='r', linestyle='--', label='Initial Capital')
plt.title('Extreme Liquidation Flush - Long Only Equity Curve')
plt.xlabel('Date')
plt.ylabel('Account Value ($)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('/home/ubuntu/Projects/skytrade6/gemini-2/equity_curve_long_only.png')
print("Saved equity_curve_long_only.png")
