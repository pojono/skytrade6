import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/ubuntu/Projects/skytrade6/gemini-2/portfolio_trades.csv')
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time'] = pd.to_datetime(df['exit_time'])

INITIAL_CAPITAL = 10000
POS_SIZE = 2000

events = []
for idx, row in df.iterrows():
    events.append({'time': row['exit_time'], 'pnl': row['net_ret'] * POS_SIZE})

df_events = pd.DataFrame(events).sort_values('time')
df_events['equity'] = INITIAL_CAPITAL + df_events['pnl'].cumsum()

plt.figure(figsize=(12, 6))
plt.plot(df_events['time'], df_events['equity'], label='Equity', color='blue')
plt.axhline(y=INITIAL_CAPITAL, color='r', linestyle='--', label='Initial Capital')
plt.title('Extreme Liquidation Flush - Equity Curve (Jan 2024 - Mar 2026)')
plt.xlabel('Date')
plt.ylabel('Account Value ($)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('/home/ubuntu/Projects/skytrade6/gemini-2/equity_curve.png')
print("Saved equity_curve.png")
