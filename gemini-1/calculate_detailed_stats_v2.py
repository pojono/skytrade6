import pandas as pd
import numpy as np

trades_file = "/home/ubuntu/Projects/skytrade6/gemini-1/strategy_capitulation_bounce.csv"
df = pd.read_csv(trades_file)
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time'] = pd.to_datetime(df['exit_time'])
df = df.sort_values('exit_time').reset_index(drop=True)

# Monthly Breakdown Analysis to understand the 944% outlier
df_time = df.set_index('exit_time')
monthly_group = df_time.resample('1ME').agg(
    trades=('pnl', 'count'),
    win_rate=('pnl', lambda x: (x > 0).mean() if len(x) > 0 else 0),
    total_pnl=('pnl', 'sum'),
    max_trade=('pnl', 'max'),
    min_trade=('pnl', 'min')
)

print(monthly_group)
