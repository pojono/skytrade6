import pandas as pd

df = pd.read_csv("/home/ubuntu/Projects/skytrade6/gemini-1/V42_TRADES.csv")
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time'] = pd.to_datetime(df['exit_time'])
df['month'] = df['exit_time'].dt.to_period('M')

feb_mar = df[df['month'].astype(str).isin(['2026-02', '2026-03'])]
oct_jan = df[df['month'].astype(str).isin(['2025-10', '2025-11', '2025-12', '2026-01'])]

print("=== OCT-JAN (The Boom Phase) ===")
print(oct_jan.groupby('dir').agg(
    trades=('pnl', 'count'),
    win_rate=('pnl', lambda x: (x>0).mean())
))

print("\n=== FEB-MAR (The Chop Phase) ===")
print(feb_mar.groupby('dir').agg(
    trades=('pnl', 'count'),
    win_rate=('pnl', lambda x: (x>0).mean())
))

print("\n=== FEB-MAR Exit Types ===")
print(feb_mar['type'].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))

# Let's see the average time held for losers vs winners in Feb/Mar
print("\n=== Average Hold Time (Feb/Mar) ===")
feb_mar['hold_hours'] = (feb_mar['exit_time'] - feb_mar['entry_time']).dt.total_seconds() / 3600
print(feb_mar.groupby('type')['hold_hours'].mean())

