import pandas as pd
from simulate_portfolio import get_symbol_trades, GOLDEN_CLUSTER

all_trades = []
for sym in GOLDEN_CLUSTER:
    t = get_symbol_trades(sym)
    if t: all_trades.extend(t)

df = pd.DataFrame(all_trades)
df['size_mult'] = 0.01 / df['atr_pct'].fillna(0.05)
df['pnl_pct'] = df['size_mult'] * df['net_ret'] * 100

print(df[['symbol', 'entry_time', 'atr_pct', 'size_mult', 'net_ret', 'pnl_pct']].sort_values('entry_time').head(15))
print(f"\nAvg PnL % per trade: {df['pnl_pct'].mean():.4f}%")
