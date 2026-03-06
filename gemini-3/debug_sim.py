import pandas as pd
from simulate_portfolio import get_symbol_trades

trades = get_symbol_trades('BTCUSDT')
df = pd.DataFrame(trades)
print(df[['entry_time', 'signal', 'atr_pct', 'net_ret']].head(10))
