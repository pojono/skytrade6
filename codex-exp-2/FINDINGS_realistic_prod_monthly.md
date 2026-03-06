# Realistic Production Monthly PnL

- Strategy filters: frozen production config (`ls>=2.0`, `taker>=0.5`, `oi>=20M`, `breadth>=0.60`, `top_n=5`).
- Portfolio: allocation=1.00, leverage=3.0x, initial equity=$1000.
- Execution model per symbol-trade: depth-based slippage (entry+exit), spread, maker/taker fee regime.
- Fee schedule: maker/maker=8.0 bps RT, mixed=14.0 bps RT, taker/taker=20.0 bps RT.
- Fallback cost for missing/stale depth: 35.0 bps RT.

## Coverage
- Symbol-trades modeled: 264
- Timestamp decisions modeled: 155
- Execution mode mix (%): {'fallback_no_liquidity': 66.3, 'fallback_no_depth': 24.6, 'maker_maker': 9.1}

## Result
- Final equity: $1,273.85
- Total return: 27.4%

## Files
- `realistic_prod_symbol_trades.csv`
- `realistic_prod_timestamp_returns.csv`
- `realistic_prod_monthly_breakdown.csv`