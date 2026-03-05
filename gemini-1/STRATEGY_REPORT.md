# Strategy: Confirmed Capitulation Bounce

## Hypothesis
High fees (0.04% maker / 0.1% taker) destroy mean-reversion strategies that target small moves. To survive, we need gross moves of at least 10-15%. These moves reliably occur after extreme liquidation cascades (Open Interest flushes).

## Rules
- **Signal**: 8-hour Price Drop > 12% AND 8-hour Open Interest Drop > 15%.
- **Confirmation**: Wait for a 1-hour bounce > 3% to avoid catching a falling knife.
- **Execution**: Taker entry at market. Limit TP at +18%. Stop market SL at -12%.
- **Time Stop**: 48 hours.

## Performance (Out of Sample, Last 180 Days)
- Total Trades: 188
- Win Rate: 52.66%
- Average Net PnL: 3.0999%
- Total Net PnL (1x Leverage): 582.7816%
- Daily Sharpe: 0.81
