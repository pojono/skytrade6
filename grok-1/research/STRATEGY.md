# Momentum Continuation

Long on 3 consecutive up bars (close > open), short on 3 consecutive down bars. Hold 3 bars.

## Rationale
Momentum tends to continue in trends.

## Edge Calculation
- Momentum edge: ~50-100% return (from past backtests).
- FR premium: +20-40bps net (from hold research).
- Combined: Higher Sharpe, lower DD, survivable after 10bps taker fees.
- Round-trip fees: 20bps (taker both legs), aim for >30bps net per trade.

## Data Requirements
- FR history (funding_rate.csv) from Bybit/Binance/OKX.
- 1h/4h klines for signals.
- Symbols: Top 20 common USDT perps (BTC, ETH, SOL, etc.).

## Execution
- Backtest first on historical data (2025-2026 range).
- If profitable, create live script (similar to fr_scalp_scanner.py but with momentum triggers).

## Risks
- FR regimes may not perfectly align with momentum.
- Multi-exchange basis risk (if cross-venue).
- Overfitting on historical data.

## Next Steps
1. Download data for top symbols.
2. Build unified dataset loader.
3. Implement FR regime and momentum signals.
4. Backtest and optimize.
