# Strict Fill Replay

- Variant: current best microstructure-gated 30d sleeve (accepted fills only)
- Bybit fills: exact simulation against archived L2 order book snapshots
- Binance fills: approximate simulation against 1% cumulative `bookDepth` snapshots
- Each leg uses half of the trade allocation as quote notional

## Coverage

- Replayed fills: 193
- Fully fillable under this strict replay: 193
- Fill success rate: 100.00%

## PnL Comparison On Fillable Trades

- Modeled avg net: 7.9684 bps
- Strict-fill avg net: -1.6466 bps
- Modeled total PnL: $3917.39
- Strict-fill total PnL: $-805.32
- Strict-fill win rate: 43.52%

## Strict Execution Cost

- Avg strict execution slippage: 14.4799 bps
- Median strict execution slippage: 14.2627 bps

## Notes

- This is stricter than the generic modeled slippage because it requires both legs to be fillable from archived depth snapshots.
- Bybit is modeled more faithfully than Binance because only Bybit has true L2 snapshots here.
- Binance remains an approximation using 1% cumulative depth buckets, so this is not yet a perfect venue-accurate simulator.