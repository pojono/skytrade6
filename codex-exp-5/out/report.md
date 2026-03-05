# Extreme Spread CRV Report

## Strategy

- Symbol: CRVUSDT
- Entry idea: fade the 1-minute Binance vs Bybit close spread when it is extremely stretched and the supporting positioning/basis signal points in the same direction.
- Exit: hold for exactly one aligned minute bar, then close on the next synchronized close.
- Direction: short the rich exchange, long the cheap exchange.

## Parameters

- Recent common days scanned: 210
- Spread threshold: 32.00 bps
- Min long/short diff: 0.15
- Min OI diff: 5.00 bps
- Min carry diff: 2.00 bps
- Min score: 14.00
- Daily cap: 3
- Maker round-trip fee: 8.00 bps
- Taker round-trip fee: 20.00 bps
- Train months: 2025-11, 2025-12, 2026-01
- Test months: 2026-02, 2026-03

## Aggregate

- Total trades: 194
- Train trades: 104
- Test trades: 90
- Train avg net after maker fees: 14.0451 bps
- Test avg net after maker fees: 21.8842 bps
- Train avg net after taker fees: 2.0451 bps
- Test avg net after taker fees: 9.8842 bps
- Test win rate after taker fees: 62.22%

## Monthly

| Month | Trades | Avg Net Maker bps | Avg Net Taker bps |
|---|---:|---:|---:|
| 2025-11 | 8 | 14.4757 | 2.4757 |
| 2025-12 | 47 | 12.5682 | 0.5682 |
| 2026-01 | 49 | 15.3913 | 3.3913 |
| 2026-02 | 84 | 20.4391 | 8.4391 |
| 2026-03 | 6 | 42.1158 | 30.1158 |
