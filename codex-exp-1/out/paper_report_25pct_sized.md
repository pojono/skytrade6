# Paper Trading Report

## Configuration

- Input: codex-exp-1/out/candidate_trades_v3.csv
- Starting capital: 100000.00
- Per-trade allocation: 25.00%
- Max open positions: 1
- Max open per symbol: 1
- Max symbol allocation: 25.00%
- Daily cap per symbol: 3
- Selector mode: spread
- Daily loss stop: 1.00%
- Monthly loss stop: 3.00%
- Base fee: 6.00 bps
- Extra slippage: 1.00 bps
- Spread slippage coeff: 0.1000
- Velocity slippage coeff: 0.0500
- Size slippage coeff: 1.5000
- Base allocation ref: 10.00%

## Results

- Filled trades: 1421
- Final capital: 105183.40
- Total PnL: 5183.40
- Average net edge per fill: 1.4242 bps
- Win rate: 50.32%

## Symbol Contribution

| Symbol | Filled Trades | PnL Dollars |
|---|---:|---:|
| CRVUSDT | 618 | 3295.85 |
| GALAUSDT | 458 | 2630.23 |
| SEIUSDT | 345 | -742.68 |

## Monthly

| Month | Filled Trades | Avg Net bps | PnL Dollars | Equity End |
|---|---:|---:|---:|---:|
| 2025-08 | 133 | -0.5765 | -191.62 | 99808.38 |
| 2025-09 | 136 | -1.1893 | -402.95 | 99405.43 |
| 2025-10 | 199 | -1.2043 | -594.42 | 98811.01 |
| 2025-11 | 217 | 1.1290 | 606.39 | 99417.40 |
| 2025-12 | 256 | 1.7440 | 1114.89 | 100532.29 |
| 2026-01 | 221 | 1.6317 | 909.55 | 101441.84 |
| 2026-02 | 246 | 5.5764 | 3536.87 | 104978.71 |
| 2026-03 | 13 | 5.9984 | 204.69 | 105183.40 |
