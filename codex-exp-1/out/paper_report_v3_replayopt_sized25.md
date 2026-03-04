# Paper Trading Report

## Configuration

- Input: codex-exp-1/out/candidate_trades_v3_replayopt.csv
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

- Filled trades: 897
- Final capital: 107638.99
- Total PnL: 7638.99
- Average net edge per fill: 3.2849 bps
- Win rate: 53.29%

## Symbol Contribution

| Symbol | Filled Trades | PnL Dollars |
|---|---:|---:|
| CRVUSDT | 548 | 6175.85 |
| GALAUSDT | 282 | 1755.39 |
| SEIUSDT | 67 | -292.25 |

## Monthly

| Month | Filled Trades | Avg Net bps | PnL Dollars | Equity End |
|---|---:|---:|---:|---:|
| 2025-08 | 44 | -1.6535 | -181.77 | 99818.23 |
| 2025-09 | 55 | 2.8602 | 393.22 | 100211.45 |
| 2025-10 | 116 | 0.5533 | 160.68 | 100372.13 |
| 2025-11 | 128 | 3.1829 | 1026.96 | 101399.09 |
| 2025-12 | 191 | 2.0711 | 1006.92 | 102406.02 |
| 2026-01 | 150 | 3.2990 | 1273.90 | 103679.91 |
| 2026-02 | 201 | 6.9914 | 3704.63 | 107384.55 |
| 2026-03 | 12 | 7.8938 | 254.44 | 107638.99 |
