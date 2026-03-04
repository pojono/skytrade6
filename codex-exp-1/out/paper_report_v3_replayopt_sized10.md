# Paper Trading Report

## Configuration

- Input: codex-exp-1/out/candidate_trades_v3_replayopt.csv
- Starting capital: 100000.00
- Per-trade allocation: 10.00%
- Max open positions: 1
- Max open per symbol: 1
- Max symbol allocation: 10.00%
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
- Final capital: 105089.18
- Total PnL: 5089.18
- Average net edge per fill: 5.5349 bps
- Win rate: 61.54%

## Symbol Contribution

| Symbol | Filled Trades | PnL Dollars |
|---|---:|---:|
| CRVUSDT | 548 | 3704.05 |
| GALAUSDT | 282 | 1345.53 |
| SEIUSDT | 67 | 39.59 |

## Monthly

| Month | Filled Trades | Avg Net bps | PnL Dollars | Equity End |
|---|---:|---:|---:|---:|
| 2025-08 | 44 | 0.5965 | 26.24 | 100026.24 |
| 2025-09 | 55 | 5.1102 | 281.51 | 100307.75 |
| 2025-10 | 116 | 2.8033 | 326.67 | 100634.42 |
| 2025-11 | 128 | 5.4329 | 702.15 | 101336.58 |
| 2025-12 | 191 | 4.3211 | 839.67 | 102176.24 |
| 2026-01 | 150 | 5.5490 | 853.86 | 103030.11 |
| 2026-02 | 201 | 9.2414 | 1931.26 | 104961.36 |
| 2026-03 | 12 | 10.1438 | 127.81 | 105089.18 |
