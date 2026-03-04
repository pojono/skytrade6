# Paper Trading Report

## Configuration

- Input: codex-exp-1/out/candidate_trades_v3.csv
- Starting capital: 100000.00
- Per-trade allocation: 50.00%
- Max open positions: 1
- Max open per symbol: 1
- Max symbol allocation: 50.00%
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

- Filled trades: 1282
- Final capital: 85976.89
- Total PnL: -14023.11
- Average net edge per fill: -2.3537 bps
- Win rate: 34.71%

## Symbol Contribution

| Symbol | Filled Trades | PnL Dollars |
|---|---:|---:|
| GALAUSDT | 414 | -2969.63 |
| CRVUSDT | 559 | -4707.24 |
| SEIUSDT | 309 | -6346.24 |

## Monthly

| Month | Filled Trades | Avg Net bps | PnL Dollars | Equity End |
|---|---:|---:|---:|---:|
| 2025-08 | 133 | -4.3265 | -2836.88 | 97163.12 |
| 2025-09 | 120 | -5.1625 | -2964.32 | 94198.81 |
| 2025-10 | 98 | -6.2225 | -2830.94 | 91367.87 |
| 2025-11 | 217 | -2.6210 | -2564.22 | 88803.65 |
| 2025-12 | 256 | -2.0060 | -2254.52 | 86549.13 |
| 2026-01 | 199 | -3.0597 | -2597.49 | 83951.64 |
| 2026-02 | 246 | 1.8264 | 1900.19 | 85851.83 |
| 2026-03 | 13 | 2.2484 | 125.06 | 85976.89 |
