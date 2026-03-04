# Candidate Robustness Report

## Configuration

- Symbols: CRVUSDT, GALAUSDT, SEIUSDT, FILUSDT
- Recent days: 180
- Test days: 60
- Spread threshold: 10.00 bps
- Fee: 6.00 bps
- Filter: ls>=0.15, oi>=5.00, carry>=2.00

## Aggregate

- Train avg net: 5.4335 bps on 18783 signals
- Test avg net: 7.0751 bps on 18403 signals
- Positive months: 7/7
- Top symbol share of total net PnL: 74.77%

## Monthly

| Month | Signals | Avg Net bps |
|---|---:|---:|
| 2025-09 | 1471 | 3.0690 |
| 2025-10 | 3854 | 5.7525 |
| 2025-11 | 5561 | 5.9332 |
| 2025-12 | 7735 | 5.2723 |
| 2026-01 | 6771 | 5.1661 |
| 2026-02 | 10997 | 8.1208 |
| 2026-03 | 797 | 9.4317 |

## Symbol Contribution

| Symbol | Test Signals | Test Avg Net bps | Total Net Sum |
|---|---:|---:|---:|
| CRVUSDT | 9224 | 10.1878 | 173655.22 |
| GALAUSDT | 5117 | 6.3246 | 46960.77 |
| FILUSDT | 2079 | 0.5492 | 8064.66 |
| SEIUSDT | 1983 | 1.3749 | 3581.29 |

## Extra Slippage Sweep

| Extra Slippage bps | Train Avg Net | Test Avg Net |
|---|---:|---:|
| 0.00 | 5.4335 | 7.0751 |
| 1.00 | 4.4335 | 6.0751 |
| 2.00 | 3.4335 | 5.0751 |
| 3.00 | 2.4335 | 4.0751 |
| 4.00 | 1.4335 | 3.0751 |

## Leave-One-Out

| Excluded Symbol | Train Avg Net | Test Avg Net |
|---|---:|---:|
| CRVUSDT | 4.1660 | 3.9472 |
| GALAUSDT | 5.9606 | 7.3642 |
| SEIUSDT | 5.5762 | 7.7636 |
| FILUSDT | 5.2399 | 7.9063 |

## Drop Top Two Contributors

- Excluded: CRVUSDT, GALAUSDT
- Train avg net: 6.1679 bps on 1261 signals
- Test avg net: 0.9523 bps on 4062 signals
