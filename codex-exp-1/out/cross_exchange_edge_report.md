# Cross-Exchange Edge Report

## Configuration

- Minimum overlap days: 90
- Minimum entry spread: 10.00 bps
- Fee assumption: 6.00 bps round trip
- Out-of-sample window: last 15 days
- Worker slots: 4
- Recent-day cap: 45

## Portfolio-Level Summary

- Symbols analyzed: 111
- Total signals: 1715476
- Total avg net PnL per signal: -3.1715 bps
- Train signals: 1144388
- Train avg net PnL per signal: -3.2170 bps
- Test signals: 571088
- Test avg net PnL per signal: -3.0803 bps
- Test-profitable symbols: 42/111

## Top 15 Symbols By Test Avg Net PnL

| Symbol | Test Signals | Test Avg Net bps | Test Hit Rate | Total Avg Net bps |
|---|---:|---:|---:|---:|
| CRVUSDT | 16201 | 12.6417 | 55.76% | 11.5936 |
| PAXGUSDT | 1 | 12.0997 | 100.00% | -1.5554 |
| ZECUSDT | 3 | 9.1217 | 100.00% | -3.2890 |
| GALAUSDT | 12768 | 7.5367 | 58.94% | 6.6483 |
| SOLUSDT | 1 | 7.4130 | 100.00% | -1.9938 |
| ETHUSDT | 1 | 6.8446 | 100.00% | 17.7733 |
| BCHUSDT | 1 | 6.4399 | 100.00% | 6.8533 |
| PENGUUSDT | 1 | 4.8643 | 100.00% | 6.0488 |
| BTCUSDT | 2 | 4.6288 | 100.00% | 4.6288 |
| HYPEUSDT | 6 | 4.4594 | 83.33% | -2.9339 |
| TONUSDT | 5 | 4.3746 | 80.00% | 3.4199 |
| TRUMPUSDT | 12 | 3.7939 | 75.00% | 1.5599 |
| XLMUSDT | 1 | 3.7384 | 100.00% | 17.5384 |
| TRXUSDT | 1 | 3.4279 | 100.00% | -5.4074 |
| 1000RATSUSDT | 6250 | 3.4278 | 63.09% | -0.7823 |

## Notes

- This uses synchronized 1-minute closes and a one-bar holding period.
- Streaming row-merge keeps memory bounded per file pair.
- Strong train results with weak test results should be treated as noise, not edge.
