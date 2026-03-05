# Trade Entry Findings: Full Available 2026 Test Basket

This extends the earlier top-3 trade-level check to all 2026 test-period strategy entries that currently have downloaded trade files on both Binance and Bybit.

## Aggregate

- Signal rows checked: `42`
- Symbols covered: `26`
- Binance maker-fill proxy within 60s: `100.0%`
- Bybit maker-fill proxy within 60s: `90.5%`
- Binance first-trade drift: `+0.14 bps`
- Bybit first-trade drift: `-0.47 bps`
- Binance 60s VWAP drift: `-1.39 bps`
- Bybit 60s VWAP drift: `-1.59 bps`

## Highest VWAP Drift Names

| Symbol | Signals | Binance Fill | Bybit Fill | Binance VWAP | Bybit VWAP |
|---|---:|---:|---:|---:|---:|
| ENAUSDT | 1 | 100% | 100% | +11.09 bps | +14.18 bps |
| LINKUSDT | 4 | 100% | 100% | +10.64 bps | +14.98 bps |
| XLMUSDT | 1 | 100% | 100% | +10.34 bps | +7.03 bps |
| AVAXUSDT | 1 | 100% | 0% | +8.24 bps | +8.41 bps |
| SOLUSDT | 2 | 100% | 100% | +7.67 bps | +4.94 bps |
| ARBUSDT | 1 | 100% | 100% | +3.99 bps | +7.21 bps |
| XRPUSDT | 2 | 100% | 50% | +3.72 bps | +4.39 bps |
| TRUMPUSDT | 1 | 100% | 100% | +3.18 bps | -1.84 bps |
| 1000PEPEUSDT | 1 | 100% | 100% | +1.84 bps | +1.04 bps |
| WIFUSDT | 5 | 100% | 100% | +1.07 bps | -1.08 bps |

Files:

- `trade_entry_feasibility_full.csv`
- `trade_entry_feasibility_summary_full.csv`