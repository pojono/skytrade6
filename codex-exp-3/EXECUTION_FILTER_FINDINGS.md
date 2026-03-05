# Execution Filter Findings

This note folds the observed symbol-level 60-second entry drift back into the strategy and compares a filtered basket.

## Rule

- Per-symbol entry drag = average of positive 60-second VWAP drift on Binance and Bybit.
- Blacklist any symbol with Bybit maker-fill rate < 100% or average positive drag >= 8.0 bps.

## Test Comparison (after 8 bps maker fees)

- Baseline funding-adjusted: 39.29 bps on 27 timestamps, win rate 59.3%
- With entry drag applied: 36.67 bps on 27 timestamps, win rate 59.3%
- Blacklist only: 39.10 bps on 23 timestamps, win rate 60.9%
- Blacklist plus drag: 38.22 bps on 23 timestamps, win rate 60.9%

## Blacklisted Symbols

| Symbol | Avg Positive Drag | Bybit Fill Rate |
|---|---:|---:|
| LINKUSDT | 12.81 bps | 100% |
| ENAUSDT | 12.64 bps | 100% |
| XLMUSDT | 8.69 bps | 100% |
| AVAXUSDT | 8.33 bps | 0% |
| XRPUSDT | 4.06 bps | 50% |
| PAXGUSDT | 0.53 bps | 0% |
| BARDUSDT | 0.00 bps | 0% |

Files:

- `execution_adjusted_symbol_trades.csv`
- `execution_adjusted_comparison.csv`