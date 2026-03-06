# Large Dataset Re-Test (2026-03-06)

## Scope

- Source: `datalake/binance` + `datalake/bybit`
- Rebuilt samples from raw files with existing exp-3 pipeline.
- Focus remained on conservative net returns (PnL in repo treated as optimistic by default).

## Data Coverage After Expansion

- Rebuilt Binance sample: `267,990` rows
- Symbols in rebuilt sample: `142`
- Timestamp range: `2024-01-15 00:05 UTC` to `2026-03-04 16:05 UTC`
- Revalidated tradable entries (Bybit-repriced): `286` rows
- Traded symbols in strategy output: `75`
- Strategy trade timestamp range: `2024-01-19 00:05 UTC` to `2026-03-04 00:05 UTC`

## Core Revalidation (exp-2 rules, cross-venue repricing)

- Files updated:
  - `revalidated_exp2_symbol_trades.csv`
  - `revalidated_exp2_portfolio.csv`
  - `revalidated_exp2_execution_sweep.csv`
  - `revalidated_exp2_funding_sweep.csv`
- Current test window used by pipeline: `2026-01-01+`
- Current test sample size: `31` timestamps
- Test average after 8 bps total costs: `26.77 bps`
- Test win rate after 8 bps: `61.3%`

## Selection Approach Comparison (OOS test subset)

| Approach | Test Avg bps | Win Rate | Timestamps |
|---|---:|---:|---:|
| strict_threshold | 54.93 | 54.5% | 11 |
| empirical_winrate | 24.58 | 58.1% | 31 |
| baseline_soft | 24.37 | 61.3% | 31 |
| blended_rank | 22.70 | 58.1% | 31 |

Interpretation:
- `strict_threshold` has best bps but very small activity.
- `baseline_soft` is the best balance of breadth and stability among higher-activity variants.

## Mode Comparison

Point-in-time test summary (`strategy_modes_compare.py`):

| Mode | Test Avg bps | Win Rate | Timestamps |
|---|---:|---:|---:|
| high_conviction | 54.93 | 54.5% | 11 |
| hybrid_regime | 41.95 | 57.9% | 19 |
| default_soft | 24.37 | 61.3% | 31 |

Rolling walk-forward monthly summary (`rolling_mode_walkforward.py`):

| Mode | Avg Monthly bps | Positive Months | Months |
|---|---:|---:|---:|
| hybrid_regime | 73.27 | 2 | 2 |
| high_conviction | 70.23 | 2 | 2 |
| default_soft | 42.98 | 2 | 2 |

Interpretation:
- On rolling walk-forward, `hybrid_regime` remains strongest.
- `high_conviction` has larger dispersion due to fewer signals.

## Practical Conclusion

- The enlarged dataset supports the same core edge directionally.
- The strategy still survives maker-fee-level friction (`0.04%` per side) in current tests.
- Main limitation remains OOS depth (`2026-01` to `2026-02` materially, with very small `2026-03` so far).

## Next Recommended Step

- Expand older history specifically for currently active high-impact symbols to increase independent OOS months, then rerun identical scripts without changing rules.
