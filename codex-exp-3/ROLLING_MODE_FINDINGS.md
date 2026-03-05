# Rolling Walk-Forward Mode Comparison

Monthly walk-forward setup:

- For each test month, thresholds are fit only on data before that month.
- Comparison modes: `default_soft`, `high_conviction`, `hybrid_regime`.
- Metric is `net_ret_after_costs` from candidate dataset (already fee + entry-drag adjusted), aggregated as equal-weight portfolio per timestamp.

## Summary

| Mode | Months | Timestamps | Avg Monthly bps | Median bps | Std bps | Min bps | Positive Months | Avg Win Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid_regime | 2 | 13 | 73.27 | 73.27 | 6.52 | 68.66 | 2 | 58.3% |
| high_conviction | 2 | 8 | 70.23 | 70.23 | 76.87 | 15.88 | 2 | 56.7% |
| default_soft | 2 | 27 | 42.98 | 42.98 | 14.21 | 32.93 | 2 | 56.0% |

## Monthly Detail

| Month | Mode | Timestamps | Avg bps | Win Rate |
|---|---|---:|---:|---:|
| 2026-01 | default_soft | 21 | 32.93 | 61.9% |
| 2026-01 | high_conviction | 5 | 124.58 | 80.0% |
| 2026-01 | hybrid_regime | 9 | 68.66 | 66.7% |
| 2026-02 | default_soft | 6 | 53.03 | 50.0% |
| 2026-02 | high_conviction | 3 | 15.88 | 33.3% |
| 2026-02 | hybrid_regime | 4 | 77.88 | 50.0% |
