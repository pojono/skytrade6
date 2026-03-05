# Selection Approach Comparison

This compares several ways to choose among the same base signal candidates.

Target metric:

- 4h cross-venue return
- minus 8 bps maker fees
- minus the symbol-level average entry-drag estimate

## Test Results

| Approach | Rows | Timestamps | Avg bps | Win Rate | Symbols |
|---|---:|---:|---:|---:|---:|
| strict_threshold | 9 | 8 | 83.82 | 62.5% | 7 |
| baseline_soft | 42 | 27 | 37.39 | 59.3% | 25 |
| blended_rank | 42 | 27 | 35.47 | 55.6% | 25 |
| empirical_winrate | 31 | 27 | 25.77 | 59.3% | 20 |
