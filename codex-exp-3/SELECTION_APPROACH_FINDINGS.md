# Selection Approach Comparison

This compares several ways to choose among the same base signal candidates.

Target metric:

- 4h cross-venue return
- minus 8 bps maker fees
- minus the symbol-level average entry-drag estimate

## Test Results

| Approach | Rows | Timestamps | Avg bps | Win Rate | Symbols |
|---|---:|---:|---:|---:|---:|
| strict_threshold | 12 | 11 | 54.93 | 54.5% | 10 |
| empirical_winrate | 38 | 31 | 24.58 | 58.1% | 26 |
| baseline_soft | 47 | 31 | 24.37 | 61.3% | 28 |
| blended_rank | 47 | 31 | 22.70 | 58.1% | 28 |
