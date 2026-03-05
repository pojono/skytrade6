# Strategy Mode Comparison

This compares three live-usable variants built on the same candidate set.

Modes:

- `default_soft`: current soft execution-penalty ranking
- `high_conviction`: stricter thresholds before ranking
- `hybrid_regime`: use high-conviction only when breadth is very strong (`breadth_mom >= 0.75`), otherwise use default_soft

High-conviction thresholds:

- `execution_adjusted_score >= 3.018`
- `ls_z >= 2.676`
- `breadth_mom >= 0.65`

## Test Results

| Mode | Rows | Timestamps | Avg bps | Win Rate | Symbols | 2026-01 | 2026-02 |
|---|---:|---:|---:|---:|---:|---:|---:|
| high_conviction | 9 | 8 | 83.82 | 62.5% | 7 | 124.58 | 15.88 |
| hybrid_regime | 15 | 13 | 71.49 | 61.5% | 12 | 68.66 | 77.88 |
| default_soft | 42 | 27 | 37.39 | 59.3% | 25 | 32.93 | 53.03 |
