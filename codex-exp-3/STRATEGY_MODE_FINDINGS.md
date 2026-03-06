# Strategy Mode Comparison

This compares three live-usable variants built on the same candidate set.

Modes:

- `default_soft`: current soft execution-penalty ranking
- `high_conviction`: stricter thresholds before ranking
- `hybrid_regime`: use high-conviction only when breadth is very strong (`breadth_mom >= 0.75`), otherwise use default_soft

High-conviction thresholds:

- `execution_adjusted_score >= 2.909`
- `ls_z >= 2.534`
- `breadth_mom >= 0.65`

## Test Results

| Mode | Rows | Timestamps | Avg bps | Win Rate | Symbols | 2026-01 | 2026-02 |
|---|---:|---:|---:|---:|---:|---:|---:|
| high_conviction | 12 | 11 | 54.93 | 54.5% | 10 | 84.22 | 15.88 |
| hybrid_regime | 22 | 19 | 41.95 | 57.9% | 17 | 53.32 | 77.88 |
| default_soft | 47 | 31 | 24.37 | 61.3% | 28 | 24.81 | 53.03 |
