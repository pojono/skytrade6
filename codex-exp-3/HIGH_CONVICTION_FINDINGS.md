# High-Conviction Mode

This is the strict, lower-frequency version of the current signal.

## Rule

- Require `execution_adjusted_score >= 3.018`
- Require `ls_z >= 2.676`
- Require `breadth_mom >= 0.65`
- Rank remaining names by `execution_adjusted_score` and take up to `3` per timestamp

## Aggregate

- Selected symbol rows: 52
- Selected timestamps: 39
- Unique traded symbols: 35

## Train/Test

- Train: 31 timestamps, 97.59 bps, win rate 64.5%
- Test: 8 timestamps, 83.82 bps, win rate 62.5%

## Tougher OOS Slices

- 2026-01: 5 timestamps, 124.58 bps, win rate 80.0%
- 2026-02: 3 timestamps, 15.88 bps, win rate 33.3%

## Monthly Breakdown

| Month | Timestamps | Avg bps | Win Rate |
|---|---:|---:|---:|
| 2026-01 | 5 | 124.58 | 80.0% |
| 2026-02 | 3 | 15.88 | 33.3% |
