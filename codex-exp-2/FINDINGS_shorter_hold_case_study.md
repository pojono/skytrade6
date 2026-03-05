# Shorter Hold Case Study

This tests whether the covered-universe signal behaves better with 30m, 60m, or 90m exits instead of the original 4-hour hold.
For each cohort and hold, the delayed-confirmation rule is chosen on the train split only, then applied to the test split.

## covered_all

- `30m` hold: base test `-21.99 bps`, delayed test `0.52 bps on 13 rows`, improvement `22.51 bps`
- `60m` hold: base test `-22.53 bps`, delayed test `23.80 bps on 12 rows`, improvement `46.33 bps`
- `90m` hold: base test `-46.21 bps`, delayed test `-20.14 bps on 12 rows`, improvement `26.07 bps`

## robust_plus_sol

- `30m` hold: base test `-21.32 bps`, delayed test `19.17 bps on 3 rows`, improvement `40.49 bps`
- `60m` hold: base test `-19.07 bps`, delayed test `47.16 bps on 3 rows`, improvement `66.23 bps`
- `90m` hold: base test `17.53 bps`, delayed test `87.50 bps on 3 rows`, improvement `69.97 bps`

## total_positive

- `30m` hold: base test `-29.04 bps`, delayed test `-25.24 bps on 8 rows`, improvement `3.80 bps`
- `60m` hold: base test `-25.95 bps`, delayed test `11.78 bps on 6 rows`, improvement `37.73 bps`
- `90m` hold: base test `-36.99 bps`, delayed test `-19.95 bps on 6 rows`, improvement `17.03 bps`

## train_positive_2plus

- `30m` hold: base test `-22.51 bps`, delayed test `19.18 bps on 7 rows`, improvement `41.70 bps`
- `60m` hold: base test `-21.83 bps`, delayed test `43.95 bps on 7 rows`, improvement `65.77 bps`
- `90m` hold: base test `-21.01 bps`, delayed test `54.32 bps on 7 rows`, improvement `75.33 bps`

## Bottom Line

- Best absolute delayed result is `robust_plus_sol` with a `90m` hold: 87.50 bps on 3 rows.
- Best result with at least 5 delayed test rows is `train_positive_2plus` with a `90m` hold: 54.32 bps on 7 rows.