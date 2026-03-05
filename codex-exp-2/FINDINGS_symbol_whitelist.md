# Symbol Whitelist Classifier

This is a second-stage symbol filter built only from each symbol's train-period behavior under the base strategy.

## Base Strategy

- `ls_z >= 2.0`
- `taker_z >= 0.5`
- `oi_med_3d >= $20M`
- `breadth_mom >= 0.60`
- `median_ls_z >= 0.0`
- `top_n = 3` per timestamp before whitelist

## Best Whitelist Rule

- `min_train_trades >= 2`
- `min_train_avg_bps >= 10.0`
- `min_train_hit_rate >= 0.40`
- `train_worst_bps >= -9999.0`
- Whitelist size: 15
- Symbols with test support: 6

## Holdout Impact

- Base strategy holdout avg: 26.95 bps
- Base strategy holdout hit rate: 55.6%
- Whitelisted holdout avg: -11.94 bps
- Whitelisted holdout hit rate: 57.1%
- Whitelisted holdout trade rows: 8
- Whitelisted holdout decisions: 7

## Rank-Based Symbol Classifier

Symbols are ranked by: `train_avg_bps * train_hit_rate * sqrt(train_trades)` using train data only.

- Best aggressive top-K: `K=3`
- Aggressive holdout avg: 117.62 bps
- Aggressive holdout decisions: 1
- Best robust top-K with at least 3 holdout decisions: `K=9`
- Robust holdout avg: 2.80 bps
- Robust holdout decisions: 3

## Interpretation

- The threshold whitelist does not improve holdout; simple static thresholds are not enough.
- The rank-based classifier can improve holdout only by collapsing to a tiny, sparse list.
- If the only improvement comes from 1-3 decisions, the result is not reliable enough for deployment.
- Symbol selection is clearly a major lever, but the current train-only classifier is still too weak.

## Whitelisted Symbols

| Symbol | Train Trades | Train Avg | Train Hit | Test Trades | Test Avg |
|---|---:|---:|---:|---:|---:|
| NEARUSDT | 4 | 283.79 | 75.0% | 0 | nan |
| HBARUSDT | 2 | 152.90 | 100.0% | 0 | nan |
| TIAUSDT | 5 | 129.22 | 60.0% | 1 | 117.62 |
| POLUSDT | 2 | 87.50 | 100.0% | 0 | nan |
| FILUSDT | 2 | 70.78 | 100.0% | 0 | nan |
| SOLUSDT | 2 | 67.21 | 50.0% | 2 | -54.61 |
| ETCUSDT | 4 | 64.86 | 75.0% | 0 | nan |
| PENGUUSDT | 4 | 64.37 | 75.0% | 0 | nan |
| AAVEUSDT | 2 | 56.19 | 50.0% | 0 | nan |
| KAITOUSDT | 3 | 46.19 | 66.7% | 0 | nan |
| ENAUSDT | 2 | 38.85 | 50.0% | 1 | -55.54 |
| OPUSDT | 2 | 37.84 | 50.0% | 0 | nan |
| WLDUSDT | 2 | 29.61 | 50.0% | 2 | 8.94 |
| ETHUSDT | 4 | 20.52 | 75.0% | 1 | -67.64 |
| AVAXUSDT | 7 | 10.05 | 71.4% | 1 | -28.90 |
