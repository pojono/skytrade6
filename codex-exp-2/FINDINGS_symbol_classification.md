# Symbol Concentration And Classification

This report evaluates whether the currently best fee-aware strategy is broad or concentrated by symbol.

## Configuration Tested

- `ls_z >= 2.0`
- `taker_z >= 0.5`
- `oi_med_3d >= $20M`
- `breadth_mom >= 0.60`
- `median_ls_z >= 0.0`
- `top_n = 3` per timestamp
- fee assumption: `20 bps` round-trip all taker

## Aggregate

- Selected trade rows: 174
- Unique symbols selected: 63
- Net symbol contribution total: 1513.19 bps
- Positive contribution pool: 7452.04 bps
- Negative contribution pool: 5938.85 bps
- Top 3 symbols = 163.5% of final net and 33.2% of gross positive contribution
- Top 5 symbols = 218.9% of final net and 44.4% of gross positive contribution
- Positive-contributor HHI: 0.0608

## Interpretation

- Concentration is real. A few winners contribute more than the final net because many other symbols give back gains.
- This means the current edge is not broad enough to trust blindly across all triggered names.
- Symbol selection quality matters almost as much as the base signal.

## Class Counts

- `robust_positive`: 2
- `mixed_or_sparse_positive`: 38
- `negative`: 23

## Robust Positive Symbols

These had at least 4 total trades, at least 2 train trades, at least 1 test trade, and were positive in both train and test.

| Symbol | Trades | Train | Test | Train Avg | Test Avg | Total |
|---|---:|---:|---:|---:|---:|---:|
| TIAUSDT | 6 | 5 | 1 | 129.22 | 117.62 | 763.70 |
| WLDUSDT | 4 | 2 | 2 | 29.61 | 8.94 | 77.10 |

## Top Positive Contributors

| Symbol | Trades | Avg | Total | Class |
|---|---:|---:|---:|---|
| NEARUSDT | 4 | 283.79 | 1135.15 | mixed_or_sparse_positive |
| TIAUSDT | 6 | 127.28 | 763.70 | robust_positive |
| PIPPINUSDT | 1 | 575.59 | 575.59 | mixed_or_sparse_positive |
| XPLUSDT | 2 | 261.93 | 523.87 | mixed_or_sparse_positive |
| FARTCOINUSDT | 6 | 52.28 | 313.67 | mixed_or_sparse_positive |
| HBARUSDT | 2 | 152.90 | 305.80 | mixed_or_sparse_positive |
| JELLYJELLYUSDT | 1 | 276.01 | 276.01 | mixed_or_sparse_positive |
| FORMUSDT | 1 | 272.48 | 272.48 | mixed_or_sparse_positive |
| ETCUSDT | 4 | 64.86 | 259.42 | mixed_or_sparse_positive |
| PENGUUSDT | 4 | 64.37 | 257.47 | mixed_or_sparse_positive |

## Worst Contributors

| Symbol | Trades | Avg | Total | Class |
|---|---:|---:|---:|---|
| DOTUSDT | 6 | -194.11 | -1164.65 | negative |
| BTCUSDT | 4 | -133.20 | -532.80 | negative |
| VIRTUALUSDT | 4 | -129.49 | -517.97 | negative |
| SUIUSDT | 6 | -70.05 | -420.28 | negative |
| JUPUSDT | 1 | -397.66 | -397.66 | negative |
| XRPUSDT | 4 | -97.50 | -389.99 | mixed_or_sparse_positive |
| DASHUSDT | 1 | -334.27 | -334.27 | negative |
| TONUSDT | 6 | -52.89 | -317.32 | negative |
| AEROUSDT | 2 | -123.05 | -246.09 | negative |
| ARBUSDT | 3 | -70.85 | -212.55 | negative |

## Bottom Line

- The strategy is promising but concentrated.
- Only a very small subset currently looks robust across both train and holdout.
- A production version should probably add a second-stage symbol classifier or whitelist rather than trade every triggered name.