# Stress Matrix: Replay-Optimized 25% Sleeve

- Input: /home/ubuntu/Projects/skytrade6/codex-exp-1/out/candidate_trades_v3_replayopt.csv
- Strategy is frozen; only execution costs change across scenarios.

## Results

| Scenario | PnL Dollars | Avg Net bps | Win Rate | Max DD % | Neg Weeks | Neg Months |
|---|---:|---:|---:|---:|---:|---:|
| base | 7638.99 | 3.2849 | 53.29% | 0.25% | 5 | 1 |
| higher_fees | 2918.30 | 1.2849 | 47.94% | 0.79% | 14 | 2 |
| higher_fixed_slip | 5252.21 | 2.2849 | 50.39% | 0.35% | 11 | 2 |
| higher_variable_slip | 6140.96 | 2.6599 | 51.84% | 0.29% | 7 | 2 |
| higher_size_slip | 5843.90 | 2.5349 | 51.51% | 0.31% | 7 | 2 |
| harsh_combo | -2419.67 | -1.0901 | 42.14% | 3.88% | 23 | 6 |
| very_harsh_combo | -11406.67 | -5.4044 | 30.92% | 11.68% | 30 | 8 |

## Survival Summary

- Positive scenarios: 5 / 7
- Base case PnL: $7638.99

- Harsh combo remains negative at $-2419.67
- Very harsh combo remains negative at $-11406.67
