# Anti-Overfit Follow-Ups

- Train months: 2025-08, 2025-09, 2025-10, 2025-11, 2025-12, 2026-01
- Test months: 2026-02, 2026-03

## 1. Train-Selected Symbol Sleeve

- Symbols kept from filtered train replay: CRVUSDT, GALAUSDT
- Baseline 3-symbol unfiltered test: 259 fills, 60.62% win rate, 7.8476 bps, $2052.88
- Filtered 3-symbol test: 210 fills, 60.95% win rate, 9.1116 bps, $1931.40
- Train-selected sleeve test: 176 fills, 64.20% win rate, 10.6558 bps, $1892.67

## 2. Simple Probability Ranker

- Selected predicted cutoff: 3.3940
- Train replay: 577 fills, 69.67% win rate, 6.3095 bps, $3707.07
- Test replay: 192 fills, 59.38% win rate, 8.9773 bps, $1738.18

## 3. Replay-PnL Threshold Search

Compact search space to reduce overfitting:

- `min_score` in {6, 8, 10}
- `sei_score_extra` in {8, 10, 12}
- `max_velocity` in {10, 12, 14}
- `min_spread_abs` in {10, 12, 14}
- `min_ls = 0.15`, `min_oi = 5`, `min_carry = 2` fixed

Selected replay-optimized config (train only):

```json
{
  "max_velocity": 12.0,
  "min_carry": 2.0,
  "min_ls": 0.15,
  "min_oi": 5.0,
  "min_score": 6.0,
  "min_spread_abs": 14.0,
  "sei_score_extra": 10.0
}
```

- Train replay: 684 fills, 61.26% win rate, 4.3649 bps, $3030.11
- Test replay: 213 fills, 62.44% win rate, 9.2923 bps, $1998.51
