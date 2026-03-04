# Findings Phase 10: Anti-Overfit Follow-Ups

This phase tested three follow-up ideas using a fixed holdout:

- Train months: `2025-08` through `2026-01`
- Test months: `2026-02`, `2026-03`

The goal was to improve trade quality without selecting rules on the test period.

Primary output: `out/anti_overfit_followups.md`

## 1. Train-Selected Symbol Sleeve

Method:

- Start from the existing Phase 9 meta-filter.
- Run the filtered replay on train months only.
- Keep only symbols with positive train contribution.
- Evaluate that reduced sleeve on the held-out test months.

Train-selected symbols:

- `CRVUSDT`
- `GALAUSDT`

Held-out test comparison:

- Baseline 3-symbol unfiltered:
  - `259` fills
  - `60.62%` win rate
  - `7.8476 bps`
  - `$2,052.88`
- Filtered 3-symbol:
  - `210` fills
  - `60.95%` win rate
  - `9.1116 bps`
  - `$1,931.40`
- Train-selected 2-symbol sleeve:
  - `176` fills
  - `64.20%` win rate
  - `10.6558 bps`
  - `$1,892.67`

Conclusion:

- Removing `SEIUSDT` does improve held-out win rate and per-trade edge.
- It does not improve held-out total dollars at the current `10%` sizing.

## 2. Simple Probability Ranker

Method:

- Train a simple additive expected-edge ranker on train months only.
- Features:
  - symbol
  - score bucket
  - spread bucket
  - velocity bucket
- Choose the kept fraction using train replay dollars only.
- Evaluate the selected cutoff on held-out test months.

Selected train-only cutoff:

- predicted score cutoff: `3.3940`
- effectively keeps the top `40%` of train-ranked signals

Held-out result:

- Train replay:
  - `577` fills
  - `69.67%` win rate
  - `6.3095 bps`
  - `$3,707.07`
- Test replay:
  - `192` fills
  - `59.38%` win rate
  - `8.9773 bps`
  - `$1,738.18`

Conclusion:

- Strong train result did not carry through proportionally on test.
- Held-out dollars are lower than the plain baseline and also lower than the Phase 9 filter.
- This is the clearest sign that ranking adds overfit risk faster than it adds practical value in the current sample.

## 3. Replay-PnL Threshold Search

Method:

- Use a compact local search, not a wide sweep, to reduce overfitting pressure.
- Optimize on train replay dollars only, then apply the chosen config to test months.

Compact search space:

- `min_score` in `{6, 8, 10}`
- `sei_score_extra` in `{8, 10, 12}`
- `max_velocity` in `{10, 12, 14}`
- `min_spread_abs` in `{10, 12, 14}`
- `min_ls = 0.15`
- `min_oi = 5`
- `min_carry = 2`

Selected train-only config:

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

Held-out result:

- Train replay:
  - `684` fills
  - `61.26%` win rate
  - `4.3649 bps`
  - `$3,030.11`
- Test replay:
  - `213` fills
  - `62.44%` win rate
  - `9.2923 bps`
  - `$1,998.51`

Conclusion:

- This is the best compromise found in this phase.
- It improves held-out win rate and per-trade edge versus the baseline.
- It still trails the plain 3-symbol baseline slightly in total test dollars:
  - baseline: `$2,052.88`
  - replay-optimized filter: `$1,998.51`

That gap is small enough that this variant may still be preferable if:

- execution quality matters more than raw turnover
- higher per-trade edge allows safer modest sizing later

## Overall Conclusion

The anti-overfit result is clear:

- More selection can improve trade quality.
- More selection has not yet improved absolute held-out dollars at the current conservative deployment.

Current honest ranking:

1. Best held-out total dollars at `10%` sizing: plain 3-symbol baseline.
2. Best held-out quality / confidence tradeoff: replay-optimized filter (`min_score 6`, `min_spread_abs 14`, `max_velocity 12`, `sei_score_extra 10`).
3. Highest held-out win rate: train-selected `CRVUSDT + GALAUSDT` sleeve.

This is a useful stopping point because it prevents false optimism:

- the edge is real
- quality filters help
- but the simplest frozen baseline still wins on dollars in the holdout

The next meaningful test, if needed, is not a more complex classifier. It is to re-run the baseline and the replay-optimized filter under the size-aware slippage model and compare whether the higher-quality variant degrades more slowly with size.
