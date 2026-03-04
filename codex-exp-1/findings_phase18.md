# Findings Phase 18: Longer-Sample Microstructure Validation

This phase extends the high-resolution study from the initial 7-day window to a full 30-day window:

- `2026-02-01` through `2026-03-02`

That larger sample is the first meaningful check on whether the microstructure gate survives beyond a tiny slice.

## Data Extension

High-resolution archives were extended for:

- `CRVUSDT`
- `GALAUSDT`
- `SEIUSDT`

Across the full 30-day range for:

- Bybit `orderbook`, `trades`
- Binance `bookDepth`, `trades`

## 30-Day Microstructure Overlay

Source:

- `out/microstructure_window_analysis_30d.md`

Window summary:

- `3,090` filtered trade candidates
- `1,496` winners
- `1,594` losers
- mean modeled net PnL: `4.2159 bps`
- median modeled net PnL: `-1.6044 bps`

The microstructure differences are smaller than in the first 7-day study, but still directionally consistent:

- winners still have slightly higher recent Bybit activity
- winners still have slightly higher recent Binance activity
- winners still see a slightly tighter Bybit book

This is what we want to see from a longer sample: weaker than the tiny pilot, but still pointing the same way.

## Holdout Design

The same constrained microstructure filter test was rerun on the 30-day enriched file.

Train days:

- `2026-02-01` through `2026-02-23`

Test days:

- `2026-02-24` through `2026-03-02`

So the holdout is now a full 7-day block, not a 2-day sliver.

## Baseline

No extra microstructure gate:

- Train:
  - `2,529` trades
  - `3.7644 bps`
  - `47.49%` win rate
- Test:
  - `561` trades
  - `6.2514 bps`
  - `52.58%` win rate

## Selected Train-Only Filter

The constrained train-only selector chose:

```json
{
  "min_bybit_trade_count_5s": 2.0,
  "min_binance_trade_count_5s": 0.0,
  "max_bybit_book_spread_bps": 4.5,
  "max_bybit_trade_imbalance_5s": 1.0
}
```

Result:

- Train:
  - `901` trades
  - `4.7455 bps`
  - `51.28%` win rate
- Test:
  - `199` trades
  - `8.1528 bps`
  - `59.30%` win rate

This is materially better than the baseline on the 7-day holdout:

- higher held-out edge
- higher held-out win rate
- fewer trades, but much cleaner trades

## Hypothesis-Driven Gate

The simpler hand-constrained gate from the earlier 7-day study was also retested:

```json
{
  "min_bybit_trade_count_5s": 4.0,
  "min_binance_trade_count_5s": 0.0,
  "max_bybit_book_spread_bps": 4.5,
  "max_bybit_trade_imbalance_5s": 1.0
}
```

Result:

- Train:
  - `694` trades
  - `4.5536 bps`
  - `50.29%` win rate
- Test:
  - `158` trades
  - `8.7627 bps`
  - `58.86%` win rate

Interpretation:

- this keeps even fewer trades
- the held-out edge is even higher
- the held-out win rate remains strong

So the simple “activity + tighter Bybit book” gate still holds up on the larger sample.

## What Changed From The 7-Day Pilot

In the small 7-day test:

- the train-only optimizer was not convincing
- the hypothesis-driven gate looked better

In the 30-day test:

- both the constrained train-only filter and the hypothesis-driven gate improve the larger holdout

That is an important upgrade in confidence.

## Current Best Microstructure Interpretation

The evidence now supports a practical statement:

- the base 1-minute filtered strategy is stronger when local microstructure is not thin
- requiring at least modest Bybit pre-trade activity
- and requiring a tighter Bybit book

appears to improve out-of-sample trade quality on a meaningfully larger high-resolution sample.

## Practical Candidate Gate

The simplest defensible microstructure add-on right now is:

- `bybit_trade_count_5s >= 4`
- `bybit_book_spread_bps <= 4.5`

This is not yet the final production rule, but it is the cleanest optional execution-quality gate found so far.

## Why This Matters

This is the first time the high-resolution data changed the research in a non-trivial way:

- not by adding a new exchange
- not by adding more symbols
- but by making the entry quality more realistic

That moves the strategy from:

- “1-minute signal with modeled execution”

toward:

- “1-minute signal with microstructure-aware entry gating”

which is much closer to what a real implementation needs.
