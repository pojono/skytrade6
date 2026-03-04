# Findings Phase 17: Constrained Holdout Test For A Microstructure Gate

This phase takes the first-pass microstructure overlay and asks a stricter question:

- can a simple microstructure gate improve the held-out last two days?

Script:

- `test_microstructure_filter.py`

Outputs:

- `out/microstructure_filter_search.csv`
- `out/microstructure_filter_report.md`
- `out/microstructure_window_filtered.csv`

## Holdout Split

Recent 7-day window:

- `2026-02-24` through `2026-03-02`

Train days:

- `2026-02-24` through `2026-02-28`

Test days:

- `2026-03-01`
- `2026-03-02`

This is a small sample, so the result is directional only.

## Baseline

No extra microstructure gate:

- Train:
  - `362` trades
  - `6.9477 bps`
  - `53.04%` win rate
- Test:
  - `199` trades
  - `4.9847 bps`
  - `51.76%` win rate

## Train-Optimized Filter

The constrained train-only selector chose:

```json
{
  "min_bybit_trade_count_5s": 0.0,
  "min_binance_trade_count_5s": 0.0,
  "max_bybit_book_spread_bps": 4.5,
  "max_bybit_trade_imbalance_5s": 0.2
}
```

Result:

- Train:
  - `224` trades
  - `8.9789 bps`
  - `58.93%` win rate
- Test:
  - `151` trades
  - `4.6207 bps`
  - `51.66%` win rate

Interpretation:

- this improves train quality
- it does not improve the held-out test

So this is not a convincing production filter.

## Hypothesis-Driven Gate

A simpler, human-led gate was then evaluated. It was chosen from the descriptive findings, not from the holdout:

- require more Bybit local trade activity
- require a tighter Bybit book

Gate:

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
  - `92` trades
  - `9.4215 bps`
  - `58.70%` win rate
- Test:
  - `66` trades
  - `7.8444 bps`
  - `59.09%` win rate

Interpretation:

- fewer trades
- better held-out edge
- better held-out win rate

This is the first microstructure-aware gate that looks directionally promising in the holdout.

## What We Learned

The important result is not “machine search found the best filter.”

The important result is:

- descriptive microstructure signals were useful
- a simple hand-constrained activity + tight-book gate improved the small holdout
- the train-optimized selector alone was not enough

That reinforces the earlier anti-overfit lesson:

- pure train optimization can still choose the wrong filter
- hypothesis-driven constraints are safer than broad re-optimization

## Current Meaning

This is not yet enough evidence to replace the main strategy spec.

But it is enough to justify a next pass where the microstructure gate becomes:

- a clearly defined optional execution-quality filter
- tested on a longer high-resolution window

The best current candidate for that next pass is:

- `min_bybit_trade_count_5s >= 4`
- `bybit_book_spread_bps <= 4.5`

because those are the two signals that were both:

- supported by the descriptive analysis
- directionally positive in the tiny holdout
