# Findings Phase 11: Size-Aware Comparison of Baseline vs Replay-Optimized Filter

This phase compares only the two credible candidates under the same size-aware slippage model:

- plain 3-symbol baseline
- compact replay-optimized filter from Phase 10

The goal was to test whether the higher-quality filter degrades more slowly as position size increases.

## Fixed Config

Common live-style assumptions:

- one open position
- one open position per symbol
- selector mode `spread`
- daily cap `3` per symbol
- daily loss stop `1%`
- monthly loss stop `3%`
- fee `6 bps`
- extra slippage `1 bps`
- spread slippage coeff `0.10`
- velocity slippage coeff `0.05`
- size slippage coeff `1.5 bps`
- base allocation reference `10%`

Replay-optimized pre-trade filter:

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

Filtered trade file used for replay:

- `out/candidate_trades_v3_replayopt.csv`

## 10% Allocation

Baseline, from `out/paper_report_v3_base_sized10.md`:

- `1,421` fills
- `58.69%` win rate
- `3.6742 bps`
- `$5,358.62`

Replay-optimized filter, from `out/paper_report_v3_replayopt_sized10.md`:

- `897` fills
- `61.54%` win rate
- `5.5349 bps`
- `$5,089.18`

Conclusion at `10%`:

- The baseline still wins on total dollars.
- The replay-optimized filter wins on trade quality:
  - higher win rate
  - higher average net edge
  - much smaller `SEIUSDT` drag

## 25% Allocation

Baseline, from `out/paper_report_v3_base_sized25.md`:

- `1,421` fills
- `50.32%` win rate
- `1.4242 bps`
- `$5,183.40`

Replay-optimized filter, from `out/paper_report_v3_replayopt_sized25.md`:

- `897` fills
- `53.29%` win rate
- `3.2849 bps`
- `$7,638.99`

Conclusion at `25%`:

- The replay-optimized filter clearly wins.
- Incremental advantage vs baseline:
  - `+2.97` percentage points win rate
  - `+1.8607 bps` average net edge
  - `+$2,455.59` total PnL

This is the first strong evidence that stricter signal quality matters more once size-dependent execution penalties are included.

## Why This Matters

At low sizing, the baseline benefits from higher turnover.

At larger sizing, weaker trades become much more expensive:

- the baseline has more marginal trades
- `SEIUSDT` turns into a clear drag under larger size
- the replay-optimized filter keeps fewer but stronger trades

That makes the filtered strategy more scalable, even though it was slightly worse at the smallest deployment size.

## Current Practical Interpretation

There are now two honest variants for different deployment goals:

1. Best small-size dollar generator:
   - plain 3-symbol baseline
   - preferred around `10%` allocation

2. Best moderate-size scalable variant:
   - replay-optimized filter
   - preferred around `25%` allocation under the current size-aware slippage model

## Current Best Moderate-Size Candidate

At `25%` allocation under size-aware slippage, the replay-optimized filter is now the strongest version tested:

- `897` fills
- `53.29%` win rate
- `3.2849 bps` average net edge
- final capital: `$107,638.99`
- total PnL: `$7,638.99`

That is roughly:

- `7.64%` total return over the same sample
- materially better than the baseline’s `5.18%` under the same larger-size assumptions

## Remaining Risk

The filtered `25%` run is better, but it is still not “free money”:

- August is still negative
- `SEIUSDT` is still slightly negative in dollars
- the model still does not include funding, impact, or venue liquidity limits

So this is an improved deployment candidate, not a final production guarantee.
