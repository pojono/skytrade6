# Codex Exp 2: Fee-Aware LS Momentum Findings

Independent research from raw `datalake/binance` daily CSVs only.

This note ignores prior repo PnL claims and rebuilds the signal from source data. Existing repo findings were used only as idea generation, not as evidence.

## Objective

Find a strategy that can plausibly survive:

- maker fee: `0.04%` (`4 bps`)
- taker fee: `0.10%` (`10 bps`)

Primary benchmark in this research:

- strict round-trip all taker: `20 bps`

Additional benchmarks:

- taker entry + maker exit: `14 bps`
- all maker: `8 bps`
- stress case: `24 bps`

## Data Used

- Source: `datalake/binance`
- Fields from `*_metrics.csv`:
  - `sum_open_interest_value`
  - `count_toptrader_long_short_ratio`
  - `sum_taker_long_short_vol_ratio`
- Fields from spot-adjusted futures price series:
  - `close` from standard `*_kline_1m.csv`

Coverage used by the experiment:

- Sample rows after feature construction: `129,774`
- Eligible symbols after liquidity filter in train: `85`
- Eligible symbols after liquidity filter in test: `67`
- Train period: through `2025-12-31`
- Holdout test period: from `2026-01-01`

## Research Process

Features per symbol:

- 14-day rolling z-score of top-trader long/short ratio (`ls_z`)
- 14-day rolling z-score of taker long/short volume ratio (`taker_z`)
- 4-hour momentum (`mom_4h`)
- 3-day rolling median open interest value (`oi_med_3d`)

Signal cadence:

- evaluate every 4 hours at `HH:05 UTC`

Base trade model:

- fixed 4-hour hold
- no stop
- no compounding
- no leverage model
- returns averaged across selected names per timestamp

Cross-sectional ranking:

- `abs(ls_z) + 0.35 * abs(taker_z)`

## What Failed

These are important because they removed a lot of false comfort:

### 1. Symmetric Long/Short LS Momentum Failed

The first implementation used mirrored long and short rules:

- long when `ls_z` and `taker_z` were strong positive with positive momentum
- short when `ls_z` and `taker_z` were strong negative with negative momentum

This did not survive realistic fees.

Best early symmetric result was still:

- about `-9.77 bps/trade` on the 2026 holdout after `20 bps` all-taker

Conclusion:

- the bearish mirror is weak or false
- the useful part of this feature family is not symmetric
- top-trader bearish positioning does not create a clean short edge the same way bullish positioning creates a long edge

### 2. Loose Thresholds Produced More Trades but Worse Quality

Lower thresholds such as:

- `ls_z >= 0.5`
- `taker_z >= 0.0`

increased trade count materially but degraded edge. Most of those parameter sets were negative after 20 bps.

Conclusion:

- this is not a broad “always-on” factor
- it behaves like a rare-event signal

### 3. Some Parameter Sets Looked Good Only in Holdout

There were configurations with strong 2026 test performance but weak or negative pre-2026 train performance.

Example pattern:

- test looked strong
- train was flat or negative

Those were rejected as likely unstable or overfit to a small recent pocket.

Conclusion:

- ranking by holdout only is too dangerous here
- consistency across train and test matters more than peak test bps

## What Worked

The profitable shape is:

- long-only
- high-conviction
- risk-on regime filtered

### Winning Hypothesis

When all of these line up:

- top traders are unusually net long
- aggressive taker flow is also net long
- the symbol itself already has positive 4-hour momentum
- the broader liquid universe is risk-on

then the next 4 hours continue higher often enough to survive full taker fees.

This is a trend continuation / confirmation effect, not a reversal effect.

## Best Surviving Configuration

Selected from the grid by requiring:

- positive train performance after `20 bps`
- positive holdout test performance after `20 bps`
- then ranking by the weaker of the two periods

Parameters:

- `ls_threshold=2.0`
- `taker_threshold=0.5`
- `min_oi_value=20,000,000`
- `top_n=3`
- `breadth_threshold=0.60`
- `median_ls_threshold=0.0`

Interpretation:

- only act on very large positive LS deviations
- require taker flow confirmation
- only trade names with at least moderate liquidity
- allow up to 3 names at a signal time
- only trade when at least 60% of the liquid universe has positive 4-hour momentum
- avoid bearish broad positioning regimes

## Performance Of Best Configuration

Trade counts:

- train trades: `83`
- holdout test trades: `27`

Net average return per trade:

- after `20 bps` all-taker: `+18.85 bps` in train
- after `20 bps` all-taker: `+26.95 bps` in holdout test
- consistency score (min of train/test): `+18.85 bps`

Holdout test fee scenarios:

- after `14 bps` taker entry + maker exit: `+32.95 bps`
- after `8 bps` all-maker: `+38.95 bps`
- after `24 bps` stress case: `+22.95 bps`

Holdout hit rate:

- after `20 bps`: `55.6%`

Monthly holdout breakdown:

| Month | Trades | Avg After 20bps | Avg After 14bps | Avg After 24bps | Hit Rate |
|---|---:|---:|---:|---:|---:|
| 2026-01 | 21 | 22.87 bps | 28.87 bps | 18.87 bps | 57.1% |
| 2026-02 | 6 | 41.23 bps | 47.23 bps | 37.23 bps | 50.0% |

## Other Positive Configurations

These also survived both train and test after `20 bps`, but were weaker or less balanced:

| LS Z | Taker Z | OI Floor | Top N | Breadth | Train | Test |
|---:|---:|---:|---:|---:|---:|---:|
| 2.0 | 0.5 | 20M | 2 | 0.60 | 18.48 bps | 19.12 bps |
| 2.0 | 0.5 | 20M | 1 | 0.60 | 16.33 bps | 12.24 bps |
| 2.0 | 0.5 | 20M | 3 | 0.55 | 11.33 bps | 24.00 bps |
| 2.0 | 0.5 | 20M | 2 | 0.55 | 10.99 bps | 16.71 bps |
| 2.0 | 0.0 | 20M | 1 | 0.65 | 5.01 bps | 40.44 bps |

Interpretation:

- `taker_z >= 0.5` is more reliable than `taker_z >= 0.0`
- the `0.60` breadth filter is more balanced than `0.55`
- taking the top 3 names gave the best mix of diversification and strength

## Practical Strategy Description

At each `HH:05 UTC`:

1. Compute `ls_z`, `taker_z`, `mom_4h`, `oi_med_3d` for each Binance perpetual symbol.
2. Keep only symbols with `oi_med_3d >= $20M`.
3. Compute market breadth over the liquid universe.
4. If breadth is below `60%` positive 4-hour momentum, do nothing.
5. Keep only symbols where:
   - `ls_z >= 2.0`
   - `taker_z >= 0.5`
   - `mom_4h > 0`
6. Rank by `abs(ls_z) + 0.35 * abs(taker_z)`.
7. Buy up to the top 3 symbols.
8. Exit after 4 hours.

## Why This Is Plausible

- The signal is slow enough that a maker exit is operationally realistic.
- It is based on positioning plus aggressive flow, not just price.
- It is selective enough to avoid constant churn.
- It remains positive even under a `24 bps` fee stress assumption in the tested sample.

## Why This Is Not Yet Production-Ready

This research only prices explicit fees. It does not include:

- slippage
- queue position / missed maker fills
- spread crossing on entry
- partial fills
- latency
- funding paid/received during the 4-hour hold
- capital allocation constraints across overlapping positions
- survivorship / listing-bias checks

So the result is:

- a credible candidate edge
- not a final deployable strategy

## Bottom Line

The repo’s broad PnL numbers should not be trusted, and the first broad strategy shape did not hold up when rechecked.

The strongest useful finding from a fresh rebuild is narrower:

- a rare-event, long-only Binance futures momentum strategy driven by extreme top-trader long bias and confirmed by taker-flow
- filtered to risk-on market regimes
- positive in both pre-2026 and 2026 holdout samples
- positive after full `20 bps` taker round-trip fees
- still positive in a `24 bps` stress case on the tested holdout

## Files

- Research code: `codex-exp-2/research_fee_aware_ls_momentum.py`
- Raw sample export: `codex-exp-2/samples_4h.csv`
- Grid search results: `codex-exp-2/grid_results.csv`
