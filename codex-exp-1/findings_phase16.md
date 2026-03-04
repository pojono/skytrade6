# Findings Phase 16: First High-Resolution Microstructure Overlay

This phase uses the newly downloaded sub-minute data to enrich the frozen replay-optimized trade stream over the recent 7-day window:

- `2026-02-24` through `2026-03-02`

Primary script:

- `analyze_microstructure_window.py`

Outputs:

- `out/microstructure_window_analysis.csv`
- `out/microstructure_window_analysis.md`

## Scope

This is a first-pass descriptive overlay, not a new backtest engine.

It enriches the recent replay-optimized trade candidates with:

- Bybit true L2 order book state from `orderbook.jsonl`
- Bybit last-5-second trade flow
- Binance last-5-second trade flow
- Binance 1% aggregated depth from `bookDepth.csv`

The window contains:

- `561` filtered trade candidates
- `295` winners
- `266` losers

under the current frozen `25%` modeled cost assumptions.

## Main Microstructure Signals

### 1. More Pre-Trade Activity Is Better

Winners had more short-horizon trade activity before entry:

- Bybit trade count in prior 5s:
  - winners: `4.68`
  - losers: `3.56`
- Binance trade count in prior 5s:
  - winners: `3.87`
  - losers: `2.62`

This is one of the clearest signals in the first overlay.

Interpretation:

- trades entered during more active local microstructure were better
- “dead” or low-activity setups were weaker

### 2. Slightly Tighter Bybit Book Helps

Bybit top-of-book spread was modestly tighter on winners:

- winners: `3.52 bps`
- losers: `3.70 bps`

Interpretation:

- cleaner local liquidity on Bybit may help the one-minute mean-reversion trade survive execution better

### 3. Bybit Short-Horizon Trade Imbalance Matters

Prior 5-second Bybit trade imbalance differed meaningfully:

- winners: `-0.036`
- losers: `+0.084`

Interpretation:

- the sign suggests loser trades were more often preceded by short-horizon aggressive buy pressure on Bybit
- this may indicate chasing into still-expanding dislocations rather than fading exhaustion

This needs a directional second pass, but it is a real difference.

### 4. Binance Depth Signal Is Weak But Positive

Binance 1% depth imbalance:

- winners: `0.0546`
- losers: `0.0439`

Interpretation:

- there is a mild positive difference, but it is much weaker than the trade-flow signals
- this is not yet strong enough to be a primary filter

### 5. Binance Depth Timing Is Coarse

Nearest Binance `bookDepth` snapshot lag at entry:

- winners: about `23.8s`
- losers: about `22.3s`

Interpretation:

- Binance `bookDepth` archive is useful as a coarse depth context
- it is not true tick-by-tick top-of-book data
- most of the actionable sub-minute information in this first pass comes from:
  - Bybit L2 updates
  - Bybit trades
  - Binance trades

## Quick Practical Read

The most actionable first conclusion is:

- low-activity entries are likely weaker

A simple next microstructure filter to test is:

- require at least a minimum combined trade count over the prior 5 seconds

This is supported by the quick decile check:

- top decile of Bybit 5-second trade count (`>= 10` trades):
  - win rate: `54.10%`
  - mean net: `8.05 bps`
- rest of the sample:
  - win rate: `52.40%`
  - mean net: `6.03 bps`

That is not enough to declare a final production rule, but it is a credible direction for a second-pass execution-aware filter.

## What This Means

The high-resolution data did add value immediately:

- it confirmed the 1-minute signal is not uniform internally
- it identified concrete pre-trade microstructure conditions associated with better outcomes
- it suggests that execution-aware filtering should focus on:
  - recent trade activity
  - local book tightness
  - short-horizon trade pressure

The next correct step is to turn these descriptive features into a constrained, holdout-tested microstructure filter rather than guessing new thresholds by eye.
