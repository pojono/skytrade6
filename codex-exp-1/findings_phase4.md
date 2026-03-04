# Phase 4 Findings

## Goal

Move from signal validation to implementation-oriented testing.

This phase adds:

1. Timestamped trade export for the frozen basket
2. A simple paper-trading simulator
3. Capital allocation, daily trade caps, and open-position limits

## Trade Export

Script:

- `codex-exp-1/export_candidate_trades.py`

Output:

- `codex-exp-1/out/candidate_trades_v3.csv`

Exported trade count for the frozen 3-symbol basket:

- `35,928` filtered candidate trades

Each row contains:

- entry and exit timestamps
- gross PnL in bps
- entry spread
- spread velocity proxy
- long/short, OI, and carry filter values
- a ranking score used for execution priority

## Paper Simulation Setup

Script:

- `codex-exp-1/paper_trade_candidate.py`

Base simulation parameters:

- starting capital: `$100,000`
- per-trade allocation: `10%`
- daily cap per symbol: `3`
- base fee: `6 bps`
- moderate dynamic slippage:
  - fixed extra: `1 bps`
  - spread coefficient: `0.10`
  - velocity coefficient: `0.05`

## No-Peek Selection Rule

The daily-cap logic was corrected to avoid intraday lookahead.

Current rule:

- cap trades in encounter order within each symbol/day
- do not cherry-pick the best future trades later in the same day

This removes an accidental future-information leak from the walk-forward validation.

## Portfolio Result: 3 Open Slots

With `max_open_positions = 3`:

- filled trades: `1,422`
- final capital: `$105,302.27`
- total PnL: `$5,302.27`
- average net edge: `3.6340 bps`
- win rate: `58.72%`

Symbol contribution:

- `CRVUSDT`: `$2,751.18`
- `GALAUSDT`: `$2,052.13`
- `SEIUSDT`: `$498.95`

## Portfolio Result: 1 Open Slot

With `max_open_positions = 1`:

- filled trades: `1,421`
- final capital: `$105,356.75`
- total PnL: `$5,356.75`
- average net edge: `3.6729 bps`

Interpretation:

- concurrency is not required for the edge
- prioritizing the top-scoring trade each minute is slightly better
- this makes the strategy easier to deploy conservatively

## Refreshed No-Peek Walk-Forward

Using the same frozen 3-symbol basket with:

- daily cap: `3`
- moderate dynamic slippage:
  - fixed extra: `1 bps`
  - spread coefficient: `0.10`
  - velocity coefficient: `0.05`

And the corrected no-peek cap logic:

- train avg net: `2.7280 bps` on `1,163` trades
- test avg net: `7.7022 bps` on `259` trades
- positive months: `8/8`

This confirms the edge survives after removing the intraday lookahead flaw.

## Current Best Implementation Shape

The cleanest version now is:

1. Basket:
   - `CRVUSDT`
   - `GALAUSDT`
   - `SEIUSDT`
2. Signal:
   - 1-minute cross-exchange spread reversion
3. Filters:
   - `ls >= 0.15`
   - `oi >= 5 bps`
   - `carry >= 2 bps`
4. Execution:
   - daily cap `3` per symbol
   - `1` open position at a time is acceptable
   - moderate dynamic slippage still leaves positive expectancy

## Practical Conclusion

This is the first version that looks reasonably implementable:

- fixed symbols
- fixed rules
- positive under capped walk-forward validation
- positive under moderate dynamic slippage
- positive after the no-peek cap correction
- positive in a simple capital-based paper simulation

It is still a research candidate, but it is now close to a deployable paper-trading spec.
