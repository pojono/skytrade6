# Phase 5 Findings

## Goal

Tighten the frozen 3-symbol strategy into a more explicit live-style execution policy.

This phase adds:

1. Explicit same-timestamp selector modes
2. Per-symbol open-position caps
3. Per-symbol allocation caps

## New Execution Controls

Added to `codex-exp-1/paper_trade_candidate.py`:

- `--selector-mode`
  - `score`
  - `spread`
  - `velocity`
- `--max-open-per-symbol`
- `--max-symbol-allocation`

This makes same-minute signal conflicts deterministic and configurable.

## Conservative Live-Style Test

Frozen basket:

1. `CRVUSDT`
2. `GALAUSDT`
3. `SEIUSDT`

Execution constraints:

- starting capital: `$100,000`
- per-trade allocation: `10%`
- max open positions: `1`
- max open per symbol: `1`
- max symbol allocation: `10%`
- daily cap per symbol: `3`
- moderate dynamic slippage:
  - fixed extra: `1 bps`
  - spread coefficient: `0.10`
  - velocity coefficient: `0.05`

## Selector Comparison

### `score` selector

Report:

- `codex-exp-1/out/paper_report_v3_slot1_score.md`

Result:

- filled trades: `1,421`
- final capital: `$105,356.75`
- total PnL: `$5,356.75`
- average net edge: `3.6729 bps`

### `spread` selector

Report:

- `codex-exp-1/out/paper_report_v3_slot1_spread.md`

Result:

- filled trades: `1,421`
- final capital: `$105,358.62`
- total PnL: `$5,358.62`
- average net edge: `3.6742 bps`

## Interpretation

The difference is small, but `spread` priority is marginally better than `score` priority in the current conservative single-slot setup.

That makes practical sense:

- the strategy is a spread-reversion strategy
- when only one trade can be chosen, prioritizing the largest spread often captures the strongest immediate dislocation

## Current Preferred Execution Policy

Use:

1. Basket:
   - `CRVUSDT`
   - `GALAUSDT`
   - `SEIUSDT`
2. Selector:
   - `spread`
3. Exposure:
   - max open positions: `1`
   - max open per symbol: `1`
   - max symbol allocation: `10%`
4. Trade density:
   - max `3` trades per symbol per day

## Practical Conclusion

The strategy is now defined tightly enough to paper trade in a realistic way:

- fixed symbols
- fixed entry logic
- fixed filters
- fixed same-timestamp selector
- fixed exposure caps

This is the cleanest paper-trading specification found so far.
