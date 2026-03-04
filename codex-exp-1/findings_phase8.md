# Phase 8 Findings

## Goal

Re-test leverage after making slippage worsen with position size.

This is the correct next step, because the earlier leverage sweep was optimistic.

## New Size-Aware Slippage Model

Added to `codex-exp-1/paper_trade_candidate.py`:

- `--size-slip-coeff`
- `--base-allocation-ref`

The idea:

- the conservative `10%` allocation is the baseline
- larger allocations pay extra slippage as they move above that baseline

In this test:

- size slippage coefficient: `1.5 bps`
- base allocation reference: `10%`

So:

- `25%` allocation pays a moderate size penalty
- `50%` allocation pays a much larger one

## 25% Allocation With Size-Aware Slippage

Report:

- `codex-exp-1/out/paper_report_25pct_sized.md`

Result:

- filled trades: `1,421`
- final capital: `$105,183.40`
- total PnL: `$5,183.40`
- average net edge: `1.4242 bps`
- win rate: `50.32%`

Interpretation:

- still positive
- meaningfully weaker than the naive no-size-penalty result
- still plausible as a candidate sizing band

## 50% Allocation With Size-Aware Slippage

Report:

- `codex-exp-1/out/paper_report_50pct_sized.md`

Result:

- filled trades: `1,282`
- final capital: `$85,976.89`
- total PnL: `-$14,023.11`
- average net edge: `-2.3537 bps`
- win rate: `34.71%`

Interpretation:

- the strategy breaks at this size under the chosen size-slippage model
- larger size increases slippage enough to destroy the edge

## Practical Sizing Conclusion

After adding size-aware slippage:

1. `10%` allocation remains the conservative baseline
2. `25%` allocation still works, but with much less margin
3. `50%` allocation is no longer acceptable under this model

That means the realistic sizing ceiling is much lower than the earlier naive leverage sweep suggested.

## Current Best Capital Policy

A practical deployment range is:

- `10%` to `25%` allocation per trade

And **not**:

- `50%+` until a better execution model proves otherwise

## Why This Matters

This is the difference between:

- a mathematically scalable backtest
- and a strategy that can plausibly survive execution friction

The updated result is more believable, even though it is less exciting.
