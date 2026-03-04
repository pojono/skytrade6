# Phase 3 Findings

## Goal

Move from a promising signal study to a stricter execution-aware strategy candidate.

This phase adds:

1. Month-based walk-forward validation
2. Per-symbol daily trade caps
3. Dynamic slippage stress tied to spread stretch and spread velocity

## Best Stable Basket Before Dynamic Slippage

Using the fixed basket:

1. `CRVUSDT`
2. `GALAUSDT`
3. `SEIUSDT`
4. `FILUSDT`

With:

- spread threshold: `10 bps`
- base fee: `6 bps`
- filters:
  - `ls >= 0.15`
  - `oi >= 5 bps`
  - `carry >= 2 bps`

And a daily cap of `3` trades per symbol:

- recent history: `210` days
- test months: last `2`

Result:

- train avg net: `5.5574 bps` on `1,448` trades
- test avg net: `7.9771 bps` on `349` trades
- positive months: `8/8`

This is the strongest flat-slippage result so far.

## Dynamic Slippage Model

To make execution more realistic, slippage was changed from a flat add-on to:

`dynamic_slippage = fixed_extra + spread_coeff * stretch + velocity_coeff * spread_velocity`

Where:

- `stretch = max(0, entry_spread_abs - min_signal_bps)`
- `spread_velocity = abs(current_entry_spread - prior_spread)`

This penalizes entries more when the spread is very stretched or moving fast.

## Moderate Dynamic Slippage

Configuration:

- fixed extra slippage: `1 bps`
- spread coefficient: `0.10`
- velocity coefficient: `0.05`
- daily cap: `3`

Result:

- train avg net: `0.5768 bps`
- test avg net: `5.1459 bps`
- positive months: `7/8`

Interpretation:

- still tradable in principle
- much less robust than the flat-slippage result
- edge becomes thin in training history

## Harsh Dynamic Slippage

Configuration:

- fixed extra slippage: `2 bps`
- spread coefficient: `0.20`
- velocity coefficient: `0.10`
- daily cap: `3`

Result:

- train avg net: `-4.4038 bps`
- test avg net: `2.3148 bps`
- positive months: `6/8`

Interpretation:

- the strategy fails under this harsher execution model
- current edge is not robust to very aggressive slippage assumptions

## Symbol-Level Weak Point

Under moderate dynamic slippage:

- `FILUSDT` turns negative: `-5.7879 bps`

Under harsh dynamic slippage:

- `FILUSDT` is deeply negative: `-18.8230 bps`
- `SEIUSDT` also turns negative

That makes `FILUSDT` the first obvious candidate for removal in a stricter live basket.

## 3-Symbol Refinement

After removing `FILUSDT`, the tighter basket became:

1. `CRVUSDT`
2. `GALAUSDT`
3. `SEIUSDT`

Using the same settings:

- daily cap: `3`
- spread threshold: `10 bps`
- filters unchanged

Flat walk-forward result over `210` days with last `2` months as test:

- train avg net: `4.8067 bps` on `1,163` trades
- test avg net: `9.7423 bps` on `259` trades
- positive months: `8/8`

Moderate dynamic slippage result:

- fixed extra: `1 bps`
- spread coefficient: `0.10`
- velocity coefficient: `0.05`

Result:

- train avg net: `2.6781 bps`
- test avg net: `6.5136 bps`
- positive months: `8/8`

This is the cleanest strategy version found so far.

## Current Best Interpretation

The robust conclusion is:

1. A broad cross-exchange portfolio still does not work
2. A narrow 4-symbol basket works under capped trade density and flat/slightly elevated slippage
3. Dynamic slippage reveals fragility, especially in `FILUSDT`

So the best current strategy is the 3-symbol capped basket above.

## Candidate For Next Iteration

Most promising live candidate now:

1. `CRVUSDT`
2. `GALAUSDT`
3. `SEIUSDT`

The next work should focus on position sizing, exposure caps, and per-trade export rather than more symbol discovery.
