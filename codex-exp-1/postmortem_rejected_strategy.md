# Postmortem: Why This Strategy Was Rejected

This document records why the cross-exchange filtered sleeve in `codex-exp-1` is not considered a deployable strategy, despite looking promising during earlier research stages.

## Short Conclusion

The strategy was rejected because it failed the strictest execution realism test that was run.

In plain terms:

- the signal looked real enough in coarse historical replay
- but once entry and exit were replayed against stricter order book mechanics, the multi-symbol edge disappeared

That makes the current version unsuitable for production.

## What The Strategy Was

The final researched version was:

- a short-horizon cross-exchange mean-reversion strategy
- trading Binance vs Bybit dislocations
- using a narrow symbol set:
  - `CRVUSDT`
  - `GALAUSDT`
  - `SEIUSDT`
- one-minute hold
- `25%` allocation
- single-slot exposure
- with optional microstructure-aware entry gating

## Why It Looked Promising At First

The strategy passed several earlier layers:

- broad-universe filtering narrowed it to a small survivor cluster
- the filtered sleeve beat the baseline under the earlier 1-minute replay
- it stayed positive under moderate cost stress
- high-resolution microstructure data improved trade selection
- the best optional microstructure gate improved account-level replay over a 30-day overlap window

If the research had stopped there, it would have looked like a viable candidate.

That would have been the wrong conclusion.

## What Actually Broke It

The decisive test was the strict fill replay.

That replay:

- took the actual accepted fills from the current best microstructure-gated sleeve
- simulated Bybit fills against archived L2 order book snapshots
- simulated Binance fills against archived depth buckets
- required both entry and exit to be fillable

All fills were mechanically fillable.

But the economics failed:

- modeled average net: `+7.9684 bps`
- strict-fill average net: `-1.6466 bps`
- modeled total PnL: `+$3,917.39`
- strict-fill total PnL: `-$805.32`

So the strategy did not fail because it could not get orders filled.
It failed because the fills were too expensive.

## Root Cause

The main bottleneck was Bybit execution cost.

Average strict slippage components:

- Bybit entry: `5.1678 bps`
- Bybit exit: `5.0532 bps`
- Binance entry: `2.2938 bps`
- Binance exit: `1.9651 bps`

Total average strict execution slippage:

- `14.4799 bps`

Most of that came from Bybit.

This means:

- the strategy was paying too much crossing the book on Bybit
- the remaining spread edge was not large enough to absorb that cost

## Why This Is Not A “Just Trade One Coin” Fix

Under strict replay:

- `CRVUSDT` stayed positive on average
- `GALAUSDT` failed
- `SEIUSDT` failed

That might tempt a “just trade CRV” conclusion.

That is not good enough.

A strategy that only survives on one symbol is too fragile:

- too concentrated
- too regime-dependent
- too easy to lose if that one local edge degrades

So the multi-symbol portfolio thesis failed.

That is the correct standard to use.

## What Was Learned

Even though the strategy is rejected, the work was still useful.

We learned:

- broad cross-exchange spread trading is not robust across the universe
- narrower symbol-specific effects can exist
- microstructure matters
- 1-minute replay alone can be materially too optimistic
- strict fill realism can completely reverse the conclusion
- Bybit crossing cost is the dominant execution bottleneck in this setup

The most important lesson is:

- signal research without sufficiently realistic execution testing is not enough

## Why We Are Stopping Here

We are stopping because the current path no longer justifies more incremental tuning.

At this point:

- more parameter fitting would be low-value
- more symbol tweaks would not solve the execution bottleneck
- keeping the same execution style and hoping for a better result would be self-deception

This is where the research should stop unless the structure changes materially.

## What Would Need To Be Different To Reopen This Idea

This idea should only be revisited if at least one of these changes materially:

1. A different execution style.
   - passive or semi-passive placement
   - lower effective Bybit crossing cost
   - better queue / fill logic

2. A different venue mix.
   - another venue changes the fair-value or routing logic
   - another venue provides cheaper execution than the current pair

3. A different holding structure.
   - not the same pure one-minute in-and-out crossing behavior

4. A stricter candidate standard from the start.
   - only ideas that pass strict fill replay on a multi-symbol basket should survive

Without one of those changes, this exact strategy should be treated as closed.

## Final Status

The correct final label for this strategy is:

- researched
- informative
- rejected for production

It remains useful as:

- a negative result
- an execution-cost case study
- a reminder that apparent statistical edge can vanish under realistic fills

That is the right conclusion to preserve.
