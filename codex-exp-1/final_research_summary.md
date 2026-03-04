# Cross-Exchange Edge Research Summary

This document summarizes the full research path in `codex-exp-1`, from broad universe discovery to the current best candidate.

## Objective

Research whether cross-exchange dislocations between Binance and Bybit USDT perpetuals contain a repeatable, net-positive edge across a large symbol universe, and narrow that into a deployable strategy if one exists.

## Data Scope

The work was done on the local `datalake`:

- `116` common Binance/Bybit symbols with overlapping minute data
- minute-based cross-exchange signal construction
- Binance `kline`, `mark`, `index`, and `metrics`
- Bybit `kline`, `mark`, `premium`, `funding`, `open_interest`, `long_short`

Important adjustment:

- sparse funding prints are not useful for minute entries
- dense carry proxies were used instead:
  - Binance: `mark / index - 1`
  - Bybit: minute premium index

## Research Path

### 1. Broad Universe Scan

The initial idea was to trade 1-minute Binance/Bybit spread dislocations across `90+` symbols.

What happened:

- unfiltered broad-universe spread reversion was negative after fees
- filtered broad-universe variants were still negative
- broad “trade everything” was rejected

Core lesson:

- there is no robust portfolio-wide edge from the raw spread signal alone

### 2. Survivor Discovery

A small cluster of symbols stayed positive after adding:

- spread threshold
- one-bar hold
- long/short divergence
- open-interest divergence
- carry divergence

That reduced the universe to a small candidate basket.

### 3. Execution Realism

As the replay became more realistic, weaker symbols dropped out:

- no-peek intraday cap logic
- explicit same-timestamp selector
- live-style capital allocation
- daily and monthly loss stops
- dynamic slippage tied to spread and velocity
- size-aware slippage tied to allocation size

This made it clear that:

- some edges were artifacts of weak execution assumptions
- weaker symbols diluted returns and worsened slippage sensitivity

### 4. Classifier / Meta-Filter Work

A pre-trade quality filter was tested on top of the frozen 3-symbol strategy.

What it did:

- improved per-trade edge
- improved win rate slightly
- reduced turnover

What it did not do at small size:

- it did not beat the plain 3-symbol baseline on total dollars at `10%` sizing

Important anti-overfit result:

- more complex classifiers improved trade quality
- they did not beat the simplest baseline on held-out total dollars at conservative size

### 5. Size-Aware Re-Test

Once size-aware slippage was introduced, the deployment conclusion changed:

- the plain baseline still won at `10%`
- the higher-quality filtered sleeve degraded more slowly
- at `25%`, the filtered sleeve clearly beat the baseline on both return and downside

This was the key turning point in the research.

## What Failed

These ideas did not survive robust testing:

- broad multi-symbol spread reversion across the universe
- assuming the same rule works across all symbols
- naive leverage without size-dependent slippage
- more complex pre-trade filtering as an automatic improvement at low utilization

## What Survived

The research supports a narrow, symbol-specific edge with explicit filters and realistic constraints.

## Final Candidate Hierarchy

There are two honest variants depending on target deployment size.

### Small-Size Variant

Best dollar generator near `10%` allocation:

- plain 3-symbol baseline
- symbols:
  - `CRVUSDT`
  - `GALAUSDT`
  - `SEIUSDT`

At `10%` with the current live-style model:

- `1,421` fills
- `58.69%` win rate
- `3.6742 bps` average net edge
- `$5,358.62` total PnL

This variant wins at small size because higher turnover still outweighs lower trade quality.

### Best Current Moderate-Size Variant

Best current research candidate near `25%` allocation:

- replay-optimized filtered sleeve
- same 3-symbol base universe
- stricter pre-trade filter

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

At `25%` with size-aware slippage:

- `897` fills
- `53.29%` win rate
- `3.2849 bps` average net edge
- `$7,638.99` total PnL

Why this becomes superior:

- it trades fewer, stronger signals
- it degrades more slowly when size increases
- weaker trades are filtered before size-dependent execution penalties hit

## Final Strategy Spec

The current best moderate-size candidate is:

- universe:
  - `CRVUSDT`
  - `GALAUSDT`
  - `SEIUSDT`
- signal:
  - cross-exchange 1-minute Binance/Bybit spread dislocation
- base hold:
  - one bar
- live-style selection:
  - one open position
  - one open position per symbol
  - selector mode `spread`
  - daily cap `3` per symbol
- risk rails:
  - daily loss stop `1%`
  - monthly loss stop `3%`
- execution costs:
  - fee `6 bps`
  - extra slippage `1 bps`
  - spread slippage coeff `0.10`
  - velocity slippage coeff `0.05`
  - size slippage coeff `1.5 bps`
  - base allocation reference `10%`
- preferred deployment size:
  - `25%` per trade

## Downside Snapshot Of The Best Current Candidate

For the replay-optimized `25%` sleeve:

- max realized drawdown: `$270.08`
- max realized drawdown: `0.25%`
- positive weeks: `26 / 31`
- negative weeks: `5 / 31`
- positive months: `7 / 8`
- negative months: `1 / 8`

Compared to the baseline `25%` sleeve:

- baseline PnL: `$5,183.40`
- baseline max realized drawdown: `$1,399.34` (`1.40%`)
- baseline positive weeks: `18 / 31`
- baseline negative months: `3 / 8`

So the filtered `25%` sleeve is better on both:

- return
- realized downside

## Anti-Overfit Conclusions

The strongest anti-overfit lessons from this work:

- broad parameter sweeps can produce attractive but fragile results
- fixing holdout months and selecting on train only removes many false “improvements”
- more filtering is not automatically better at low size
- the right question is not only “does edge per trade improve?”
- the right question is “does held-out, live-style account PnL improve under the intended sizing?”

This is why the final recommendation differs by size:

- at `10%`, simpler is better
- at `25%`, higher-quality filtering becomes better

## Current Best Interpretation

The repo now supports a defensible claim:

- there is a real but narrow cross-exchange edge
- it is concentrated, not universal
- it survives cost stress and stricter replay assumptions
- it is more attractive as a moderate-size filtered sleeve than as a broad-market strategy

The repo does not support these claims:

- that the edge is universal across the futures universe
- that high leverage is safe
- that the current replay is equivalent to live execution

## Main Remaining Risks

The current best candidate is still a modeled historical replay, not a production system.

Missing realism still includes:

- live fill uncertainty
- order book depth / market impact
- funding / carry transfer under actual execution
- intrabar adverse path risk beyond the one-minute modeled exit
- exchange-specific constraints and operational failure modes

The very smooth realized equity curve should be treated as encouraging, not definitive.

## Historical Stress Test

The frozen replay-optimized `25%` sleeve was also stress-tested under worse execution assumptions.

Result:

- it survives moderate cost deterioration
- it fails under truly harsh execution assumptions

Examples:

- higher fees only (`8 bps`): still positive at `$2,918.30`
- higher fixed slippage (`2 bps` extra): still positive at `$5,252.21`
- higher variable slippage: still positive at `$6,140.96`
- higher size slippage (`2.0 bps`): still positive at `$5,843.90`
- harsh combined stress: negative at `-$2,419.67`
- very harsh combined stress: negative at `-$11,406.67`

So the edge is real enough to survive moderate execution degradation, but not severe degradation.

## Strict Fill Reality Check

A later strict fill replay on the actual accepted trades of the best microstructure-gated sleeve produced a much harsher result:

- all `193` tested fills were mechanically fillable in the replay
- but average strict execution slippage was about `14.48 bps`
- strict-fill average net fell to `-1.6466 bps`
- strict-fill total PnL fell to `-$805.32`

This is the strongest execution realism test in the repo so far, and it fails.

So the best current interpretation is:

- the signal itself is still interesting
- the current execution assumptions are too optimistic
- the strategy is not cleared for production-style deployment yet

## Best Next Step

The most useful next step is no longer more in-sample tuning.

The next step should be:

- freeze the current `25%` filtered spec
- build a forward-only daily paper runner
- append only new days
- judge future performance without reusing the same historical sample

That is the correct transition from research to validation.
