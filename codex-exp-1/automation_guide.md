# Automation Guide For The Cross-Exchange Filtered Sleeve

This guide explains how to automate the current best research candidate in a practical, implementation-oriented way.

It is not a claim that the strategy is production-safe today.
It is a guide for building the automation correctly, with the current research assumptions frozen.

## Strategy To Automate

The strategy to automate is the current best moderate-size candidate from the research:

- symbol universe:
  - `CRVUSDT`
  - `GALAUSDT`
  - `SEIUSDT`
- deployment style:
  - one open position at a time
  - one open position per symbol
  - `25%` capital per trade
- selector:
  - choose the highest spread candidate if multiple symbols fire at the same timestamp
- holding period:
  - one minute / one bar

This is a cross-exchange mean-reversion sleeve between Binance and Bybit.

## Core Idea

At each minute:

1. Compare Binance and Bybit prices for the same symbol.
2. Detect whether one venue is materially rich or cheap relative to the other.
3. Confirm that the spread is large enough and has the right supporting context.
4. Enter a paired position expecting the spread to compress over the next bar.
5. Exit after one minute.

The edge is not “market direction.”
The edge is “temporary cross-exchange dislocation with confirmation.”

## Signal Mechanics

### Raw Spread Signal

For each symbol and minute:

- compute the Binance vs Bybit spread in basis points
- use the absolute spread for trigger strength
- keep the sign to know which venue is rich and which is cheap

Interpretation:

- positive signed spread: Binance is richer than Bybit
- negative signed spread: Bybit is richer than Binance

### Base Trigger

The base spread condition is:

- absolute spread must be at least `14 bps`

This is stricter than the earliest research threshold and reflects the replay-optimized filtered sleeve.

### Confirmation Filters

The trade must also satisfy:

- `min_score >= 6`
- `SEIUSDT` requires an additional `10` score points
- `max_velocity <= 12`
- `min_ls >= 0.15`
- `min_oi >= 5`
- `min_carry >= 2`

Frozen filter:

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

### Optional Microstructure Gate

The best current optional execution-quality gate from the high-resolution study is:

```json
{
  "min_bybit_trade_count_5s": 2.0,
  "min_binance_trade_count_5s": 0.0,
  "max_bybit_book_spread_bps": 4.5,
  "max_bybit_trade_imbalance_5s": 1.0
}
```

Operational meaning:

- require at least modest recent Bybit trade activity
- require a reasonably tight local Bybit book
- do not reject on Binance 5-second trade count in the current best version
- do not require a strongly negative Bybit trade imbalance yet; only reject extreme positive pressure

This should be treated as an optional gate layered after the 1-minute signal and before order submission.

If you want the current strongest implementation path, use this gate.
If you want the simpler frozen base sleeve, omit it.

### What The Confirmation Terms Mean

`score`

- a precomputed composite strength metric already used in the research export
- higher score means stronger alignment across the current signal features

`velocity`

- how fast the spread is changing
- too much velocity suggests unstable dislocation or poor fill conditions

`ls`

- signed long/short divergence between venues
- used as a positioning stretch confirmation

`oi`

- signed open-interest divergence in basis points
- confirms positioning imbalance is moving in the same direction as the spread

`carry`

- dense carry proxy, not sparse funding prints
- derived from minute-level proxies:
  - Binance: `mark / index - 1`
  - Bybit: premium index minute values

## Trade Direction Logic

When the spread fires:

- short the rich venue
- long the cheap venue

Examples:

- if Binance is rich vs Bybit:
  - short Binance
  - long Bybit
- if Bybit is rich vs Binance:
  - short Bybit
  - long Binance

This is a market-neutral relative-value trade, not a directional bet.

## Entry Mechanics

At each signal timestamp:

1. Build the candidate list across `CRVUSDT`, `GALAUSDT`, `SEIUSDT`.
2. Keep only candidates passing the frozen filter.
3. If using the optional microstructure gate:
   - compute the 5-second microstructure features at the decision timestamp
   - reject candidates that fail the gate
4. If multiple candidates pass in the same minute:
   - rank by spread magnitude
   - take the highest-spread candidate only
5. Open the paired trade.

In practice, the microstructure gate is applied after the 1-minute setup is known but before any order is sent.
It is a final execution-quality veto, not a primary signal generator.

Because the strategy is single-slot:

- if a position is already open, skip new entries

Because the strategy is capped by symbol:

- no more than one open trade per symbol

Because the strategy is capped by day:

- no more than `3` trades per symbol per day

## Exit Mechanics

The research assumption is simple and frozen:

- exit after one minute

Operationally, that means:

- track the exact entry timestamp
- schedule the exit for the next completed 1-minute bar
- close both legs together

In production, the two critical requirements are:

- exit symmetry: both legs must be closed as one logical action
- hedge integrity: do not leave one venue open if the other leg fails

## Position Sizing Mechanics

### Capital Allocation

Use:

- `25%` of current equity per trade

Since this is a paired trade:

- treat the trade as one logical unit
- the notional should be balanced across both venues

Practical interpretation:

- if account equity is `$100,000`
- allocate `$25,000` notional to the trade
- split that across the two legs according to hedge-neutral sizing

### Hedge Sizing

The simplest production sizing should be:

- equal USD notional on both venues

If fees, contract multipliers, or tick sizes differ:

- round to the nearest valid contract size
- prefer slight under-hedging over over-sizing one leg

The system must account for:

- contract step size
- minimum notional
- tick size
- leverage and margin mode per venue

## Cost Model To Respect

The research assumes these effective costs:

- round-trip fee: `6 bps`
- fixed extra slippage: `1 bp`
- spread slippage coefficient: `0.10`
- velocity slippage coefficient: `0.05`
- size slippage coefficient: `1.5 bps`
- base allocation reference: `10%`

Interpretation:

- the larger the signal stretch, the more slippage is assumed
- the faster the spread is moving, the more slippage is assumed
- size above the `10%` baseline is penalized further

In production, you should not literally hardcode these as “fake fees.”
You should use them as:

- the minimum edge hurdle for allowing a trade
- a benchmark for whether live fills are good enough

## Risk Controls

The frozen risk rails are:

- daily loss stop: `1%`
- monthly loss stop: `3%`

Operational meaning:

- if realized PnL for the day reaches `-1%` of day-start equity:
  - stop opening new trades for that day
- if realized PnL for the month reaches `-3%` of month-start equity:
  - stop opening new trades for the rest of that month

Existing open positions should still be managed to exit safely.

## Operational Architecture

A practical automation stack should have five components.

### 1. Market Data Collector

Responsibilities:

- subscribe to or poll 1-minute relevant data from Binance and Bybit
- build synchronized minute snapshots per symbol
- compute:
  - price spread
  - spread velocity
  - score inputs
  - long/short divergence
  - open-interest divergence
  - carry divergence
- if using the optional microstructure gate, also maintain rolling 5-second state:
  - Bybit recent trade count
  - Binance recent trade count
  - Bybit top-of-book spread
  - Bybit short-horizon trade imbalance

Requirements:

- strict timestamp normalization
- per-venue clock sanity checks
- missing-bar detection

### 2. Signal Engine

Responsibilities:

- apply the frozen filter
- if enabled, apply the optional microstructure gate
- build candidate trades at each minute
- rank simultaneous candidates by spread
- emit at most one approved trade signal per timestamp

Requirements:

- deterministic logic
- no future leakage
- explicit logging for every accepted and rejected candidate

### 3. Execution Engine

Responsibilities:

- place both legs
- confirm fills
- handle partial fills
- hedge failures immediately
- schedule the one-minute exit
- close both legs

Requirements:

- exchange adapters for both venues
- idempotent order submission
- retry rules
- kill-switch behavior

### 4. Risk Engine

Responsibilities:

- enforce single-slot constraint
- enforce per-symbol daily caps
- enforce daily and monthly stop rules
- reject entries if exposure or operational state is invalid

Requirements:

- position reconciliation against exchange state
- independent calculation of realized PnL
- no reliance on exchange UI or manual tracking

### 5. Monitoring And Logging

Responsibilities:

- record every signal, order, fill, skip, reject, and stop trigger
- compare actual live fills to modeled expectations
- track realized spread capture
- flag when live execution quality drifts beyond historical assumptions

Requirements:

- durable logs
- alerting on hedge mismatch and API failure
- daily summary and month-to-date summary

## Live Minute-By-Minute Loop

A safe automation loop should work like this:

1. Wait for the current minute to complete.
2. Build synchronized per-symbol feature rows.
3. Generate all valid 1-minute candidates.
4. Drop candidates blocked by:
   - open position already active
   - per-symbol daily cap
   - daily stop
   - monthly stop
5. If the optional microstructure gate is enabled:
   - compute recent 5-second trade counts
   - compute Bybit local book spread
   - compute Bybit short-horizon trade imbalance
   - reject candidates that fail the microstructure gate
6. Rank remaining candidates by spread.
7. Submit both legs for the top candidate.
8. Verify hedge fill state.
9. Hold exactly one minute.
10. Exit both legs.
11. Update realized PnL and risk state.
12. Append a permanent log row.

## Critical Failure Cases

This strategy is operationally fragile in a few specific ways.

### Partial Fill On One Venue

If one leg fills and the other does not:

- immediately reduce or cancel the unmatched leg
- do not keep a naked directional position

### Stale Data

If one venue’s minute snapshot is stale:

- do not trade

The edge depends on synchronized comparison.

### API Failure During Exit

If the planned one-minute exit fails:

- switch to emergency close logic
- prioritize flattening risk over price quality

### Symbol Suspension / Contract Change

If contract metadata changes:

- disable that symbol until validated

## Logging Schema You Should Keep

For every signal decision, log:

- timestamp
- symbol
- signed spread
- absolute spread
- score
- velocity
- ls
- oi
- carry
- if using the optional microstructure gate:
  - bybit_trade_count_5s
  - binance_trade_count_5s
  - bybit_book_spread_bps
  - bybit_trade_imbalance_5s
- accepted / rejected
- rejection reason

For every trade, log:

- entry timestamp
- exit timestamp
- venue sides
- intended notional
- actual fills
- realized PnL
- realized slippage vs modeled expectation

Without this, you cannot tell whether the edge failed or the execution failed.

## Deployment Sequence

Do not jump directly from research to full automation.

Use this sequence:

1. Build a real-time signal-only service first.
   - No orders
   - Just print candidate signals and reasons

2. Build shadow execution next.
   - Simulate orders using live market snapshots
   - Compare expected vs likely fills

3. Run forward-only paper trading.
   - Keep using the frozen rules
   - Do not retune on the fly

4. Run tiny-capital live execution.
   - Treat this as execution validation
   - Not as a profit-maximization phase

5. Only then consider scaling.
   - Scale only if live fills resemble the moderate-stress historical assumptions

## What Must Not Change Without Re-Validation

If you change any of these, the research result is no longer the same strategy:

- symbol universe
- spread threshold
- filter thresholds
- holding period
- selector logic
- allocation size
- slippage assumptions
- stop rules
- whether the optional microstructure gate is enabled
- the microstructure gate thresholds

Any such change requires a new validation pass.

## Best Practical Starting Point

If you want to automate this now, the best first build is:

- one service that computes minute signals
- one service that records forward-only paper trades
- one service that compares modeled vs actual executable prices

That is enough to answer the only question that matters next:

- does the edge survive in genuinely new data with real execution constraints?

## Bottom Line

This strategy is automatable.

The mechanics are simple enough:

- synchronized minute features
- deterministic filter
- single-slot paired execution
- one-minute exits
- strict risk caps

The hard part is not the signal formula.
The hard part is:

- reliable two-venue execution
- hedge integrity
- proving live fills stay inside the moderate-stress envelope

That is where the automation effort should focus.
