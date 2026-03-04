# Candidate Strategy: Clustered Cross-Exchange Reversion

## Status

This is the first strategy candidate that remains net positive after explicit costs under repeated tests.

It is **not** a universal all-symbol strategy.
It is a **clustered basket strategy** restricted to a narrow set of symbols.

## Trading Universe

Original validated basket:

1. `CRVUSDT`
2. `GALAUSDT`
3. `SEIUSDT`
4. `FILUSDT`
5. `XPLUSDT`

Refined flat-slippage basket:

1. `CRVUSDT`
2. `GALAUSDT`
3. `SEIUSDT`
4. `FILUSDT`

This refined basket is stored in `codex-exp-1/out/candidate_basket_v2.txt`.

Current best execution-aware basket:

1. `CRVUSDT`
2. `GALAUSDT`
3. `SEIUSDT`

This basket is stored in `codex-exp-1/out/candidate_basket_v3.txt`.

## Entry Logic

At 1-minute frequency:

1. Compute cross-exchange close spread:
   - `spread_bps = 10000 * (binance_close / bybit_close - 1)`
2. Only consider entries when:
   - `abs(spread_bps) >= 10`
3. Define the expensive venue by the sign of `spread_bps`
4. Enter a one-bar reversion trade:
   - Binance rich: short Binance / long Bybit
   - Binance cheap: long Binance / short Bybit

## Confirmation Filters

Use the state from the entry bar:

1. Cross-venue long/short divergence:
   - `signed_ls_diff >= 0.15`
2. Cross-venue open-interest change divergence:
   - `signed_oi_diff_bps >= 5`
3. Cross-venue carry divergence:
   - `signed_carry_diff_bps >= 2`

Definitions:

- `signed_ls_diff` compares Binance long/short ratio vs Bybit long/short ratio, signed toward the expensive venue
- `signed_oi_diff_bps` compares Binance OI change vs Bybit OI change, signed toward the expensive venue
- `signed_carry_diff_bps` compares:
  - Binance `mark/index` basis
  - Bybit premium index close
  signed toward the expensive venue

Interpretation:

Fade the spread only when the expensive venue also looks more crowded and richer on carry.

## Holding Period

- Hold for 1 bar
- Exit on the next synchronized 1-minute close

## Tested Performance

Validation window (intermediate validation):

- recent history used: `90` overlap days
- out-of-sample: last `30` days

Weighted basket result across the 5 symbols:

### Base Cost: 6 bps round trip

- train avg net: `5.0758 bps` on `14,574` signals
- test avg net: `7.5168 bps` on `13,060` signals

### Harsher Cost: 8 bps round trip

- train avg net: `3.0758 bps`
- test avg net: `5.5168 bps`

### Harsher Cost: 10 bps round trip

- train avg net: `1.0758 bps`
- test avg net: `3.5168 bps`

### Stricter Trigger: 12 bps spread, 8 bps cost

- train avg net: `4.1600 bps` on `11,843` signals
- test avg net: `7.1200 bps` on `10,726` signals

## Stronger Walk-Forward Validation

Fixed basket used:

- `CRVUSDT`, `GALAUSDT`, `SEIUSDT`, `FILUSDT`

Walk-forward setup:

- recent history used: `210` days
- test months: last `2`
- daily cap: `3` trades per symbol

Flat slippage result:

- train avg net: `5.5574 bps` on `1,448` trades
- test avg net: `7.9771 bps` on `349` trades
- positive months: `8/8`

With an extra `2 bps` flat slippage:

- train avg net: `3.5574 bps`
- test avg net: `5.9771 bps`
- positive months: `8/8`

## Dynamic Slippage Stress

Moderate dynamic slippage:

- fixed extra: `1 bps`
- spread coefficient: `0.10`
- velocity coefficient: `0.05`

Result:

- train avg net: `0.5768 bps`
- test avg net: `5.1459 bps`
- positive months: `7/8`

Harsh dynamic slippage:

- fixed extra: `2 bps`
- spread coefficient: `0.20`
- velocity coefficient: `0.10`

Result:

- train avg net: `-4.4038 bps`
- test avg net: `2.3148 bps`
- positive months: `6/8`

This means the strategy survives moderate execution stress but does not survive harsh dynamic slippage assumptions.

## Best Current Version

After removing `FILUSDT`, the 3-symbol basket produced the cleanest walk-forward result.

Setup:

- symbols: `CRVUSDT`, `GALAUSDT`, `SEIUSDT`
- recent history: `210` days
- test months: last `2`
- daily cap: `3`

Flat result:

- train avg net: `4.7873 bps` on `1,163` trades
- test avg net: `10.7735 bps` on `259` trades
- positive months: `8/8`

Moderate dynamic slippage:

- fixed extra: `1 bps`
- spread coefficient: `0.10`
- velocity coefficient: `0.05`

Result:

- train avg net: `2.7280 bps`
- test avg net: `7.7022 bps`
- positive months: `8/8`

These results use the corrected no-peek daily-cap logic.

This is the strongest current implementation candidate.

## Implementation Snapshot

Timestamped trade export:

- `codex-exp-1/out/candidate_trades_v3.csv`

Simple paper simulation with:

- starting capital: `$100,000`
- allocation: `10%` per trade
- daily cap: `3` per symbol
- moderate dynamic slippage

Results with `3` open slots:

- filled trades: `1,422`
- final capital: `$105,302.27`
- total PnL: `$5,302.27`
- average net edge: `3.6340 bps`
- win rate: `58.72%`

Results with `1` open slot:

- filled trades: `1,421`
- final capital: `$105,356.75`
- total PnL: `$5,356.75`
- average net edge: `3.6729 bps`

Monthly paper log:

- `codex-exp-1/out/paper_monthly_v3_slot1.csv`

This suggests a conservative single-position implementation is viable and may be preferable.

## Preferred Live-Style Selector

The paper simulator now supports explicit same-timestamp selector modes.

In the conservative single-slot setup:

- `score` selector:
  - final capital: `$105,356.75`
  - average net edge: `3.6729 bps`
- `spread` selector:
  - final capital: `$105,358.62`
  - average net edge: `3.6742 bps`

The difference is small, but `spread` is currently the preferred selector.

Recommended execution controls:

- max open positions: `1`
- max open per symbol: `1`
- max symbol allocation: `10%`
- daily cap: `3` per symbol
- daily loss stop: `1%`
- monthly loss stop: `3%`

With these risk rails enabled, the current historical paper result remains:

- final capital: `$105,358.62`
- total PnL: `$5,358.62`
- average net edge: `3.6742 bps`

This means the chosen stop thresholds act as safety constraints rather than active performance drivers in the tested sample.

## Capital Efficiency And Leverage

The conservative setup underuses capital because:

- trades last only 1 minute
- only 1 position is open
- only 10% of capital is deployed per trade

So the strategy has a positive edge but low clock-time utilization.

Using the same frozen 3-symbol strategy with the same assumptions, only scaling position size:

- `10%` allocation:
  - total return: `5.36%`
  - rough annualized return: `9.54%`
- `25%` allocation:
  - total return: `13.94%`
  - rough annualized return: `25.59%`
- `50%` allocation:
  - total return: `29.80%`
  - rough annualized return: `57.69%`
- `100%` allocation:
  - total return: `67.41%`
  - rough annualized return: `145.93%`
- `200%` notional:
  - total return: `184.30%`
  - rough annualized return: `520.15%`

This shows leverage can help a lot in the model, but these larger-size results are optimistic because the simulator still does not fully scale slippage, funding, and margin risk with size.

For practical next steps, `25%` to `50%` allocation are the most reasonable sizing bands to validate further.

## Size-Aware Leverage Re-Test

Once slippage is made size-dependent, the leverage picture changes materially.

Using:

- size slippage coefficient: `1.5 bps`
- base allocation reference: `10%`

Results:

- `25%` allocation:
  - final capital: `$105,183.40`
  - total PnL: `$5,183.40`
  - average net edge: `1.4242 bps`
- `50%` allocation:
  - final capital: `$85,976.89`
  - total PnL: `-$14,023.11`
  - average net edge: `-2.3537 bps`

So the realistic sizing conclusion is:

- `10%` remains the safe baseline
- `25%` is still viable but materially tighter
- `50%` is too large under the current size-aware slippage model

This is the current practical sizing range for the strategy.

## What This Means

Current evidence supports:

- the edge is real enough to survive explicit cost stress
- the edge is concentrated in a specific symbol cluster
- the confirmation filters matter
- the best current version uses trade caps
- the best current version is robust under flat and moderate slippage, not under harsh slippage

Current evidence does **not** support:

- trading the full symbol universe
- assuming the same rule works on every coin

## Optional Pre-Trade Meta Filter

Phase 9 adds an optional quality-first filter before the live-style paper engine:

```json
{
  "max_velocity": 12.0,
  "min_carry": 2.0,
  "min_ls": 0.15,
  "min_oi": 5.0,
  "min_score": 8.0,
  "min_spread_abs": 12.0,
  "sei_score_extra": 10.0
}
```

At the same conservative `10%` sizing and existing live controls, this changes the replay from:

- baseline: `1,421` fills, `3.6742 bps` avg edge, `58.69%` win rate, `$5,358.62` PnL

to:

- filtered: `956` fills, `4.8466 bps` avg edge, `59.73%` win rate, `$4,741.42` PnL

So the meta-filter improves trade quality and slightly improves win rate, but reduces turnover enough that total dollars fall at the current low-utilization deployment.

## Anti-Overfit Status

Phase 10 tested three stricter pre-trade selection ideas on a fixed holdout (`2026-02` and `2026-03`) without selecting on the test period:

- train-selected 2-symbol sleeve (`CRVUSDT + GALAUSDT`)
- simple probability ranker
- compact replay-PnL threshold search

Result:

- none of the stricter variants beat the plain 3-symbol baseline on held-out total dollars at `10%` sizing
- stricter variants did improve held-out win rate and per-trade edge

Best held-out baseline:

- `259` fills
- `60.62%` win rate
- `7.8476 bps`
- `$2,052.88`

Best held-out quality tradeoff (compact train-only replay filter):

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

Held-out result for that variant:

- `213` fills
- `62.44%` win rate
- `9.2923 bps`
- `$1,998.51`

So the plain baseline remains the best dollar generator at current conservative sizing, while the replay-optimized filter is the cleaner high-confidence variant.

## Size-Aware Deployment Split

Phase 11 compared the plain baseline and the replay-optimized filter under the same size-aware slippage model (`size_slip_coeff = 1.5`, `base_allocation_ref = 10%`).

At `10%` allocation:

- baseline: `1,421` fills, `58.69%` win rate, `3.6742 bps`, `$5,358.62`
- replay-optimized filter: `897` fills, `61.54%` win rate, `5.5349 bps`, `$5,089.18`

At `10%`, the baseline still wins on absolute dollars.

At `25%` allocation:

- baseline: `1,421` fills, `50.32%` win rate, `1.4242 bps`, `$5,183.40`
- replay-optimized filter: `897` fills, `53.29%` win rate, `3.2849 bps`, `$7,638.99`

At `25%`, the replay-optimized filter clearly wins because the higher-quality signals degrade more slowly under size-aware slippage.

That changes the practical recommendation:

- If staying near `10%` allocation, the plain baseline remains the best dollar generator.
- If scaling toward `25%` allocation, the replay-optimized filter is the better candidate.

## Downside Snapshot For The 25% Filtered Variant

Phase 12 analyzed the frozen replay-optimized `25%` variant directly from its fill stream.

From [downside_report_v3_replayopt_sized25.md](/home/ubuntu/Projects/skytrade6/codex-exp-1/out/downside_report_v3_replayopt_sized25.md):

- max realized drawdown: `$270.08`
- max realized drawdown percent: `0.25%`
- weeks observed: `31`
- positive weeks: `26`
- negative weeks: `5`
- months observed: `8`
- positive months: `7`
- negative months: `1`

This is a strong realized-equity stability profile in the current sample, but it should be read as model-based realized PnL stability, not as a full live-trading worst-case path estimate.

## Next Validation Before Production

1. Add a more realistic fill model tied to venue liquidity, if order book data is used later
2. Extend rolling paper-trading logs into a continuous forward process fed by new data
3. Add a simple risk-stop layer at the portfolio level
4. Freeze the basket and re-test without reselecting symbols
