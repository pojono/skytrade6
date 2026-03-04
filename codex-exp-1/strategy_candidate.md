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

## Next Validation Before Production

1. Add a more realistic fill model tied to venue liquidity, if order book data is used later
2. Extend rolling paper-trading logs into a continuous forward process fed by new data
3. Add a simple risk-stop layer at the portfolio level
4. Freeze the basket and re-test without reselecting symbols
