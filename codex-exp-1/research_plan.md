# Cross-Exchange Research Plan

## Goal

Extract tradeable, repeatable cross-exchange signals across a broad universe rather than optimizing a single-symbol anecdote.

The target is not "find any backtest that looks good." The target is:

- broad symbol coverage
- stable behavior across time
- positive net expectancy after explicit costs
- simple enough mechanics to survive production

## Research Thesis

When the same perp trades simultaneously on Binance and Bybit, short-lived imbalances can emerge due to fragmented liquidity, exchange-specific positioning, and asynchronous information absorption.

That creates three candidate edge families:

1. Cross-exchange spread mean reversion
2. Lead-lag price discovery
3. Cross-venue positioning divergence

This experiment starts with (1), because it is the cleanest to test with the current local files.

## Phase 1: Universe Qualification

Use `universe_scan.py` to identify symbols that:

- exist on both exchanges
- have at least 90 overlapping dates
- have the core files required for aligned analysis

Core files:

- Binance: `_kline_1m.csv`
- Bybit: `_kline_1m.csv`

Secondary files for later phases:

- Binance: `_metrics.csv`
- Bybit: `_funding_rate.csv`, `_open_interest_5min.csv`, `_long_short_ratio_5min.csv`

Deliverables:

- coverage summary CSV
- eligible symbol list

## Phase 2: First Edge Test

Use `cross_exchange_edge_scan.py` to test a simple spread-reversion trade:

- Compute synchronized 1-minute close spread:
  - `spread_bps = 10000 * (binance_close / bybit_close - 1)`
- Trigger only when absolute spread exceeds a threshold
- Direction:
  - Binance rich: short Binance / long Bybit
  - Binance cheap: long Binance / short Bybit
- Hold for one bar and measure spread compression

Per-trade gross PnL proxy:

- `gross_pnl_bps = sign(spread_t) * (spread_t - spread_t+1)`

Then stress it with:

- round-trip fee assumptions
- minimum signal threshold sweeps
- symbol-level ranking
- pooled performance across the universe

The core question:

Does large spread dislocation compress often enough, and by enough magnitude, to clear costs?

## Phase 3: Improve Signal Quality

If Phase 2 shows raw edge but weak net performance, tighten entries using filters:

1. Volatility filter
   - ignore chaotic minutes with unstable microstructure
2. Spread z-score filter
   - use symbol-relative spread normalization
3. Trend filter
   - avoid fading strong directional breaks
4. Freshness filter
   - avoid repeated entries during the same dislocation cluster
5. Liquidity proxy filter
   - prioritize symbols with smaller gap noise and steadier volume

## Phase 4: Positioning Divergence Layer

Use cross-exchange positioning features to filter or bias trades:

- Binance metrics:
  - open interest
  - top-trader long/short ratio
  - overall long/short ratio
  - taker volume ratio
- Bybit:
  - funding rate
  - open interest
  - long/short ratio

Key hypothesis:

If price spread is stretched and positioning is lopsided on the expensive venue, reversion should be stronger.

## Validation Rules

No strategy graduates unless it passes all of these:

1. In-sample and out-of-sample split by date
2. Broad contribution across many symbols, not one or two outliers
3. Positive expectancy after explicit fee deduction
4. Reasonable win rate and left-tail control
5. Stable behavior under threshold perturbation

Red flags:

- PnL concentrated in a single symbol
- profitability disappears with small fee changes
- performance relies on only a few isolated dates
- edge exists only at unrealistic entry frequency

## Immediate Next Tests

After the first scan, the next scripts to add should be:

1. Threshold sweep and walk-forward splits
2. Spread z-score normalization
3. Lead-lag test (which venue leads by 1 to 5 minutes)
4. Positioning-filtered spread reversion
5. Multi-symbol portfolio construction with capital caps

## Data Notes

- The current local datalake already covers 116 common symbols, which is enough for the first pass.
- Bybit has deeper optional bulk data, but Phase 1 and Phase 2 do not require downloading more.
- If coverage gaps matter later, extend the local dataset with the existing `datalake` download scripts.
