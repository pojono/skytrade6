# Market Structure Research Plan

## Goal

Build a strategy family that exploits market-wide structure only if it survives explicit fees and remains broad, simple, and repeatable.

The bar is not a good-looking backtest. The bar is:

- broad symbol participation
- stable behavior across dates
- edge that survives realistic maker and taker costs
- mechanics simple enough to execute consistently

## Core Questions

1. How often do many coins move in the same direction?
2. How synchronized are short-horizon returns across the universe?
3. Does strong breadth predict forward market continuation?
4. Does cross-sectional momentum outperform raw market timing?
5. Does market breadth improve cross-sectional momentum as a regime filter?

## Phase 1: Universe Qualification

Use `universe_scan.py` to identify symbols that:

- exist on both Binance and Bybit
- have enough overlapping daily kline files
- have at least baseline derivative metadata available for later filtering

Core requirement:

- Binance: `_kline_1m.csv`
- Bybit: `_kline_1m.csv`

Secondary fields for later phases:

- Binance: `_metrics.csv`
- Bybit: `_funding_rate.csv`, `_open_interest_5min.csv`, `_long_short_ratio_5min.csv`

## Phase 2: Market Structure Measurement

Use `analyze_market_structure.py` to measure:

- pairwise 1-minute return correlation
- breadth distribution by horizon
- how often 70%+ or 80%+ of the universe moves together
- forward equal-weight market returns after strong positive or negative breadth

This phase answers whether a market factor is large enough to be worth trading.

## Phase 3: Strategy Family Tests

Evaluate three simple signal families at multiple horizons:

1. Breadth trend
   - trade the equal-weight market in the direction of extreme breadth
2. Cross-sectional momentum
   - long the strongest symbols, short the weakest symbols
3. Breadth-gated cross-sectional momentum
   - run cross-sectional momentum only when breadth is already extreme

Each family must be evaluated gross and net under:

- maker round-trip assumptions
- taker round-trip assumptions

## Validation Rules

No strategy is credible unless it passes all of these:

1. Positive net expectancy after explicit fees
2. Non-trivial sample size
3. Broad contribution across many timestamps
4. Stability across nearby horizons and parameter shifts
5. No dependence on one or two extreme outliers

## Next Phase If Raw Edge Exists

If Phase 3 finds a viable baseline, the next additions should be:

1. Add Binance `metrics.csv` filters:
   - open interest expansion
   - taker long/short volume imbalance
   - long/short ratio shifts
2. Add Bybit cross-checks:
   - funding
   - open interest
   - long/short ratio
3. Add walk-forward splits by month
4. Add execution realism:
   - maker-only fill assumptions
   - taker-only stress
   - holding period slippage sensitivity
