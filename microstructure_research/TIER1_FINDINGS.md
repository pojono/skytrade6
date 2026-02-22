# Tier 1 Feature-Target Scan: Findings

**Date**: 2025-02-22
**Symbols**: DOGEUSDT, SOLUSDT
**Timeframe**: 4h
**Period**: 1 year (2025-01-01 to 2026-01-01)
**Candles per symbol**: 2,196
**Features**: 763 (418 raw + 345 z-score)
**Targets**: 120 (63 continuous + 57 binary)

---

## Methodology

Three metrics computed for every (feature, target) pair:

1. **Spearman rank correlation** — for 63 continuous targets (ret, cum_ret, sharpe, etc.)
2. **AUC (Area Under ROC)** — for 57 binary targets (profitable_long/short, direction, etc.)
3. **Mutual Information** — for 11 key targets (captures nonlinear relationships)

Total: ~91,560 pairs per symbol.

---

## Summary Statistics

| Metric | DOGE | SOL |
|---|---|---|
| Pairs with \|Spearman\| > 0.02 | 2,327 | 2,130 |
| Pairs with \|Spearman\| > 0.05 | 386 | 409 |
| Pairs with \|Spearman\| > 0.10 | 34 | 34 |
| Pairs with AUC > 0.52 or < 0.48 | 583 | 418 |
| Pairs with AUC > 0.53 or < 0.47 | 202 | 110 |
| Pairs with AUC > 0.55 or < 0.45 | 0 | 2 |

Signal strength is weak individually (max |Spearman| ~0.16, max AUC ~0.555) but this is
typical for financial data. Value comes from combining features and from temporal stability.

---

## Cross-Symbol Consistent Features

The most important finding: features that show the **same sign and significance on both
DOGE and SOL** are likely real signals, not noise.

### Spearman: Continuous Targets

#### tgt_cum_ret_10 (strongest signals)

| Feature | DOGE r | SOL r | avg\|r\| | Interpretation |
|---|---|---|---|---|
| twap | -0.124 | -0.116 | 0.120 | Mean-reversion: high price → lower returns |
| close | -0.124 | -0.116 | 0.120 | Same pattern |
| low | -0.126 | -0.112 | 0.119 | Same pattern |
| vwap / vwap_buy / vwap_sell | -0.124 | -0.116 | 0.120 | Same pattern |
| fair_value / fair_price | -0.125 | -0.116 | 0.120 | Same pattern |
| poc_price | -0.125 | -0.116 | 0.120 | Same pattern |
| fvg_bearish_count_10 | +0.131 | +0.080 | 0.105 | Bearish gaps → bullish reversal |
| avg_buy_size | +0.089 | +0.110 | 0.100 | Large buyers → bullish |
| graph_edge_count_z | +0.135 | +0.056 | 0.095 | Network complexity → returns |

#### tgt_ret_5 (medium-term returns)

| Feature | DOGE r | SOL r | avg\|r\| | Interpretation |
|---|---|---|---|---|
| close_vs_prev_fair_value_bps | -0.075 | -0.043 | 0.059 | Overextension → reversal |
| fair_value_change_bps | -0.074 | -0.039 | 0.056 | Same pattern |
| vol_asymmetry | +0.067 | +0.044 | 0.055 | Volume asymmetry → direction |
| drawdown_drawup_asymmetry | +0.069 | +0.037 | 0.053 | Asymmetric moves → continuation |

#### tgt_ret_1 (short-term returns)

| Feature | DOGE r | SOL r | avg\|r\| | Interpretation |
|---|---|---|---|---|
| ew_impulse_quality | -0.075 | -0.036 | 0.056 | Impulse exhaustion → reversal |
| vwap_spread_bps_z | +0.068 | +0.043 | 0.055 | Spread deviation → direction |
| busiest_quartile | -0.073 | -0.034 | 0.054 | Activity timing → reversal |

### AUC: Binary Targets

#### tgt_profitable_long_3 / tgt_profitable_short_3

| Feature | DOGE AUC (L/S) | SOL AUC (L/S) | Interpretation |
|---|---|---|---|
| session_asia | 0.461 / 0.541 | 0.445 / 0.555 | Asia session favors shorts |
| market_pressure | 0.545 / 0.455 | 0.543 / 0.455 | Order flow → direction |
| absorption_ratio | 0.545 / 0.455 | 0.543 / 0.455 | Same as market_pressure |
| bernoulli | 0.545 / 0.455 | 0.543 / 0.455 | Same cluster |
| hour_cos | 0.459 / 0.540 | 0.461 / 0.538 | Time-of-day cycle |
| avg_sell_size | 0.530 / 0.471 | 0.542 / 0.456 | Large sellers → bearish |
| avg_trade_size | 0.532 / 0.468 | 0.538 / 0.459 | Institutional activity |

#### tgt_profitable_long_5 / tgt_profitable_short_5

| Feature | DOGE AUC (L/S) | SOL AUC (L/S) | Interpretation |
|---|---|---|---|
| avg_buy_size | 0.530 / 0.467 | 0.544 / 0.455 | Large buyers → bullish |
| golden_ratio_half_dist_z | 0.545 / 0.457 | 0.529 / 0.470 | Fibonacci structure |
| avg_trade_size | 0.526 / 0.470 | 0.545 / 0.454 | Institutional activity |
| market_pressure | 0.536 / 0.462 | 0.533 / 0.466 | Order flow |
| twap | 0.470 / 0.533 | 0.462 / 0.536 | Price level (inverted) |

### Mutual Information: Nonlinear Signals

Top MI features consistent across both symbols:

| Feature | Best Target | DOGE MI | SOL MI | Notes |
|---|---|---|---|---|
| fib_wave_avg_dist_z | profitable_long_5 | 0.089 | 0.052 | Nonlinear Fibonacci signal |
| fvg_bearish_size_bps_z | profitable_short_5 | 0.060 | 0.069 | FVG size predicts direction |
| fvg_bullish_size_bps_z | profitable_long_5 | 0.086 | 0.071 | Same pattern, opposite side |
| ew_correction_quality_z | profitable_long_5 | 0.039 | 0.046 | Elliott wave quality |
| consecutive_direction_z | profitable_short_5 | 0.043 | 0.026 | Momentum persistence |
| price_clustering_z | profitable_long_3 | — | 0.040 | Stronger on SOL |

---

## Feature Groups Identified

### Group 1: Price Level Mean-Reversion (strongest, most consistent)
- `twap`, `close`, `low`, `vwap`, `fair_value`, `poc_price`, etc.
- Negative Spearman for all return/Sharpe targets
- Consistent on both DOGE and SOL
- Interpretation: high absolute price → lower future returns (mean-reversion)
- **Caveat**: these are raw price levels, may be regime-dependent

### Group 2: Order Flow / Market Pressure
- `market_pressure`, `absorption_ratio`, `bernoulli`
- AUC ~0.545 for profitable_long, ~0.455 for profitable_short
- Nearly identical on both coins
- Interpretation: buy-side pressure predicts profitable longs

### Group 3: Trade Size (Institutional Footprint)
- `avg_buy_size`, `avg_sell_size`, `avg_trade_size`
- Consistent AUC signal on both coins
- Spearman +0.09 to +0.11 for cum_ret_10
- Interpretation: larger average trade size → institutional presence → directional signal

### Group 4: Time-of-Day / Session
- `session_asia`, `hour_cos`, `hour_sin`
- Strongest AUC signals (session_asia: 0.555 on SOL for profitable_short_3)
- Highly consistent cross-symbol
- Interpretation: Asia session has systematic short bias

### Group 5: Fair Value Gap (FVG)
- `fvg_bearish_count_10`, `fvg_bullish_count_10_z`
- `fvg_bearish_size_bps_z`, `fvg_bullish_size_bps_z`
- Strong Spearman for longer horizons, strong MI for binary targets
- Interpretation: unfilled gaps create future price magnets

### Group 6: Fibonacci / Golden Ratio
- `golden_ratio_half_dist`, `golden_ratio_half_dist_z`, `fib_wave_avg_dist_z`
- AUC ~0.547 for profitable_long_5 (DOGE)
- Top MI feature for binary targets on both coins
- Interpretation: price near Fibonacci levels has directional bias

### Group 7: Microstructure (short-term only)
- `ew_impulse_quality`, `vwap_spread_bps_z`, `busiest_quartile`
- Only significant for tgt_ret_1 (next candle)
- Interpretation: intra-candle microstructure predicts immediate next move

---

## Key Observations

1. **No single feature exceeds AUC 0.555** — individual features are weak. Strategy must combine multiple features.

2. **Short-term vs long-term features are completely different**:
   - ret_1: microstructure features (impulse quality, VWAP spread)
   - ret_5: fair value deviation, volume asymmetry
   - cum_ret_10: price levels, graph features, FVG counts

3. **Price levels are the strongest Spearman signal** but may be spurious — raw price in a trending market will always correlate with future returns. The z-score versions are more trustworthy.

4. **Session/time features are the most robust AUC signal** — simple, interpretable, and consistent. Asia session short bias is real.

5. **MI reveals nonlinear structure** that Spearman misses — `fib_wave_avg_dist_z` and `fvg_size_bps_z` have strong MI but weak Spearman, suggesting threshold/nonlinear effects.

6. **Feature redundancy is high** — price levels (twap, close, vwap, etc.) are all >0.95 correlated. Need Tier 4 clustering to reduce.

---

## Recommended Feature Shortlist for Tier 2

Based on cross-symbol consistency and avoiding redundancy:

1. `twap` (representative of price level group)
2. `market_pressure` (representative of order flow group)
3. `avg_buy_size` (institutional footprint)
4. `session_asia` (time-of-day)
5. `hour_cos` (time-of-day cycle)
6. `fvg_bearish_count_10` (fair value gaps)
7. `fvg_bearish_size_bps_z` (FVG size, nonlinear)
8. `golden_ratio_half_dist_z` (Fibonacci structure)
9. `fib_wave_avg_dist_z` (Fibonacci, top MI)
10. `ew_impulse_quality` (microstructure, short-term)
11. `vwap_spread_bps_z` (spread deviation)
12. `close_vs_prev_fair_value_bps` (fair value deviation)
13. `vol_asymmetry` (volume asymmetry)
14. `graph_edge_count_z` (network complexity)
15. `consecutive_direction_z` (momentum persistence)

These 15 features cover all 7 groups with minimal redundancy.

---

## Next Steps

- **Tier 2**: Test temporal stability of these 15 features (rolling 30-day windows)
- **Tier 4**: Cluster all 763 features to confirm no important signals were missed
- **Tier 5**: Walk-forward single-feature models on the shortlist
