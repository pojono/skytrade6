# Feature Analysis Findings: Complete Summary

**Date**: 2025-02-22
**Symbols**: DOGEUSDT, SOLUSDT
**Primary Timeframe**: 4h (also tested 2h, 1h)
**Period**: 1 year (2025-01-01 to 2026-01-01)
**Candles per symbol (4h)**: 2,196

---

## What We Started With

- **763 features** (418 raw microstructure + 345 z-score normalizations)
- **120 targets** (63 continuous + 57 binary)
- Features computed from tick-level data: order flow, trade sizes, volume profile, price structure, Fibonacci, Elliott Wave, spectral analysis, graph theory, entropy, and more
- Targets include: returns at multiple horizons, cumulative returns, Sharpe ratios, drawdowns, PnL after fees, profitable trade indicators, breakout/crash events, regime changes

---

## The Analysis Pipeline

| Tier | Method | Purpose | Features In → Out |
|---|---|---|---|
| **1** | Spearman, AUC, Mutual Information | Find any signal | 763 → ~200 |
| **2** | Rolling 30d windows, regime split | Temporal stability | ~200 → 111 (cross-symbol) |
| **4** | Hierarchical clustering (\|r\|>0.7) | Remove redundancy | 111 → 44 |
| **5** | Walk-forward OOS (Ridge/Logistic) | Real predictive power | 44 → ~15 |

Total computation: ~91,560 pairs per symbol per timeframe (Tier 1), ~6,600 stability checks (Tier 2), 5,280 WFO runs per symbol (Tier 5).

---

## Finding 1: No Single Feature Predicts Returns Out-of-Sample

**This is the most important finding.**

Every continuous target — `tgt_ret_*`, `tgt_cum_ret_*`, `tgt_sharpe_*`, `tgt_pnl_*` — had **negative R²** out-of-sample for every single feature on both coins. Zero exceptions.

The Spearman correlations from Tier 1 (up to |r| = 0.157) were **in-sample artifacts**. They looked promising but did not survive walk-forward validation.

**Implication**: If you want to predict return magnitude, you need multi-feature models. No shortcut exists.

---

## Finding 2: Binary Targets Have Real OOS Signal

The `tgt_profitable_long/short` targets (which incorporate fee costs) showed genuine out-of-sample predictive power. Multiple features achieved positive AUC deviation on both DOGE and SOL simultaneously.

### Top Cross-Symbol OOS Survivors for `tgt_profitable_long_5`

| Feature | DOGE AUC_dev | SOL AUC_dev | DOGE % positive folds | SOL % positive folds |
|---|---|---|---|---|
| close (price level) | +0.147 | +0.148 | 100% | 100% |
| hilbert_std_amplitude | +0.053 | +0.068 | 70% | 100% |
| avg_buy_size | +0.038 | +0.046 | 80% | 80% |
| max_monotonic_run | +0.020 | +0.045 | 60% | 90% |
| fvg_bearish_count_10 | +0.032 | +0.025 | 70% | 70% |
| golden_ratio_half_dist_z | +0.021 | +0.029 | 60% | 70% |
| absorption_ratio | +0.036 | +0.011 | 80% | 60% |
| size_gini_z | +0.025 | +0.025 | 70% | 60% |

### Top Cross-Symbol OOS Survivors for `tgt_profitable_short_5`

| Feature | DOGE AUC_dev | SOL AUC_dev | DOGE % positive folds | SOL % positive folds |
|---|---|---|---|---|
| close (price level) | +0.147 | +0.145 | 100% | 100% |
| hilbert_std_amplitude | +0.031 | +0.067 | 70% | 100% |
| avg_buy_size | +0.039 | +0.049 | 80% | 80% |
| fvg_bearish_count_10 | +0.039 | +0.025 | 70% | 60% |
| max_monotonic_run | +0.021 | +0.042 | 60% | 80% |
| golden_ratio_half_dist_z | +0.023 | +0.033 | 60% | 80% |
| absorption_ratio | +0.039 | +0.012 | 80% | 60% |
| size_gini_z | +0.025 | +0.025 | 70% | 60% |

**Implication**: Single features can predict whether a trade will be profitable (direction after fees), but the signal is weak individually (AUC ~0.52-0.55 excluding price level). The value comes from combining them.

---

## Finding 3: Price Level Is the Strongest Predictor — But It's a Trap

`close` (and its cluster: twap, vwap, fair_value, poc_price, etc.) dominates every analysis:
- Tier 1: Highest Spearman for all return targets
- Tier 2: 100% sign consistency, SNR ~3.0, zero wrong-sign streaks
- Tier 5: AUC_dev +0.15, 100% positive folds on both coins

But this is **momentum/trend following** disguised as a feature. It says: "when price is high (trending up), future profitable longs are more likely." This works in trending markets and fails in mean-reverting ones.

**Recommendation**: Include in models as a regime indicator, but don't rely on it as the primary signal. The microstructure features below are more robust.

---

## Finding 4: The Real Microstructure Signals

These features have genuine, cross-symbol, temporally stable, out-of-sample predictive power:

### Tier A: Strongest (OOS on both coins, multiple targets)

| Feature | What It Measures | Why It Works |
|---|---|---|
| **hilbert_std_amplitude** | Amplitude of price oscillations | High oscillation → directional move coming |
| **avg_buy_size** | Average size of buy trades | Large buyers → institutional presence → direction |
| **absorption_ratio** | Buy absorption of sell pressure | Order flow imbalance → direction |
| **fvg_bearish_count_10** | Fair value gaps in last 10 candles | Unfilled gaps create price magnets |

### Tier B: Moderate (OOS on both coins, fewer targets)

| Feature | What It Measures | Why It Works |
|---|---|---|
| **golden_ratio_half_dist_z** | Distance to Fibonacci levels | Price near key levels has directional bias |
| **max_monotonic_run** | Longest consecutive same-direction candles | Momentum persistence signal |
| **session_asia** | Whether candle is in Asia session | Systematic session bias |
| **hour_cos** | Time-of-day cycle | Intraday patterns |
| **vpin** | Volume-synchronized probability of informed trading | Informed trader detection |
| **size_gini_z** | Inequality of trade sizes | Concentrated vs distributed activity |

### Tier C: Weak but consistent

| Feature | What It Measures |
|---|---|
| **downtick_pct** | Percentage of downtick trades |
| **sell_urgency_ratio** | Urgency of sell-side activity |
| **vol_price_feedback** | Volume-price feedback loop strength |
| **area_above_vwap_pct_z** | Price position relative to VWAP |

---

## Finding 5: 4h Is the Best Timeframe

| Metric | 1h | 2h | 4h |
|---|---|---|---|
| Max \|Spearman\| | 0.079 | 0.089 | **0.157** |
| Cross-symbol features (\|r\|>0.02) | 51 | 63 | **163** |
| AUC pairs > 0.53 | 3 | 31 | **312** |

Signal strength roughly doubles with each timeframe step. 1h signals are barely above noise. 4h provides the best signal-to-noise ratio and lowest fee impact (fewer trades).

However, 1h has unique value for **time-of-day features** (`hour_sin`, `hour_cos`) which lose resolution at 4h (only 6 candles/day).

---

## Finding 6: Most Features Are Noise

| Stage | Features | % of Original |
|---|---|---|
| Starting set | 763 | 100% |
| Any Tier 1 signal | ~200 | 26% |
| Cross-symbol Tier 2 stable | 111 | 15% |
| Non-redundant clusters | 44 | 6% |
| OOS predictive (Tier 5) | ~15 | **2%** |

98% of our engineered features have no real predictive power. This is normal for financial ML — most hypotheses about market structure don't survive rigorous testing.

---

## Finding 7: z² and |z| Transforms Don't Help

We tested squared and absolute z-score transforms to capture U-shaped relationships. Result: **no improvement over raw z-scores**. The relationships are directional (monotonic), not symmetric. Tree-based models can learn thresholds natively; for linear models, the raw z-scores are already optimal.

---

## Finding 8: Feature Redundancy Is Extreme

The 14 price-level features (close, twap, vwap, fair_value, poc_price, value_area_high, value_area_low, etc.) are all >0.95 correlated. Similarly:
- `market_pressure`, `absorption_ratio`, `bernoulli` are essentially the same feature
- `avg_buy_size`, `avg_sell_size`, `avg_trade_size` form a cluster

After clustering at |r|>0.7, 111 features collapsed to 44 groups. This means **most of our feature engineering produced variations of the same ~44 independent signals**.

---

## Recommended Next Steps

### Immediate: Multi-Feature Model

Build a Logistic Regression model combining the ~15 OOS survivors to predict `tgt_profitable_long_5` and `tgt_profitable_short_5`:

**Proposed feature set (excluding price level):**
1. `hilbert_std_amplitude` — oscillation amplitude
2. `avg_buy_size` — institutional footprint
3. `absorption_ratio` — order flow
4. `fvg_bearish_count_10` — fair value gaps
5. `golden_ratio_half_dist_z` — Fibonacci structure
6. `max_monotonic_run` — momentum persistence
7. `session_asia` — time-of-day
8. `hour_cos` — time-of-day cycle
9. `vpin` — informed trading probability
10. `size_gini_z` — trade size inequality
11. `downtick_pct` — sell pressure
12. `vol_price_feedback` — volume-price dynamics

**With price level (optional, adds trend component):**
13. `close` — regime/trend indicator

Expected combined AUC: 0.55-0.60 (modest but potentially tradeable with proper position sizing).

### Medium-term

- Generate features for BTC, ETH, XRP and validate cross-symbol consistency on 5 coins
- Test multi-feature model with walk-forward optimization
- Add position sizing based on model confidence
- Backtest with realistic fee model (maker 0.02%, taker 0.055%)

### Long-term

- Consider multi-timeframe approach: 4h features for direction, 1h for entry timing
- Explore ensemble methods (LightGBM) which can capture nonlinear interactions
- Add regime detection to switch between trend-following and mean-reversion models

---

## Technical Notes

- All analysis used 1 year of tick-level data aggregated to candles
- Walk-forward: 60-day train, 30-day test, 1-day purge gap
- Tier 2 stability: 30-day rolling windows, 15-day step, 23 windows per year
- Clustering: hierarchical (average linkage), Spearman distance
- Cross-symbol validation: feature must show same sign and significance on both DOGE and SOL
- Regime split: high-vol vs low-vol using realized volatility median
- Fee model for profitable targets: maker 0.02% + taker 0.055% = ~4 bps round-trip
