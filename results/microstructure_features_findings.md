# Microstructure Features: Tick-to-Candle Aggregation

**Date:** 2026-02-21
**Symbol:** BTCUSDT
**Period:** 2026-01-01 → 2026-01-31 (31 days, ~48M trades)
**Script:** `microstructure_features.py`

## Overview

Computed **87 microstructure features** from tick-level Bybit futures trades, aggregated into 5 timeframes (15m, 30m, 1h, 2h, 4h). Ran univariate correlation analysis against forward 1-candle returns.

## Key Findings

### 1. Shorter timeframes have more statistically significant features

- **15m**: 12 features with p < 0.05 (strongest: `order_flow_imbalance` ρ=-0.0614***)
- **30m**: 5 features with p < 0.05
- **1h**: 1 feature with p < 0.05
- **2h**: 2 features with p < 0.05
- **4h**: 1 feature with p < 0.05 (but larger magnitude: `max_trades_per_second` ρ=0.168*)

### 2. Most consistent predictive features across timeframes

| Feature | Direction | Interpretation |
|---|---|---|
| **order_flow_imbalance** | negative | Heavy buying → next candle tends to reverse (mean-reversion) |
| **close_to_vwap** | negative | Price above VWAP → next candle tends to pull back |
| **second_half_return** | negative | Late-candle momentum → reversal next candle |
| **uptick/downtick_pct** | negative/positive | Tick direction imbalance → mean-reversion |
| **buy_sell_trade_ratio** | negative | More buy trades → reversal |
| **cvd_close_vs_range** | negative | CVD ending near high of range → reversal |

### 3. Higher timeframes show different character

At **2h/4h**, the strongest features shift to:
- **buy_time_std** (2h, ρ=-0.124*): Concentrated buy timing → reversal
- **max_trades_per_second** (4h, ρ=0.168*): Peak burst activity → continuation (!)
- **kyle_lambda** (4h, ρ=0.129): Higher price impact → continuation
- **return_autocorr_1** (4h, ρ=-0.134): Negative tick autocorrelation → reversal
- **volatility features** (4h): Higher realized vol, down_vol, roll_spread → continuation

### 4. Mean-reversion dominates at short timeframes

At 15m/30m, almost all significant features point to **mean-reversion**: aggressive buying/selling in the current candle predicts a pullback in the next candle. This is consistent with market microstructure theory (price impact is temporary).

### 5. At 4h, volatility/activity features suggest continuation

Higher timeframes show a different regime: elevated activity (burst trades, volatility, price impact) predicts **continuation**, not reversal. This suggests that large moves at 4h scale are driven by genuine information flow, not temporary impact.

## Feature Categories (87 total)

1. **Basic OHLCV** (9): open, high, low, close, range, return, total_volume, total_notional, total_trades
2. **Volume by side** (9): buy/sell volume, delta, ratios, OFI, trade imbalance
3. **Trade size stats** (10): avg/median/max/std size, large trade metrics, size imbalance
4. **VWAP/TWAP** (7): vwap, twap, buy/sell vwap, spreads
5. **Time distribution** (12): time-weighted position, time at high/low, volume quartiles, buy/sell time concentration, time to high/low
6. **Trade arrival** (7): trades/sec, inter-trade stats, burst rate, volume/sec std
7. **Consecutive runs** (4): side switches, max buy/sell runs
8. **CVD intra-candle** (5): final, max, min, range, close vs range
9. **Volatility** (9): realized vol, skew, kurtosis, autocorr, up/down vol, GK, Parkinson
10. **Price path** (3): path length, efficiency ratio, path/range
11. **Tick direction** (3): uptick/downtick pct, net direction
12. **Liquidity** (4): Kyle's lambda, Amihud, Roll spread, avg notional
13. **Size distribution** (5): volume skew/kurtosis, size percentiles
14. **Intra-candle momentum** (6): first/second half return, reversal, OFI shift

## Z-Score Features (added 2026-02-21)

Added **61 rolling z-score features** (20-candle window, min 10 periods) for all scale-dependent raw features. Z-scores normalize each feature relative to its recent history, capturing *deviation from recent norm* rather than raw magnitude.

### Z-score impact: significant features (p < 0.05 vs fwd_ret_1)

| Timeframe | Raw sig | Z-score sig | Best raw |ρ| | Best z |ρ| |
|---|---|---|---|---|
| 15m | 24 | 12 | 0.0788 | 0.0677 |
| 30m | 6 | 3 | 0.0818 | 0.0881 |
| 1h | 1 | 2 | 0.0734 | 0.0826 |
| **2h** | **2** | **15** | 0.1235 | 0.1288 |
| **4h** | **1** | **14** | 0.1682 | **0.2474** |

### Key insight: z-scores dominate at higher timeframes

At **2h/4h**, z-score features massively outperform raw features:
- **4h**: `max_trades_per_second_z` (ρ=0.247, p<0.001) — strongest signal overall, 47% stronger than raw
- **2h**: `sell_trades_z` (ρ=0.129, p=0.014), `large_trade_count_z` (ρ=0.121, p=0.021)
- Z-scores unlock 14 significant features at 4h vs just 1 for raw

At **15m**, raw features are stronger — the signal is more about the absolute level of order flow imbalance than deviation from norm.

### Top z-score features at 4h (all significant)

| Feature | Spearman ρ | p-value |
|---|---|---|
| `max_trades_per_second_z` | 0.2474 | 0.0009 |
| `volume_per_second_std_z` | 0.1792 | 0.017 |
| `trade_rate_std_z` | 0.1781 | 0.018 |
| `realized_vol_z` | 0.1704 | 0.024 |
| `max_sell_run_z` | 0.1676 | 0.026 |
| `path_length_over_range_z` | 0.1667 | 0.027 |
| `up_volatility_z` | 0.1640 | 0.030 |
| `down_volatility_z` | 0.1633 | 0.030 |

**Interpretation**: At 4h scale, what matters is not the absolute level of activity/volatility, but whether it's *unusually high relative to recent history*. Abnormally elevated burst rates, volatility, and run lengths predict continuation — consistent with information-driven moves.

## Expanded Feature Set (2026-02-21, v2)

Added **42 new raw features** across 9 novel categories, plus their z-scores (96 z-score features total). Final count: **225 features per timeframe** (129 raw + 96 z-score).

### New feature categories

| # | Category | Count | Description |
|---|---|---|---|
| 15 | **Entropy** | 4 | Volume entropy, inter-trade time entropy, side sequence entropy, price tick entropy |
| 16 | **Toxicity** | 4 | VPIN, toxic flow ratio, effective spread, price impact asymmetry |
| 17 | **Clustering** | 4 | Size clustering (round lots), price clustering, temporal clustering (Hawkes-like), volume autocorrelation |
| 18 | **Acceleration** | 5 | Volume acceleration, OFI acceleration, price acceleration, price curvature, vol-of-vol |
| 19 | **Cross-side** | 6 | Absorption ratio, aggression imbalance, response asymmetry, sweep detection (up/down/max) |
| 20 | **Fractal** | 4 | Hurst exponent (R/S), fractal dimension (Higuchi), sub-regime trending %, sub-regime transitions |
| 21 | **Tail/Extreme** | 6 | Max drawdown/drawup, DD/DU asymmetry, tail volume ratio, flash event count/pct |
| 22 | **Time-of-day** | 5 | Hour sin/cos, DOW sin/cos, distance to funding rate |
| 23 | **Cross-candle** | 5 | Return reversal, consecutive direction, volume surprise, range surprise, OFI persistence |

### Significant new features (p < 0.05 vs fwd_ret_1)

**15m (13 new significant features):**
| Feature | Spearman ρ | p-value |
|---|---|---|
| `drawdown_drawup_asymmetry_z` | 0.0804 | 1.2e-05 |
| `drawdown_drawup_asymmetry` | 0.0794 | 1.5e-05 |
| `aggression_imbalance` | -0.0724 | 7.7e-05 |
| `aggression_imbalance_z` | -0.0721 | 8.4e-05 |
| `price_curvature` | 0.0559 | 2.3e-03 |
| `ofi_persistence` | -0.0458 | 0.013 |

**30m:** `drawdown_drawup_asymmetry` (ρ=0.066*), `vpin_z` (ρ=-0.063*), `vpin` (ρ=-0.059*)

**1h:** `return_reversal_z` (ρ=-0.087*), `size_clustering` (ρ=-0.073*)

**2h:** `max_down_sweep_z` (ρ=0.140**), `sweep_count_z` (ρ=0.112*), `inter_trade_entropy_z` (ρ=-0.110*), `range_surprise` (ρ=0.107*)

**4h:** `flash_event_pct_z` (ρ=-0.193*), `flash_event_pct` (ρ=-0.161*), `size_clustering` (ρ=-0.158*), `price_clustering_z` (ρ=0.149*)

### Key new insights

1. **Drawdown/drawup asymmetry** is the strongest new signal at 15m (ρ=0.080, p<0.00001). When drawdowns exceed drawups within a candle, the next candle tends to be positive — classic oversold bounce.

2. **Aggression imbalance** (ρ=-0.072, p<0.0001 at 15m): When buyers are more aggressive (buying above VWAP), the next candle reverses. This is a novel toxicity-like signal.

3. **VPIN** shows predictive power at 30m (ρ=-0.059 to -0.063): Higher informed trading probability → next candle reversal. Classic adverse selection result.

4. **Sweep detection** matters at 2h (`max_down_sweep_z` ρ=0.140**): Unusually large downward sweeps predict positive next candle — liquidity sweeps get reversed.

5. **Flash events** are strongly predictive at 4h (`flash_event_pct_z` ρ=-0.193*): Unusually high flash event rate → negative next candle. This is a novel signal — extreme tick-level volatility at 4h scale signals genuine stress.

6. **Size clustering** (ρ=-0.158 at 4h): More round-lot trading → negative next candle. Retail herding is a contrarian signal.

7. **Price curvature** (ρ=0.056** at 15m): Convex price paths (accelerating) predict continuation. Concave (decelerating) predicts reversal.

## Output Files

- `microstructure_BTCUSDT_{tf}_2026-01-01_2026-01-31.csv` — full feature tables per timeframe (225 features: 129 raw + 96 z-score)
- `microstructure_correlations_BTCUSDT_2026-01-01_2026-01-31.csv` — all correlations

## Next Steps

1. **Multi-variate analysis**: Combine top features into a composite signal
2. **Out-of-sample test**: Run on Feb 2026 or different symbols
3. **Regime conditioning**: Check if feature predictiveness varies by volatility regime
4. **Lag analysis**: Check fwd_ret_2 through fwd_ret_10 for longer-horizon predictability
5. **Cross-symbol**: Run on ETH, SOL, DOGE, XRP to check universality
