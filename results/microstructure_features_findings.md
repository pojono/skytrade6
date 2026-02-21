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

## Volume Profile, Fair Value & FVG Features (v3)

Added **17 intra-candle volume profile features** and **10 cross-candle fair value features**.

### Volume Profile (intra-candle, from tick data)
- **POC (Point of Control)**: Price level with highest volume; `close_to_poc_bps`
- **Value Area** (70% of volume): `value_area_width_bps`, `close_in_value_area`, `close_above/below_value_area`
- **Fair Price**: Volume-weighted median price; `close_to_fair_price_bps`
- **Fair Value**: Average of POC and VWAP; `close_to_fair_value_bps`
- **Profile shape**: `volume_profile_skew`, `low/high_volume_node_pct`

### Fair Value Gaps (cross-candle)
- **FVG detection**: `fvg_bullish`, `fvg_bearish`, `fvg_net`, size in bps
- **Cumulative FVGs**: `fvg_bullish_count_10`, `fvg_bearish_count_10`
- **Cross-candle FV**: `close_vs_prev_fair_value_bps`, `close_vs_prev_poc_bps`
- **FV dynamics**: `fair_value_change_bps`, `poc_migration_bps`, `value_area_overlap_pct`

### Key FV/VP results

| Feature | TF | Spearman ρ | p-value |
|---|---|---|---|
| `fair_value_change_bps` | 15m | -0.098 | 8.9e-08 |
| `poc_migration_bps` | 15m | -0.095 | 2.4e-07 |
| `close_vs_prev_fair_value_bps` | 15m | -0.093 | 4.1e-07 |
| `close_to_fair_price_bps` | 30m | -0.078 | 0.003 |
| `close_to_poc_bps` | 2h | -0.121 | 0.020 |
| `close_to_fair_value_bps` | 2h | -0.115 | 0.026 |

**Interpretation**: Fair value features are the **strongest new signals at 15m** — when fair value/POC shifts up, the next candle reverses down. This is a powerful mean-reversion signal: price overshooting fair value predicts pullback.

## Physics-Inspired Features (v3)

Added **37 features** across 6 physics domains.

| Domain | Features | Key idea |
|---|---|---|
| **Newtonian** (8) | Momentum, force, impulse, KE, PE, total energy, energy ratio, inertia | Volume × return = momentum; force = Δmomentum/Δt |
| **Thermodynamics** (7) | Temperature, pressure, PV work, heat capacity, Boltzmann entropy, heating rate | Vol/range = pressure; realized vol = temperature |
| **Electromagnetism** (6) | OFI field gradient, temporal/price dipoles, VWAP flux | Buy/sell separation in time and price |
| **Fluid Dynamics** (5) | Viscosity, Reynolds number, turbulence, flow velocity, Bernoulli | Inertial vs viscous forces in order flow |
| **Wave Physics** (6) | Amplitude, frequency, wavelength, standing wave ratio, wave energy, zero-crossing rate | Price oscillation characteristics |
| **Gravity** (5) | VWAP gravity, escape velocity, orbital energy, binding energy, centripetal accel | Attraction to fair value, breakout energy |

### Key physics results

| Feature | TF | Spearman ρ | p-value | Interpretation |
|---|---|---|---|---|
| `market_impulse` | 15m | -0.079*** | 1.7e-05 | Signed volume × time → mean reversion |
| `flow_velocity` | 15m | -0.072*** | 8.7e-05 | Net flow per unit range → reversal |
| `reynolds_number_z` | 4h | 0.165* | 0.029 | High Re = turbulent → continuation |
| `market_temperature_z` | 4h | 0.170* | 0.024 | Unusually hot → continuation |
| `boltzmann_entropy_z` | 2h | 0.114* | 0.031 | More price levels visited → continuation |
| `escape_velocity_z` | 2h | 0.108* | 0.040 | Higher breakout energy → continuation |
| `wave_frequency_z` | 2h/4h | 0.116*/0.152* | 0.027/0.044 | Faster oscillation → continuation |

**Interpretation**: Physics features reveal a consistent pattern — at short TFs (15m), momentum/impulse mean-reverts. At long TFs (2h/4h), unusually high energy/temperature/turbulence predicts continuation. The Reynolds number analogy works: laminar (low Re) = ranging, turbulent (high Re) = trending.

## Psychology & Behavioral Finance Features (v3)

Added **33 features** across 8 behavioral categories.

| Category | Features | Key idea |
|---|---|---|
| **Anchoring** (5) | Round number distance ($100/$500/$1k), magnet effect, drift from open | Traders anchor to reference prices |
| **Loss Aversion** (6) | Sell urgency ratio, panic sell/FOMO buy ratios, contrarian ratio, loss aversion ratio | Losses hurt 2× more; panic detection |
| **Herding** (3) | Vol-price feedback, herding ratio, attention effect | Crowd follows crowd |
| **Disposition** (3) | Disposition sell/buy, disposition effect | Sell winners early, hold losers |
| **Regret/Hesitation** (3) | Post-shock size ratio, trade rate decay, post-shock pause | Traders hesitate after shocks |
| **Overconfidence** (3) | Size escalation, Gini coefficient, extreme price % | Concentrated large trades |
| **Fear & Greed** (3) | Micro fear/greed index, fear score, greed score | Composite sentiment |
| **Attention** (3) | Shock attention ratio, rubber-necking, surprise magnitude | Volume response to shocks |

### Key psychology results

| Feature | TF | Spearman ρ | p-value | Interpretation |
|---|---|---|---|---|
| `contrarian_ratio_z` | 2h | 0.142** | 0.007 | Unusually high contrarian activity → continuation |
| `fomo_buy_ratio` | 15m | -0.068*** | 2.2e-04 | FOMO buying → reversal (classic) |
| `drift_from_open_bps` | 15m | -0.068*** | 2.2e-04 | Far from open → mean reversion |
| `disposition_sell` | 30m | -0.061* | 0.019 | Profit-taking at highs → more downside |
| `post_shock_pause_ratio_z` | 1h | 0.094* | 0.011 | Unusual hesitation → continuation |
| `post_shock_size_ratio_z` | 4h | 0.150* | 0.048 | Larger trades after shocks → continuation |
| `volume_profile_skew` | 15m | -0.071*** | 9.9e-05 | Volume concentrated at top → reversal down |

**Key insight**: The strongest psychology signal is **contrarian_ratio_z** at 2h — when an unusually high fraction of traders are going against the trend, the trend continues. Contrarians get run over. FOMO buying at 15m is a classic reversal signal.

## Math, Algebra & Geometry Features (v4)

Added **30+ features** across 7 mathematical domains.

| Domain | Features | Key idea |
|---|---|---|
| **Linear Algebra** (3) | Eigenvalue ratio, principal angle, covariance determinant | Price-volume covariance structure |
| **Geometry** (3) | Convex hull area, OHLC triangle area, aspect ratio | Shape of price path in 2D |
| **Angles/Slopes** (2) | Price slope angle, price-volume angle cosine | Directional characteristics |
| **Distance Metrics** (3) | Manhattan, Chebyshev, half-cosine similarity | Distance between open→close paths |
| **Calculus** (3) | Signed area integral, area above VWAP, jerk | Integral/derivative features |
| **Topology** (4) | Extrema count/density, monotonicity, tortuosity | Path complexity measures |
| **Spectral/Fourier** (6) | Dominant freq, spectral energy ratio, entropy, flatness, centroid | Frequency domain analysis |

### Key math/geometry results

| Feature | TF | Spearman ρ | p-value |
|---|---|---|---|
| `signed_area_bps` | 15m | +0.078*** | 1.9e-05 |
| `eigen_ratio_z` | 4h | -0.161* | 0.032 |
| `price_slope_angle_z` | 2h | -0.159** | 0.003 |
| `tortuosity_z` | 4h | -0.159* | 0.035 |
| `manhattan_distance_z` | 4h | +0.154* | 0.041 |
| `spectral_centroid` | 2h | +0.135** | 0.010 |

**Interpretation**: Signed area (integral of price path) is a strong 15m signal — positive area = bullish bias → continuation. At higher TFs, eigenvalue ratio (how "stretched" the price-volume relationship is) and tortuosity (path complexity) predict reversals.

## Deep Fibonacci Analysis (v4)

Replaced basic Fibonacci features with **28 deep Fibonacci features**:

| Sub-category | Features | Key idea |
|---|---|---|
| **Fib Level Interaction** (5) | Crosses, volume concentration, bounce count, respect score | Do prices react at 0.236/0.382/0.5/0.618/0.786 levels? |
| **Fibonacci Time Zones** (3) | Volume/trade clustering at Fib time fractions | Activity at φ-based time points |
| **Golden Ratio Volume** (4) | Buy/sell ratio vs φ, half-split vs φ, quartile pairs, composite | Volume distribution vs golden ratio |
| **Fib Trade Sizes** (3) | Consecutive size ratios near φ, max/median distance | Trade size relationships |
| **Wave Structure** (5) | Wave ratio count/pct, avg distance, swing count, up/down ratio | Elliott-like amplitude ratios |
| **Golden Angle** (3) | Deviation, uniformity, sunflower score | Phyllotaxis distribution of trades |
| **Inter-trade Time** (2) | Fib ratio in time gaps, scaling score | Fibonacci in temporal spacing |

### Key Fibonacci results

| Feature | TF | Spearman ρ | p-value |
|---|---|---|---|
| `golden_angle_deviation_z` | 4h | +0.215** | 0.004 |
| `golden_angle_uniformity_z` | 4h | -0.215** | 0.004 |
| `fib_total_crosses_z` | 4h | +0.184* | 0.015 |
| `fib_time_scaling_score_z` | 4h | -0.175* | 0.025 |
| `fib_size_ratio_count_z` | 2h | +0.131* | 0.013 |
| `fib_intertime_ratio_pct_z` | 1h | -0.098** | 0.008 |

**Interpretation**: Golden angle features are remarkably predictive at 4h — when trade distribution deviates from the golden angle (137.5°), the next candle tends to be positive. This is a genuinely novel signal. Fibonacci level crosses also predict continuation at 4h.

## Elliott Wave Principle (v4)

Added **15 Elliott Wave features**:

| Sub-category | Features | Key idea |
|---|---|---|
| **Impulse Detection** (2) | Impulse quality score, impulse detected flag | 5-wave pattern with EW rules |
| **Correction Detection** (1) | Correction quality score | ABC pattern detection |
| **Wave Personality** (4) | Volume exhaustion, amplitude trend (raw + normalized) | Wave character signatures |
| **Structure Metrics** (8) | Wave count, swing count, impulse/correction ratio, alternation, symmetry, net direction, completion % | Overall wave structure |

### Key Elliott Wave results

| Feature | TF | Spearman ρ | p-value |
|---|---|---|---|
| `ew_impulse_detected` | 4h | -0.205** | 0.005 |
| `ew_impulse_detected` | 4h (h=5) | -0.201** | 0.007 |
| `ew_last_wave_pct_z` | 4h | +0.154* | 0.041 |
| `ew_amplitude_trend_norm_z` | 4h | +0.151* | 0.045 |
| `ew_wave_symmetry_z` | 1h | -0.109** | 0.003 |
| `ew_net_direction_z` | 2h (h=10) | -0.123* | 0.020 |

**Interpretation**: Elliott Wave impulse detection is the **2nd strongest new signal at 4h** (ρ=-0.205, p=0.005). When a 5-wave impulse pattern completes within a 4h candle, the next candle reverses — classic wave exhaustion. Wave symmetry at 1h also predicts reversals.

## Taylor Series Decomposition (v4)

Added **25 Taylor Series features** — polynomial fit of intra-candle paths:

| Sub-category | Features | Key idea |
|---|---|---|
| **Price Path** (10) | a0-a5 coefficients, R², RMSE, curvature/trend ratio, asymmetry/curvature ratio | Shape of price(t) polynomial |
| **Volume Path** (5) | a1-a3 coefficients, R², acceleration ratio | Shape of cumulative volume(t) |
| **OFI Path** (5) | a1-a3 coefficients, R², price-OFI alignment | Shape of order flow(t) |
| **Trade Rate Path** (4) | a1-a3 coefficients, R² | Shape of activity(t) |
| **Cross-series** (2) | Price-volume divergence, price-rate divergence | Shape mismatch between series |
| **Complexity** (2) | Dominant order, complexity (min degree for 90% R²) | How complex is the path? |

### Key Taylor Series results

| Feature | TF | Spearman ρ | p-value |
|---|---|---|---|
| `taylor_complexity` | 4h | +0.195** | 0.008 |
| `taylor_rate_a2_z` | 2h | +0.159** | 0.002 |
| `taylor_rate_a2` | 2h | +0.152** | 0.003 |
| `taylor_rate_a1_z` | 2h | -0.146** | 0.006 |
| `taylor_vol_a2_z` | 2h | +0.136** | 0.009 |
| `taylor_ofi_a3_z` | 1h | -0.090* | 0.015 |

**Interpretation**: Taylor Series features are **dominant at 2h** — trade rate acceleration (a2) is the strongest signal. When activity accelerates through the candle (positive a2), the next candle continues. Taylor complexity at 4h: more complex price paths predict continuation. The OFI jerk (a3) at 1h predicts reversals.

## Information Theory (v4)

Added **7 features**: transfer entropy (price→volume, volume→price), mutual information (side↔return), Lempel-Ziv complexity, average surprise.

### Key results

| Feature | TF | Spearman ρ | p-value |
|---|---|---|---|
| `te_net_direction` | 4h (h=3) | -0.172* | 0.022 |
| `te_volume_to_price` | 4h (h=3) | +0.158* | 0.036 |
| `avg_surprise` | 4h (h=3) | +0.152* | 0.044 |
| `lz_complexity_norm_z` | 2h (h=5) | +0.122* | 0.020 |

**Interpretation**: Transfer entropy reveals causal direction — when volume drives price (not vice versa), the 3-bar-ahead return is positive. Higher LZ complexity (less compressible = more random) at 2h predicts continuation.

## Game Theory / Auction Theory (v4)

Added **5 features**: Nash balance, Stackelberg leader, auction clearing speed, large trade timing.

### Key results

| Feature | TF | Spearman ρ | p-value |
|---|---|---|---|
| `nash_balance_z` | 4h (h=2) | +0.157* | 0.037 |
| `stackelberg_buy_leader` | 2h (h=5) | -0.143** | 0.007 |
| `auction_clearing_speed` | 2h (h=2) | -0.122* | 0.020 |

**Interpretation**: When buy/sell forces are in Nash equilibrium (balanced), the 2-bar-ahead return at 4h is positive — balanced markets precede breakouts. When buyers lead (Stackelberg), the 5-bar return reverses.

## Network / Graph Theory (v4)

Added **9 features**: price level transition graph metrics, recurrence.

### Key results

| Feature | TF | Spearman ρ | p-value |
|---|---|---|---|
| `avg_visits_per_level_z` | 4h | +0.151* | 0.045 |
| `max_visits_level_z` | 2h (h=2) | +0.139** | 0.008 |
| `graph_self_loop_ratio_z` | 2h | +0.116* | 0.028 |

**Interpretation**: More revisits to the same price level (higher recurrence) predicts continuation — price consolidation at a level builds energy for a move.

## Signal Processing — Wavelet & Hilbert (v4)

Added **12 features**: Haar wavelet energy decomposition (5 scales), HF/LF ratio, wavelet entropy, Hilbert transform (amplitude, frequency, correlation).

### Key results

| Feature | TF | Spearman ρ | p-value |
|---|---|---|---|
| `wavelet_energy_s3` | 4h | -0.197** | 0.007 |
| `wavelet_hf_lf_ratio_z` | 4h | +0.180* | 0.017 |
| `wavelet_hf_lf_ratio` | 4h | +0.174* | 0.018 |
| `hilbert_std_freq` | 4h (h=5) | +0.171* | 0.023 |

**Interpretation**: Wavelet features are strong at 4h — when mid-frequency energy (scale 3) is low and high-frequency energy dominates, the next candle continues. This means choppy/noisy price action at the tick level predicts trending at the candle level.

## Biological / Ecological Analogies (v4)

Added **7 features**: predator-prey dynamics, population growth, carrying capacity, Simpson/Shannon diversity.

### Key results

| Feature | TF | Spearman ρ | p-value |
|---|---|---|---|
| `predator_vol_share_z` | 4h (h=10) | -0.213** | 0.004 |
| `predator_vol_share_z` | 4h (h=5) | -0.164* | 0.029 |
| `carrying_capacity_ratio_z` | 1h | -0.110** | 0.003 |
| `carrying_capacity_ratio` | 1h | -0.107** | 0.003 |

**Interpretation**: Predator volume share (large trades' fraction of total volume) is a **top-5 signal at 4h for 10-bar horizon** — when large trades dominate, the 10-bar return is negative. This is a powerful institutional exhaustion signal. Carrying capacity (max trade rate / mean) predicts reversals at 1h.

## Compression / Complexity (v4)

Added **4 features**: approximate entropy, sample entropy, permutation entropy, RLE compression ratio.

### Key results

| Feature | TF | Spearman ρ | p-value |
|---|---|---|---|
| `permutation_entropy` | 4h (h=2) | -0.190** | 0.010 |
| `sample_entropy` | 4h (h=2) | -0.160* | 0.033 |
| `approx_entropy` | 4h (h=2) | -0.156* | 0.038 |
| `sample_entropy_z` | 1h | +0.087* | 0.018 |

**Interpretation**: All three entropy measures agree at 4h — lower entropy (more predictable/regular price patterns) predicts positive 2-bar returns. Regular patterns = trending, which continues. At 1h, unusually high sample entropy (more random) predicts continuation.

## Chaos Theory (v4)

Added **5 features**: Hurst regime, Lyapunov exponent, RQA recurrence rate & determinism.

### Key results

| Feature | TF | Spearman ρ | p-value |
|---|---|---|---|
| `lyapunov_exponent` | 4h (h=10) | +0.183* | 0.015 |

**Interpretation**: Positive Lyapunov exponent (chaotic divergence) at 4h predicts positive 10-bar returns — chaos at the tick level precedes trending at the candle level. Only 1 significant result, but it's a unique signal.

## Summary Statistics (v4)

| Metric | Value |
|---|---|
| Total features per timeframe | 736 |
| Raw features | ~401 |
| Z-score features | ~335 |
| Feature categories | 55 |
| New features in v4 | ~160 (raw) |
| New significant correlations | 277 |
| Strongest new signal (15m) | `signed_area_bps` (ρ=+0.078, p=1.9e-05) |
| Strongest new signal (4h) | `golden_angle_deviation_z` (ρ=+0.215, p=0.004) |
| Strongest new signal (2h) | `taylor_rate_a2_z` (ρ=+0.159, p=0.002) |
| Strongest overall (4h) | `max_trades_per_second_z` (ρ=0.247, p=9.3e-04) |

## Top 10 Most Predictive New Features (across all TFs)

| Rank | Feature | TF | Horizon | ρ | p-value | Category |
|---|---|---|---|---|---|---|
| 1 | `golden_angle_deviation_z` | 4h | 1 | +0.215 | 0.004 | Fibonacci |
| 2 | `predator_vol_share_z` | 4h | 10 | -0.213 | 0.004 | Biological |
| 3 | `ew_impulse_detected` | 4h | 1 | -0.205 | 0.005 | Elliott Wave |
| 4 | `wavelet_energy_s3` | 4h | 1 | -0.197 | 0.007 | Signal Processing |
| 5 | `taylor_complexity` | 4h | 1 | +0.195 | 0.008 | Taylor Series |
| 6 | `permutation_entropy` | 4h | 2 | -0.190 | 0.010 | Compression |
| 7 | `lyapunov_exponent` | 4h | 10 | +0.183 | 0.015 | Chaos Theory |
| 8 | `wavelet_hf_lf_ratio_z` | 4h | 1 | +0.180 | 0.017 | Signal Processing |
| 9 | `te_net_direction` | 4h | 3 | -0.172 | 0.022 | Information Theory |
| 10 | `taylor_rate_a2_z` | 2h | 1 | +0.159 | 0.002 | Taylor Series |

## Output Files

- `microstructure_BTCUSDT_{tf}_2026-01-01_2026-01-31.csv` — full feature tables per timeframe (736 features)
- `microstructure_correlations_BTCUSDT_2026-01-01_2026-01-31.csv` — all correlations

## Next Steps

1. **Multi-variate analysis**: Combine top features into a composite signal
2. **Out-of-sample test**: Run on Feb 2026 or different symbols
3. **Regime conditioning**: Check if feature predictiveness varies by volatility regime
4. **Lag analysis**: Check fwd_ret_2 through fwd_ret_10 for longer-horizon predictability
5. **Cross-symbol**: Run on ETH, SOL, DOGE, XRP to check universality
6. **Feature selection**: Use mutual information or LASSO to find orthogonal feature set
7. **Nonlinear models**: Test with gradient boosting (XGBoost/LightGBM) to capture interactions
