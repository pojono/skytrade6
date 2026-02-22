# Feature Analysis V2 — Expanded Feature Set Comparison

## Overview

| Metric | V1 | V2 | Change |
|--------|----|----|--------|
| Raw features | 763 | 1,040 | +277 (+36%) |
| Total columns (incl z-scores + targets) | ~900 | 1,160 | +260 |
| Tier 4 clusters (non-redundant) | 44 | 61 | +17 (+39%) |
| Tier 5 survivors (per coin) | 44 | 61 | +17 |

## New Feature Categories Added in V2

| Category | Raw Features | Description |
|----------|-------------|-------------|
| Market Structure | ~20 | HH/HL/LH/LL counts, ms_score, streaks |
| Smart Money Concepts | ~18 | BOS, CHoCH, order blocks, liquidity sweeps |
| Support/Resistance | ~25 | Rolling S/R, pivot points, level touches |
| Breakout Detection | ~25 | Up/down/any at 5/10/20, vol confirm, failed breakouts |
| Liquidity Pools | ~14 | Equal highs/lows, pool strength, voids, stop hunts |
| Wick Analysis | ~9 | Upper/lower wick ratios, rejection, dominance |
| Candle Patterns | ~14 | Engulfing, hammer, doji, inside/outside bar, stars |
| Gap Analysis | ~8 | Gap bps, fill %, fill rate |
| ATR | ~17 | True range, ATR ratios, percentiles |
| Donchian Channel | ~8 | Position, width, change, mid distance |

## 20 New Features Surviving All 5 Tiers (Both Coins)

These features passed: Tier 1 (correlation), Tier 2 (temporal stability), Tier 4 (non-redundant), Tier 5 (walk-forward OOS):

| Feature | Category | Description |
|---------|----------|-------------|
| `is_swing_high` | Market Structure | Binary: is this candle a local swing high? |
| `is_swing_low` | Market Structure | Binary: is this candle a local swing low? |
| `higher_low` | Market Structure | Binary: current low > previous low |
| `hh_streak_z` | Market Structure | Z-score of consecutive higher-high streak |
| `resistance_touches_10` | Support/Resistance | Count of touches near rolling 10-bar high |
| `resistance_touches_20` | Support/Resistance | Count of touches near rolling 20-bar high |
| `support_touches_10` | Support/Resistance | Count of touches near rolling 10-bar low |
| `near_support_5` | Support/Resistance | Binary: close near 5-bar support |
| `near_support_20` | Support/Resistance | Binary: close near 20-bar support |
| `dist_to_s1_bps` | Support/Resistance | Distance to pivot S1 level in bps |
| `failed_breakout_up_10` | Breakout | Broke above 10-bar high then reversed |
| `failed_breakout_down_10` | Breakout | Broke below 10-bar low then reversed |
| `atr_percentile_14` | ATR | Where current ATR(14) sits vs recent range |
| `donchian_mid_dist_bps_20` | Donchian | Distance from close to 20-bar Donchian midline |
| `liq_pool_above_20` | Liquidity Pools | Strength of liquidity pool above price |
| `ob_demand` | Smart Money | Order block demand zone detection |
| `engulfing_bearish` | Candle Patterns | Bearish engulfing pattern |
| `gap_abs_bps` | Gap Analysis | Absolute gap size in bps |
| `market_pressure` | Physics | Market pressure (volume × price change) |
| `uptick_pct` | Microstructure | Percentage of uptick trades |

## Top Cross-Symbol OOS Performers (New Features Only)

### tgt_profitable_long_5 (key target)

| Feature | DOGE OOS | SOL OOS | Avg | DOGE %+ | SOL %+ |
|---------|----------|---------|-----|---------|--------|
| `is_swing_low` | +0.047 | +0.057 | 0.052 | 100% | 100% |
| `is_swing_high` | +0.044 | +0.049 | 0.047 | 80% | 100% |
| `resistance_touches_20` | +0.035 | +0.046 | 0.040 | 60% | 90% |
| `donchian_mid_dist_bps_20` | +0.050 | +0.010 | 0.030 | 80% | 70% |
| `resistance_touches_10` | +0.009 | +0.046 | 0.028 | 70% | 70% |
| `market_pressure` | +0.036 | +0.011 | 0.024 | 80% | 60% |
| `uptick_pct` | +0.016 | +0.014 | 0.015 | 70% | 50% |

### tgt_profitable_short_5 (key target)

| Feature | DOGE OOS | SOL OOS | Avg | DOGE %+ | SOL %+ |
|---------|----------|---------|-----|---------|--------|
| `is_swing_low` | +0.047 | +0.055 | 0.051 | 100% | 100% |
| `is_swing_high` | +0.046 | +0.048 | 0.047 | 80% | 100% |
| `resistance_touches_20` | +0.035 | +0.047 | 0.041 | 70% | 90% |
| `donchian_mid_dist_bps_20` | +0.051 | +0.008 | 0.030 | 80% | 60% |
| `resistance_touches_10` | +0.013 | +0.041 | 0.027 | 60% | 60% |
| `market_pressure` | +0.039 | +0.012 | 0.026 | 80% | 60% |
| `uptick_pct` | +0.018 | +0.014 | 0.016 | 80% | 50% |

## Key Findings

### 1. Swing Points Are the Strongest New Signal
`is_swing_high` and `is_swing_low` are the top new features — **100% positive OOS folds** on both coins for multiple targets. These simple binary flags (is the current candle a local turning point?) carry genuine predictive power.

### 2. Support/Resistance Touches Work
`resistance_touches_20` consistently shows +0.04 OOS AUC deviation with 70-90% positive folds. The more times a level is tested, the more predictive it becomes — consistent with the liquidity pool thesis.

### 3. Failed Breakouts Have Signal
`failed_breakout_up_10` and `failed_breakout_down_10` both survived all 5 tiers. Failed breakouts (price breaks a level then reverses) are a classic smart money concept and the data confirms they carry information.

### 4. Donchian Position Adds Value
`donchian_mid_dist_bps_20` — how far price is from the 20-bar channel midline — survived with decent OOS scores. This is essentially a mean-reversion signal.

### 5. ATR Percentile Is Useful
`atr_percentile_14` survived — knowing whether current volatility is historically high or low relative to recent history adds predictive value.

### 6. Candle Patterns: Engulfing Works, Others Don't
Only `engulfing_bearish` survived all 5 tiers. Hammer, doji, morning star, etc. did not make it through temporal stability or redundancy filtering. This suggests most candle patterns are noise, but engulfing has genuine signal.

### 7. Liquidity Pool Features Show Promise
`liq_pool_above_20` and `ob_demand` survived, confirming that smart money concepts (liquidity accumulation at key levels) carry some predictive information, though the effect sizes are smaller than swing points.

## V1 Features That Remain Strong

The original top performers are still dominant:
- `close` (price level proxy): +0.14 OOS, 100% positive folds
- `hilbert_std_amplitude`: +0.05-0.07 OOS
- `avg_buy_size`: +0.04 OOS
- `session_asia`: +0.05 OOS
- `fvg_bearish_count_10`: +0.03 OOS

## Recommendations for Next Steps

1. **Multi-feature model**: Combine top 10-15 features (both v1 and v2 survivors) into Ridge/Logistic ensemble
2. **Feature interactions**: Test `is_swing_low × resistance_touches_20` and similar combinations
3. **Focus targets**: `tgt_profitable_long_5` and `tgt_profitable_short_5` show the most consistent cross-symbol signal
4. **Consider regime conditioning**: Several features (market_pressure, atr_percentile_14) work differently in high-vol vs low-vol — regime-aware models may improve
