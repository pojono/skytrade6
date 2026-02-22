# Pipeline V3 — Statistical Power Rework

## Problem Identified

With 1 year of 4h data (2,196 candles), the original pipeline had insufficient statistical power:

| Component | V2 (old) | Min detectable \|r\| | V3 (new) | Min detectable \|r\| |
|-----------|----------|---------------------|----------|---------------------|
| Tier 2 window | 180 candles (30d) | **0.145** | 540 candles (90d) | **0.084** |
| Tier 5 train (fold 1) | 360 candles (60d) | 0.103 | 720 candles (120d) | 0.073 |
| Tier 5 train (last fold) | 360 candles (60d) | 0.103 | 1800 candles (300d) | **0.046** |
| Tier 5 test | 180 candles (30d) | 0.145 | 270 candles (45d) | 0.119 |

## Changes Made

### Tier 2: Temporal Stability
- **Window**: 30d → 90d (540 candles per window)
- **Step**: 15d → 30d (10 windows instead of 23)
- **Max wrong streak**: ≤4 → ≤3 (stricter with fewer windows)
- **Result**: More features pass because per-window estimates are now reliable

### Tier 5: Walk-Forward OOS
- **Architecture**: Fixed sliding → **Expanding window**
- **Min train**: 60d → 120d (720 candles minimum)
- **Test**: 30d → 45d (270 candles)
- **Purge**: 1d → 2d
- **Folds**: 10 → 5 (fewer but much more meaningful)
- Fold 1: 120d train, 45d test
- Fold 5: 300d train, 45d test
- **Result**: Later folds train on 5× more data than v2

## Results Comparison

| Metric | V2 | V3 |
|--------|----|----|
| Tier 4 clusters | 61 | **182** |
| Cross-symbol survivors (both coins) | 61 | **182** |
| Kept from v2 | — | 48/61 (79%) |
| New features discovered | — | 134 |
| Dropped from v2 | — | 13 |

### Features Dropped (failed with proper statistical power)
`aggression_imbalance_z`, `dist_to_s1_bps`, `ew_impulse_quality_z`, `failed_breakout_down_10`, `golden_ratio_half_dist_z`, `higher_low`, `resistance_touches_10`, `sell_urgency_ratio`, `support_touches_10`, `taylor_price_a0`, `taylor_rate_a1`, `uptick_pct`, `vol_asymmetry_z`

These 13 features passed v2's weak statistical tests but failed with proper window sizes — likely spurious signals.

## Top Cross-Symbol OOS Features (V3, tgt_profitable_long_5)

| Rank | Feature | DOGE OOS | SOL OOS | Avg | D%+ | S%+ | New? |
|------|---------|----------|---------|-----|-----|-----|------|
| 1 | `close` | +0.095 | +0.114 | 0.104 | 100% | 100% | |
| 2 | `golden_ratio_half_dist` | +0.069 | +0.040 | 0.054 | 100% | 80% | ★ |
| 3 | `is_swing_low` | +0.053 | +0.055 | 0.054 | 100% | 100% | |
| 4 | `vol_price_feedback` | +0.052 | +0.047 | 0.050 | 80% | 80% | |
| 5 | `hilbert_std_amplitude` | +0.041 | +0.056 | 0.049 | 60% | 100% | |
| 6 | `is_swing_high` | +0.047 | +0.046 | 0.046 | 100% | 100% | |
| 7 | `donchian_mid_dist_bps_20` | +0.049 | +0.032 | 0.040 | 80% | 80% | |
| 8 | `max_monotonic_run` | +0.028 | +0.050 | 0.039 | 60% | 80% | |
| 9 | `area_above_vwap_pct_z` | +0.039 | +0.037 | 0.038 | 80% | 100% | |
| 10 | `ob_net_10_z` | +0.039 | +0.036 | 0.038 | 80% | 80% | ★ |

★ = newly discovered in v3 (wasn't in v2's 61 clusters)

## Key Takeaways

1. **79% of v2 features survived** — the core signal was real, not an artifact of small windows
2. **13 features were false positives** — they only appeared significant due to noisy small-window estimates
3. **134 new features discovered** — larger windows let weaker but genuine signals pass Tier 2
4. **Expanding WFO is critical** — fold 5 trains on 300d vs v2's fixed 60d, giving much better model estimates
5. **Top features are robust** — `is_swing_low/high`, `close`, `hilbert_std_amplitude` survive both v2 and v3 with consistent OOS scores
6. **57 cross-symbol survivors** for `tgt_profitable_long_5` alone — much richer feature pool for multi-feature modeling
