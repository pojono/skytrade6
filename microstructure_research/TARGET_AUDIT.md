# Target Audit — ML Predictability Validation

**Date:** 2026-02-23 (updated 2026-02-24)
**Assets:** SOLUSDT, XRPUSDT, DOGEUSDT
**Timeframes:** 4h (~4392 candles, 2024-01-01 → 2026-01-01), 1h (~8784 candles, 2025-01-01 → 2026-01-01)
**Method:** 3-fold expanding-window WFO, LightGBM, top-30 features per target
**Scoring:** Binary = AUC - 0.5 (>0 = better than random), Continuous = Spearman correlation

---

## ⚠️ Contamination Fix (2026-02-24)

**All swing-derived features were contaminated by lookahead bias.** The swing point detection
used `.shift(-1)` which peeks at the NEXT candle — classic lookahead. This inflated 32 features
(`is_swing_high`, `is_swing_low`, `bos_*`, `choch_*`, `ob_*`, `liq_sweep_*`, `dist_to_swing_*`)
and affected 26/54 targets that had contaminated features in their top-30 selection.

**Fix:** Replaced with lagged 3-bar pivots that only use past data:
```
# OLD (contaminated): swing_high = (h > h.shift(1)) & (h > h.shift(-1))
# NEW (clean):        swing_high = (h.shift(1) > h.shift(2)) & (h.shift(1) > h)
```

All features regenerated. All scores below are from **clean** features.

---

## Pipeline

1. **SOL audit:** Test all 120 targets → 84 STRONG + 2 GOOD = 86 predictable
2. **Deduplication:** Remove targets correlated |r| > 0.95 → 86 → **54 unique**
3. **Cross-coin validation (4h):** Test 54 targets on SOL + XRP + DOGE (clean features)
4. **1h comparison:** Test same 54 targets on SOL + DOGE at 1h (XRP has no 1h data)

## Summary

| Step | Count |
|------|-------|
| Total targets | 120 |
| Usable base rate (5–95%) | 107 |
| STRONG + GOOD (SOL) | 86 |
| After deduplication (|r| > 0.95) | **54** |
| Positive avg score (4h, 3 coins) | **50** |
| Positive avg score (1h, 2 coins) | **47** |

---

## Deduplication: 17 Clusters of Redundant Targets

Many targets are mathematically identical or near-identical. We keep the best representative from each cluster.

| Cluster | Keep (representative) | Drop (duplicates) |
|---------|----------------------|-------------------|
| 1 | tgt_best_short_entry_pct | best_long_entry_pct, slippage_long/short_bps |
| 2 | tgt_relative_ret_1 | alpha_1, pnl_long/short_bps_1, ret_1 |
| 3 | tgt_range_bps_1 | fwd_spread_bps_1 |
| 4 | tgt_optimal_action_1 | best_side_1, ret_sign_1, profitable_long/short_1 |
| 5 | tgt_max_drawdown_long_3 | max_drawup_short_3 |
| 6 | tgt_max_drawup_long_3 | max_drawdown_short_3 |
| 7 | tgt_realized_vol_10 | inventory_cost_10 |
| 8 | tgt_alpha_3 | relative_ret_3 |
| 9 | tgt_alpha_5 | relative_ret_5 |
| 10 | tgt_max_drawup_long_5 | max_drawdown_short_5 |
| 11 | tgt_max_drawdown_long_5 | max_drawup_short_5 |
| 12 | tgt_best_side_3 | optimal_action_3, profitable_long/short_3 |
| 13 | tgt_pnl_short_bps_3 | pnl_long_bps_3, cum_ret_3 |
| 14 | tgt_max_drawup_long_10 | max_drawdown_short_10 |
| 15 | tgt_max_drawup_short_10 | max_drawdown_long_10 |
| 16 | tgt_cum_ret_5 | pnl_long/short_bps_5 |
| 17 | tgt_optimal_action_5 | best_side_5, kelly_5, profitable_long/short_5 |

---

## Cross-Coin + Timeframe Comparison: 54 Targets (Clean Features)

Sorted by 4h average score. All scores from clean (non-contaminated) features.

| Target | Type | 4h Avg | 1h Avg | Δ (1h−4h) |
|--------|------|--------|--------|-----------|
| tgt_range_bps_1 | cont | **+0.456** | **+0.666** | +0.210 |
| tgt_fwd_spread_bps_3 | cont | **+0.431** | **+0.625** | +0.194 |
| tgt_consolidation_5 | bin | **+0.384** | +0.378 | -0.006 |
| tgt_consolidation_3 | bin | **+0.349** | +0.335 | -0.015 |
| tgt_vol_expansion_10 | bin | **+0.317** | +0.259 | -0.058 |
| tgt_range_bps_3 | cont | **+0.316** | **+0.504** | +0.188 |
| tgt_realized_vol_5 | cont | **+0.289** | **+0.440** | +0.151 |
| tgt_inventory_cost_5 | cont | **+0.276** | **+0.482** | +0.205 |
| tgt_range_bps_5 | cont | **+0.268** | **+0.460** | +0.192 |
| tgt_realized_vol_10 | cont | **+0.266** | **+0.440** | +0.174 |
| tgt_inventory_cost_3 | cont | **+0.263** | **+0.472** | +0.209 |
| tgt_breakout_up_3 | bin | +0.259 | +0.237 | -0.023 |
| tgt_breakout_down_3 | bin | +0.250 | +0.227 | -0.023 |
| tgt_breakout_up_5 | bin | +0.243 | +0.233 | -0.010 |
| tgt_breakout_down_5 | bin | +0.234 | +0.217 | -0.017 |
| tgt_breakout_up_10 | bin | +0.229 | +0.217 | -0.011 |
| tgt_vol_expansion_5 | bin | +0.220 | +0.186 | -0.034 |
| tgt_breakout_down_10 | bin | +0.217 | +0.207 | -0.010 |
| tgt_vol_regime_10 | cont | +0.208 | +0.254 | +0.046 |
| tgt_ret_magnitude_1 | cont | +0.204 | +0.346 | +0.142 |
| tgt_crash_10 | bin | +0.193 | +0.166 | -0.027 |
| tgt_tail_event_5 | bin | +0.192 | +0.156 | -0.037 |
| tgt_alpha_5 | cont | +0.172 | +0.233 | +0.061 |
| tgt_max_drawup_long_3 | cont | +0.169 | +0.281 | +0.112 |
| tgt_relative_ret_1 | cont | +0.161 | +0.208 | +0.047 |
| tgt_vol_regime_5 | cont | +0.159 | +0.255 | +0.096 |
| tgt_max_drawup_long_5 | cont | +0.152 | +0.234 | +0.082 |
| tgt_max_drawdown_long_3 | cont | +0.148 | +0.309 | +0.162 |
| tgt_tail_event_3 | bin | +0.146 | +0.125 | -0.021 |
| tgt_crash_5 | bin | +0.145 | +0.102 | -0.043 |
| tgt_alpha_3 | cont | +0.140 | +0.233 | +0.092 |
| tgt_ret_magnitude_3 | cont | +0.126 | +0.260 | +0.134 |
| tgt_max_drawdown_long_5 | cont | +0.122 | +0.281 | +0.160 |
| tgt_liquidation_cascade_5 | bin | +0.119 | +0.193 | +0.074 |
| tgt_liquidation_cascade_3 | bin | +0.119 | +0.174 | +0.056 |
| tgt_max_drawup_long_10 | cont | +0.117 | +0.175 | +0.058 |
| tgt_ret_magnitude_5 | cont | +0.115 | +0.214 | +0.099 |
| tgt_crash_3 | bin | +0.104 | +0.078 | -0.025 |
| tgt_tail_event_1 | bin | +0.094 | +0.093 | -0.001 |
| tgt_regime_change_5 | bin | +0.090 | +0.157 | +0.067 |
| tgt_best_side_3 | cont | +0.080 | +0.020 | -0.060 |
| tgt_autocorr_break_5 | bin | +0.072 | +0.080 | +0.008 |
| tgt_max_drawup_short_10 | cont | +0.067 | +0.267 | +0.200 |
| tgt_regime_change_10 | bin | +0.058 | +0.226 | +0.168 |
| tgt_ret_2 | cont | +0.033 | -0.005 | -0.039 |
| tgt_ret_sign_10 | cont | +0.030 | +0.007 | -0.024 |
| tgt_best_short_entry_pct | cont | +0.030 | +0.037 | +0.007 |
| tgt_recovery_time_5 | cont | +0.029 | +0.042 | +0.013 |
| tgt_optimal_action_5 | cont | +0.029 | +0.011 | -0.018 |
| tgt_optimal_action_1 | cont | +0.024 | -0.003 | -0.027 |
| tgt_ret_risk_adj_1 | cont | +0.020 | +0.003 | -0.017 |
| tgt_pnl_short_bps_3 | cont | +0.018 | -0.002 | -0.020 |
| tgt_cum_ret_5 | cont | +0.012 | -0.013 | -0.025 |
| tgt_adverse_selection_1 | bin | +0.003 | -0.007 | -0.010 |

---

## Impact of Contamination Fix

Several targets that appeared STRONG with contaminated features are now revealed as weak:

| Target | Old Score (contaminated) | New Score (clean) | Verdict |
|--------|-------------------------|-------------------|---------|
| tgt_best_short_entry_pct | +0.503 | +0.030 | **Was fake — swing features inflated it** |
| tgt_relative_ret_1 | +0.422 | +0.161 | Dropped 62%, still predictable |
| tgt_ret_risk_adj_1 | +0.372 | +0.020 | **Was fake — nearly unpredictable** |
| tgt_optimal_action_1 | +0.343 | +0.024 | **Was fake — nearly unpredictable** |
| tgt_range_bps_1 | +0.391 | +0.456 | Improved (was held back by contaminated features) |
| tgt_consolidation_5 | +0.385 | +0.384 | Unchanged — was never using swing features |
| tgt_breakout_up_3 | +0.346 | +0.259 | Dropped 25%, still strong |

**3 targets that were top-10 are now near-zero.** The contamination was severe for return-prediction
targets where `is_swing_high/low` had |r| > 0.20 with the target.

---

## 1h vs 4h Comparison

**1h data:** SOL + DOGE only (no XRP 1h data), 2025-01-01 → 2026-01-01, 1 WFO fold.
**4h data:** SOL + XRP + DOGE, 2024-01-01 → 2026-01-01, 3 WFO folds.

### Overall
- **4h mean score:** 0.166 (across 54 targets, 3 coins)
- **1h mean score:** 0.218 (across 54 targets, 2 coins)
- **1h better:** 29 targets, **1h worse:** 25 targets

### Key Patterns

**Continuous targets are MUCH better at 1h** — more data points, more signal:
- Range/vol/cost targets gain +0.15 to +0.21 at 1h (e.g., `range_bps_1`: 0.456 → 0.666)
- Drawdown/magnitude targets gain +0.10 to +0.16 (e.g., `max_drawdown_long_5`: 0.122 → 0.281)

**Binary targets are slightly worse at 1h** — noisier at shorter timeframe:
- Breakout targets lose ~0.01 to 0.02 (e.g., `breakout_up_3`: 0.259 → 0.237)
- Crash/tail targets lose ~0.02 to 0.06 (e.g., `vol_expansion_10`: 0.317 → 0.259)

### Top 10 Most Predictable (1h)
| Target | 1h Score | Type |
|--------|----------|------|
| tgt_range_bps_1 | +0.666 | continuous |
| tgt_fwd_spread_bps_3 | +0.625 | continuous |
| tgt_range_bps_3 | +0.504 | continuous |
| tgt_inventory_cost_5 | +0.482 | continuous |
| tgt_inventory_cost_3 | +0.472 | continuous |
| tgt_range_bps_5 | +0.460 | continuous |
| tgt_realized_vol_5 | +0.440 | continuous |
| tgt_realized_vol_10 | +0.440 | continuous |
| tgt_consolidation_5 | +0.378 | binary |
| tgt_ret_magnitude_1 | +0.346 | continuous |

---

## Unpredictable / Unusable Targets

### Unpredictable (score ≤ 0 on SOL)
- tgt_trend_strength_5/10, tgt_cum_ret_10, tgt_pnl_long/short_bps_10
- tgt_ret_risk_adj_3, tgt_ret_3

### Unusable Base Rate (>95% or <5%)
- tgt_fill_prob_long/short_1/3/5 (99.8–100%)
- tgt_mean_reversion_5/10 (99.8–99.9%)
- tgt_breakout_any_3/5/10 (96.7–99.6%)
- tgt_mid_reversion_1/3 (99.6–99.7%)

### Revealed as Weak After Contamination Fix
- tgt_best_short_entry_pct (4h avg: +0.030)
- tgt_ret_risk_adj_1 (4h avg: +0.020)
- tgt_optimal_action_1 (4h avg: +0.024)
- tgt_adverse_selection_1 (4h avg: +0.003)

---

## Key Insights

### 1. Contamination Fix Removed 3 Fake Top Targets
The `.shift(-1)` lookahead in swing features inflated `tgt_best_short_entry_pct`, `tgt_ret_risk_adj_1`,
and `tgt_optimal_action_1` from apparent scores of +0.34 to +0.50 down to near-zero (+0.02 to +0.03).
**The old strategy results that used these targets may have been overfitted to lookahead.**

### 2. Volatility/Structure Targets Remain Strong
After the fix, the most predictable targets are all volatility/structure-based:
- **Range/spread** (0.27–0.46): `range_bps_*`, `fwd_spread_bps_3`
- **Consolidation** (0.35–0.38): `consolidation_3/5`
- **Breakout** (0.22–0.26): `breakout_up/down_3/5/10`
- **Vol expansion** (0.22–0.32): `vol_expansion_5/10`

### 3. 1h Is Better for Continuous Targets
Continuous targets (Spearman correlation) benefit from 4× more data at 1h.
The top 8 targets at 1h all score > 0.44, vs max 0.46 at 4h.
**Recommendation: use 1h for vol/range prediction, 4h for binary signals.**

### 4. Predictability Decays with Horizon (Clean)
| Category | 1-bar | 3-bar | 5-bar | 10-bar |
|----------|-------|-------|-------|--------|
| breakout_up | — | +0.26 | +0.24 | +0.23 |
| breakout_down | — | +0.25 | +0.23 | +0.22 |
| crash | — | +0.10 | +0.14 | +0.19 |
| range_bps | +0.46 | +0.32 | +0.27 | — |
| realized_vol | — | — | +0.29 | +0.27 |

### 5. Strategies Need Re-evaluation
The directional and straddle strategies used `tgt_profitable_long/short_3` as meta-targets
(duplicates of `tgt_best_side_3`, score +0.08). The feature selection in those strategies
did NOT exclude contaminated features. **Strategy backtests should be re-run with clean features.**

---

## Files

- `predictable_targets.json` — 54 deduplicated targets with metadata
- `results/target_audit_SOLUSDT_4h.csv` — Full 120-target SOL audit (old, contaminated)
- `results/target_audit_crosscoin_4h.csv` — Cross-coin comparison (old, contaminated)
- `results/target_audit_1h_vs_4h.csv` — **Clean** 1h vs 4h comparison (54 targets × 5 coin/tf combos)
- `target_audit.py` — SOL audit script
- `target_audit_crosscoin.py` — Cross-coin validation script
- `target_audit_1h_vs_4h.py` — 1h vs 4h comparison script (clean features)
