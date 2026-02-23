# Target Audit — ML Predictability Validation

**Date:** 2026-02-23
**Asset:** SOLUSDT 4h, 4392 candles (2024-01-01 → 2026-01-01)
**Method:** 3-fold expanding-window WFO, LightGBM, top-30 features per target
**Scoring:** Binary = AUC - 0.5 (>0 = better than random), Continuous = Spearman correlation

---

## Summary

| Verdict | Count | Criteria |
|---------|-------|----------|
| **STRONG** | 84 | All folds positive, mean score > 0.05 |
| **GOOD** | 2 | All folds positive, mean score 0.02–0.05 |
| **WEAK** | 7 | Mean > 0.02 but inconsistent across folds |
| **MARGINAL** | 7 | Mean 0–0.02, unreliable |
| **UNPREDICTABLE** | 7 | Mean ≤ 0, no signal |
| **SKIP (base rate)** | 13 | Base rate <5% or >95%, unusable for ML |
| **Total** | **120** | |

**Key finding:** 84 of 120 targets (70%) are STRONGLY predictable. Our strategies currently use only ~15 of them.

---

## Tier 1: Top Predictable Targets (score > 0.20)

These have very strong, consistent OOS signal across all folds.

| Target | Type | Score | Base Rate | Currently Used? |
|--------|------|-------|-----------|-----------------|
| tgt_best_long_entry_pct | cont | +0.503 | — | No |
| tgt_best_short_entry_pct | cont | +0.503 | — | No |
| tgt_slippage_long_bps | cont | +0.499 | — | No |
| tgt_slippage_short_bps | cont | +0.499 | — | No |
| tgt_alpha_1 | cont | +0.422 | — | **Yes** |
| tgt_relative_ret_1 | cont | +0.422 | — | **Yes** |
| tgt_range_bps_1 | cont | +0.391 | — | No |
| tgt_fwd_spread_bps_3 | cont | +0.386 | — | No |
| tgt_consolidation_5 | binary | +0.385 | 6.1% | No |
| tgt_ret_risk_adj_1 | cont | +0.372 | — | No |
| tgt_ret_1 | cont | +0.363 | — | No |
| tgt_pnl_long_bps_1 | cont | +0.363 | — | No |
| tgt_pnl_short_bps_1 | cont | +0.363 | — | No |
| tgt_consolidation_3 | binary | +0.353 | 14.8% | **Yes** (regime) |
| tgt_breakout_up_3 | binary | +0.346 | 66.2% | **Yes** |
| tgt_optimal_action_1 | cont | +0.343 | — | No |
| tgt_best_side_1 | cont | +0.335 | — | No |
| tgt_ret_sign_1 | cont | +0.335 | — | No |
| tgt_fwd_spread_bps_1 | cont | +0.332 | — | No |
| tgt_breakout_down_3 | binary | +0.327 | 64.1% | **Yes** |
| tgt_vol_expansion_10 | binary | +0.322 | 49.2% | **Yes** |
| tgt_breakout_down_5 | binary | +0.298 | 71.7% | **Yes** |
| tgt_range_bps_3 | cont | +0.290 | — | No |
| tgt_max_drawdown_long_3 | cont | +0.282 | — | No |
| tgt_max_drawup_short_3 | cont | +0.282 | — | No |
| tgt_breakout_down_10 | binary | +0.271 | 78.7% | **Yes** |
| tgt_breakout_up_5 | binary | +0.263 | 73.1% | **Yes** |
| tgt_vol_regime_10 | cont | +0.261 | — | No |
| tgt_max_drawup_long_3 | cont | +0.259 | — | No |
| tgt_max_drawdown_short_3 | cont | +0.259 | — | No |
| tgt_realized_vol_5 | cont | +0.258 | — | No |
| tgt_breakout_up_10 | binary | +0.254 | 81.2% | **Yes** |
| tgt_inventory_cost_5 | cont | +0.234 | — | No |
| tgt_vol_expansion_5 | binary | +0.233 | 39.4% | **Yes** |
| tgt_vol_regime_5 | cont | +0.229 | — | No |
| tgt_realized_vol_10 | cont | +0.224 | — | No |
| tgt_crash_10 | binary | +0.219 | 16.0% | **Yes** (regime) |
| tgt_tail_event_5 | binary | +0.219 | 35.5% | **Yes** (regime) |
| tgt_alpha_3 | cont | +0.202 | — | No |
| tgt_relative_ret_3 | cont | +0.202 | — | No |

## Tier 2: Good Predictable Targets (score 0.05–0.20)

| Target | Type | Score | Base Rate | Currently Used? |
|--------|------|-------|-----------|-----------------|
| tgt_profitable_short_1 | binary | +0.197 | 48.2% | **Yes** |
| tgt_alpha_5 | cont | +0.189 | — | No |
| tgt_relative_ret_5 | cont | +0.189 | — | No |
| tgt_profitable_long_1 | binary | +0.189 | 49.4% | **Yes** |
| tgt_liquidation_cascade_5 | binary | +0.188 | 9.8% | No |
| tgt_liquidation_cascade_3 | binary | +0.185 | 5.7% | No |
| tgt_max_drawdown/drawup_5 | cont | ~+0.179 | — | No |
| tgt_tail_event_3 | binary | +0.175 | 24.9% | **Yes** (regime) |
| tgt_crash_5 | binary | +0.171 | 8.6% | No |
| tgt_crash_3 | binary | +0.167 | 5.4% | No |
| tgt_tail_event_1 | binary | +0.150 | 9.6% | No |
| tgt_ret_magnitude_1/3/5 | cont | +0.094–0.145 | — | No |
| tgt_cum_ret_3 | cont | +0.136 | — | No (used for threshold cal.) |
| tgt_max_drawdown/drawup_10 | cont | ~+0.113 | — | No |
| tgt_adverse_selection_1 | binary | +0.103 | 51.0% | **Yes** |
| tgt_profitable_short_3 | binary | +0.103 | 48.7% | **Yes** (meta target) |
| tgt_regime_change_5 | binary | +0.096 | 61.0% | No |
| tgt_profitable_long_3 | binary | +0.090 | 49.9% | **Yes** (meta target) |
| tgt_cum_ret_5 | cont | +0.083 | — | No |
| tgt_recovery_time_5 | cont | +0.081 | — | No |
| tgt_autocorr_break_5 | binary | +0.078 | 35.5% | No |
| tgt_best_side_5 | cont | +0.074 | — | No |
| tgt_kelly_5 | cont | +0.073 | — | No |
| tgt_regime_change_10 | binary | +0.066 | 62.7% | No |
| tgt_profitable_long_5 | binary | +0.056 | 49.2% | **Yes** |
| tgt_profitable_short_5 | binary | +0.053 | 49.8% | **Yes** |

## Tier 3: Weak/Marginal (score 0–0.05)

Not recommended as base models — too noisy for reliable signal.

| Target | Type | Score | Verdict |
|--------|------|-------|---------|
| tgt_ret_2 | cont | +0.042 | GOOD |
| tgt_adverse_selection_3 | binary | +0.033 | WEAK |
| tgt_recovery_time_10 | cont | +0.029 | WEAK |
| tgt_ret_sign_10 | cont | +0.025 | GOOD |
| tgt_profitable_long_10 | binary | +0.025 | WEAK |
| tgt_sharpe_10 | cont | +0.024 | WEAK |
| tgt_optimal_horizon | cont | +0.025 | WEAK |
| tgt_ret_risk_adj_5 | cont | +0.022 | WEAK |
| tgt_sharpe_5 | cont | +0.115 | WEAK (inconsistent) |
| tgt_profitable_short_10 | binary | +0.015 | MARGINAL |
| tgt_ret_10 | cont | +0.013 | MARGINAL |
| tgt_ret_sign_3/5 | cont | +0.008–0.012 | MARGINAL |
| tgt_kelly_10 | cont | +0.005 | MARGINAL |
| tgt_adverse_selection_5 | binary | +0.001 | MARGINAL |

## Tier 4: Unpredictable (score ≤ 0)

DO NOT USE — no OOS signal or anti-predictable.

| Target | Type | Score |
|--------|------|-------|
| tgt_trend_strength_10 | cont | -0.004 |
| tgt_cum_ret_10 | cont | -0.013 |
| tgt_pnl_long/short_bps_10 | cont | -0.013 |
| tgt_ret_risk_adj_3 | cont | -0.015 |
| tgt_ret_3 | cont | -0.024 |
| tgt_trend_strength_5 | cont | -0.029 |

## Tier 5: Unusable Base Rate (skipped)

These have >95% or <5% positive rate — ML classifiers can't learn from them.

| Target | Base Rate | Issue |
|--------|-----------|-------|
| tgt_fill_prob_long_5 | 100.0% | Always true |
| tgt_fill_prob_long_3 | 99.9% | Always true |
| tgt_mean_reversion_10 | 99.9% | Always true |
| tgt_fill_prob_short_1/3/5 | 99.8% | Always true |
| tgt_fill_prob_long_1 | 99.8% | Always true |
| tgt_mean_reversion_5 | 99.8% | Always true |
| tgt_breakout_any_5 | 98.8% | Always true |
| tgt_mid_reversion_1/3 | 99.6–99.7% | Always true |
| tgt_breakout_any_3 | 96.7% | Always true |
| tgt_breakout_any_10 | 99.6% | Always true |

---

## What We Currently Use vs What's Available

### Currently Used (15 targets as base models + 2 meta targets):
- breakout_up/down_3/5/10 (6 models) — all STRONG ✓
- vol_expansion_5/10 (2 models) — all STRONG ✓
- profitable_long/short_1/5 (4 models) — all STRONG ✓
- alpha_1, relative_ret_1 (2 continuous) — STRONG ✓
- adverse_selection_1 (1 model) — STRONG ✓
- consolidation_3, tail_event_3/5, crash_10 (4 regime) — all STRONG ✓
- Meta targets: profitable_long/short_3 — STRONG ✓

### High-Value Unused Targets (potential new base models):

| Target | Score | Why Valuable |
|--------|-------|-------------|
| **tgt_best_long/short_entry_pct** | +0.503 | Optimal entry timing — could improve entry execution |
| **tgt_slippage_long/short_bps** | +0.499 | Predict slippage — could adjust position sizing |
| **tgt_range_bps_1/3** | +0.290–0.391 | Predict volatility magnitude — size positions accordingly |
| **tgt_fwd_spread_bps_1/3** | +0.332–0.386 | Predict future spread — execution cost awareness |
| **tgt_ret_risk_adj_1** | +0.372 | Risk-adjusted return — better than raw return |
| **tgt_ret_1** | +0.363 | 1-bar return — very strong short-term signal |
| **tgt_optimal_action_1** | +0.343 | Optimal long/short/flat — direct trading signal |
| **tgt_ret_sign_1** | +0.335 | Direction prediction — strong at 1-bar |
| **tgt_vol_regime_5/10** | +0.229–0.261 | Regime classification — could replace vol gate |
| **tgt_realized_vol_5/10** | +0.224–0.258 | Forward vol — position sizing |
| **tgt_max_drawdown_long/short_3** | +0.259–0.282 | Risk prediction — stop-loss calibration |
| **tgt_inventory_cost_3/5/10** | +0.216–0.234 | Holding cost — optimal hold period |
| **tgt_alpha_3/5** | +0.189–0.202 | Multi-bar alpha — longer horizon signals |
| **tgt_liquidation_cascade_3/5** | +0.185–0.188 | Cascade risk — avoid dangerous entries |
| **tgt_regime_change_5/10** | +0.066–0.096 | Regime shifts — adapt strategy |

### Pattern: Predictability Decays with Horizon

| Category | 1-bar | 3-bar | 5-bar | 10-bar |
|----------|-------|-------|-------|--------|
| alpha/relative_ret | +0.422 | +0.202 | +0.189 | — |
| ret | +0.363 | -0.024 | +0.009 | +0.013 |
| profitable_long | +0.189 | +0.090 | +0.056 | +0.025 |
| profitable_short | +0.197 | +0.103 | +0.053 | +0.015 |
| cum_ret | — | +0.136 | +0.083 | -0.013 |
| max_drawdown_long | — | +0.282 | +0.178 | +0.113 |

**Conclusion:** 1-bar and 3-bar targets are most predictable. Signal decays significantly by 10 bars. This confirms our hold-3 with early exit is optimal.

---

## Recommendations

### Immediate (add to existing strategies):
1. Add `tgt_ret_1` and `tgt_ret_sign_1` as base models — very strong directional signal
2. Add `tgt_optimal_action_1` — direct long/short/flat signal
3. Add `tgt_range_bps_1` — volatility magnitude for sizing
4. Add `tgt_max_drawdown_long/short_3` — risk awareness for the meta-model

### Medium-term (new strategy ideas):
5. Build execution-aware strategy using `tgt_slippage`, `tgt_fwd_spread`, `tgt_best_entry_pct`
6. Use `tgt_liquidation_cascade` as a risk filter (avoid entries before cascades)
7. Use `tgt_vol_regime` as a smarter replacement for the vol gate

### Targets to Remove from predictable_targets.json:
- `tgt_profitable_long_10` (WEAK, score +0.025)
- `tgt_profitable_short_10` (MARGINAL, score +0.015)

### Targets to Add to predictable_targets.json:
All 84 STRONG targets should be in the validated list. Key additions:
- `tgt_ret_1`, `tgt_ret_sign_1`, `tgt_optimal_action_1/3/5`
- `tgt_range_bps_1/3/5`, `tgt_ret_magnitude_1/3/5`
- `tgt_max_drawdown/drawup_long/short_3/5/10`
- `tgt_realized_vol_5/10`, `tgt_vol_regime_5/10`
- `tgt_inventory_cost_3/5/10`, `tgt_fwd_spread_bps_1/3`
- `tgt_slippage_long/short_bps`, `tgt_best_long/short_entry_pct`
- `tgt_liquidation_cascade_3/5`, `tgt_regime_change_5/10`
- `tgt_crash_3/5`, `tgt_consolidation_5`, `tgt_tail_event_1`
- `tgt_alpha_3/5`, `tgt_relative_ret_3/5`
- `tgt_ret_risk_adj_1`, `tgt_best_side_1/3/5`
- `tgt_kelly_5`, `tgt_recovery_time_5`, `tgt_autocorr_break_5`
