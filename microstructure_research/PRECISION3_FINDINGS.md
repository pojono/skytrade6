# Precision3 Strategy — Findings & Audit

**Date:** 2026-02-23
**Strategy:** Precision3 (5 models, no meta-stacking)
**Version:** v3 — after full audit and contaminated feature removal
**Status:** ❌ **STRATEGY DOES NOT WORK** — previous results were driven by lookahead features

---

## Critical Bug Found: Feature Lookahead via Swing Points

### The Bug

In `microstructure_features.py`, the swing point detection uses `.shift(-1)` (future data):

```python
# line 4001-4002 of microstructure_features.py
swing_high = (h > h.shift(1)) & (h > h.shift(-1))  # ← LOOKS AT NEXT CANDLE
swing_low  = (l < l.shift(1)) & (l < l.shift(-1))   # ← LOOKS AT NEXT CANDLE
```

This contaminates **36 features** derived from swing points:
`is_swing_high`, `is_swing_low`, `last_swing_high`, `last_swing_low`,
`bos_*`, `choch_*`, `ob_demand*`, `ob_supply*`, `ob_net*`,
`liq_sweep_*`, `dist_to_swing_*`, `ew_swing_count*`, `fib_swing_count*`

### Impact on Feature Selection

When `auto_select_features` ranks features by Spearman correlation with `tgt_ret_1`:
- **#1:** `is_swing_low` (corr=0.261) ← CONTAMINATED
- **#2:** `is_swing_high` (corr=0.228) ← CONTAMINATED
- **#3:** `dist_to_swing_high_bps` (corr=0.071) ← CONTAMINATED
- #4: `overlap_asia_europe` (corr=0.057) — first clean feature
- **6 of top 30** features for direction model are contaminated
- **1 of top 30** for stop-loss model (`is_swing_low` at #4)

The contaminated features have 4-5x higher correlation than any clean feature,
so the model was essentially learning "if next candle confirms a swing point,
the return is predictable" — which is trivially true but unknowable in real time.

---

## Results Comparison: Before vs After Bug Fix

### v2 (WITH contaminated features — INVALID)

| Metric | SOLUSDT | DOGEUSDT | XRPUSDT |
|--------|---------|----------|---------|
| Net Return | +400.9% | +292.0% | +98.6% |
| Sharpe | 7.33 | 4.64 | 2.23 |
| Profit Factor | 3.07 | 2.01 | 1.42 |
| Max Drawdown | -12.7% | -15.8% | -15.6% |
| Periods Positive | 12/12 | 12/12 | 9/12 |

### v3 (WITHOUT contaminated features — VALID)

| Metric | SOLUSDT | DOGEUSDT | XRPUSDT |
|--------|---------|----------|---------|
| Net Return | **+37.2%** | **-21.4%** | **-8.2%** |
| Sharpe | **0.50** | **-0.38** | **-0.83** |
| Profit Factor | **1.10** | **0.93** | **0.85** |
| Max Drawdown | **-56.6%** | **-50.5%** | **-39.5%** |
| Periods Positive | **7/12** | **3/12** | **7/12** |

### Conclusion

**The strategy is not profitable without the contaminated features.**
SOL drops from +401% to +37% (barely positive, Sharpe 0.50).
DOGE and XRP become net losers. The "edge" was entirely driven by
features that peek at the next candle.

---

## Full Audit Checklist

### ✅ Items that passed audit

- **WFO boundaries:** 18-bar (3-day) purge gap between selection and trade windows, no overlap confirmed for all 12 periods
- **Trade entry:** Signal at bar close → entry at bar+1 open (no lookahead)
- **Trade exit:** Exit at open of exit bar (not close)
- **Stop-loss:** Detected at bar close using low/high, exit at next bar open (conservative)
- **Fee handling:** 4 bps round-trip applied per trade (reasonable for major exchanges)
- **Feature selection (v2+):** Spearman correlation on training data only, re-selected per CV fold
- **Inner CV:** Features re-selected on CV-train split only
- **Targets:** All use `.shift(-N)` correctly (forward-looking by design)
- **Rolling features:** All use backward-looking windows (pandas default)
- **Lag features:** Use `.shift(1,2,3)` — past only
- **Z-score features:** Rolling mean/std — backward-looking only

### ❌ Items that FAILED audit

- **Swing point features:** `is_swing_high/low` use `.shift(-1)` — **FUTURE DATA**
- **36 derived features** inherit this contamination (BOS, CHoCH, order blocks, liquidity sweeps, etc.)
- **These features dominate feature selection** for the direction model (top 2 by correlation)

### ⚠️ Minor concerns (not bugs)

- **WFO is sliding window, not expanding:** Each period uses a fixed 360-day window, not all data from start. Docstring says "expanding" but implementation is sliding. Not a correctness issue.
- **Stop-loss exit pricing:** When stop is hit, exit is at next bar open (which could be better or worse than stop price). Minor impact since only ~4-8% of trades hit stops.

---

## Design Philosophy (still valid)

5 purpose-built models, NO meta-model stacking:

| Model | Target | Type | Role |
|-------|--------|------|------|
| Direction | `tgt_ret_1` | Regressor | Predict 1-bar return → long/short/flat |
| Vol Sizing | `tgt_realized_vol_5` | Regressor | Inverse-vol position sizing |
| Stop Loss | `tgt_max_drawdown_long_3` | Regressor | Adaptive stop-loss placement |
| Filter 1 | `tgt_consolidation_3` | Classifier | Skip low-opportunity bars |
| Filter 2 | `tgt_crash_10` | Classifier | Halve size before crashes |

The architecture is sound. The problem is that the clean features don't have
enough predictive power for `tgt_ret_1` to generate a tradeable edge.

---

## Lessons Learned

1. **Always audit `.shift(-N)` in feature code** — any negative shift in a feature (not target) is lookahead.
2. **Suspiciously good results should trigger deeper audits** — Sharpe 7.33 on 12 OOS periods is too good to be true.
3. **Feature correlation analysis should be part of every backtest** — if the top features have 4-5x higher correlation than the rest, investigate why.
4. **The "it improved after removing lookahead" red flag** — when v2 (removing JSON feature lists) improved over v1, that was a sign something else was wrong. The contaminated features were still present and being auto-selected.

---

## Next Steps

1. **Fix `microstructure_features.py`** — replace `.shift(-1)` in swing detection with a purely backward-looking approach (e.g., confirmed swing = bar[i-2] is a swing if bar[i-1] already reversed)
2. **Regenerate all feature parquet files** with clean swing features
3. **Re-evaluate all strategies** (1, 2, 3) that may have used contaminated features
4. **Investigate whether Strategy 1/2 results are also inflated** by the same bug
