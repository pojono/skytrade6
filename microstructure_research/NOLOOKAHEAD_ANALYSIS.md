# No-Lookahead Pipeline: Cross-Coin Analysis (SOL & XRP)

## Pipeline Design

**Goal:** Validate that microstructure features can predict targets on truly unseen data with zero lookahead bias.

**Method:** Walk-forward expanding window with 3 independent splits:
- **Selection period:** 432d → 522d → 612d (expanding)
- **Purge gap:** 30 days (no data leakage)
- **Test period:** 90 days each (completely unseen)
- **Total unseen test data:** 270 days across 3 non-overlapping windows

**Pipeline per split:**
1. **Tier 2** — Stability scan (90d rolling windows, sign consistency, min effect size)
2. **Tier 4** — Hierarchical clustering → ~320 representative features
3. **Tier 5** — Single-feature walk-forward optimization (expanding window, 120d min train, 45d test)
4. **Tier 6** — Multi-feature holdout (Ridge/LogReg with top 20 features per target)

---

## Summary Results

| Metric | SOLUSDT | XRPUSDT |
|--------|---------|---------|
| **Data range** | 2024-01-01 → 2026-01-01 (732d) | 2024-01-01 → 2026-01-01 (732d) |
| **Splits** | 3 | 3 |
| **Tier A targets** (holdout>0.03, all splits positive) | **38** | **34** |
| **Tier B targets** (holdout>0.01, all positive) | 0 | 1 |
| **Tier C targets** (weak/inconsistent) | 2 | 6 |
| **Tier D targets** (unpredictable) | 0 | 0 |
| **Pipeline time** | 106 min | 108 min |

---

## Top Targets: Cross-Coin Comparison

### Tier 1: Structural/Breakout Targets (R² > 0.25)

| Target | SOL mean | SOL std | XRP mean | XRP std | Cross-coin? |
|--------|----------|---------|----------|---------|-------------|
| `tgt_breakout_any_10` | +0.354 | 0.150 | **+0.465** | 0.019 | ✅ Both strong |
| `tgt_consolidation_5` | **+0.367** | 0.047 | +0.375 | 0.059 | ✅ Both strong |
| `tgt_breakout_any_5` | +0.343 | 0.074 | +0.397 | 0.054 | ✅ Both strong |
| `tgt_consolidation_3` | +0.356 | 0.033 | +0.345 | 0.008 | ✅ Both strong |
| `tgt_breakout_up_3` | +0.335 | 0.027 | +0.344 | 0.013 | ✅ Both strong |
| `tgt_vol_expansion_10` | +0.327 | 0.045 | +0.317 | 0.030 | ✅ Both strong |
| `tgt_breakout_up_5` | +0.315 | 0.019 | +0.329 | 0.019 | ✅ Both strong |
| `tgt_breakout_down_3` | +0.315 | 0.002 | +0.320 | 0.007 | ✅ Both strong |
| `tgt_breakout_any_3` | +0.308 | 0.063 | +0.368 | 0.032 | ✅ Both strong |
| `tgt_breakout_down_5` | +0.305 | 0.009 | +0.298 | 0.002 | ✅ Both strong |
| `tgt_breakout_up_10` | +0.295 | 0.036 | +0.307 | 0.037 | ✅ Both strong |
| `tgt_breakout_down_10` | +0.282 | 0.031 | +0.271 | 0.011 | ✅ Both strong |
| `tgt_crash_10` | +0.250 | 0.029 | +0.205 | 0.037 | ✅ Both strong |
| `tgt_vol_expansion_5` | +0.249 | 0.049 | +0.245 | 0.026 | ✅ Both strong |

**Key finding:** All structural/breakout targets are strongly predictable on BOTH coins with remarkably similar scores. This is the most robust category.

### Tier 2: Risk/Event Targets (R² 0.10–0.25)

| Target | SOL mean | XRP mean | Cross-coin? |
|--------|----------|----------|-------------|
| `tgt_tail_event_5` | +0.229 | +0.212 | ✅ |
| `tgt_liquidation_cascade_3` | +0.219 | +0.165 | ✅ |
| `tgt_tail_event_3` | +0.206 | +0.177 | ✅ |
| `tgt_liquidation_cascade_5` | +0.203 | +0.114 | ✅ |
| `tgt_crash_5` | +0.196 | +0.132 | ✅ |
| `tgt_crash_3` | +0.194 | +0.084 | ✅ (SOL stronger) |
| `tgt_profitable_short_1` | +0.194 | +0.209 | ✅ |
| `tgt_profitable_long_1` | +0.191 | +0.216 | ✅ |
| `tgt_tail_event_1` | +0.174 | +0.089 | ✅ (SOL stronger) |
| `tgt_alpha_1` | +0.144 | +0.106 | ✅ |
| `tgt_relative_ret_1` | +0.144 | +0.106 | ✅ |
| `tgt_profitable_short_3` | +0.128 | +0.100 | ✅ |
| `tgt_profitable_long_3` | +0.125 | +0.096 | ✅ |
| `tgt_regime_change_5` | +0.100 | -0.019 | ❌ SOL only |
| `tgt_adverse_selection_1` | +0.080 | +0.102 | ✅ |

### Tier 3: Weak but Consistent (R² 0.03–0.10)

| Target | SOL mean | XRP mean | Cross-coin? |
|--------|----------|----------|-------------|
| `tgt_profitable_short_5` | +0.096 | +0.097 | ✅ |
| `tgt_profitable_long_5` | +0.092 | +0.098 | ✅ |
| `tgt_profitable_short_10` | +0.075 | +0.063 | ✅ |
| `tgt_profitable_long_10` | +0.073 | +0.070 | ✅ |
| `tgt_autocorr_break_5` | +0.062 | +0.062 | ✅ (identical!) |
| `tgt_alpha_5` / `tgt_relative_ret_5` | +0.045 | — | SOL only |
| `tgt_alpha_3` / `tgt_relative_ret_3` | +0.044 | — | SOL only |
| `tgt_adverse_selection_5` | +0.030 | +0.033 | ✅ |

### Unpredictable Targets

| Target | SOL | XRP | Notes |
|--------|-----|-----|-------|
| `tgt_mid_reversion_1` | +0.082 (2 splits) | **-0.137** | XRP strongly negative |
| `tgt_mid_reversion_3` | +0.055 (2 splits) | -0.033 | Inconsistent |
| `tgt_mean_reversion_5` | +0.241 (2 splits) | +0.014 (inconsistent) | SOL only, not all splits |
| `tgt_regime_change_10` | +0.022 (inconsistent) | **-0.103** | Unpredictable |
| `tgt_ret_sign_*` | 0 splits | 1 split max | ❌ Not predictable |
| `tgt_fill_prob_*` | Inconsistent | Inconsistent | ❌ Unstable |

---

## Key Findings

### 1. Structural Predictability is Real and Cross-Coin
The top 14 targets (all breakout/consolidation/vol_expansion) achieve **R² 0.25–0.47 on completely unseen data** and are consistent across both SOL and XRP. This is not overfitting — the scores are stable across 3 independent temporal splits with 30-day purge gaps.

### 2. Risk Events are Predictable but Coin-Specific
Tail events, crashes, and liquidation cascades are predictable on both coins, but SOL shows stronger signal (R² 0.17–0.23 vs XRP 0.08–0.21). This may reflect SOL's higher volatility providing more training signal.

### 3. Profitability Targets are Weakly but Consistently Predictable
Short-horizon profitability (`profitable_long/short_1`) achieves R² ~0.19–0.22 on both coins. Longer horizons decay to R² ~0.06–0.10 but remain positive.

### 4. Alpha/Return Targets are Barely Predictable
`tgt_alpha_1` and `tgt_relative_ret_1` achieve R² ~0.10–0.14 with very few features (3–6). This is a genuine but small edge. Longer horizons (3, 5) are weaker.

### 5. Mean Reversion and Mid Reversion are NOT Reliably Predictable
These targets show inconsistent or negative holdout scores. `tgt_mid_reversion_1` is actually **anti-predictable** on XRP (R² = -0.14). This is a critical finding — mean reversion signals from microstructure features are unreliable.

### 6. Return Sign is NOT Predictable
`tgt_ret_sign_*` targets fail completely — they appear in at most 1 split and never consistently. Microstructure features cannot reliably predict the direction of returns.

### 7. Fill Probability is Unstable
`tgt_fill_prob_*` targets show high variance across splits and are not reliably predictable.

---

## Stability Analysis (Per-Split Heatmaps)

### SOLUSDT
- Top targets show remarkable consistency: `tgt_consolidation_5` ranges from +0.315 to +0.405 across splits
- `tgt_breakout_down_3` is the most stable: +0.313, +0.317, +0.313 (std = 0.002!)
- Later splits (more training data) don't always score higher — suggesting the signal is genuine, not just from more data

### XRPUSDT
- `tgt_breakout_any_10` is extremely stable: +0.481, +0.469, +0.444
- `tgt_breakout_down_3` also very stable: +0.314, +0.321, +0.327
- `tgt_consolidation_3` remarkably tight: +0.336, +0.351, +0.347

---

## Actionable Targets for Trading

### High Confidence (use for live trading signals)
These targets are predictable on both coins, all 3 splits, with R² > 0.20:

1. **Breakout detection** (`breakout_any/up/down` at 3/5/10 horizons) — R² 0.27–0.47
2. **Consolidation detection** (`consolidation_3/5`) — R² 0.31–0.38
3. **Volatility expansion** (`vol_expansion_5/10`) — R² 0.25–0.33
4. **Crash prediction** (`crash_10`) — R² 0.21–0.25
5. **Tail event prediction** (`tail_event_3/5`) — R² 0.18–0.23
6. **Profitability** (`profitable_long/short_1`) — R² 0.19–0.22

### Medium Confidence (use with caution)
7. **Alpha/returns** (`alpha_1`, `relative_ret_1`) — R² 0.10–0.14
8. **Adverse selection** (`adverse_selection_1`) — R² 0.08–0.10

### Do NOT Use
- `tgt_ret_sign_*` — not predictable
- `tgt_mid_reversion_*` — anti-predictable on some coins
- `tgt_fill_prob_*` — unstable
- `tgt_regime_change_10` — negative on XRP

---

## Pipeline Parameters
- Tier 2: 90d window, 30d step, sign consistency ≥ 0.55, min effect size 0.02
- Tier 4: Spearman correlation threshold 0.7, hierarchical clustering
- Tier 5: Expanding window, 120d min train, 45d test, 2d purge, mean OOS > 0.02, 60% positive folds
- Tier 6: Ridge/LogReg with top 20 features, trained on full selection period
- Split design: 30d purge, 90d test, 3 splits with expanding selection
