# Research Findings v11 — Direct Range Prediction & Quantile Regression

**Date:** 2026-02-16
**Exchange:** Bybit Futures (VIP0)
**Symbol:** BTCUSDT (results generalize across symbols per v9/v10)
**Period:** 2025-01-01 → 2026-01-31 (396 days, ~114K bars)
**Method:** Walk-forward time-series CV, 45+ backward-looking features, CPU-only ML
**Runtime:** 18.3 min

---

## Motivation

In v9/v10 we proved we can predict **volatility** (R²=0.34 at 1h, r=0.59). But a grid bot doesn't need volatility — it needs the **expected price range** (high − low) to set grid width. This experiment tests:

1. **Phase 1:** Can we predict range directly? Is it better than converting from vol?
2. **Phase 2:** Can we predict quantiles of range (P10, P50, P90) for confidence bands?

The goal is to answer: *"If BTC is at $100K right now, what price band should my grid cover for the next 1h / 4h?"*

---

## Key Data Properties

| Property | 1h (12 bars) | 4h (48 bars) |
|----------|-------------|-------------|
| Mean range (% of price) | 0.629% | 1.299% |
| Median range (% of price) | 0.489% | 1.027% |
| Mean range ($, at $100K) | **$629** | **$1,299** |
| Median range ($, at $100K) | **$489** | **$1,027** |
| Range/Vol ratio | 5.87× | 11.18× |
| Correlation(range, vol) | **0.894** | **0.899** |
| Asymmetry (upside/range) | 0.499 | 0.496 |

**Key observation:** Range and vol are 89–90% correlated, but not identical. Range captures the full high-low swing including intra-bar extremes that return-based vol misses.

**Asymmetry is almost exactly 0.50** — on average, upside and downside are symmetric. This is important: it means we can't predict direction from these features (confirmed in experiments below).

---

## Phase 1: Direct Range Prediction

### 1h Horizon Results

| Method | R² | Corr | MAE | Coverage |
|--------|-----|------|-----|----------|
| **Range (Ridge)** | **0.326** | **0.579** | 0.00215 | 0.599 |
| Range (LGBM) | 0.303 | 0.557 | 0.00223 | 0.644 |
| Vol→Range (Ridge) | 0.324 | 0.578 | 0.00209 | 0.564 |
| Upside (Ridge) | 0.129 | 0.371 | 0.00194 | — |
| Downside (Ridge) | 0.130 | 0.366 | 0.00204 | — |
| Asymmetry (Ridge) | -0.007 | 0.012 | 0.278 | 0.502 |

### 4h Horizon Results

| Method | R² | Corr | MAE | Coverage |
|--------|-----|------|-----|----------|
| **Range (Ridge)** | 0.190 | 0.467 | 0.00483 | 0.564 |
| Range (LGBM) | 0.135 | 0.414 | 0.00510 | 0.622 |
| **Vol→Range (Ridge)** | **0.196** | **0.475** | 0.00469 | 0.535 |
| Upside (Ridge) | 0.037 | 0.273 | 0.00403 | — |
| Downside (Ridge) | 0.070 | 0.286 | 0.00435 | — |
| Asymmetry (Ridge) | -0.018 | 0.008 | 0.273 | 0.503 |

### Phase 1 Takeaways

1. **Direct range prediction ≈ vol-derived range.** At 1h, Ridge range R²=0.326 vs vol-derived R²=0.324. At 4h, vol-derived is marginally better (0.196 vs 0.190). The 89% correlation between range and vol means predicting either one gives you the other.

2. **Ridge beats LGBM again.** Consistent with v9 — linear models outperform tree-based models for this task.

3. **Upside and downside are individually harder to predict** (R²=0.13 at 1h, 0.04–0.07 at 4h). This makes sense: predicting the total range is easier than predicting which direction the range extends.

4. **Asymmetry is completely unpredictable** (R²≈0, corr≈0, directional accuracy=50.2%). Our backward-looking features contain **zero information about whether the next move will be up or down**. This is a strong result — it means these features are purely volatility indicators, not directional predictors.

---

## Phase 2: Quantile Regression

### Calibration Results — 1h Horizon

| Quantile | LGBM (actual below) | Linear (actual below) | Target | LGBM Cal Error |
|----------|---------------------|----------------------|--------|----------------|
| P10 | 0.155 | **0.120** | 0.10 | 0.055 |
| P25 | 0.302 | — | 0.25 | 0.058 |
| **P50** | 0.527 | **0.524** | 0.50 | 0.041 |
| P75 | 0.740 | — | 0.75 | 0.021 |
| **P90** | 0.876 | **0.905** | 0.90 | 0.024 |

### Calibration Results — 4h Horizon

| Quantile | LGBM (actual below) | Linear (actual below) | Target | LGBM Cal Error |
|----------|---------------------|----------------------|--------|----------------|
| P10 | 0.181 | **0.123** | 0.10 | 0.081 |
| P25 | 0.321 | — | 0.25 | 0.072 |
| **P50** | 0.522 | **0.502** | 0.50 | 0.043 |
| P75 | 0.722 | — | 0.75 | 0.028 |
| **P90** | 0.848 | **0.904** | 0.90 | 0.052 |

### Band Coverage

| Band | 1h Coverage (expected) | 4h Coverage (expected) |
|------|----------------------|----------------------|
| P10–P90 | 72.4% (80%) | 66.6% (80%) |
| P25–P75 | 44.2% (50%) | 40.5% (50%) |

### Phase 2 Takeaways

1. **Linear quantile regression is better calibrated than LGBM.** At P90, linear achieves 0.905 actual (target 0.90) vs LGBM's 0.876. LGBM tends to be overconfident at the tails — its P10 captures 15.5% instead of 10%.

2. **P50 and P90 are well-calibrated.** Both models achieve P50 actual ≈ 0.52 (close to 0.50) and P90 actual ≈ 0.88–0.90. These are usable for grid sizing.

3. **P10 is slightly miscalibrated.** LGBM's P10 captures 15–18% instead of 10%. This means the "minimum expected range" is slightly overestimated — the model is conservative at the lower bound.

4. **Band coverage is ~10% below ideal.** P10–P90 covers 72% instead of 80%. This is because the tails are harder to predict — extreme ranges (very calm or very volatile periods) are less predictable.

---

## Practical Grid Sizing

### 1h Grid (BTC at ~$100K)

| Strategy | Grid Width | Coverage | Notes |
|----------|-----------|----------|-------|
| **Adaptive P50** | **$460** | 53.8% | Price stays within grid ~54% of the time |
| **Adaptive P90** | **$786** | 86.8% | Price stays within grid ~87% of the time |
| Fixed (median) | $440 | 54.6% | Static grid at historical median |

### 4h Grid (BTC at ~$100K)

| Strategy | Grid Width | Coverage | Notes |
|----------|-----------|----------|-------|
| **Adaptive P50** | **$911** | 50.6% | Typical expected range |
| **Adaptive P90** | **$1,670** | 84.6% | Conservative — captures most moves |
| Fixed (median) | $924 | 53.7% | Static grid at historical median |

### Adaptive vs Fixed Grid Efficiency

The real value of adaptive grids is **not in average coverage** (which is similar to fixed) but in **regime-appropriate sizing**:

#### 1h Horizon
| Regime | Adaptive P50 | Fixed Grid | Difference |
|--------|-------------|-----------|------------|
| **Low-vol periods** | $277 | $441 | **Saves $164** (37% tighter) |
| **High-vol periods** | $697 | $437 | **Wider by $260** (60% wider) |

#### 4h Horizon
| Regime | Adaptive P50 | Fixed Grid | Difference |
|--------|-------------|-----------|------------|
| **Low-vol periods** | $638 | $926 | **Saves $288** (31% tighter) |
| **High-vol periods** | $1,159 | $920 | **Wider by $239** (26% wider) |

**This is the key result:** During calm markets, the adaptive grid is 31–37% tighter than fixed, concentrating capital on levels that actually get filled. During volatile markets, it's 26–60% wider, preventing the grid from being overrun.

---

## Feature Importance for Range Prediction

### 1h Horizon — Top 10

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | parkvol_24h | 5.5% |
| 2 | parkvol_8h | 4.9% |
| 3 | vol_sma_24h | 4.6% |
| 4 | parkvol_1h | 4.0% |
| 5 | price_vs_sma_24h | 3.8% |
| 6 | rvol_24h | 3.7% |
| 7 | vol_accel_4h | 3.7% |
| 8 | adx_4h | 3.6% |
| 9 | bar_eff_4h | 3.6% |
| 10 | rvol_8h | 3.5% |

Same top features as vol prediction (v9/v10) — confirms range and vol are driven by the same underlying signal.

---

## Summary of Findings

### What Works

| Capability | Accuracy | Practical Use |
|-----------|----------|---------------|
| **Predict 1h range magnitude** | R²=0.33, r=0.58 | Set adaptive grid width |
| **Predict 4h range magnitude** | R²=0.19, r=0.47 | Set wider grid parameters |
| **P50 quantile (median range)** | Cal error 2.7–4.3% | "Typical" grid width |
| **P90 quantile (wide range)** | Cal error 1.2–5.2% | "Safe" grid width (87% coverage) |
| **Regime-adaptive sizing** | 31–37% tighter in calm, 26–60% wider in chaos | Capital efficiency |

### What Doesn't Work

| Capability | Result | Why |
|-----------|--------|-----|
| **Predict direction (asymmetry)** | R²≈0, dir_acc=50% | Features are vol indicators, not directional |
| **Predict upside/downside separately** | R²=0.04–0.13 | Decomposing range into up/down loses signal |
| **Direct range vs vol-derived** | Identical accuracy | Range and vol are 89% correlated |

### What This Means for the Grid Bot

1. **Use Ridge vol prediction (from v9/v10) × scaling factor** to estimate range. No need for a separate range model — vol-derived range is equally good.

2. **Use linear quantile regression for P90** to set the "safe" grid boundary. This gives 87% coverage — only 13% of periods will exceed the grid.

3. **Don't try to predict direction** with these features. Grid should be symmetric around current price. Directional bias would require different features (funding rate, order flow, sentiment).

4. **Recommended grid sizing formula:**
   ```
   grid_width = predicted_vol × k × safety_factor
   where:
     k ≈ 5.6 (1h) or 10.6 (4h)  — empirical range/vol ratio
     safety_factor = 1.0 (P50, typical) or 1.7 (P90, conservative)
   ```

---

## Files

| File | Description |
|------|-------------|
| `regime_ml_range.py` | Phase 1 + Phase 2 experiment suite |
| `results/regime_ML_range_BTC.txt` | Complete BTCUSDT output |
| `FINDINGS_v10_multi_horizon.md` | Previous multi-horizon vol results |
