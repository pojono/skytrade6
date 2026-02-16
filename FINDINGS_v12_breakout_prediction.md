# Research Findings v12 — Breakout & S/R Level Prediction

**Date:** 2026-02-16
**Symbol:** BTCUSDT
**Period:** 2025-01-01 → 2026-01-31 (396 days, ~114K bars at 5m)
**Method:** Walk-forward time-series CV, automated feature selection, CPU-only ML
**Runtime:** 11.8 min

---

## Motivation

In v9–v11 we proved we can predict **volatility magnitude** and **price range**. The next question: can we predict **breakouts** — moments when price makes an unusually large move — and whether **support/resistance levels** will hold or break?

This is harder than vol prediction because it's partially directional and event-driven rather than continuous.

---

## Feature Engineering

### Starting Point: 102 Candidate Features

We designed 102 features across 7 categories:

| Category | Count | Examples |
|----------|-------|---------|
| **Existing vol** (from v9) | 19 | parkvol, rvol, vol ratios, vol accel |
| **Existing trend** (from v9) | 26 | efficiency, ADX, momentum, imbalance |
| **ATR / Normalized ATR** | 10 | atr_1h..24h, natr_1h..24h |
| **Bollinger / Compression** | 9 | bb_width, bb_pctile, range_compression |
| **Price position / S/R** | 16 | price_position, dist_to_high/low, touches |
| **Approach / Consolidation** | 10 | approach_speed, consolidation ratios |
| **Candle / Distribution** | 12 | body_ratio, wick_ratio, skew, kurtosis |

### Automated Feature Selection Pipeline

For each experiment, we applied:
1. **Target correlation filter:** Drop features with |corr(feature, target)| < 0.02
2. **Inter-correlation filter:** For pairs with |corr| > 0.90, keep the one with higher target correlation

Results:
- **1h breakout:** 102 → 48 features (35 dropped for weak signal, 19 for redundancy)
- **4h breakout:** 102 → 49 features
- **S/R 24h/1h:** 104 → 60 features (S/R-specific features added)

### Top Features by Target Correlation

**Phase 1 — Breakout occurrence (1h):**

| Rank | Feature | Target Corr | Category |
|------|---------|------------|----------|
| 1 | atr_1h | 0.155 | ATR |
| 2 | range_compression_1h | 0.149 | Compression |
| 3 | bb_pctile_2h | 0.147 | Bollinger |
| 4 | bb_width_2h | 0.144 | Bollinger |
| 5 | bar_eff_1h | 0.143 | Trend |

**Phase 2 — S/R break (24h lookback, 1h forward):**

| Rank | Feature | Target Corr | Category |
|------|---------|------------|----------|
| 1 | **consolidation_2h_vs_24h** | **0.297** | Consolidation |
| 2 | **approach_speed_30m** | **0.274** | Approach |
| 3 | consolidation_4h_vs_24h | 0.268 | Consolidation |
| 4 | vol_ratio_bar | 0.262 | Volume |
| 5 | approach_speed_1h | 0.260 | Approach |

**Key insight:** The new breakout-specific features (consolidation, approach speed, Bollinger) show **stronger target correlations** than the existing vol features for breakout prediction. The feature engineering added real value.

---

## Phase 1: Breakout Occurrence Prediction

### Definition

A **breakout** = forward range exceeds K × ATR, where K is calibrated per horizon to produce ~20-30% breakout rate:
- **1h:** K=5.0 → 21.9% breakout rate
- **4h:** K=10.0 → 26.6% breakout rate

### Results

| Horizon | Model | AUC | F1 | Precision | Recall | Brier |
|---------|-------|-----|-----|-----------|--------|-------|
| **1h** | Logistic Reg | **0.654** | 0.397 | 0.317 | 0.548 | 0.219 |
| **1h** | LGBM | 0.641 | 0.377 | 0.329 | 0.453 | 0.209 |
| **4h** | Logistic Reg | **0.687** | 0.456 | 0.425 | 0.507 | 0.205 |
| **4h** | LGBM | 0.671 | 0.424 | 0.431 | 0.425 | 0.203 |

### Feature Importance (LGBM, 4h)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | vol_sma_24h | 6.8% |
| 2 | atr_24h | 5.6% |
| 3 | atr_8h | 4.8% |
| 4 | touches_high_24h | 4.6% |
| 5 | vol_trend_4h | 4.6% |
| 6 | ret_kurtosis_8h | 4.5% |
| 7 | touches_low_24h | 4.4% |
| 8 | range_compression_4h | 3.8% |
| 9 | bar_eff_4h | 3.6% |
| 10 | consolidation_4h_vs_24h | 3.4% |

### Breakout Magnitude Prediction

Once we know a breakout occurs, can we predict how big it will be?

| Horizon | R² | Correlation |
|---------|-----|------------|
| 1h | -0.020 | 0.099 |
| 4h | -0.101 | 0.134 |

**No.** Breakout magnitude is essentially unpredictable from backward-looking features. We can detect *that* a breakout is likely, but not *how big* it will be.

### Phase 1 Interpretation

- **AUC 0.65–0.69** is a weak-to-moderate signal. Better than random (0.50) but not strong.
- **Logistic Regression beats LGBM** — consistent with all our previous experiments. The signal is linear.
- **4h is more predictable than 1h** — longer horizons give the model more time for the breakout to materialize.
- **Precision is low (~32-43%)** — many false positives. The model flags breakouts that don't happen.
- **This is fundamentally harder than vol prediction** (AUC 0.65 vs R²=0.34 for vol). Breakouts are discrete events, vol is continuous.

---

## Phase 2: Support/Resistance Level Prediction

### Definition

- **S/R levels:** Rolling high/low over lookback window (8h or 24h)
- **Near S/R:** Price within 5% of the range boundary
- **Break:** Price exceeds the level + zone in the forward window
- **Reject:** Price stays within the level

### Results

| Lookback | Forward | Events | Break Rate | LR AUC | LGBM AUC | Best F1 |
|----------|---------|--------|-----------|--------|----------|---------|
| 8h | 1h | 10,688 | 78.2% | **0.711** | 0.686 | 0.824 (LGBM) |
| 8h | 4h | 10,688 | 88.7% | 0.659 | 0.636 | 0.915 (LGBM) |
| **24h** | **1h** | **7,263** | **61.7%** | **0.703** | **0.683** | **0.701 (LGBM)** |
| 24h | 4h | 7,263 | 76.7% | 0.662 | 0.617 | 0.808 (LGBM) |

### Best Configuration: 24h Lookback, 1h Forward

This is the most balanced and informative configuration:
- **61.7% break rate** — reasonably balanced classes
- **LR AUC = 0.703** — moderate discriminative power
- **LGBM F1 = 0.701** with balanced precision (0.706) and recall (0.700)

### Feature Importance (24h/1h, LGBM)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | vol_ratio_bar | 3.0% |
| 2 | vol_trend_4h | 3.0% |
| 3 | ret_kurtosis_8h | 2.7% |
| 4 | adx_4h | 2.6% |
| 5 | upper_wick_ratio_4h | 2.6% |
| 6 | consolidation_4h_vs_24h | 2.5% |
| 7 | ret_skew_8h | 2.5% |
| 8 | consolidation_2h_vs_24h | 2.4% |
| 9 | vol_accel_4h | 2.3% |
| 10 | dist_to_high_4h | 2.3% |

**Notable:** `upper_wick_ratio` and `ret_skew` are new candle/distribution features that weren't in our original vol feature set. They capture rejection patterns at levels.

### Phase 2 Interpretation

- **8h lookback levels are too weak** — 78-89% break rate means these levels barely hold. Not useful for prediction.
- **24h lookback levels are more meaningful** — 62% break rate gives a balanced prediction problem.
- **AUC 0.70 is moderate** — we can distinguish break vs reject somewhat, but not reliably enough for high-confidence trading signals.
- **The high F1 in 8h/4h (0.915) is misleading** — it's just predicting the majority class (89% breaks).

---

## Comparison: Breakout vs Volatility Prediction

| Capability | Metric | Value | Verdict |
|-----------|--------|-------|---------|
| **Vol prediction (v9)** | R² | 0.34 | Strong |
| **Range prediction (v11)** | R² | 0.33 | Strong |
| **Breakout occurrence** | AUC | 0.65–0.69 | Weak-Moderate |
| **Breakout magnitude** | R² | -0.02 to -0.10 | None |
| **S/R break vs reject** | AUC | 0.68–0.71 | Moderate |
| **Direction (asymmetry, v11)** | R² | ≈0 | None |

**Volatility prediction remains our strongest capability.** Breakout prediction adds some value but is fundamentally harder because it's a discrete, partially directional event.

---

## What the New Features Revealed

### Features That Matter for Breakouts (Not for Vol)

| Feature | Breakout Importance | Vol Importance | Interpretation |
|---------|-------------------|---------------|----------------|
| **consolidation ratios** | Top 1-3 for S/R | Not used | Tight range → energy stored |
| **approach_speed** | Top 2-5 for S/R | Not used | Fast approach → more likely to break |
| **ret_kurtosis** | Top 1-6 | Not used | Fat tails → breakout regime |
| **upper_wick_ratio** | Top 5 for S/R | Not used | Rejection candles signal level strength |
| **touches_high/low** | Top 4-7 | Not used | More touches → weaker level |

### Features That Matter for Both

| Feature | Both Tasks | Interpretation |
|---------|-----------|----------------|
| vol_sma_24h | Top for both | Overall vol regime |
| atr_4h/8h | Top for both | Recent price movement scale |
| bar_eff_4h | Top for both | Trend efficiency |
| adx_4h | Top for both | Directional strength |

---

## Practical Applications

### What We Can Use

1. **Breakout alert system** — When model probability > 0.6, flag "elevated breakout risk in next 4h"
   - AUC 0.69 means this will have ~40% false positive rate, but catches ~50% of real breakouts
   - Useful as a **warning**, not a trading signal

2. **S/R level confidence** — When price approaches a 24h level, predict break probability
   - AUC 0.70 gives moderate confidence
   - Can adjust grid asymmetry: if break likely, extend grid in breakout direction

3. **Combine with vol prediction** — Breakout probability × predicted vol magnitude = expected move size
   - Vol prediction gives the "how much" (R²=0.34)
   - Breakout prediction gives the "will it happen" (AUC=0.69)

### What We Cannot Use

1. **Breakout magnitude** — Unpredictable (R²<0). Can't size positions based on expected breakout size.
2. **Breakout direction** — Still no directional signal from these features.
3. **High-confidence breakout signals** — AUC 0.65-0.70 is not reliable enough for aggressive position taking.

---

## Summary

| Finding | Implication |
|---------|------------|
| Breakout occurrence is weakly predictable (AUC 0.65-0.69) | Useful for alerts, not for trading signals |
| S/R break vs reject is moderately predictable (AUC 0.70) | Can inform grid asymmetry near levels |
| Breakout magnitude is unpredictable (R²<0) | Cannot size for expected breakout |
| New features (consolidation, approach speed, kurtosis) add value | Feature engineering was worthwhile |
| Linear models still beat LGBM | Signal is linear, not complex |
| **Vol prediction remains our strongest edge** | Focus grid bot on vol-adaptive sizing |

---

## Files

| File | Description |
|------|-------------|
| `breakout_ml.py` | Phase 1 + Phase 2 experiment suite with 102 candidate features |
| `results/breakout_ML_BTC.txt` | Complete BTCUSDT output |
| `FINDINGS_v11_range_prediction.md` | Previous range prediction results |
