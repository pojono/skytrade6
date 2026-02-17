# FINDINGS v21: HMM vs GMM for Regime Classification & Detection

**Date:** 2025-01-01 to 2026-01-31 (13 months)
**Symbols:** BTCUSDT (initial evaluation)
**Bars:** 113,761 × 5-minute
**Library:** hmmlearn 0.3.3 (GaussianHMM, diagonal covariance)

---

## 1. Executive Summary

HMM provides **meaningful improvements over GMM** in two areas:

1. **Noise filtering:** HMM reduces transitions from 5,803 → 3,453 (−40%) and short episodes (≤15min) from 53% → 32.5%. This is the biggest practical win.
2. **Prediction:** HMM-derived features boost AUC from 0.796 → 0.812 (+2%) at 1h horizon. HMM features dominate the top-15 importance list (7 of 15).

Detection speed is comparable — both achieve 0-bar median lag. HMM forward filter is slightly better when measured against HMM ground truth (98.7% accuracy, 0.3 mean lag vs GMM's 2.0 false switches/day).

**Verdict:** HMM is worth adopting. The noise reduction alone justifies it — fewer false regime switches means fewer unnecessary strategy changes.

---

## 2. Classification: HMM vs GMM

### Model Selection (K=2,3,4)

| Model | K | Silhouette | Transitions | Short ≤3 bars | Expected Duration |
|-------|---|-----------|-------------|---------------|-------------------|
| GMM | 2 | **0.187** | 5,803 | 53.0% | N/A (no temporal model) |
| GMM | 3 | 0.045 | 7,543 | 52.9% | N/A |
| GMM | 4 | 0.014 | 10,405 | 57.1% | N/A |
| **HMM** | **2** | **0.181** | **3,453** | **32.5%** | Quiet: 24 bars (2h), Volatile: 40 bars (3.4h) |
| HMM | 3 | 0.040 | 4,275 | 31.7% | 15–50 bars |
| HMM | 4 | 0.007 | 5,395 | 31.7% | 14–53 bars |

**K=2 wins for both models** — same conclusion as v20. Silhouette scores are nearly identical (0.187 vs 0.181), confirming the same 2-regime structure is found.

### HMM K=2 Transition Matrix

```
          → Quiet    → Volatile
Quiet:     0.9587     0.0413
Volatile:  0.0247     0.9753
```

- **P(stay quiet)** = 95.9% → expected quiet duration = **24.2 bars (2 hours)**
- **P(stay volatile)** = 97.5% → expected volatile duration = **40.5 bars (3.4 hours)**
- Volatile regime is more persistent than quiet — once volatility kicks in, it tends to last longer

### Agreement Between GMM and HMM

- **97.9% agreement** (ARI = 0.918)
- They find essentially the same regimes
- The 2.1% disagreement is mostly HMM "absorbing" short GMM flickers into the surrounding regime

### Episode Duration Comparison

| Metric | GMM | HMM | Improvement |
|--------|-----|-----|-------------|
| **Total transitions** | 5,803 | 3,453 | **−40%** |
| **Transitions/day** | 14.7 | 8.7 | **−41%** |
| Quiet median duration | 6 bars (30m) | 17 bars (85m) | **+183%** |
| Volatile median duration | 2 bars (10m) | 4 bars (20m) | **+100%** |
| Episodes ≤3 bars (noise) | 53.0% | 32.5% | **−39%** |
| Episodes ≤10 bars | 69.8% | 51.7% | **−26%** |

**This is the key win.** HMM's transition matrix penalizes rapid switching, so short noise flickers get absorbed into the dominant regime. Median quiet episode goes from 30 min to 85 min — much more practical for strategy switching.

---

## 3. Detection Speed: HMM Forward vs GMM Posterior

### Against GMM Ground Truth

| Method | Accuracy | Med Lag | Mean Lag | P90 Lag | %<3 bars | False Sw/day |
|--------|----------|---------|----------|---------|----------|-------------|
| GMM posterior | **1.000** | 0 | 0.0 | 0 | 100% | 0.0 |
| GMM + EMA(3) | 0.987 | 0 | 3.2 | 2 | 90.2% | 0.0 |
| HMM forward | 0.976 | 0 | 3.1 | 5 | 85.1% | 0.0 |
| HMM forward+EMA(3) | 0.966 | 0 | 5.6 | 10 | 77.3% | 0.0 |

Against GMM's own labels, GMM posterior is trivially perfect (it's the same model). HMM forward is slightly slower because it's tracking different (smoother) labels.

### Against HMM Ground Truth (More Meaningful)

| Method | Accuracy | Med Lag | Mean Lag | P90 Lag | %<3 bars | False Sw/day | Total Trans |
|--------|----------|---------|----------|---------|----------|-------------|-------------|
| GMM posterior | 0.979 | 0 | 0.0 | 0 | 99.9% | **5.9** | 5,803 |
| GMM + EMA(3) | 0.983 | 0 | 1.1 | 1 | 96.8% | 2.4 | 4,403 |
| **HMM forward** | **0.987** | **0** | **0.3** | **1** | **99.2%** | **2.0** | 4,231 |
| HMM forward+EMA(3) | 0.981 | 0 | 2.0 | 2 | 93.8% | **0.0** | 3,457 |

**HMM forward is the best online method:**
- Highest accuracy (98.7%)
- Lowest mean lag (0.3 bars)
- 99.2% detected within 3 bars
- Only 2.0 false switches/day (vs GMM's 5.9)

**HMM forward + EMA(3) achieves zero false switches** while maintaining 98.1% accuracy and 0-bar median lag. This is the best production option.

### HMM Forward Threshold Sensitivity

| Threshold | Accuracy | Med Lag | %<3 bars | False Sw/day |
|-----------|----------|---------|----------|-------------|
| 0.5 | 98.7% | 0 | 99.2% | 2.0 |
| 0.7 | 98.8% | 0 | 98.3% | 1.9 |
| 0.9 | 98.9% | 0 | 96.6% | 2.2 |

Remarkably stable across thresholds — the HMM forward probabilities are well-calibrated.

---

## 4. Prediction: HMM Features Improve ML Models

### Pure HMM Transition Probability

The HMM transition matrix gives a constant P(switch) ≈ 4.1% per bar for quiet and 2.5% for volatile. This is too uniform to be useful as a standalone predictor — it doesn't vary with market conditions.

### ML with HMM-Augmented Features

| Horizon | Features | Model | Accuracy | Precision | Recall | F1 | AUC |
|---------|----------|-------|----------|-----------|--------|-----|-----|
| 1h | v20 (50 feat) | RF | 0.679 | 0.387 | 0.779 | 0.517 | 0.787 |
| 1h | v20 (50 feat) | GB | 0.786 | 0.524 | 0.325 | 0.401 | 0.796 |
| 1h | **v21 (57 feat)** | **RF** | **0.716** | **0.418** | **0.740** | **0.535** | **0.809** |
| 1h | **v21 (57 feat)** | **GB** | **0.802** | **0.571** | **0.414** | **0.480** | **0.812** |
| 2h | v20 (50 feat) | RF | 0.669 | 0.529 | 0.766 | 0.626 | 0.758 |
| 2h | v20 (50 feat) | GB | 0.699 | 0.583 | 0.582 | 0.583 | 0.759 |
| 2h | **v21 (57 feat)** | **RF** | **0.676** | **0.537** | **0.726** | **0.617** | **0.768** |
| 2h | **v21 (57 feat)** | **GB** | **0.715** | **0.620** | **0.540** | **0.577** | **0.778** |

**HMM features improve AUC by +1.0 to +2.2 percentage points** across all model/horizon combinations. The improvement is consistent but moderate.

### Top 15 Features (v21 RF, 1h Horizon)

| Rank | Feature | Importance | HMM? |
|------|---------|-----------|------|
| 1 | **hmm_p_volatile** | 0.0929 | ★ |
| 2 | parkvol_1h | 0.0869 | |
| 3 | **hmm_p_volatile_ema3** | 0.0795 | ★ |
| 4 | rvol_2h | 0.0757 | |
| 5 | **hmm_p_vol_roc6** | 0.0602 | ★ |
| 6 | rvol_1h | 0.0583 | |
| 7 | trade_intensity_ratio | 0.0516 | |
| 8 | **hmm_p_volatile_ema12** | 0.0488 | ★ |
| 9 | **hmm_p_switch** | 0.0483 | ★ |
| 10 | **hmm_p_vol_roc12** | 0.0475 | ★ |
| 11 | parkvol_4h | 0.0408 | |
| 12 | **hmm_p_vol_roc3** | 0.0398 | ★ |
| 13 | rvol_4h | 0.0342 | |
| 14 | vol_ratio_bar | 0.0253 | |
| 15 | vol_sma_24h | 0.0195 | |

**7 of the top 15 features are HMM-derived** (marked with ★). The #1 most important feature is `hmm_p_volatile` — the HMM forward-filtered probability of being in the volatile regime. This is a temporally-smoothed version of the raw volatility features, and the ML model finds it more informative than any single raw feature.

---

## 5. Noise Filtering Analysis

| Metric | GMM | HMM Viterbi | HMM Forward | HMM Fwd+Confirm(3) |
|--------|-----|-------------|-------------|---------------------|
| Transitions/day | 14.7 | 8.7 | 10.7 | 4.8 |
| Episodes ≤3 bars | 53.0% | 32.5% | 40.6% | 6.5% |
| Accuracy vs Viterbi | 97.9% | 100% | 98.7% | 94.8% |
| Median lag | 0 | 0 | 0 | 2 |

**HMM Forward + 3-bar confirmation** is the sweet spot for production:
- Only 6.5% noise episodes (vs 53% with GMM)
- 4.8 transitions/day (vs 14.7 with GMM)
- 94.8% accuracy with 2-bar median lag
- Eliminates virtually all noise flickers

---

## 6. Practical Recommendations

### For Production Regime Detection

**Use HMM Forward + 3-bar confirmation:**
1. Fit HMM (K=2, diag covariance) on historical data
2. Run forward algorithm on each new bar → P(volatile)
3. Switch regime only when P(new_regime) > 0.5 for 3 consecutive bars
4. Result: ~5 regime changes/day, 95% accuracy, 10-min median lag

### For Prediction

**Use v21 HMM-augmented features with Gradient Boosting:**
- Add `hmm_p_volatile`, `hmm_p_switch`, EMA and ROC of HMM probability
- AUC 0.812 at 1h horizon (vs 0.796 without HMM features)
- Still not reliable enough for hard switching, but better as a warning signal

### What HMM Does NOT Solve

- **The fundamental prediction limit remains:** regime switches are caused by exogenous shocks, not endogenous dynamics
- **AUC improvement is moderate** (+1-2%), not transformative
- **The transition matrix is too static** — P(switch) is constant per state, doesn't adapt to market conditions

---

## 7. Comparison: v20 (GMM) vs v21 (HMM)

| Aspect | v20 (GMM) | v21 (HMM) | Winner |
|--------|-----------|-----------|--------|
| Regimes found | 2 | 2 | Tie |
| Silhouette | 0.187 | 0.181 | Tie |
| Agreement | — | 97.9% | Same regimes |
| Transitions/day | 14.7 | 8.7 | **HMM** (−41%) |
| Noise episodes (≤15min) | 53.0% | 32.5% | **HMM** (−39%) |
| Detection accuracy | 98.7% (EMA3) | 98.7% (forward) | Tie |
| Detection lag | 0 bars | 0 bars | Tie |
| False switches/day | 0 (EMA3) | 0 (fwd+EMA3) | Tie |
| Prediction AUC (1h) | 0.796 | 0.812 | **HMM** (+2%) |
| Prediction AUC (2h) | 0.759 | 0.778 | **HMM** (+2.5%) |
| Computation time | ~20s | ~1100s | GMM (55× faster) |
| Complexity | Simple | Moderate | GMM simpler |

**Bottom line:** HMM is better for classification (cleaner labels) and marginally better for prediction. The main cost is computation time (55× slower), but this only matters for fitting — online inference (forward algorithm) is fast.

---

## Files

| File | Description |
|------|-------------|
| `regime_hmm.py` | HMM vs GMM experiment suite (4 experiments) |
| `results/regime_hmm_v21.txt` | Complete BTC output |
| `FINDINGS_v20_regime_classification.md` | Previous GMM-based analysis |
