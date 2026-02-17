# FINDINGS v21 Summary: Regime Detection Research

**Date:** 2025-01-01 to 2026-01-31 (13 months)
**Symbols:** BTCUSDT, ETHUSDT, SOLUSDT, DOGEUSDT, XRPUSDT
**Data:** 5-minute bars from Bybit tick data (~114K bars per symbol)
**Studies:** v8 (feature-based detection), v9 (ML regime), v19 (13-month validation), v20 (unsupervised classification), v21 (HMM)

---

## 1. What Regimes Exist?

**Two regimes — that's it.**

Tested K=2 through K=8 using GMM and KMeans with BIC, AIC, silhouette, and Calinski-Harabasz metrics. K=2 wins on every metric, for every symbol.

| Regime | % of Time | Volatility | Trade Intensity | Efficiency |
|--------|-----------|-----------|-----------------|------------|
| **Quiet** | 60-65% | Low (rvol ~0.07%) | Low (0.65× avg) | Low (mean-reverting) |
| **Volatile** | 35-40% | High (rvol ~0.18%, 2-3× quiet) | High (1.77× avg) | Slightly higher |

- K=3+ just subdivides volatile into "medium" and "extreme" — no genuinely new structure
- Identical 2-regime pattern across all 5 cryptos (BTC, ETH, SOL, DOGE, XRP)
- Top distinguishing features: `parkvol_1h` (Parkinson volatility) and `trade_intensity_ratio`
- Minimum history needed: ~2 hours (24 bars) for stable classification

## 2. Feature Engineering

35 backward-looking features computed from 5-minute OHLCV bars:

**Volatility (5 windows: 1h, 2h, 4h, 8h, 24h):**
- Realized volatility (rolling std of returns)
- Parkinson volatility (high-low based, more efficient estimator)
- Vol ratios (short/long — detect expansion/contraction)
- Vol acceleration (is volatility increasing or decreasing?)

**Trend/Efficiency (4 windows: 1h, 2h, 4h, 8h):**
- Kaufman efficiency ratio (|net return| / sum |bar returns|)
- Return autocorrelation (lag-1)
- ADX-like directional movement
- Momentum (cumulative returns)
- Sign persistence (fraction of same-sign returns)

**Microstructure (rolling):**
- Trade intensity (current count / 24h average)
- Bar efficiency (net move / total path within bar)
- Volume imbalance persistence
- Large trade fraction
- Inter-trade interval burstiness (ITI CV)

**Price structure:**
- Price vs SMA deviations (2h, 4h, 8h, 24h)

All features are strictly backward-looking. Ground-truth labels use forward-looking windows (never leaked into features).

## 3. Detection: How Fast Can We Know the Current Regime?

**Near-instantly — detection is a solved problem.**

### Methods Tested

| Method | Accuracy | Median Lag | Mean Lag | False Sw/day |
|--------|----------|------------|----------|-------------|
| GMM single-bar posterior | 98.7% | 0 bars | 0.0 | 5.9 |
| GMM + EMA(3) smoothing | 98.3% | 0 bars | 1.1 | 0.0 |
| **HMM forward filter** | **98.7%** | **0 bars** | **0.3** | **2.0** |
| **HMM forward + EMA(3)** | **98.1%** | **0 bars** | **2.0** | **0.0** |
| HMM forward + 3-bar confirm | 94.8% | 2 bars | 11.9 | ~0 |
| Rolling window GMM (1-24h) | 50-70% | varies | varies | varies |

- All posterior-based methods achieve **0-bar median detection lag**
- The regime is apparent from the current bar's features alone
- The real challenge is **noise filtering**, not speed

### Recommended Production Setup

**HMM forward filter + 3-bar confirmation:**
- ~5 regime changes/day (vs 14.7 raw GMM)
- 94.8% accuracy
- 10-minute median detection lag
- Only 6.5% noise episodes (vs 53% with raw GMM)

## 4. GMM vs HMM

HMM (Hidden Markov Model) adds temporal modeling: P(state_t | state_{t-1}, features_t) instead of GMM's independent P(state_t | features_t).

### Classification

| Metric | GMM | HMM | |
|--------|-----|-----|---|
| Regimes found | 2 | 2 | Tie |
| Silhouette score | 0.187 | 0.181 | Tie |
| Agreement | — | 97.9% | Same regimes |
| **Transitions/day** | **14.7** | **8.7** | **HMM −41%** |
| **Noise episodes (≤15min)** | **53%** | **32.5%** | **HMM −39%** |
| Quiet median duration | 30 min | 85 min | HMM +183% |
| Volatile median duration | 10 min | 20 min | HMM +100% |
| Computation time | ~20s | ~1100s | GMM 55× faster |

HMM finds the same regimes but with much cleaner labels. The transition matrix penalizes rapid switching, absorbing short noise flickers into the surrounding regime.

### HMM Transition Matrix (BTC)

```
          → Quiet    → Volatile
Quiet:     0.9587     0.0413     (expected duration: 2 hours)
Volatile:  0.0247     0.9753     (expected duration: 3.4 hours)
```

Volatile regime is more persistent — once volatility kicks in, it tends to last longer.

## 5. Prediction: Can We Predict Regime Switches?

**Partially — AUC ~0.8, but not reliable enough for hard switching.**

### Models Tested

- Logistic Regression (C=0.1, balanced classes)
- Random Forest (100 trees, max_depth=8)
- Gradient Boosting (100 trees, max_depth=4)

### Results (Best: Gradient Boosting with HMM-augmented features)

| Horizon | AUC | Precision | Recall | F1 |
|---------|-----|-----------|--------|-----|
| 30 min (6 bars) | ~0.82 | ~0.35 | ~0.70 | ~0.45 |
| **1 hour (12 bars)** | **0.812** | **0.571** | **0.414** | **0.480** |
| 2 hours (24 bars) | 0.778 | 0.620 | 0.540 | 0.577 |
| 4 hours (48 bars) | ~0.75 | ~0.65 | ~0.55 | ~0.60 |

- AUC 0.8 is genuinely good for financial prediction
- Consistent across all 5 cryptos (±2% AUC)
- **Precision is the bottleneck** — at 57%, nearly half the alerts are false positives

### HMM Features Improve Prediction

Adding HMM-derived features (forward-filtered probability, switch probability, rate-of-change of HMM probability) boosts AUC by +1.5 to +2.5 percentage points:

| Features | Model | AUC (1h) | AUC (2h) |
|----------|-------|----------|----------|
| v20 (50 features, no HMM) | GB | 0.796 | 0.759 |
| **v21 (57 features, +HMM)** | **GB** | **0.812** | **0.778** |

7 of the top 15 most important features are HMM-derived. The #1 feature is `hmm_p_volatile` (HMM forward-filtered probability of volatile regime).

### Why Prediction Has a Hard Ceiling

- Regime switches are caused by **exogenous shocks** (news, liquidation cascades, whale movements)
- These are not visible in backward-looking price/volume features
- The ~20% predictable signal comes from volatility clustering and gradual buildup
- To exceed AUC 0.85+, you'd need: order book depth, funding rates, on-chain data, or sentiment

### Pure HMM Transition Matrix Is Useless for Prediction

The transition matrix gives a constant P(switch) per state (~4% for quiet, ~2.5% for volatile). It doesn't vary with market conditions, so it has no discriminative power as a standalone predictor.

## 6. Practical Recommendations

### For a Trading System

1. **Classification:** Use HMM (K=2, diagonal covariance) for cleaner regime labels
2. **Detection:** HMM forward filter + 3-bar confirmation → ~5 transitions/day, 95% accuracy, 10-min lag
3. **Prediction:** Use as a **warning signal** (alert when P(switch) > 0.3), not a hard trigger
4. **Strategy switching:** React to detection, not prediction — losing 10 minutes of detection lag is acceptable; acting on a 57% precision prediction is not

### What NOT to Do

- Don't use raw GMM at 5-min resolution without smoothing — 53% of episodes are noise flickers
- Don't try to predict exact switch timing — precision is too low for reliable action
- Don't use K>2 regimes — no statistical support, adds complexity for no gain
- Don't expect 1-minute bars to help — features need 1h+ lookback windows anyway
- Don't use the HMM transition matrix alone for prediction — it's constant per state

## 7. Open Questions

1. **Would order book features break the prediction ceiling?** Depth imbalance and spread changes might lead regime switches by minutes
2. **Cross-asset leading indicators?** BTC regime changes may predict altcoin regime changes
3. **Regime-conditional strategy optimization?** Now that we can detect regimes reliably, which strategies work best in each?
4. **Adaptive grid parameters?** Grid bot spacing/levels could be tuned per regime in real-time

## 8. Files

| File | Description |
|------|-------------|
| `regime_detection.py` | Feature engineering (35 features from 5-min bars), ground-truth labeling |
| `regime_classify.py` | Unsupervised clustering experiments (GMM/KMeans, optimal K, 5 symbols) |
| `regime_speed.py` | Detection latency + prediction experiments (GMM-based) |
| `regime_hmm.py` | HMM vs GMM comparison (4 experiments, BTC) |
| `plot_regimes.py` | GMM regime visualization (Dec 2025) |
| `plot_regimes_hmm.py` | GMM vs HMM side-by-side chart (Dec 2025, 1h) |
| `plot_regimes_hmm_zoom.py` | GMM vs HMM zoomed chart (Dec 18-19, raw 5-min bars) |
| `FINDINGS_v8_regime_detection.md` | Feature-based regime detection |
| `FINDINGS_v9_ml_regime.md` | ML regime prediction (linear models) |
| `FINDINGS_v19_13m_validation.md` | 13-month validation |
| `FINDINGS_v20_regime_classification.md` | Unsupervised classification (5 symbols) |
| `FINDINGS_v21_hmm_regimes.md` | HMM vs GMM detailed results (BTC) |
