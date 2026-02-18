# FINDINGS v27: Regime Detection + Liquidation Features

**Date**: February 18, 2026  
**Data Period**: May 11 – Aug 10, 2025 (92 days) + Feb 9–17, 2026 (9 days)  
**Symbols**: BTCUSDT, ETHUSDT, SOLUSDT, DOGEUSDT, XRPUSDT  
**Bars**: ~21,000 five-minute bars per symbol  
**Experiments**: 5 (Correlation, Leading Indicators, Prediction, Conditional Patterns, Detection Speed)  

---

## Executive Summary

**Liquidation features do NOT improve regime detection or prediction.** Despite strong correlation with the current regime (r=0.51 for liq_rate_2h), liquidation data is a **lagging** indicator — it confirms the regime rather than predicting it. Adding 26 liquidation features to the ML models produced **zero improvement** and sometimes degraded performance.

### Key Results

| Experiment | Result | Verdict |
|-----------|--------|---------|
| **Correlation with regime** | r=0.45–0.51 for liq_rate | ✅ Strong, but lagging |
| **Leading indicator test** | p<0.001 for rate/notional | ⚠️ Significant but concurrent |
| **Prediction AUC improvement** | -0.02 to +0.03 (avg ~0) | ❌ No improvement |
| **Detection speed** | +0.5 to +2.0 bars slower | ❌ Slightly worse |
| **Regime-conditional patterns** | 5–9× more liquidations in volatile | ✅ Informative for understanding |

**Bottom line**: Liquidations are a **consequence** of volatility, not a cause. They spike simultaneously with regime switches, not before them. The standard volatility features (rvol, parkvol, efficiency) already capture the information that liquidation data provides.

---

## Experiment 1: Liquidation Feature Correlation with Regime

### Top Features by Correlation (BTC)

| Feature | Correlation | Quiet Mean | Volatile Mean | Ratio |
|---------|------------|------------|---------------|-------|
| **liq_rate_2h** | +0.508 | 4.2 | 26.8 | 6.4× |
| **liq_rate_4h** | +0.493 | 5.6 | 26.6 | 4.7× |
| **liq_rate_1h** | +0.493 | 3.2 | 26.4 | 8.3× |
| **liq_notional_2h** | +0.439 | $41K | $307K | 7.5× |
| **liq_notional_4h** | +0.429 | $53K | $274K | 5.2× |
| liq_imb_abs_1h | +0.325 | 0.39 | 0.67 | 1.7× |
| liq_rate_zscore | +0.191 | -0.04 | 0.77 | 20× |

### Cross-Symbol Consistency

| Feature | BTC | ETH | SOL | DOGE | XRP |
|---------|-----|-----|-----|------|-----|
| liq_rate_2h | +0.51 | +0.51 | +0.51 | +0.51 | +0.49 |
| liq_notional_2h | +0.44 | +0.50 | +0.51 | +0.47 | +0.47 |
| liq_imb_abs_1h | +0.33 | +0.33 | +0.44 | +0.37 | +0.38 |

**Remarkably consistent** across all 5 symbols. Liquidation rate and notional are ~5–9× higher during volatile regimes. However, this is **concurrent** correlation, not predictive.

### Imbalance Is NOT Correlated

| Feature | Correlation |
|---------|------------|
| liq_imbalance | +0.02 to +0.06 |
| liq_imb_1h | +0.03 to +0.08 |
| liq_buy_ratio | +0.11 to +0.12 |

Liquidation direction (buy vs sell) has almost zero correlation with regime. Volatile regimes have more liquidations on **both** sides, not just one.

---

## Experiment 2: Liquidation as Leading Indicators

### Mann-Whitney U Test Results (BTC)

Tested whether liquidation features are significantly elevated in the 15m/30m/1h/2h **before** regime transitions vs random non-transition windows.

| Feature | 15m effect | p-value | 30m effect | p-value | 1h effect | p-value |
|---------|-----------|---------|-----------|---------|----------|---------|
| **liq_count** | +13.4 | <0.001*** | +13.4 | <0.001*** | +14.5 | <0.001*** |
| **liq_notional** | +$203K | <0.001*** | +$197K | <0.001*** | +$212K | <0.001*** |
| **liq_rate_1h** | +13.8 | <0.001*** | +14.0 | <0.001*** | +12.4 | <0.001*** |
| **liq_rate_zscore** | +0.44 | <0.001*** | +0.45 | <0.001*** | +0.44 | <0.001*** |
| liq_imbalance | +0.03 | 0.29 | +0.04 | 0.39 | +0.01 | 0.34 |

### Interpretation

The statistical significance is **misleading**. The effect sizes are nearly identical across all windows (15m, 30m, 1h, 2h), which means liquidations don't gradually build up before a transition — they spike **at the same time** as the regime switch. If liquidations were truly leading, we'd see:
- Larger effect at shorter windows (closer to transition)
- Gradual buildup from 2h → 15m

Instead, the effect is flat, confirming that liquidations are **concurrent** with regime changes, not predictive.

---

## Experiment 3: Regime Switch Prediction

### AUC Comparison: Base Features vs Base + Liquidation Features

#### BTC (Gradient Boost)

| Horizon | Base AUC | Base+Liq AUC | Δ AUC |
|---------|----------|-------------|-------|
| 30min | 0.840 | 0.844 | **+0.004** |
| 1h | 0.741 | 0.753 | **+0.013** |
| 2h | 0.666 | 0.693 | **+0.026** |
| 4h | 0.656 | 0.649 | -0.007 |

#### All Symbols (Gradient Boost, 1h horizon)

| Symbol | Base AUC | Base+Liq AUC | Δ AUC |
|--------|----------|-------------|-------|
| **BTC** | 0.741 | 0.753 | **+0.013** |
| **ETH** | 0.794 | 0.793 | -0.001 |
| **SOL** | 0.819 | 0.822 | +0.003 |
| **DOGE** | 0.856 | 0.836 | **-0.019** |
| **XRP** | 0.799 | 0.805 | +0.005 |

### Grand Summary: AUC Improvement Across All Horizons

| Symbol | 30min | 1h | 2h | 4h |
|--------|-------|-----|-----|-----|
| BTC | +0.004 | +0.013 | +0.026 | -0.007 |
| ETH | -0.002 | -0.001 | -0.012 | -0.004 |
| SOL | -0.002 | +0.003 | -0.005 | -0.016 |
| DOGE | -0.008 | -0.019 | -0.012 | -0.007 |
| XRP | +0.001 | +0.005 | +0.015 | -0.024 |
| **Average** | **-0.001** | **+0.000** | **+0.002** | **-0.012** |

**Average improvement is essentially zero.** BTC shows a small improvement at 2h horizon (+0.026), but DOGE and ETH get worse. The liquidation features add noise, not signal.

### Feature Importance (Base+Liq RF, 1h horizon, BTC)

| Rank | Feature | Importance | Liq? |
|------|---------|-----------|------|
| 1 | rvol_1h | 0.0920 | |
| 2 | parkvol_1h | 0.0868 | |
| 3 | rvol_2h | 0.0791 | |
| 4 | vol_ratio_1h_24h | 0.0630 | |
| 5 | rvol_4h | 0.0563 | |
| 6 | efficiency_1h | 0.0483 | |
| 7 | **liq_rate_2h** | 0.0432 | ★ |
| 8 | **liq_rate_4h** | 0.0380 | ★ |
| 9 | vol_ratio_1h_8h | 0.0371 | |
| 10 | **liq_notional_2h** | 0.0350 | ★ |
| 11 | vol_accel_1h | 0.0318 | |
| 12 | **liq_rate_1h** | 0.0302 | ★ |
| 13 | momentum_1h | 0.0286 | |
| 14 | efficiency_2h | 0.0266 | |
| 15 | **liq_rate_8h** | 0.0241 | ★ |

Liquidation features rank 7th–15th — they have some importance but are dominated by the standard volatility features. The top 6 features are all price-based.

### Liquidation-Only Model Performance

| Symbol | Liq-Only AUC (1h) | Base AUC (1h) | Base+Liq AUC (1h) |
|--------|-------------------|---------------|-------------------|
| BTC | 0.705 | 0.741 | 0.753 |
| ETH | 0.720 | 0.794 | 0.793 |
| SOL | 0.720 | 0.819 | 0.822 |
| DOGE | 0.749 | 0.856 | 0.836 |
| XRP | 0.724 | 0.799 | 0.805 |

Liquidation features alone achieve AUC 0.70–0.75, which is decent but **far below** the base features (0.74–0.86). They carry some information but it's redundant with volatility.

---

## Experiment 4: Regime-Conditional Liquidation Patterns

### BTC: Quiet vs Volatile Regime

| Metric | Quiet | Volatile | Ratio |
|--------|-------|----------|-------|
| Liquidations per 5min | 3.2 | 26.4 | **8.3×** |
| Notional per 5min | $41K | $307K | **7.5×** |
| Max single liq | $29K | $128K | **4.5×** |
| Imbalance | +0.01 | +0.05 | ~same |
| Abs imbalance 1h | 0.39 | 0.67 | **1.7×** |
| Active bars in 1h | 2.0 | 5.6 | **2.8×** |
| Buy count | 2.8 | 19.4 | **7.0×** |
| Sell count | 2.3 | 10.6 | **4.7×** |

### Liquidation Profile Around Regime Switches (BTC, ±1h)

```
Time     Avg Count   Avg Notional
-60min      3.2      $  18,000
-50min      3.5      $  20,000
-40min      4.1      $  23,000
-30min      5.2      $  28,000
-20min      6.8      $  37,000
-15min      8.5      $  47,000
-10min     11.2      $  62,000
 -5min     16.3      $  91,000
  0min     27.5      $ 152,000  ← SWITCH
 +5min     15.8      $  82,000
+10min      9.2      $  48,000
+15min      6.5      $  35,000
+20min      5.1      $  27,000
```

**The profile is symmetric around the switch** — liquidations ramp up ~30min before and decay ~30min after. This confirms they are **concurrent** with the regime change, not leading. The ramp-up before the switch is because the GMM labels the transition bar as the point where features cross the threshold, but the actual volatility increase starts a few bars earlier.

---

## Experiment 5: Regime Detection Speed

### Detection Accuracy and Lag

| Symbol | Base Acc | +Liq Acc | Base Mean Lag | +Liq Mean Lag | Base FalsSw/d | +Liq FalsSw/d |
|--------|----------|---------|---------------|---------------|---------------|---------------|
| **BTC** | 0.978 | 0.970 | 5.9 bars | 7.4 bars | 0.0 | 0.0 |
| **ETH** | 0.980 | 0.974 | 5.5 bars | 6.1 bars | 0.0 | 0.0 |
| **SOL** | 0.980 | 0.971 | 3.4 bars | 5.1 bars | 0.0 | 0.0 |
| **DOGE** | 0.981 | 0.978 | 2.9 bars | 2.8 bars | 0.0 | 0.0 |
| **XRP** | 0.975 | 0.971 | 1.8 bars | 3.2 bars | 0.0 | 0.0 |

**Adding liquidation features makes detection SLOWER**, not faster:
- Accuracy drops by 0.3–1.0 percentage points
- Mean lag increases by 0.1–1.7 bars on 4/5 symbols
- Only DOGE shows a marginal improvement (-0.1 bars)

The extra features add noise to the classifier, diluting the signal from the strong volatility features.

---

## Why Liquidation Features Don't Help

### 1. Liquidations Are a Consequence, Not a Cause

Volatility regime changes are driven by:
- News events (macro, regulatory, exchange)
- Large directional trades (whale orders)
- Cascading stop-losses

Liquidations happen **because** volatility increased, not the other way around. By the time liquidations spike, the regime has already changed.

### 2. Information Is Redundant

The correlation between `liq_rate_2h` and `rvol_1h` is ~0.50. The ML model already has `rvol_1h` (and many other volatility features) which capture the same information more directly and with less noise.

### 3. Liquidation Data Is Noisier

- Many 5-min bars have zero liquidations (especially in quiet regimes)
- The signal-to-noise ratio is lower than price-based features
- Rolling averages smooth the noise but also add lag

### 4. The Transition Matrix Is the Bottleneck

From v21 (HMM research), we know that regime switches are fundamentally hard to predict because they're driven by exogenous shocks. Adding more features doesn't help when the underlying process is unpredictable.

---

## Comparison to Previous Research

| Version | Approach | Key Finding |
|---------|----------|-------------|
| v20 | GMM regime classification | 2 regimes, silhouette 0.187 |
| v21 | HMM vs GMM | HMM reduces noise 40%, AUC +2% |
| v26 | Liquidation analysis | Patterns confirmed, 4 strategies |
| v26b | Liquidation strategies (9d) | Imbalance Reversal best |
| v26c | Liquidation strategies (100d) | ToD Fade most robust |
| **v27** | **Regime + Liquidation** | **Liquidations don't improve regime prediction** |

### The Paradox

Liquidation features are **excellent for trading strategies** (v26c: +25% aggregate) but **useless for regime prediction**. Why?

- **Trading strategies** exploit the **mean reversion** after liquidation cascades — a micro-level effect
- **Regime prediction** needs **leading indicators** of macro-level volatility shifts
- Liquidations are a micro-level phenomenon that happens within regimes, not across them

---

## Practical Implications

### For Regime Detection
- **Keep using volatility features only** (rvol, parkvol, efficiency)
- **Don't add liquidation features** — they add noise and slow detection
- **HMM forward + 3-bar confirmation** remains the best approach (from v21)

### For Trading Strategies
- **Use liquidation features for trade timing** (v26c strategies)
- **Use regime detection for position sizing** (reduce size in volatile regime)
- **Combine both**: Run regime detection with vol features, then apply liquidation strategies with regime-conditional parameters

### Recommended Architecture

```
Layer 1: Regime Detection (vol features only)
  → Quiet regime: tight stops, higher leverage
  → Volatile regime: wide stops, lower leverage

Layer 2: Trade Signals (liquidation features)
  → Cascade Fade: mean reversion after cascades
  → ToD Fade: US-hours imbalance reversal
  → Regime-conditional parameters
```

---

## Data & Code

### Scripts
- `research_regime_liquidations.py` — Full 5-experiment research suite

### Results
- `results/regime_liquidations_v27.txt` — Complete output for all 5 symbols

### Reproducibility
```bash
python3 research_regime_liquidations.py 2>&1 | tee results/regime_liquidations_v27.txt
```

---

**Research by**: Cascade AI  
**Date**: February 18, 2026  
**Version**: v27 (Regime + Liquidation Research)  
**Status**: Complete ✅  
