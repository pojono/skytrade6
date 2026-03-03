# XS-8c — Fragility Deep Dive

**Date:** 2026-03-03  
**Data:** 68 Bybit perps, 2025-07-01 → 2026-02-28 (8 months)  
**Grid:** 5-minute cross-sectional snapshots (69,984 rows)  
**Features:** breadth_extreme, entropy, pca_var1, crowd_fund, crowd_oi  
**Targets:** directional tails, BTC-only, multi-horizon (30m/1h/2h/4h)  
**Model:** LogisticRegression, walk-forward (train 60% / test 40%)  
**Script:** `xs8c_deep_dive.py`  
**Runtime:** 13.4min (recomputed signed targets from raw 1m data)

---

## TL;DR

**crowd_oi is a market fragility indicator, not a stress indicator.**

Key discoveries:
1. **Fragility is stronger on the downside** — down-tail Q5/Q1 = 1.88× vs up-tail 1.56× at 1h
2. **BTC-only prediction is weak** — best AUC 0.56, cross-sectional features don't predict single-asset tails
3. **crowd_oi × pca_var1 interaction is additive** — no hidden synergy, both contribute independently
4. **2h horizon has highest AUC (0.654)** but saturated base rate (79%); 1h remains best trade-off

The indicator works because **low OI crowding = thin liquidity buffer = fragile market**.
This is structural microstructure, not ML-magic.

---

## 1. Directional Asymmetry (Angle A)

### Full Results: 12×ATR, ≥10% of coins

| Target | Base | AUC | Q5/Q1 | Q1 | Q5 | crowd_oi coef |
|--------|------|-----|-------|----|----|---------------|
| **any 30m** | 7.9% | 0.556 | 1.78× | 5.5% | 9.8% | -4.46 |
| up 30m | 3.7% | 0.547 | 1.56× | 2.6% | 4.0% | -4.47 |
| down 30m | 3.7% | 0.553 | 1.68× | 2.9% | 4.8% | -2.86 |
| **any 1h** | 34.6% | 0.591 | 1.83× | 25.8% | 47.3% | -4.62 |
| up 1h | 15.5% | 0.559 | 1.56× | 12.1% | 18.9% | -3.89 |
| **down 1h** | 14.1% | 0.562 | **1.88×** | 9.6% | 18.1% | -3.14 |
| **any 2h** | 79.3% | **0.654** | 1.31× | 70.0% | 91.5% | -6.87 |
| up 2h | 42.7% | 0.571 | 1.49× | 35.7% | 53.3% | -4.39 |
| down 2h | 35.3% | 0.572 | 1.65× | 27.1% | 44.8% | -2.90 |
| up 4h | 72.0% | 0.603 | 1.26× | 65.7% | 83.0% | -4.37 |
| down 4h | 61.9% | 0.575 | 1.31× | 57.3% | 75.0% | -3.06 |

### Key Finding: Downside asymmetry confirmed

At **every horizon**, down-tails have higher Q5/Q1 than up-tails:

| Horizon | Q5/Q1 Down | Q5/Q1 Up | Gap |
|---------|-----------|----------|-----|
| 30m | 1.68× | 1.56× | +0.12 |
| **1h** | **1.88×** | **1.56×** | **+0.32** |
| 2h | 1.65× | 1.49× | +0.16 |
| 4h | 1.31× | 1.26× | +0.05 |

**1h down-tail has the strongest directional asymmetry.** The crowd_oi coefficient is weaker for down (-3.14 vs -4.62 for any), but the *discrimination* is better — Q1 drops to 9.6% (vs 12.1% for up), meaning low-fragility regimes are even safer on the downside.

**Interpretation:** When OI crowding is low (fragile market), downside tails are ~2× more likely in Q5 vs Q1. This aligns with liquidity gap theory — sell-side liquidity evaporates faster than buy-side during thin positioning.

---

## 2. BTC-Only Prediction (Angle B)

### Results: cross-sectional features → BTC tail

| Target | Base | AUC | Q5/Q1 | crowd_oi coef |
|--------|------|-----|-------|---------------|
| btc_any_30m_2x | 53.3% | **0.564** | 1.06× | -3.95 |
| btc_any_60m_2x | 65.5% | 0.552 | 1.15× | -3.72 |
| btc_any_30m_3x | 37.7% | 0.558 | 1.22× | -4.34 |
| btc_down_30m_3x | 18.8% | 0.549 | 1.51× | -2.24 |
| btc_down_60m_5x | 32.4% | 0.526 | 1.24× | -1.45 |
| btc_up_30m_5x | 17.0% | 0.532 | 1.32× | -3.35 |

Best: **btc_any_30m_2x** — AUC 0.564, but Q5/Q1 only 1.06× (no quintile separation).

Monthly walk-forward: mean AUC = 0.558, std = 0.016 (stable but weak).

### Verdict: **NOT TRADEABLE for BTC directly**

Cross-sectional features (computed from 68 coins) carry almost no predictive power for a *single* asset's tail. The AUC ~0.56 is above chance but Q5/Q1 ~1.06–1.32× is too flat.

**Why:** BTC's tails are driven by BTC-specific catalysts (ETF flows, macro, whale orders), not by altcoin OI crowding patterns. The cross-sectional fragility indicator captures *market-wide* vulnerability, which only weakly correlates with BTC-specific moves.

**Actionable:** Use the fragility indicator as a *portfolio-level* risk filter, not as a BTC directional signal.

---

## 3. Conditional Interaction Matrix (Angle C)

### Tail probability by crowd_oi × pca_var1 tercile

(12×ATR, ≥10% coins, 1h horizon)

|  | PCA_low | PCA_mid | PCA_high |
|--|---------|---------|----------|
| **OI_low** | **45.9%** (7,043) | 38.7% (7,835) | 33.8% (8,506) |
| **OI_mid** | 40.6% (8,821) | 36.1% (8,122) | 30.3% (8,416) |
| **OI_high** | 33.4% (7,476) | 28.3% (7,286) | **22.7%** (6,358) |

### Interaction Analysis

| Metric | Value |
|--------|-------|
| Base rate | 34.6% |
| OI_low marginal | 39.1% |
| PCA_low marginal | 39.9% |
| OI_low & PCA_low combined | 45.9% |
| Additive prediction | 44.4% |
| **Synergy** | **+1.6%** (weak) |

**Both dimensions are monotonic.** Every step from OI_high→OI_low adds ~6% tail probability. Every step from PCA_high→PCA_low adds ~5%. The combined corner (OI_low + PCA_low = 45.9%) is almost exactly the additive prediction (44.4%).

### crowd_oi × breadth_extreme

|  | BX_low | BX_mid | BX_high |
|--|--------|--------|---------|
| **OI_low** | 37.5% (7,928) | **43.3%** (7,854) | 36.4% (7,602) |
| **OI_mid** | 33.8% (8,614) | 40.8% (8,717) | 32.3% (8,028) |
| **OI_high** | 27.0% (6,832) | 32.9% (6,645) | 25.8% (7,643) |

**Interesting:** breadth_extreme has a non-monotonic pattern (mid > low ≈ high). This matches the XS-8b finding that breadth contributes little linearly — its effect is concentrated in the middle tercile.

### Verdict: **No hidden interaction — effects are additive**

This confirms XS-8b's finding that nonlinear models don't help. The market responds linearly to crowd_oi and pca_var1. There's no threshold or regime switch — just a smooth gradient from safe (OI_high + PCA_high = 22.7% tail) to fragile (OI_low + PCA_low = 45.9% tail).

---

## 4. Multi-Horizon Comparison (Angle D)

| Horizon | Base | AUC | Q5/Q1 | crowd_oi coef | WF Mean | WF Std |
|---------|------|-----|-------|---------------|---------|--------|
| 30m | 7.9% | 0.556 | 1.78× | -4.46 | 0.565 | 0.019 |
| **1h** | **34.6%** | **0.591** | **1.83×** | **-4.62** | **0.584** | **0.011** |
| 2h | 79.3% | 0.654 | 1.31× | -6.87 | 0.638 | 0.024 |
| 4h | 96.8% | SKIP | — | — | — | — |

**Trade-off:** Longer horizons have higher AUC (more signal accumulation) but saturated base rates (less useful). 4h is 97% — useless.

**Sweet spot: 1h** — best AUC×Q5/Q1 product, most stable walk-forward (std 0.011), and the base rate (35%) is in the actionable range.

**30m** is interesting for real-time alerts (7.9% base rate, strong Q5/Q1) but too sparse — only 5.5% of Q1 periods see a tail vs 9.8% in Q5. The absolute numbers are small.

**crowd_oi coefficient grows with horizon** (-4.46 at 30m → -6.87 at 2h), confirming that OI crowding is a slow-moving regime that builds predictive power over hours.

---

## 5. Philosophical Synthesis

### What we proved

1. **Tails come from emptiness, not crowding.** Low OI = thin book = fragile. This is the opposite of the naive "crowded → liquidation cascade" hypothesis.

2. **The effect is fundamentally linear and additive.** No threshold, no regime switch, no interaction synergy. Market fragility scales smoothly with crowd_oi.

3. **Fragility hits harder on the downside.** Down-tail Q5/Q1 = 1.88× vs up-tail 1.56× at 1h. Sell-side liquidity gap > buy-side liquidity gap.

4. **This is a market-level, not asset-level signal.** BTC prediction is weak. The signal captures cross-sectional vulnerability — how the entire altcoin market is positioned.

5. **The ceiling is in the features, not the model.** GBT doesn't help. Interactions don't help. The 5 cross-sectional features capture a real but weak linear gradient.

### Practical Application (updated from XS-8)

| Regime | crowd_oi | pca_var1 | 1h tail prob | Action |
|--------|----------|----------|-------------|--------|
| **Safe** | High | High | ~23% | Full position size |
| **Normal** | Mid | Mid | ~35% | Standard size |
| **Fragile** | Low | Low | ~46% | Reduce size 30-50% |

For **downside protection specifically:**
- In fragile regime, Q1 down-tail rate = 9.6% vs Q5 = 18.1% (1.88×)
- Use fragility score to asymmetrically hedge: stronger downside protection in fragile regime

---

## Files

- **Script:** `flow_research/xs8c_deep_dive.py`
- **Extended data:** `flow_research/output/xs8c/xs8c_extended.parquet` (70K rows, 66 columns)
- **Previous:** `flow_research/xs8_tail_stress.py` (original), `flow_research/xs8_nonlinear.py` (model comparison)
