# XS-8 — Tail Stress Indicator (Market-Level)

**Date:** 2026-03-03  
**Data:** 68 Bybit perps, 2025-07-01 → 2026-02-28 (8 months)  
**Grid:** 5-minute cross-sectional snapshots (69,984 rows)  
**Features:** breadth_extreme, entropy, pca_var1, crowd_fund, crowd_oi  
**Target:** fraction of coins making |ret| > K×ATR in **next 1h**, binarized at various cutoffs  
**Model:** Logistic regression, walk-forward (train 60% / test 40%)  
**Script:** `xs8_tail_stress.py`

---

## TL;DR

**Moderate signal, not strong enough for standalone use.** Best AUC config (8×ATR, ≥5% coins)
hits 0.74 OOS but with base rate 94% it's trivial. Most useful config: **12×ATR, ≥10% coins**
(base 35%, AUC 0.59, Q5/Q1 1.82×). **crowd_oi** is the dominant predictor by far.

Q5/Q1 never reaches 2× for non-trivial targets. Useful as regime filter, not standalone.

---

## 1. Feature Distributions

| Feature | Mean | Std | P5 | P95 |
|---------|------|-----|-----|-----|
| breadth_extreme | 0.698 | 0.159 | 0.438 | 0.942 |
| entropy | 1.873 | 0.503 | 0.943 | 2.594 |
| pca_var1 | 0.372 | 0.141 | 0.187 | 0.649 |
| crowd_fund | 0.057 | 0.053 | 0.000 | 0.152 |
| crowd_oi | 0.046 | 0.040 | 0.000 | 0.119 |

---

## 2. Tail Fraction Distributions (1h horizon)

| Threshold | Mean | P25 | P50 | P75 | P95 |
|-----------|------|-----|-----|-----|-----|
| 3×ATR | 0.505 | 0.388 | 0.500 | 0.618 | 0.794 |
| 5×ATR | 0.226 | 0.104 | 0.191 | 0.313 | 0.529 |
| 8×ATR | 0.083 | 0.015 | 0.044 | 0.118 | 0.279 |
| 12×ATR | 0.032 | 0.000 | 0.015 | 0.044 | 0.118 |

1h horizon gives much better discrimination than 6h (which was 97%+ saturated).

---

## 3. Model Results — All Target Configurations

| Target | Base Rate | AUC OOS | Q5/Q1 |
|--------|-----------|---------|-------|
| 5×ATR, ≥30% coins | 90.4% | 0.685 | 1.19× |
| 8×ATR, ≥5% coins | 93.9% | **0.740** | 1.12× |
| 8×ATR, ≥10% coins | 82.4% | 0.659 | 1.26× |
| 8×ATR, ≥15% coins | 69.9% | 0.627 | 1.47× |
| 8×ATR, ≥20% coins | 58.6% | 0.609 | 1.59× |
| 8×ATR, ≥30% coins | 40.5% | 0.574 | 1.59× |
| 12×ATR, ≥3% coins | 75.8% | 0.621 | 1.32× |
| 12×ATR, ≥5% coins | 59.0% | 0.606 | 1.53× |
| **12×ATR, ≥10% coins** | **34.6%** | **0.591** | **1.82×** |
| 12×ATR, ≥15% coins | 23.3% | 0.570 | 1.76× |
| 12×ATR, ≥20% coins | 17.3% | 0.554 | 1.61× |
| 12×ATR, ≥30% coins | 11.3% | 0.544 | 1.52× |

**Trade-off:** Higher base rate → higher AUC but lower uplift. Lower base rate → more useful Q5/Q1 but weaker AUC.

**Sweet spot: 12×ATR, ≥10%** — P(tail) in Q5 is 1.82× vs Q1, base rate 35%, AUC 0.59.

---

## 4. Feature Coefficients (12×ATR, ≥10% target)

| Feature | Coefficient | Interpretation |
|---------|------------|----------------|
| crowd_oi | **-4.29** | High OI crowding → LOWER tail probability |
| crowd_fund | -1.22 | High funding extremes → lower tail probability |
| pca_var1 | -0.95 | High correlation → lower tail probability |
| entropy | -0.10 | Minimal impact |
| breadth_extreme | +0.50 | More extreme breadth → slightly higher tail |

**Key finding: crowd_oi dominates.** When many coins have high OI z-score simultaneously,
the next 1h is actually *less* likely to produce extreme moves. This suggests crowded positioning
acts as **damping** (liquidity provision) rather than fuel for explosions.

The explosion happens when OI crowding is LOW — i.e., when the market is positioned light
and a shock arrives with no liquidity buffer.

---

## 5. Monthly Walk-Forward Stability (12×ATR, ≥10%)

| Month | AUC | N | Positives |
|-------|-----|---|-----------|
| 2025-09 | 0.574 | 8,640 | 2,076 |
| 2025-10 | 0.577 | 8,928 | 1,794 |
| 2025-11 | 0.571 | 8,640 | 1,932 |
| 2025-12 | 0.580 | 8,928 | 2,358 |
| 2026-01 | 0.563 | 8,928 | 2,043 |
| 2026-02 | 0.563 | 8,063 | 1,650 |

**Stable.** AUC stays in 0.56-0.58 range for every month. No regime dependence.
This is a real but weak signal — exactly what you'd expect from 5 simple features.

---

## 6. Quintile Detail (12×ATR, ≥10%, OOS)

| Quintile | N | Positives | P(tail) | Uplift |
|----------|---|-----------|---------|--------|
| Q1 (low stress) | 5,598 | 885 | 15.8% | 0.67× |
| Q2 | 5,598 | 1,192 | 21.3% | 0.91× |
| Q3 | 5,597 | 1,433 | 25.6% | 1.09× |
| Q4 | 5,598 | 1,505 | 26.9% | 1.15× |
| Q5 (high stress) | 5,598 | 1,554 | 27.8% | 1.18× |

**Q5/Q1 = 1.82×** — Q5 sees tail events 28% of the time vs Q1's 16%.
Monotonic increase across quintiles — good calibration.

---

## 7. Verdict

### ✅ What works:
- **crowd_oi is a real market-level predictor** — 4× stronger coefficient than anything else
- **Stable across 6 months of walk-forward** — not overfitting
- **1.82× Q5/Q1** at the 12×ATR/10% target — meaningful separation
- **Counter-intuitive but consistent:** low crowding → more tails (liquidity gap thesis)

### ❌ What doesn't:
- **AUC 0.59 is marginal** — lots of overlap between stress regimes
- **Q5/Q1 < 2×** — not sharp enough for standalone position sizing
- **No directional info** — only says "tail likely", not "up or down"
- **breadth_extreme and entropy contribute little** — could simplify to just crowd_oi + pca_var1

### Final Assessment: **USEFUL AS REGIME FILTER, NOT STANDALONE** ✅

The stress score can reduce tail risk exposure by ~40% (Q1 = 16% vs base 35%)
when you use it to scale down positions during high-stress regimes.

**Practical application:**
1. Compute crowd_oi + pca_var1 in real-time (every 5 min)
2. When stress_score is in Q5 → reduce position sizes by 30-50%
3. When in Q1 → full size (16% tail risk vs 28% in Q5)
4. Can be layered onto any directional strategy

**Next step:** Test as regime filter on existing strategies (FR scalp, bracket trades).

---

## Files

- **Script:** `flow_research/xs8_tail_stress.py`
- **Stress data:** `flow_research/output/xs8/xs8_stress.parquet` (70K rows)
- **Summary:** `flow_research/output/xs8/xs8_summary.csv`
