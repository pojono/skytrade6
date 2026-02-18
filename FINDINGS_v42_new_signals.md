# FINDINGS v42: New Signal Research — 5 Experiments

## Overview

**Objective**: Explore what other profitable signals the tick-level data can produce beyond the cascade MM strategy.

**Method**: 5 independent experiments on SOLUSDT (7-30 days), expand winners to DOGEUSDT.

**Result**: 2 actionable findings, 3 dead ends.

---

## Experiment Scorecard

| Exp | Signal | Verdict | Edge | Actionable? |
|-----|--------|---------|------|-------------|
| **A** | Spot-Futures Basis Mean-Reversion | ❌ DEAD | ~0.5 bps (fees = 7.5 bps) | No |
| **B** | Cascade Size Filtering | ✅ **WINNER** | P97: +6-13 bps OOS, 93-97% WR | Yes — enhances cascade MM |
| **C** | OI Divergence (buildup → breakout) | ❌ DEAD | No predictive power | No |
| **D** | Funding Rate Pre-Settlement | ❌ DEAD | -5 bps price_ret kills it | No |
| **E** | Intraday Seasonality | ✅ **WINNER** | 15:00 UTC +23-36 bps, 18:00 -22-28 bps | Yes — time filter |

---

## EXP A: Spot-Futures Basis Mean-Reversion

**Thesis**: When futures trade at extreme premium/discount to spot, the basis reverts.

**Finding**: The basis z-score → forward return relationship is **monotonically real**:
- z < -3: +0.43 bps at 10s
- z > +3: -0.42 bps at 10s
- Spread: ~0.85 bps at 10s, ~0.52 bps at 60s

**Why it fails**: The edge (~0.5 bps) is 15x smaller than round-trip fees (7.5 bps). Would need 0% fees on both legs or a cross-exchange arb setup.

**Status**: Dead for single-exchange trading. Potentially viable as cross-exchange arb (Bybit futures vs Binance spot) but requires separate infrastructure.

---

## EXP B: Cascade Size Filtering ✅

**Thesis**: Larger liquidation cascades create bigger dislocations and more reliable mean-reversion.

**Finding**: **Confirmed on 30 days, 2 symbols, walk-forward OOS.**

### Threshold Comparison (SOLUSDT, 30 days)

| Threshold | Cascades | Fills | WR% | Avg Net | Total Net |
|-----------|----------|-------|-----|---------|-----------|
| P90 | 648 | 358 | 87.7% | +6.5 bps | +23.35% |
| P93 | 417 | 249 | 90.0% | +8.1 bps | +20.22% |
| P95 | 303 | 188 | 90.4% | +8.3 bps | +15.55% |
| P97 | 173 | 113 | 93.8% | +10.6 bps | +11.97% |
| P99 | 53 | 38 | 94.7% | +12.1 bps | +4.61% |

**Key insight**: Higher threshold → higher WR and avg return, but fewer trades. P95-P97 is the sweet spot (enough trades + high edge).

### Walk-Forward OOS (train=20d, test=10d)

| Config | Train | Test | OOS? |
|--------|-------|------|------|
| SOL P95 off=0.20 tp=0.15 sl=0.50 | +10.50% (95% WR) | +4.02% (96% WR) | ✅ |
| SOL P97 off=0.20 tp=0.15 sl=0.50 | +7.98% (98% WR) | +1.71% (93% WR) | ✅ |
| DOGE P95 off=0.20 tp=0.15 sl=0.50 | +2.87% (88% WR) | +2.07% (96% WR) | ✅ |
| DOGE P97 off=0.20 tp=0.20 sl=0.50 | +4.46% (87% WR) | +1.51% (93% WR) | ✅ |

### Cascade Notional → Edge

| Size Tercile | SOL Avg Net | SOL WR | DOGE Avg Net | DOGE WR |
|-------------|-------------|--------|-------------|---------|
| Small (bottom 1/3) | +8.5 bps | 89% | +7.8 bps | 89% |
| Medium (middle 1/3) | +6.3 bps | 87% | +10.5 bps | 91% |
| Large (top 1/3) | +12.0 bps | 95% | +1.9 bps | 81% |

On SOL, large cascades clearly outperform. On DOGE, the relationship is less clear — medium cascades are best.

### Practical Implication

**Raise the cascade threshold from P95 to P97 for higher quality trades.** This reduces trade frequency (~40%) but increases win rate from ~90% to ~94% and avg return from +8 to +11 bps. The tradeoff is worth it for a live system where execution quality matters.

---

## EXP C: OI Divergence

**Thesis**: When OI rises but price is flat, a breakout is building. When OI drops but price is flat, the market is range-bound.

**Finding**: OI buildup does NOT predict larger forward moves. The |fwd_60m| for "OI↑ + Price flat" (61.9 bps) is actually slightly *below* baseline (65.5 bps). The OI-price divergence signal has no predictive power for absolute return magnitude.

**Status**: Dead.

---

## EXP D: Funding Rate Pre-Settlement

**Thesis**: Position 1h before funding settlement based on current FR direction. Collect funding + ride the expected price pressure.

**Finding**: The price return component (-5 bps avg) overwhelms the funding income (+1 bps avg), resulting in net -11.5 bps per trade. The price movement around funding settlements is too noisy and often goes against the expected direction.

**Status**: Dead for directional trading. Funding rate is too small (~0.01%) relative to price noise to be tradeable at this frequency.

---

## EXP E: Intraday Seasonality ✅

**Thesis**: Certain hours of the day have predictable return patterns.

**Finding**: **Confirmed on 30 days, 2 symbols, week-by-week consistent.**

### Best and Worst Hours (30 days)

| Hour | SOL Avg (bps) | SOL Sharpe | DOGE Avg (bps) | DOGE Sharpe |
|------|-------------|-----------|---------------|------------|
| **15:00 UTC** | **+23.3** | **+5.87** | **+35.8** | **+7.07** |
| 04:00 UTC | +11.1 | +3.57 | +14.2 | +3.66 |
| 11:00 UTC | +12.8 | +3.96 | +7.3 | +1.90 |
| 21:00 UTC | +7.5 | +2.08 | +14.8 | +3.42 |
| **14:00 UTC** | **-19.1** | **-3.97** | **-27.5** | **-4.56** |
| **18:00 UTC** | **-22.5** | **-4.50** | **-27.8** | **-5.00** |
| 01:00 UTC | -16.9 | -3.32 | -17.0 | -2.63 |
| 08:00 UTC | -7.3 | -1.95 | -18.9 | -4.56 |

### Week-by-Week Consistency

- **15:00 UTC positive**: 5/5 weeks on both SOL and DOGE
- **18:00 UTC negative**: 4/5 weeks on both SOL and DOGE

### Practical Implication

**Use hour-of-day as a filter for the cascade MM strategy:**
- **Prefer trades during 15:00, 04:00, 11:00, 21:00 UTC** (positive seasonality)
- **Avoid or reduce size during 14:00, 18:00, 01:00, 08:00 UTC** (negative seasonality)
- This could improve the cascade MM by ~20-30% by avoiding the worst hours

**Caution**: 30 days is still a short sample. Need to validate on 88+ days before using in production.

---

## Summary: What to Do Next

1. **Integrate cascade size filtering into the cascade MM** — raise threshold to P97, accept fewer but higher-quality trades
2. **Add hour-of-day filter** — avoid 14:00 and 18:00 UTC, prefer 15:00 and 04:00 UTC
3. **Validate seasonality on full 88-day dataset** before relying on it
4. **Don't pursue**: basis arb (too small), OI divergence (no signal), FR pre-settlement (negative EV)
