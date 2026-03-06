# Phase 3 Findings: Execution Realism

**Date:** 2026-03-06

---

## Overview

Phase 3 stress-tested the winning combo (A: funding + mom_24h) across five execution dimensions:

1. Rebalancing frequency (8h / 16h / 24h)
2. Universe size (N = 5 / 10 / 15 / 20 / 30 per leg)
3. Sizing method (equal-weight vs vol-scaled 1/σ)
4. Maker fill rate (50–100%)
5. Market impact capacity (AUM $100K – $100M)

All simulations use the clean parquets and corrected resampling logic from Phase 2b.

---

## Test 1: Rebalancing Frequency

| Freq | Bars | Gross bps | Net bps | Ann. Ret | Ann. Vol | Sharpe | Max DD | Turnover | OOS win% |
|------|------|----------|--------|---------|---------|--------|--------|---------|---------|
| 8h | 1283 | 31.5 | **27.3** | 299% | 92% | **3.27** | -46% | 51.7% | 4/4 |
| 16h | 641 | 64.4 | 59.2 | 324% | 93% | **3.47** | -37% | 65.8% | 4/4 |
| 24h | 427 | 78.2 | 72.2 | 263% | 96% | 2.75 | -57% | 75.1% | 3/4 |

**8h and 16h are both strong. 24h degrades.**

At 16h, each bar captures roughly 2× the signal change, so gross bps nearly doubles (64 vs 31). Net Sharpe improves to 3.47 despite higher turnover (65.8%), because the larger gross alpha absorbs the cost. The 24h rebal shows higher gross but one negative OOS window (-2.09 Sharpe), likely because the 24h holding period is long enough for reversals to dominate.

**Choice:** 8h is preferred — it aligns with funding rate settlements (every 8h on Bybit), allows tighter risk management, and shows 4/4 positive OOS windows.

---

## Test 2: Universe Size (N per Leg)

Combo A at 8h rebalancing, maker fees.

| N per leg | Gross bps | Net bps | Ann. Ret | Ann. Vol | Sharpe | Max DD | OOS avg Sharpe |
|----------|----------|--------|---------|---------|--------|--------|---------------|
| 5 | 48.3 | 43.9 | 480% | 154% | 3.12 | -68% | 2.42 |
| **10** | 31.5 | **27.3** | 299% | 92% | **3.27** | -46% | **2.60** |
| 15 | 19.1 | 15.2 | 166% | 67% | 2.48 | -42% | 2.00 |
| 20 | 11.3 | 7.6 | 83% | 54% | 1.53 | -38% | 1.11 |
| 30 | 6.6 | 3.3 | 37% | 40% | 0.90 | -38% | 0.68 |

**N=10 is the sweet spot.** N=5 has higher gross alpha per trade but extreme volatility (154% ann. vol) and deeper drawdowns (-68%). N=10 achieves the best risk-adjusted returns. Beyond N=15, signal dilution dominates.

The signal decays monotonically — the top/bottom 10% of the universe (by composite rank) carry essentially all the alpha. Expanding to top/bottom 20% adds noise.

---

## Test 3: Sizing Method

| Method | Gross bps | Net bps | Ann. Ret | Ann. Vol | Sharpe | Max DD | OOS avg Sharpe |
|--------|----------|--------|---------|---------|--------|--------|---------------|
| **Equal-weight** | 31.5 | **27.3** | 299% | 92% | **3.27** | -46% | **2.60** |
| Vol-scaled (1/σ) | 24.4 | 20.2 | 221% | 87% | 2.54 | -51% | 1.76 |

**Equal-weight wins decisively.** Vol-scaling reduces gross alpha by 7 bps/trade and drops Sharpe from 3.27 to 2.54.

**Why:** The funding carry signal is positively correlated with volatility — high-funding coins tend to be in trending, volatile regimes. Vol-scaling downweights precisely those coins with the strongest signal. Equal-weight captures the full alpha, accepting higher variance.

Vol-scaling would be appropriate if the goal were pure risk parity, but here it actively removes alpha.

---

## Test 4: Maker Fill Rate Sensitivity

Baseline: 100% maker (4 bps/side). Degraded to partial maker + partial taker.

| Fill Rate | Eff. fee/side | Net bps | Ann. Ret | Sharpe |
|----------|-------------|--------|---------|--------|
| 100% maker | 4.0 bps | 27.3 | 299% | 3.27 |
| 90% maker | 4.6 bps | 26.7 | 293% | 3.19 |
| 80% maker | 5.2 bps | 26.1 | 286% | 3.12 |
| 70% maker | 5.8 bps | 25.5 | 279% | 3.05 |
| 60% maker | 6.4 bps | 24.9 | 272% | 2.97 |
| 50% maker | 7.0 bps | 24.2 | 265% | 2.90 |

**Extremely robust to fill rate.** Even 50% maker fill (7 bps/side effective) leaves Sharpe at 2.90. The strategy clears fees with comfortable margin at any realistic maker fill rate.

This is because gross alpha (31.5 bps) is large relative to fee range:
- Pure maker: 8 bps RT consumed → 23.5 bps net from signal
- Pure taker: 20 bps RT consumed → 11.5 bps net from signal

Both scenarios are profitable. Fee sensitivity is not a critical risk.

---

## Test 5: Market Impact Capacity

Using sqrt market impact model: `impact_bps = 10 × sqrt(order_size / daily_volume)`.
Assumes N=10, 8h rebal, equal-weight, 51.7% turnover per bar.

| AUM | Position Size | Order Fraction | Impact | Net after Impact |
|-----|-------------|---------------|--------|-----------------|
| $100K | $5K | 0.001% | 0.03 bps | 27.26 bps |
| $500K | $25K | 0.005% | 0.07 bps | 27.21 bps |
| $1M | $50K | 0.010% | 0.10 bps | 27.18 bps |
| $5M | $250K | 0.050% | 0.22 bps | 27.02 bps |
| $10M | $500K | 0.101% | 0.32 bps | 26.91 bps |
| $50M | $2.5M | 0.505% | 0.71 bps | 26.42 bps |
| **$100M** | $5M | 1.009% | **1.00 bps** | **26.05 bps** |

**Strategy is capacity-rich.** Even at $100M AUM, market impact consumes only 1 bps of the 27.3 bps net edge. Net Sharpe barely moves.

The strategy trades the top-10 most liquid perpetuals by relative rank — these are mid-large cap coins with daily volumes in the hundreds of millions to billions. A $5M order is a fraction of a percent of daily flow.

Practical capacity limit is likely set by portfolio concentration risk (10 long + 10 short, max drawdown risk) rather than market impact.

---

## Summary Table

| Dimension | Optimal Config | Key Finding |
|-----------|--------------|-------------|
| Signal | funding + mom_24h (equal-weight) | Super-additive combination |
| Rebal freq | **8h** | Aligns with funding, 4/4 OOS positive |
| Universe size | **N=10 per leg** | Sweet spot: alpha vs. noise |
| Sizing | **Equal-weight** | Vol-scaling removes alpha |
| Fees | Any maker fill ≥50% is fine | 31.5 bps gross vs 8–20 bps RT fee |
| Capacity | >$100M comfortable | Low impact on deep liquidity |

## Scripts

- `research_cross_section/phase3_execution.py` — all 5 tests

## Output Files

- `results/phase3_rebal_freq.csv`
- `results/phase3_universe_size.csv`
- `results/phase3_vol_sizing.csv`
- `results/phase3_fill_rate.csv`
- `results/phase3_capacity.csv`
