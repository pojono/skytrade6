# Research Findings v14 — Grid Bot Simulator

**Date:** 2026-02-16
**Symbol:** BTCUSDT
**Periods:** Q3 2025 (Jul-Sep, ranging +6.5%), Q4 2025 (Oct-Dec, crash -28%)
**Capital:** $10,000 | **Fee:** 7 bps per fill (Bybit VIP0)
**Grid:** 5 buy levels + 5 sell levels, symmetric

---

## The Fundamental Math Problem

Before any ML, the grid bot faces a hard constraint:

```
Fee per round-trip = 2 × 7 bps = 0.14%
Grid profit per round-trip = 1 × spacing
Breakeven: spacing > 0.14%
```

With 5 levels each side, total grid width = 10 × spacing:
- At 0.25% spacing → 2.5% total grid → barely above breakeven
- At 0.50% spacing → 5.0% total grid → comfortable margin
- At 1.00% spacing → 10.0% total grid → high margin, few fills

BTC median 1h range is ~0.5%. So a 0.25% grid gets ~2 fills per hour, but each fill barely covers fees. A 1.00% grid gets fills only when price moves 1%+, which happens a few times per day.

---

## Results: Q3 2025 (Ranging Period, +6.5%)

| Strategy | PnL | Grid Profits | Fees | PnL/day | Fills/day |
|----------|-----|-------------|------|---------|-----------|
| Fix 0.25% (8h) | -$1,689 | +$768 | -$1,340 | -$20 | 23 |
| Fix 0.50% (4h) | -$720 | +$161 | -$701 | -$9 | 12 |
| Fix 0.50% (8h) | -$863 | +$251 | -$599 | -$10 | 10 |
| **Fix 1.00% (24h)** | **-$167** | **+$101** | **-$171** | **-$2** | **3** |
| Adaptive (8h) | -$1,494 | +$683 | -$1,180 | -$18 | 20 |
| Wide adapt (24h) | -$415 | +$254 | -$368 | -$5 | 6 |

**Even in the most favorable period (low trend, moderate vol), no strategy is profitable.**

---

## Results: Q4 2025 (Crash Period, -28%)

| Strategy | PnL | Grid Profits | Fees | PnL/day | Fills/day |
|----------|-----|-------------|------|---------|-----------|
| Fix 0.25% (8h) | -$2,673 | +$1,012 | -$2,034 | -$32 | 35 |
| Fix 0.50% (8h) | -$1,513 | +$523 | -$1,049 | -$18 | 18 |
| **Fix 1.00% (24h)** | **+$34** | **+$425** | **-$300** | **+$0.40** | **5** |
| Adaptive (8h) | -$2,452 | +$847 | -$1,860 | -$29 | 32 |
| Wide adapt (24h) | -$203 | +$736 | -$573 | -$2 | 10 |

**One strategy barely broke even: Fix 1.00% with 24h rebalance (+$34 over 83 days).**

---

## Cross-Period Summary

| Strategy | Q3 (range) | Q4 (crash) | 6-month total |
|----------|-----------|-----------|---------------|
| Fix 0.25% (8h) | -$1,689 | -$2,673 | **-$4,362** |
| Fix 0.50% (8h) | -$863 | -$1,513 | **-$2,376** |
| Fix 1.00% (24h) | -$167 | +$34 | **-$133** |
| Adaptive (8h) | -$1,494 | -$2,452 | **-$3,946** |
| Wide adapt (24h) | -$415 | -$203 | **-$618** |

**Clear monotonic pattern: wider spacing + longer rebalance = less loss.**

---

## Why Adaptive Spacing Doesn't Help

The vol-adaptive grid (from v9 Ridge model) averages 0.28% spacing — barely above the 0.14% breakeven. It's not wide enough to generate meaningful profit per fill.

| Regime | Adaptive Spacing | Fixed 0.50% | Verdict |
|--------|-----------------|-------------|---------|
| Calm | 0.22% | 0.50% | Adaptive too tight → more fills, more fees |
| Volatile | 0.46% | 0.50% | Similar → no advantage |

The adaptive grid **tightens in calm markets** (25% tighter than fixed 0.50%), which is the opposite of what we want. In calm markets, the grid should be tight to capture small oscillations — but our spacing is already at the fee floor, so tighter = more fee drag.

The vol prediction is accurate (R²=0.34), but it's solving the wrong problem. The grid doesn't need to know "how volatile will the next hour be?" — it needs to know "will price mean-revert within my grid width?" And that's a different question we can't answer (direction R²≈0).

---

## The Three Sources of Loss

1. **Fee drag** — Each fill costs 0.07%. With 10-35 fills/day, fees are $7-$25/day on $10K capital.
2. **Inventory bleed** — Grid accumulates directional inventory during trends. Even a +6.5% quarterly trend causes persistent long inventory that loses on rebalance.
3. **Rebalance cost** — Closing inventory at market to recenter the grid is a forced loss. Less frequent rebalancing helps, but the inventory grows larger.

---

## What Would Make a Grid Bot Profitable?

### 1. Lower fees (most impactful)
At 2 bps per fill (maker rebate on some exchanges), breakeven drops to 0.04%. This changes everything:
- 0.25% spacing → 6× margin over fees (vs 1.8× at 7 bps)
- Grid profits would dominate fee costs

### 2. Maker-only orders
Grid bots naturally use limit orders. On exchanges with maker rebates, you get PAID to provide liquidity instead of paying fees. This flips the economics entirely.

### 3. Hedging inventory risk
Instead of rebalancing (closing at market), hedge the accumulated inventory with a futures position. This avoids the rebalance cost but adds complexity.

### 4. Truly range-bound assets
BTC trends too much even in "ranging" periods. A stablecoin pair (e.g., USDT/USDC) or a mean-reverting spread would be more suitable.

---

## Honest Assessment

| Question | Answer |
|----------|--------|
| Does vol prediction help grid bots? | **No** — at 7 bps fees, spacing is fee-constrained, not vol-constrained |
| Can a grid bot profit on BTC at 7 bps? | **Barely** — only with 1%+ spacing and 24h rebalance, and even then ~$0/day |
| Is the grid bot concept viable? | **Only with maker fees** (≤2 bps) or maker rebates |
| Did we waste time? | **No** — we now know the exact fee threshold where grid bots become viable |

---

## Key Lesson

**The grid bot problem is not a prediction problem — it's a fee problem.**

Our vol prediction (R²=0.34) is genuinely good. But it doesn't matter because:
- Grid spacing is constrained by fees, not by vol uncertainty
- Direction is unpredictable, so inventory always accumulates
- Rebalancing is expensive regardless of how well you predict vol

The path to a profitable grid bot is:
1. **Get maker fees** (exchange VIP tier, or use an exchange with maker rebates)
2. **Then** use vol prediction to optimize spacing
3. **Then** the adaptive grid actually has room to outperform fixed

Without step 1, steps 2-3 are irrelevant.

---

## Files

| File | Description |
|------|-------------|
| `grid_bot_sim.py` | Grid bot simulator with proper mechanics |
| `results/grid_bot_BTC_Q3.txt` | Q3 2025 results (ranging period) |
| `results/grid_bot_BTC_Q4.txt` | Q4 2025 results (crash period) |
