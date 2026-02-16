# Research Findings v14 — Grid Bot Simulator

**Date:** 2026-02-16
**Symbol:** BTCUSDT
**Period:** 2025-01-01 → 2026-01-31 (387 days, $94K → $79K, -16%)
**Capital:** $10,000 | **Fee:** 2 bps maker per fill (Bybit VIP0 futures limit orders)
**Grid:** 5 buy levels + 5 sell levels, symmetric

---

## Fee Correction

Grid bots use **limit orders only** → maker fee applies (0.02% per fill), not taker fee (0.055%).
- **Round-trip fee:** 2 × 2 bps = **0.04%** (not 0.14% as initially assumed)
- **Breakeven spacing:** > 0.04% per level
- This makes grids with 0.25%+ spacing comfortably profitable per fill

---

## 13-Month Results (387 days, BTC -16%)

### Profitable Strategies

| Strategy | PnL | Grid$ | Fees | PnL/day | Sharpe | MaxDD |
|----------|-----|-------|------|---------|--------|-------|
| **Fix 1.00% (24h)** | **+$789** | +$1,458 | -$339 | **+$2.04** | **0.83** | **-$837** |
| **Adapt /2 f050 (24h)** | **+$525** | +$2,994 | -$684 | **+$1.36** | **0.44** | -$1,241 |
| **Adapt /2 (24h)** | **+$293** | +$3,603 | -$995 | **+$0.76** | **0.24** | -$1,275 |
| **Fix 0.50% (24h)** | **+$290** | +$3,150 | -$730 | **+$0.75** | **0.27** | -$1,404 |
| **Adapt /2 f025 (24h)** | **+$190** | +$3,526 | -$929 | **+$0.49** | **0.17** | -$1,317 |

### Losing Strategies

| Strategy | PnL | Grid$ | Fees | PnL/day | Sharpe |
|----------|-----|-------|------|---------|--------|
| Fix 0.25% (24h) | -$181 | +$3,740 | -$1,379 | -$0.47 | -0.04 |
| Adapt /5 (24h) | -$959 | +$3,670 | -$2,065 | -$2.47 | -0.54 |
| Adapt /2 +Pause (24h) | -$493 | +$3,064 | -$977 | -$1.27 | -0.31 |
| All 8h rebalance variants | -$1.6K to -$3.7K | — | — | — | — |

---

## Key Findings

### 1. Fix 1.00% (24h) — Best Risk-Adjusted Return

- **Sharpe 0.83** — the highest of any strategy
- Only 4.4 fills/day → low fee drag ($339 total over 387 days)
- Grid profits +$1,458 easily cover fees
- Max drawdown only -$837 (8.4% of capital)
- **Limitation:** few fills means low absolute return (+$789 on $10K = 7.9%/year)

### 2. Adaptive Grid Actually Works Now

With proper spacing formula (spacing = predicted_range / 2):
- **Calm periods:** 0.19% spacing (62% tighter than fixed 0.50%)
- **Volatile periods:** 0.41% spacing (19% tighter than fixed 0.50%)
- The grid genuinely adapts — tighter in calm to capture more oscillations, wider in volatile to avoid overrun

**Adapt /2 (24h) beats Fix 0.25% (24h):** +$293 vs -$181. The adaptive grid avoids the trap of being too tight during vol spikes.

### 3. Adapt /2 f050 (24h) — Best Adaptive Variant

- Floor at 0.50% prevents over-tightening during calm periods
- Still adapts upward during high vol (mean spacing 0.53%)
- **+$525, Sharpe 0.44, MaxDD -$1,241**
- More fills than Fix 1.00% (8.8/day vs 4.4/day) → higher grid profits (+$2,994 vs +$1,458)
- But higher fees and inventory risk offset the extra grid profits

### 4. 24h Rebalance Dominates 8h

Every single strategy is better with 24h rebalance. The pattern is consistent:
- Fix 0.50% (8h): -$1,639 → Fix 0.50% (24h): +$290
- Adapt /2 (8h): -$2,291 → Adapt /2 (24h): +$293

**Rebalancing less frequently is the single most important parameter.** Each rebalance closes inventory at market — a forced loss during trends. Fewer rebalances = less forced loss.

### 5. Pausing During Extreme Vol Hurts

Adapt /2 +Pause (24h): -$493 vs Adapt /2 (24h): +$293. Pausing during high vol means missing wide-grid fills that are actually the most profitable (large spacing = large profit per fill).

### 6. Too-Tight Grids Still Lose

Adapt /5 (spacing = range/5, ~0.13% avg) loses -$959 despite +$3,670 in grid profits. The tight spacing generates 26.7 fills/day → $2,065 in fees + inventory bleed from frequent position changes.

---

## The Three Drivers of Grid Bot PnL

| Driver | Impact | Optimization |
|--------|--------|-------------|
| **Grid profits** | Positive: $1.5K-$7.5K | More fills = more grid profits, but diminishing returns |
| **Fees** | Negative: $0.3K-$6.2K | Fewer fills = lower fees. Maker fee (2 bps) is key |
| **Inventory losses** | Negative: $0.5K-$5K | Wider spacing + less rebalancing = less inventory bleed |

The winning formula: **maximize grid profits while keeping fees and inventory losses low.**
- Wide spacing (0.50-1.00%) → high profit per fill, low fill count
- Infrequent rebalancing (24h) → less inventory close cost
- Adaptive spacing → tighter in calm (more fills when safe), wider in volatile (less inventory risk)

---

## Does Vol Prediction Add Value?

| Comparison | Fixed | Adaptive | Δ |
|-----------|-------|----------|---|
| 0.25% fixed vs Adapt /2 (24h) | -$181 | +$293 | **+$474** |
| 0.50% fixed vs Adapt /2 f050 (24h) | +$290 | +$525 | **+$235** |
| 1.00% fixed vs Adapt /2 f050 (24h) | +$789 | +$525 | **-$264** |

**Mixed.** Adaptive beats fixed at narrow spacings (where adaptation matters most) but loses to the widest fixed grid. The vol prediction helps avoid being too tight during vol spikes, but the simplest approach (just use wide fixed spacing) still wins on Sharpe.

**Honest answer:** Vol prediction adds marginal value. The biggest driver is spacing width and rebalance frequency, not adaptive vs fixed.

---

## Annualized Returns

On $10,000 capital over 387 days:

| Strategy | Total PnL | Annual Return | Sharpe |
|----------|----------|--------------|--------|
| Fix 1.00% (24h) | +$789 | **+7.4%** | 0.83 |
| Adapt /2 f050 (24h) | +$525 | **+5.0%** | 0.44 |
| Fix 0.50% (24h) | +$290 | **+2.7%** | 0.27 |

These returns are **modest but real** — achieved through a -16% BTC decline with no directional bias. The grid bot is a pure mean-reversion strategy that profits from oscillations regardless of trend direction.

---

## Limitations & Caveats

1. **Limit order fill assumption** — We assume limit orders fill when price touches the level. In reality, there's queue priority and partial fills.
2. **No slippage** — Real fills may be slightly worse than the limit price.
3. **Single asset** — Only tested on BTCUSDT. Needs validation on ETH, SOL.
4. **Rebalance at market** — We close inventory at market price during rebalance. A real bot could use limit orders for this too.
5. **No funding rate** — Futures positions incur funding costs which aren't modeled.
6. **Capital efficiency** — $10K capital with 5 levels = $1K per level. Max inventory can reach $5K+ which ties up significant capital.

---

## Files

| File | Description |
|------|-------------|
| `grid_bot_sim.py` | Grid bot simulator (v2, proper mechanics, maker fees) |
| `results/grid_bot_BTC_13m_maker.txt` | 13-month results with correct maker fees |
| `results/grid_bot_BTC_Q3.txt` | Q3 2025 results (old 7 bps fees) |
| `results/grid_bot_BTC_Q4.txt` | Q4 2025 results (old 7 bps fees) |
