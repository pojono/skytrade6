# Partial Exits Research — Findings

**Date:** Feb 19, 2026  
**Source:** `liq_partial_exits_research.py` using WS ticker data (~100ms resolution)  
**Data:** 4 symbols (DOGE, SOL, ETH, XRP), 2,371 trades  
**Key principle:** All exits are MAKER (limit orders) except safety timeout (taker, last resort)

---

## Why Maker-Only Exits Matter

| Exit type | Fee | Round-trip (entry+exit) |
|-----------|-----|------------------------|
| Maker entry + Maker exit | 0.02% + 0.02% | **0.04%** |
| Maker entry + Taker exit | 0.02% + 0.055% | **0.075%** |

On a typical 8-10 bps gross profit, the difference between 0.04% and 0.075% fees is **~40% of the profit**. Maker-only exits are not a minor optimization — they fundamentally change the economics.

---

## Baselines

| Config | Return | Sharpe | Max DD | WR | Timeouts | Exit Fee |
|--------|--------|--------|--------|-----|----------|----------|
| **Trail 3bps (taker exit)** | **+310.58%** | +46.5 | 0.11% | 96.2% | 0 | taker |
| Trail 5bps (taker exit) | +263.18% | +39.4 | 0.15% | 94.0% | 0 | taker |
| TP 12bps (maker exit) | +159.72% | +28.2 | 1.62% | 98.1% | 46 | maker |
| TP 8bps (maker exit) | +80.55% | +19.3 | 1.71% | 99.1% | 21 | maker |
| TP 5bps (maker exit) | +20.01% | +10.2 | 1.12% | 99.7% | 6 | maker |

**The dilemma:** Pure trailing stop gives the best returns but pays taker fees on every exit. Fixed TP pays maker fees but has timeout losses. Can we get the best of both?

---

## Variant Results Summary

### Variant A: Partial TP (maker) + Trail (maker) on Remainder

**The clear winner.** Split the position: one part exits at a fixed TP (maker), the other runs a trailing limit (maker).

**Top configs from Variant A:**

| Config | Return | Sharpe | Max DD | WR | TO |
|--------|--------|--------|--------|-----|-----|
| 70% TP@12bps + 30% trail@3bps | +213.34% | **+87.6** | 0.11% | 96.2% | 0 |
| 70% TP@10bps + 30% trail@3bps | +184.82% | **+80.8** | 0.11% | 96.2% | 0 |
| 70% TP@10bps + 30% trail@5bps | +168.05% | **+70.2** | 0.15% | 94.0% | 0 |
| 50% TP@12bps + 50% trail@3bps | +241.12% | +67.1 | 0.11% | 96.2% | 0 |
| 70% TP@8bps + 30% trail@3bps | +155.21% | **+71.7** | 0.11% | 96.2% | 0 |
| 50% TP@10bps + 50% trail@3bps | +220.75% | +62.9 | 0.11% | 96.2% | 0 |
| 50% TP@8bps + 50% trail@3bps | +199.60% | +58.0 | 0.11% | 96.2% | 0 |
| 30% TP@12bps + 70% trail@3bps | +268.90% | +55.8 | 0.11% | 96.2% | 0 |

**Key patterns:**
- **More trail → more return** (30% TP gives highest returns)
- **More TP → higher Sharpe** (70% TP gives highest risk-adjusted returns)
- **3 bps trail consistently beats 5 bps** across all splits
- **TP at 10-12 bps is optimal** (8 bps leaves money on the table, 12 bps is near the TP ceiling)
- **Zero timeouts** across all configs

### Variant B: Two Fixed TPs (both maker)

| Config | Return | Sharpe | Max DD | WR | TO |
|--------|--------|--------|--------|-----|-----|
| 50% TP@8bps + 50% TP@12bps | +120.14% | +26.0 | 1.62% | 98.1% | 46 |
| 70% TP@8bps + 30% TP@15bps | +114.91% | +23.5 | 1.62% | 96.7% | 83 |
| 50% TP@8bps + 50% TP@30bps | +128.75% | +9.4 | 3.91% | 83.5% | 438 |

**Verdict: Worst variant.** The second TP level reintroduces timeouts. At TP2=30bps, 438 trades timeout — nearly 20% of all trades. Returns are lower than pure trail AND lower than Variant A. The timeout problem we solved with trailing stops comes right back.

### Variant C: TP (maker) + Tighter Trail (maker) After TP Fills

| Config | Return | Sharpe | Max DD | WR | TO |
|--------|--------|--------|--------|-----|-----|
| **50% TP@10bps, trail 5→3bps** | **+280.61%** | **+61.4** | **0.15%** | 94.0% | 0 |
| 50% TP@8bps, trail 5→3bps | +261.88% | +59.0 | 0.15% | 95.3% | 1 |
| 70% TP@10bps, trail 5→3bps | +219.28% | +73.2 | 0.15% | 94.0% | 0 |
| 70% TP@8bps, trail 5→3bps | +191.71% | +68.1 | 0.15% | 95.3% | 1 |
| 50% TP@10bps, trail 8→5bps | +254.14% | +54.3 | 0.21% | 92.5% | 1 |

**Verdict: Highest absolute returns.** Variant C at 50% TP@10bps with trail 5→3bps gives **+280.61%** — the highest return of any config tested. The two-phase trail is clever: wide trail (5 bps) protects the full position early, then tightens to 3 bps after the TP half exits, locking in profit on the remainder quickly.

**But:** More complex to implement than Variant A, and the Sharpe is lower than the best A configs.

### Variant D: Progressive Scale-Out at Milestones (maker)

| Config | Return | Sharpe | Max DD | WR | TO |
|--------|--------|--------|--------|-----|-----|
| 50%@8bps, trail 5 | +152.22% | +44.3 | 0.15% | 95.3% | 0 |
| 50%@8bps + 25%@15bps, trail 5 | +131.28% | +64.6 | 0.15% | 95.3% | 0 |
| 50%@8bps + 25%@15bps + 12.5%@30bps, trail 5 | +128.42% | **+77.6** | 0.15% | 95.3% | 0 |
| 33%@5bps + 33%@10bps + 33%@15bps, trail 5 | +79.17% | **+85.2** | 0.15% | 92.9% | 0 |

**Verdict: Best Sharpe ratios but lowest returns.** Progressive scale-out gives extremely smooth equity curves (Sharpe 77-85!) but locks in profit too early, leaving less to ride. The 3-milestone config (33%@5+33%@10+33%@15) has Sharpe +85.2 but only +79% return. Good for ultra-conservative deployment.

### Variant E: Partial Exit on Losers + Trail

| Config | Return | Sharpe | Max DD | WR | TO |
|--------|--------|--------|--------|-----|-----|
| Trail 5bps, cut 30% at -20bps | +218.73% | +32.1 | 0.22% | 84.3% | 0 |
| Trail 5bps, cut 30% at -8bps | +189.91% | +29.7 | 0.17% | 90.8% | 0 |
| Trail 8bps, cut 30% at -20bps | +152.30% | +22.6 | 0.31% | 77.3% | 0 |

**Verdict: Hurts performance.** Cutting losers early sounds good but the trailing stop already handles this. Partial loss-cutting adds fees and reduces the chance of recovery. Every config underperforms the pure trail baseline. **Don't use this.**

---

## The Big Picture — All Variants Compared

| Rank | Config | Return | Sharpe | DD | WR | Complexity |
|------|--------|--------|--------|-----|-----|-----------|
| 1 | **C: 50% TP@10bps, trail 5→3bps** | **+280.61%** | +61.4 | 0.15% | 94.0% | High |
| 2 | Trail 3bps pure (taker) | +310.58% | +46.5 | 0.11% | 96.2% | Low |
| 3 | A: 30% TP@12bps + 70% trail@3bps | +268.90% | +55.8 | 0.11% | 96.2% | Medium |
| 4 | C: 50% TP@8bps, trail 5→3bps | +261.88% | +59.0 | 0.15% | 95.3% | High |
| 5 | A: 50% TP@12bps + 50% trail@3bps | +241.12% | +67.1 | 0.11% | 96.2% | Medium |
| 6 | Trail 5bps pure (taker) | +263.18% | +39.4 | 0.15% | 94.0% | Low |
| 7 | A: 70% TP@12bps + 30% trail@3bps | +213.34% | **+87.6** | 0.11% | 96.2% | Medium |
| 8 | A: 70% TP@10bps + 30% trail@3bps | +184.82% | +80.8 | 0.11% | 96.2% | Medium |
| 9 | D: 33%@5+33%@10+33%@15, trail 5 | +79.17% | **+85.2** | 0.15% | 92.9% | Very High |

**Note:** Pure trail 3bps has the highest raw return (+310%) but pays taker fees. In reality, the maker-only configs may perform closer to or better than this live, because taker fees are guaranteed to be 0.055% while maker limit fills may get better prices.

---

## Worst-Case Comparison

| Config | Worst Trade | Worst 5 Avg | P1 PnL | Max Consec Losses |
|--------|------------|-------------|--------|-------------------|
| Trail 5bps (taker) | -0.083% | -0.072% | -0.052% | 4 |
| TP 12bps (maker) | **-1.618%** | -1.378% | -0.484% | 1 |
| A: 50% TP@8 + 50% trail@5 | -0.083% | -0.072% | -0.052% | 3 |
| A: 50% TP@8 + 50% trail@3 | **-0.063%** | **-0.052%** | **-0.032%** | 3 |
| C: 50% TP@8, trail 5→3 | -0.083% | -0.072% | -0.052% | 3 |

All partial exit configs have tiny worst-case losses (~8 bps). The fixed TP baseline has 20x worse tail risk.

---

## Recommendations

### For Maximum Return
**Variant C: 50% TP@10bps, trail 5→3bps** → +280.61%, Sharpe +61.4
- Half exits at fixed 10 bps TP (maker, cheap)
- Other half rides with 5 bps trail, tightening to 3 bps after TP fills
- Highest return of any maker-only config
- But: most complex to implement

### For Best Risk-Adjusted (Sharpe)
**Variant A: 70% TP@12bps + 30% trail@3bps** → +213.34%, Sharpe +87.6
- 70% exits at fixed 12 bps TP (maker) — simple, reliable
- 30% rides with 3 bps trailing limit (maker) — captures upside
- Extremely smooth equity curve
- Simpler than Variant C

### For Simplest Implementation
**Variant A: 50% TP@8bps + 50% trail@5bps** → +174.82%, Sharpe +50.3
- Easy to understand: half at fixed target, half trailing
- 5 bps trail is more realistic for live than 3 bps
- Still beats the pure TP baseline by +15% return and 10x less drawdown

### For Ultra-Conservative
**Variant D: 33%@5+33%@10+33%@15, trail 5** → +79.17%, Sharpe +85.2
- Lowest return but smoothest equity curve
- Sharpe +85 is extraordinary
- Good for large accounts where consistency matters more than return

### What NOT to Use
- **Variant B** (two fixed TPs) — reintroduces timeout problem
- **Variant E** (partial loss cutting) — hurts performance, adds complexity

---

## Implementation Notes

### Maker Trailing Limit
The trailing "stop" in these configs is actually a **trailing limit order**:
1. Place a limit sell (for longs) at `peak_price × (1 - trail_bps/10000)`
2. On each new peak, cancel and replace with updated level
3. When price retraces to the limit → fills as **maker**

This requires cancel+replace on each price update (~1/sec), but avoids taker fees entirely.

### Order Management for Variant A (recommended)
```
After fill:
  1. Place limit order for 70% of position at TP price (maker)
  2. Place limit order for 30% of position at trail level (maker)
  3. On each tick: if peak moved, cancel+replace trail limit
  4. If TP fills: cancel trail limit, done with that 70%
  5. Trail limit fills: cancel TP limit if still open, done
  6. Safety: 60 min timeout → market close remainder (taker, very rare)
```

---

*Source: `liq_partial_exits_research.py`, `results/liq_partial_exits_research.txt`*
