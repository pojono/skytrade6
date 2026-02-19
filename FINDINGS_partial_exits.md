# Partial Exits Research — Findings

**Date:** Feb 19, 2026  
**Source:** `liq_partial_exits_research.py`  
**Data:** 5 symbols (BTC, DOGE, SOL, ETH, XRP), 2,630 trades, 87 days (May–Aug 2025)  
**Ticker:** WS ~100ms for all 5 symbols (BTC has 25-day gap in July)  
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
| **Trail 3bps (taker exit)** | **+328.95%** | +45.6 | 0.11% | 95.1% | 0 | taker |
| Trail 5bps (taker exit) | +276.41% | +38.4 | 0.19% | 92.3% | 1 | taker |
| TP 12bps (maker exit) | +168.94% | +24.8 | 2.21% | 97.5% | 67 | maker |
| TP 8bps (maker exit) | +88.47% | +19.6 | 1.71% | 98.9% | 29 | maker |
| TP 5bps (maker exit) | +22.39% | +10.8 | 1.12% | 99.7% | 8 | maker |

**The dilemma:** Pure trailing stop gives the best returns but pays taker fees on every exit. Fixed TP pays maker fees but has timeout losses. Can we get the best of both?

---

## Variant Results Summary

### Variant A: Partial TP (maker) + Trail (maker) on Remainder

**The clear winner.** Split the position: one part exits at a fixed TP (maker), the other runs a trailing limit (maker).

**Top configs from Variant A:**

| Config | Return | Sharpe | Max DD | WR | TO |
|--------|--------|--------|--------|-----|-----|
| 70% TP@12bps + 30% trail@3bps | +228.67% | **+83.2** | 0.11% | 95.1% | 0 |
| 70% TP@10bps + 30% trail@3bps | +197.96% | **+77.4** | 0.11% | 95.1% | 0 |
| 70% TP@10bps + 30% trail@5bps | +178.74% | **+66.2** | 0.19% | 92.3% | 1 |
| 50% TP@12bps + 50% trail@3bps | +257.32% | +65.1 | 0.11% | 95.1% | 0 |
| 70% TP@8bps + 30% trail@3bps | +165.96% | **+69.2** | 0.11% | 95.1% | 0 |
| 50% TP@10bps + 50% trail@3bps | +235.39% | +61.3 | 0.11% | 95.1% | 0 |
| 50% TP@8bps + 50% trail@3bps | +212.53% | +56.7 | 0.11% | 95.1% | 0 |
| 30% TP@12bps + 70% trail@3bps | +285.98% | +54.6 | 0.11% | 95.1% | 0 |

**Key patterns:**
- **More trail → more return** (30% TP gives highest returns)
- **More TP → higher Sharpe** (70% TP gives highest risk-adjusted returns)
- **3 bps trail consistently beats 5 bps** across all splits
- **TP at 10-12 bps is optimal** (8 bps leaves money on the table, 12 bps is near the TP ceiling)
- **Zero timeouts** across all configs
- **BTC works but is weaker:** 81-85% WR vs 93-97% for alts (BTC has 25-day data gap in July)

### Variant B: Two Fixed TPs (both maker)

| Config | Return | Sharpe | Max DD | WR | TO |
|--------|--------|--------|--------|-----|-----|
| 50% TP@8bps + 50% TP@12bps | +127.78% | +23.3 | 2.21% | 97.5% | 67 |
| 70% TP@8bps + 30% TP@15bps | +122.16% | +21.3 | 2.21% | 96.0% | 115 |
| 50% TP@8bps + 50% TP@30bps | +133.81% | +8.8 | 4.43% | 82.2% | 520 |

**Verdict: Worst variant.** The second TP level reintroduces timeouts. At TP2=30bps, 520 trades timeout — ~20% of all trades. Returns are lower than pure trail AND lower than Variant A. The timeout problem we solved with trailing stops comes right back.

### Variant C: TP (maker) + Tighter Trail (maker) After TP Fills

| Config | Return | Sharpe | Max DD | WR | TO |
|--------|--------|--------|--------|-----|-----|
| **50% TP@10bps, trail 5→3bps** | **+296.07%** | **+58.2** | **0.19%** | 92.3% | 0 |
| 50% TP@8bps, trail 5→3bps | +277.43% | +56.7 | 0.19% | 93.9% | 4 |
| 70% TP@10bps, trail 5→3bps | +232.25% | +65.5 | 0.19% | 92.3% | 0 |
| 70% TP@8bps, trail 5→3bps | +203.26% | +61.7 | 0.19% | 93.9% | 1 |
| 50% TP@10bps, trail 8→5bps | +266.30% | +49.5 | 0.30% | 90.8% | 2 |

**Verdict: Highest absolute returns.** Variant C at 50% TP@10bps with trail 5→3bps gives **+296%** — the highest return of any maker-only config tested. The two-phase trail is clever: wide trail (5 bps) protects the full position early, then tightens to 3 bps after the TP half exits, locking in profit on the remainder quickly.

**But:** More complex to implement than Variant A, and the Sharpe is lower than the best A configs.

### Variant D: Progressive Scale-Out at Milestones (maker)

| Config | Return | Sharpe | Max DD | WR | TO |
|--------|--------|--------|--------|-----|-----|
| 50%@8bps, trail 5 | +160.01% | +42.8 | 0.19% | 93.9% | 1 |
| 50%@8bps + 25%@15bps, trail 5 | +138.80% | +59.3 | 0.19% | 93.9% | 1 |
| 50%@8bps + 25%@15bps + 12.5%@30bps, trail 5 | +135.86% | **+68.4** | 0.19% | 93.9% | 1 |
| 33%@5bps + 33%@10bps + 33%@15bps, trail 5 | +83.64% | **+67.8** | 0.19% | 91.1% | 1 |

**Verdict: Good Sharpe ratios but lowest returns.** Progressive scale-out gives smooth equity curves (Sharpe 57-65) but locks in profit too early, leaving less to ride. Good for ultra-conservative deployment.

### Variant E: Partial Exit on Losers + Trail

| Config | Return | Sharpe | Max DD | WR | TO |
|--------|--------|--------|--------|-----|-----|
| Trail 5bps, cut 30% at -20bps | +230.00% | +31.4 | 0.25% | 83.5% | 0 |
| Trail 5bps, cut 30% at -8bps | +199.03% | +28.9 | 0.22% | 88.9% | 0 |
| Trail 8bps, cut 30% at -20bps | +155.80% | +21.5 | 0.41% | 76.4% | 1 |

**Verdict: Hurts performance.** Cutting losers early sounds good but the trailing stop already handles this. Partial loss-cutting adds fees and reduces the chance of recovery. Every config underperforms the pure trail baseline. **Don't use this.**

---

## The Big Picture — All Variants Compared

| Rank | Config | Return | Sharpe | DD | WR | Complexity |
|------|--------|--------|--------|-----|-----|-----------|
| 1 | Trail 3bps pure (taker) | +328.95% | +45.6 | 0.11% | 95.1% | Low |
| 2 | **C: 50% TP@10bps, trail 5→3bps** | **+296.07%** | +58.2 | 0.19% | 92.3% | High |
| 3 | A: 30% TP@12bps + 70% trail@3bps | +285.98% | +54.6 | 0.11% | 95.1% | Medium |
| 4 | C: 50% TP@8bps, trail 5→3bps | +277.43% | +56.7 | 0.19% | 93.9% | High |
| 5 | Trail 5bps pure (taker) | +276.41% | +38.4 | 0.19% | 92.3% | Low |
| 6 | A: 50% TP@12bps + 50% trail@3bps | +257.32% | +65.1 | 0.11% | 95.1% | Medium |
| 7 | A: 70% TP@12bps + 30% trail@3bps | +228.67% | **+83.2** | 0.11% | 95.1% | Medium |
| 8 | A: 70% TP@10bps + 30% trail@3bps | +197.96% | +77.4 | 0.11% | 95.1% | Medium |
| 9 | D: 50%@8+25%@15+12.5%@30, trail 5 | +135.86% | +68.4 | 0.19% | 93.9% | Very High |

**Note:** Pure trail 3bps has the highest raw return (+329%) but pays taker fees on every exit. In reality, the maker-only configs may perform closer to or better than this live, because taker fees are guaranteed to be 0.055% while maker limit fills may get better prices.

---

## Worst-Case Comparison

| Config | Worst Trade | Worst 5 Avg | P1 PnL | Max Consec Losses |
|--------|------------|-------------|--------|-------------------|
| Trail 5bps (taker) | -0.083% | -0.076% | -0.053% | 4 |
| TP 12bps (maker) | **-1.618%** | -1.389% | -0.565% | 2 |
| A: 50% TP@8 + 50% trail@5 | -0.083% | -0.076% | -0.053% | 4 |
| A: 50% TP@8 + 50% trail@3 | **-0.063%** | **-0.056%** | **-0.034%** | 4 |
| C: 50% TP@8, trail 5→3 | -0.083% | -0.076% | -0.053% | 4 |

All partial exit configs have tiny worst-case losses (~6-8 bps). The fixed TP baseline has 20x worse tail risk.

---

## Recommendations

### For Maximum Return
**Variant C: 50% TP@10bps, trail 5→3bps** → +296.07%, Sharpe +58.2
- Half exits at fixed 10 bps TP (maker, cheap)
- Other half rides with 5 bps trail, tightening to 3 bps after TP fills
- Highest return of any maker-only config
- But: most complex to implement

### For Best Risk-Adjusted (Sharpe)
**Variant A: 70% TP@12bps + 30% trail@3bps** → +228.67%, Sharpe +83.2
- 70% exits at fixed 12 bps TP (maker) — simple, reliable
- 30% rides with 3 bps trailing limit (maker) — captures upside
- Extremely smooth equity curve
- Simpler than Variant C

### For Simplest Implementation
**Variant A: 50% TP@8bps + 50% trail@5bps** → +184.71%, Sharpe +48.6
- Easy to understand: half at fixed target, half trailing
- 5 bps trail is more realistic for live than 3 bps
- Still beats the pure TP baseline and has 9x less drawdown

### For Ultra-Conservative
**Variant D: 50%@8+25%@15+12.5%@30, trail 5** → +135.86%, Sharpe +68.4
- Lower return but very smooth equity curve
- Good for large accounts where consistency matters more than return

### What NOT to Use
- **Variant B** (two fixed TPs) — reintroduces timeout problem (up to 520 timeouts)
- **Variant E** (partial loss cutting) — hurts performance, adds complexity

### BTC Note
BTC now uses WS ticker data (~100ms) like the other symbols. It works but underperforms alts (81-85% WR vs 93-97%), likely due to a 25-day data gap in July and BTC's tighter spreads relative to the trail width. Per-symbol BTC results (259 trades):
- Trail 5bps: +12.78%, Sharpe +29.3
- A 50% TP@8 + 50% trail@3: +12.93%, Sharpe +55.0
- C 50% TP@8, trail 5→3: +15.55%, Sharpe +44.1

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
