# Partial Exits Research — Findings

**Date:** Feb 19, 2026  
**Source:** `liq_partial_exits_research.py`  
**Data:** 5 symbols (BTC, DOGE, SOL, ETH, XRP), 2,685 trades, 87 days (May–Aug 2025)  
**Ticker:** WS ~100ms for DOGE/SOL/ETH/XRP; REST ~5s for BTC  
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
| **Trail 3bps (taker exit)** | **+328.40%** | +44.5 | 0.18% | 94.0% | 0 | taker |
| Trail 5bps (taker exit) | +275.26% | +37.3 | 0.27% | 91.0% | 0 | taker |
| TP 12bps (maker exit) | +169.36% | +23.7 | 2.11% | 97.3% | 74 | maker |
| TP 8bps (maker exit) | +87.28% | +17.8 | 1.71% | 98.7% | 35 | maker |
| TP 5bps (maker exit) | +21.32% | +9.0 | 1.12% | 99.6% | 10 | maker |

**The dilemma:** Pure trailing stop gives the best returns but pays taker fees on every exit. Fixed TP pays maker fees but has timeout losses. Can we get the best of both?

---

## Variant Results Summary

### Variant A: Partial TP (maker) + Trail (maker) on Remainder

**The clear winner.** Split the position: one part exits at a fixed TP (maker), the other runs a trailing limit (maker).

**Top configs from Variant A:**

| Config | Return | Sharpe | Max DD | WR | TO |
|--------|--------|--------|--------|-----|-----|
| 70% TP@12bps + 30% trail@3bps | +228.22% | **+78.4** | 0.18% | 94.0% | 0 |
| 70% TP@10bps + 30% trail@3bps | +197.48% | **+73.1** | 0.18% | 94.0% | 0 |
| 70% TP@10bps + 30% trail@5bps | +177.77% | **+62.4** | 0.24% | 91.0% | 0 |
| 50% TP@12bps + 50% trail@3bps | +256.85% | +62.5 | 0.18% | 94.0% | 0 |
| 70% TP@8bps + 30% trail@3bps | +165.48% | **+65.7** | 0.18% | 94.0% | 0 |
| 50% TP@10bps + 50% trail@3bps | +234.89% | +58.9 | 0.18% | 94.0% | 0 |
| 50% TP@8bps + 50% trail@3bps | +212.03% | +54.7 | 0.18% | 94.0% | 0 |
| 30% TP@12bps + 70% trail@3bps | +285.47% | +52.9 | 0.18% | 94.0% | 0 |

**Key patterns:**
- **More trail → more return** (30% TP gives highest returns)
- **More TP → higher Sharpe** (70% TP gives highest risk-adjusted returns)
- **3 bps trail consistently beats 5 bps** across all splits
- **TP at 10-12 bps is optimal** (8 bps leaves money on the table, 12 bps is near the TP ceiling)
- **Zero timeouts** across all configs
- **BTC works but is weaker:** 73-77% WR vs 93-97% for alts (likely due to lower ticker resolution — REST 5s vs WS 100ms)

### Variant B: Two Fixed TPs (both maker)

| Config | Return | Sharpe | Max DD | WR | TO |
|--------|--------|--------|--------|-----|-----|
| 50% TP@8bps + 50% TP@12bps | +128.32% | +22.9 | 1.62% | 97.3% | 74 |
| 70% TP@8bps + 30% TP@15bps | +121.98% | +20.9 | 1.62% | 95.6% | 125 |
| 50% TP@8bps + 50% TP@30bps | +133.00% | +8.6 | 3.91% | 81.8% | 558 |

**Verdict: Worst variant.** The second TP level reintroduces timeouts. At TP2=30bps, 558 trades timeout — over 20% of all trades. Returns are lower than pure trail AND lower than Variant A. The timeout problem we solved with trailing stops comes right back.

### Variant C: TP (maker) + Tighter Trail (maker) After TP Fills

| Config | Return | Sharpe | Max DD | WR | TO |
|--------|--------|--------|--------|-----|-----|
| **50% TP@10bps, trail 5→3bps** | **+296.43%** | **+56.6** | **0.24%** | 91.0% | 0 |
| 50% TP@8bps, trail 5→3bps | +277.22% | +54.9 | 0.24% | 92.7% | 1 |
| 70% TP@10bps, trail 5→3bps | +232.25% | +65.5 | 0.24% | 91.0% | 0 |
| 70% TP@8bps, trail 5→3bps | +203.26% | +61.7 | 0.24% | 92.8% | 1 |
| 50% TP@10bps, trail 8→5bps | +266.30% | +49.5 | 0.38% | 89.5% | 2 |

**Verdict: Highest absolute returns.** Variant C at 50% TP@10bps with trail 5→3bps gives **+296.43%** — the highest return of any maker-only config tested. The two-phase trail is clever: wide trail (5 bps) protects the full position early, then tightens to 3 bps after the TP half exits, locking in profit on the remainder quickly.

**But:** More complex to implement than Variant A, and the Sharpe is lower than the best A configs.

### Variant D: Progressive Scale-Out at Milestones (maker)

| Config | Return | Sharpe | Max DD | WR | TO |
|--------|--------|--------|--------|-----|-----|
| 50%@8bps, trail 5 | +158.91% | +41.2 | 0.26% | 92.8% | 0 |
| 50%@8bps + 25%@15bps, trail 5 | +137.80% | +57.0 | 0.26% | 92.8% | 0 |
| 50%@8bps + 25%@15bps + 12.5%@30bps, trail 5 | +134.93% | **+65.4** | 0.26% | 92.8% | 0 |
| 33%@5bps + 33%@10bps + 33%@15bps, trail 5 | +82.91% | **+64.5** | 0.27% | 89.9% | 1 |

**Verdict: Good Sharpe ratios but lowest returns.** Progressive scale-out gives smooth equity curves (Sharpe 57-65) but locks in profit too early, leaving less to ride. Good for ultra-conservative deployment.

### Variant E: Partial Exit on Losers + Trail

| Config | Return | Sharpe | Max DD | WR | TO |
|--------|--------|--------|--------|-----|-----|
| Trail 5bps, cut 30% at -20bps | +229.32% | +30.7 | 0.32% | 82.1% | 0 |
| Trail 5bps, cut 30% at -8bps | +198.35% | +28.3 | 0.29% | 87.8% | 0 |
| Trail 8bps, cut 30% at -20bps | +155.19% | +21.0 | 0.52% | 75.0% | 1 |

**Verdict: Hurts performance.** Cutting losers early sounds good but the trailing stop already handles this. Partial loss-cutting adds fees and reduces the chance of recovery. Every config underperforms the pure trail baseline. **Don't use this.**

---

## The Big Picture — All Variants Compared

| Rank | Config | Return | Sharpe | DD | WR | Complexity |
|------|--------|--------|--------|-----|-----|-----------|
| 1 | Trail 3bps pure (taker) | +328.40% | +44.5 | 0.18% | 94.0% | Low |
| 2 | **C: 50% TP@10bps, trail 5→3bps** | **+296.43%** | +56.6 | 0.24% | 91.0% | High |
| 3 | A: 30% TP@12bps + 70% trail@3bps | +285.47% | +52.9 | 0.18% | 94.0% | Medium |
| 4 | C: 50% TP@8bps, trail 5→3bps | +277.22% | +54.9 | 0.24% | 92.7% | High |
| 5 | Trail 5bps pure (taker) | +275.26% | +37.3 | 0.27% | 91.0% | Low |
| 6 | A: 50% TP@12bps + 50% trail@3bps | +256.85% | +62.5 | 0.18% | 94.0% | Medium |
| 7 | A: 70% TP@12bps + 30% trail@3bps | +228.22% | **+78.4** | 0.18% | 94.0% | Medium |
| 8 | A: 70% TP@10bps + 30% trail@3bps | +197.48% | +73.1 | 0.18% | 94.0% | Medium |
| 9 | D: 50%@8+25%@15+12.5%@30, trail 5 | +134.93% | +65.4 | 0.26% | 92.8% | Very High |

**Note:** Pure trail 3bps has the highest raw return (+328%) but pays taker fees on every exit. In reality, the maker-only configs may perform closer to or better than this live, because taker fees are guaranteed to be 0.055% while maker limit fills may get better prices.

---

## Worst-Case Comparison

| Config | Worst Trade | Worst 5 Avg | P1 PnL | Max Consec Losses |
|--------|------------|-------------|--------|-------------------|
| Trail 5bps (taker) | -0.090% | -0.087% | -0.058% | 5 |
| TP 12bps (maker) | **-1.618%** | -1.378% | -0.568% | 2 |
| A: 50% TP@8 + 50% trail@5 | -0.090% | -0.087% | -0.058% | 5 |
| A: 50% TP@8 + 50% trail@3 | **-0.070%** | **-0.068%** | **-0.042%** | 3 |
| C: 50% TP@8, trail 5→3 | -0.090% | -0.087% | -0.058% | 5 |

All partial exit configs have tiny worst-case losses (~7-9 bps). The fixed TP baseline has 18x worse tail risk. BTC adds slightly more worst-case exposure due to lower ticker resolution.

---

## Recommendations

### For Maximum Return
**Variant C: 50% TP@10bps, trail 5→3bps** → +296.43%, Sharpe +56.6
- Half exits at fixed 10 bps TP (maker, cheap)
- Other half rides with 5 bps trail, tightening to 3 bps after TP fills
- Highest return of any maker-only config
- But: most complex to implement

### For Best Risk-Adjusted (Sharpe)
**Variant A: 70% TP@12bps + 30% trail@3bps** → +228.22%, Sharpe +78.4
- 70% exits at fixed 12 bps TP (maker) — simple, reliable
- 30% rides with 3 bps trailing limit (maker) — captures upside
- Extremely smooth equity curve
- Simpler than Variant C

### For Simplest Implementation
**Variant A: 50% TP@8bps + 50% trail@5bps** → +183.80%, Sharpe +46.8
- Easy to understand: half at fixed target, half trailing
- 5 bps trail is more realistic for live than 3 bps
- Still beats the pure TP baseline and has 9x less drawdown

### For Ultra-Conservative
**Variant D: 50%@8+25%@15+12.5%@30, trail 5** → +134.93%, Sharpe +65.4
- Lower return but very smooth equity curve
- Good for large accounts where consistency matters more than return

### What NOT to Use
- **Variant B** (two fixed TPs) — reintroduces timeout problem (up to 558 timeouts)
- **Variant E** (partial loss cutting) — hurts performance, adds complexity

### BTC Note
BTC works but underperforms alts (73-77% WR vs 93-97%). This is likely due to REST ticker data (5s resolution) vs WS (100ms) for alts. Once WS ticker data is collected for BTC, expect BTC performance to improve significantly. Per-symbol BTC results:
- Trail 5bps: +12.08%, Sharpe +26.1
- A 50% TP@8 + 50% trail@3: +12.42%, Sharpe +38.3
- C 50% TP@8, trail 5→3: +15.34%, Sharpe +33.4

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
