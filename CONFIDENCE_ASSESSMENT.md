# Confidence Assessment — Liquidation Cascade Strategy

**Date:** Feb 19, 2026 (Final revision)  
**Purpose:** Brutally honest evaluation of strategy readiness for live trading  
**Evidence base:**
- 10-test stress test across 4 original symbols
- Cross-symbol validation on 7 new out-of-sample symbols (11 total)
- Millisecond-precision latency analysis using WebSocket data (740K events, 25M+ ticks)

---

## Executive Summary

The strategy exploits a **real structural phenomenon**: forced liquidations create temporary price dislocations that mean-revert. This has been confirmed across 11 symbols, 3,323 trades, and ~81 days of actual data.

However, the edge is **thinner than headline numbers suggest**. Fees consume 51% of gross edge, the win rate masks an asymmetric risk profile (wins are tiny, losses are large), and we have zero bear-market data.

**Confidence level: 7/10 — Real edge, thin margins, proceed with caution.**

---

## The Numbers At A Glance

| Metric | Value |
|--------|-------|
| Symbols tested | 11 (4 original + 7 new OOS) |
| Total trades | 3,323 |
| Combined return | +145.79% |
| Win rate | 96.2% |
| Avg net PnL per trade | +0.040% |
| Avg gross PnL per trade | ~0.081% |
| Avg fee per trade | ~0.042% |
| **Fees as % of gross edge** | **51%** |
| Actual days with data | ~81 (NOT 282) |
| Calendar range | May 2025 – Feb 2026 |
| Data gap | 183 days (Aug 10, 2025 – Feb 9, 2026) |

---

## Self-Audit: What's Really Going On

### The Edge Decomposition

The strategy's return comes from two sources:

| Source | Contribution | Evidence |
|--------|-------------|----------|
| **Mean-reversion timing** (being there when dislocation happens) | **~72%** of returns | Random direction still earns +72.5% |
| **Direction signal** (fading the liquidation side) | **~28%** of returns | Z-score = 3.32 vs random |

**What this means:** The primary edge is **placing a limit order at the right moment** — when a forced liquidation has pushed price away from fair value. The direction signal helps but isn't the main driver. Even trading the wrong direction is profitable on all 4 original symbols.

**Why this is actually good news:** The edge is rooted in market microstructure (forced selling creates dislocations), not in a fragile directional prediction. This is more robust.

### The Win Rate Illusion

The 96% win rate sounds incredible but masks the real risk profile:

| Outcome | Frequency | Avg PnL | Contribution |
|---------|-----------|---------|-------------|
| Take-profit (win) | ~95.5% | +0.080% | +0.076% |
| Timeout (loss) | ~4.5% | -0.860% | -0.039% |
| **Net EV per trade** | | | **+0.040%** |

- **Risk/reward ratio: 1:11** — each loss wipes out ~11 wins
- The strategy survives because losses are rare (4.5%) and capped by the 60-min timeout
- **But**: a cluster of 3-4 consecutive timeouts would feel devastating psychologically
- Max observed consecutive losses: only 2 (across 2,538 trades on original symbols)

### The Fee Problem

This is a **thin-margin strategy**:

| Fee Tier | Maker/Taker | Combined Return | Viable? |
|----------|-------------|-----------------|---------|
| Zero fees | 0/0 bps | +205.7% | Theoretical max |
| VIP | 1/3 bps | +152.7% | ✅ Excellent |
| **Current** | **2/5.5 bps** | **+100.3%** | **✅ Good** |
| Pessimistic | 3/6 bps | +50.1% | ⚠️ Marginal |
| **Breakeven** | **~3.5/6.5 bps** | **~0%** | **❌ Dead** |

Fees eat 51% of gross edge. Any degradation in fill quality, increase in fees, or additional slippage pushes this toward breakeven fast. **VIP tier fees would nearly double net returns.**

---

## Cross-Symbol Validation (11 Symbols)

**Result: 11/11 symbols profitable.** This is the strongest evidence that the edge is structural.

| Symbol | Group | Trades | WR | Total | Sharpe | DD |
|--------|-------|--------|-----|-------|--------|-----|
| DOGEUSDT | ORIGINAL | 539 | 97.0% | +21.76% | +5.6 | 5.44% |
| SOLUSDT | ORIGINAL | 670 | 95.1% | +27.44% | +8.8 | 5.75% |
| ETHUSDT | ORIGINAL | 852 | 95.4% | +29.20% | +6.1 | 3.56% |
| XRPUSDT | ORIGINAL | 477 | 95.6% | +21.91% | +10.8 | 2.53% |
| **ADAUSDT** | **NEW OOS** | 243 | 97.9% | +16.30% | +30.8 | 0.96% |
| **BCHUSDT** | **NEW OOS** | 55 | 96.4% | +2.18% | +8.5 | 1.14% |
| **LTCUSDT** | **NEW OOS** | 107 | 99.1% | +7.59% | +34.0 | 0.89% |
| **NEARUSDT** | **NEW OOS** | 122 | 97.5% | +7.59% | +23.9 | 0.93% |
| **POLUSDT** | **NEW OOS** | 58 | 98.3% | +3.65% | +21.9 | 0.91% |
| **TONUSDT** | **NEW OOS** | 146 | 98.6% | +4.72% | +2.6 | 6.62% |
| **XLMUSDT** | **NEW OOS** | 54 | 96.3% | +3.46% | +35.4 | 0.35% |

**Key observations:**
- New OOS symbols have **higher per-trade quality** (+0.058% avg vs +0.040%)
- New symbols have **lower drawdowns** (most <1%) — less competition on mid-tier coins
- **But**: new symbols have **much lower signal frequency** (1.4 trades/day/symbol vs 7.8 for originals)
- 52/55 symbol-months positive

---

## Latency Analysis (Millisecond Precision)

The stress test showed 1-bar (60s) delay cuts returns 67%. The ms-precision analysis **resolves this concern**:

### Why 60s Delay ≠ Real-World Latency

| Delay | Median Price Move | Our 15 bps Offset Buffer | Expected Impact |
|-------|-------------------|--------------------------|-----------------|
| 50ms | 1.1 bps | 13.9 bps remaining | Negligible |
| 500ms | 1.7 bps | 13.3 bps remaining | Negligible |
| 1 second | 2.3 bps | 12.7 bps remaining | ~2-5% return drop |
| 5 seconds | 5.0 bps | 10.0 bps remaining | ~10-15% return drop |
| 10 seconds | 6.6 bps | 8.4 bps remaining | ~15-25% return drop |
| **60 seconds** | **14.4 bps** | **0.6 bps remaining** | **~67% return drop** |

**The 1-bar delay test was a worst case, not a realistic scenario.** At real-world latency (1-5 seconds from any cloud server), the strategy retains 85-95% of its edge.

### Full Latency Pipeline

```
T=0:     P95 liquidation event occurs
+252ms:  Bybit WS server pushes event (P50)
+304ms:  Your bot receives it (P50)
+305ms:  Bot computes signal (<1ms)
+305ms+N: Order sent to Bybit (N = network latency)
+305ms+2N: Order acknowledged and live on book
```

### Fill Window

- Median time for price to reach our 15 bps limit level: **12.5 seconds**
- Even at P25 (fast moves): 3.7 seconds
- **You have seconds, not milliseconds, to act**

### Latency Verdict

| Location | Total Pipeline | Impact |
|----------|---------------|--------|
| Singapore (co-located) | ~306ms | ✅ Captures ~100% of edge |
| Tokyo/HK | ~365ms | ✅ Captures ~100% |
| US West/East | ~600-700ms | ✅ Captures ~98% |
| Europe | ~800ms | ✅ Captures ~95% |
| Manual trading (2-5s) | ~5,300ms | ⚠️ Captures ~85% |

**Execution sensitivity score upgraded from 4/10 → 7/10.** Any cloud server captures nearly all the edge. No HFT infrastructure needed.

---

## Stress Test Results Summary

| Test | Result | Concern Level |
|------|--------|---------------|
| **1. Look-ahead bias** | ✅ Rolling P95 works equally well (+108.8% vs +100.3%) | None |
| **2. Fill pessimism (1-bar delay)** | ⚠️ +33.1% (−67%) — but now explained by latency analysis | Low (at real latency) |
| **3. Walk-forward OOS** | ⚪ 1/1 windows positive, but 183-day data gap | Medium |
| **4. Fee sensitivity** | ✅ Breaks even at ~3.5/6.5 bps; current fees leave margin | Low-Medium |
| **5. Random direction** | ⚠️ 72% of return is mean-reversion, not direction | Low (structural) |
| **6. Opposite direction** | ⚠️ Wrong direction still profitable on all symbols | Low (confirms MR) |
| **7. Timeout slippage** | ✅ Survives 50 bps slippage on timeouts | None |
| **8. Worst-case analysis** | ✅ Max 2 consecutive losses, worst trade -5.4% | Low |
| **9. Monthly regime** | ✅ 20/20 symbol-months positive | Low |
| **10. Data coverage** | ⚠️ Only ~81 actual days, 183-day gap | **High** |

---

## Honest Confidence Scorecard

| Dimension | Score | Notes |
|-----------|-------|-------|
| **Alpha is real** | 9/10 | 11/11 symbols profitable, structural mean-reversion |
| **Cross-symbol OOS** | 9/10 | 7/7 new symbols profitable; -1 for lower frequency on new symbols |
| **No look-ahead bias** | 9/10 | Rolling P95 works equally well |
| **Fee robustness** | 6/10 | Fees eat 51% of gross; thin margin; breakeven at ~3.5 bps maker |
| **Execution/latency** | 7/10 | Ms-analysis shows 1-5s latency loses <15%; any cloud server works |
| **Data coverage** | 4/10 | Only ~81 days across 5 calendar months; 183-day gap; no bear market |
| **OOS validation** | 7/10 | 7 new symbols = true OOS; but same time period, not new time periods |
| **Tail risk** | 8/10 | Max 2 consecutive losses, worst trade -5.4%, manageable |
| **Regime robustness** | 5/10 | All months positive but only 5 calendar months, all bull/sideways |
| **Slippage tolerance** | 9/10 | Survives 50 bps slippage on timeouts |

**Overall: 7.3/10 — Real structural edge, thin margins, limited time coverage.**

---

## What We KNOW (High Confidence)

1. **Forced liquidations create mean-reverting dislocations** — this is physics, not statistics. Forced sellers push price below fair value, it bounces back.
2. **The edge works across 11 different coins** — 7 never used in development, all profitable. The phenomenon is universal.
3. **The limit order offset (15 bps) is the primary edge** — it captures the bounce regardless of direction. Direction adds ~28% incremental value.
4. **Latency requirements are modest** — any cloud server (1-5s total pipeline) captures 85-95% of the edge. No co-location needed.
5. **No look-ahead bias** — rolling P95 threshold works equally well or better.
6. **Tail risk is bounded** — max 2 consecutive losses observed, worst single trade -5.4%, 60-min timeout caps exposure.

## What We DON'T KNOW (Critical Unknowns)

1. **Live fill rates** — backtest assumes 100% fill when price touches limit. In reality, your order competes with others in the book. If fill rate is 50%, returns halve. **This is the #1 unknown.**
2. **Bear market / low-vol performance** — we have zero data from Sep 2025 – Jan 2026. Liquidation patterns may differ in bear markets (fewer cascades? different dynamics?).
3. **Competition / crowding** — if others exploit the same signal, fill rates drop and the edge erodes. We don't know how many participants already do this.
4. **Fee stability** — Bybit could change fee structure. At 3.5 bps maker the strategy dies.
5. **Liquidation feed reliability** — Bybit WS uptime, data quality, and whether the `allLiquidation` feed is complete or sampled.
6. **Signal frequency in different regimes** — we see 7.8 trades/day/symbol in our data, but this could vary dramatically across market conditions.

## What We're Probably WRONG About

1. **The +100% headline number** — this assumes perfect fills, zero latency, and current fee tier. Realistic expectation with live execution friction: **+40% to +70% annualized** on original 4 symbols.
2. **The 96% win rate** — live WR will likely be 85-92% due to partial fills, wider spreads during volatility, and order queue position.
3. **"No latency concern"** — while ms-analysis shows the strategy is tolerant, we haven't tested with real order book dynamics. The limit order may sit in queue behind others.

---

## Recommended Path Forward

### Phase 1: Paper Trading (2-4 weeks)
1. Run the bot on Bybit testnet or with paper orders
2. **Measure these metrics:**
   - Actual fill rate (% of limit orders that execute)
   - Actual latency (event → order acknowledged)
   - Signal frequency (trades per day)
   - Win rate and avg PnL per trade
3. **Kill criteria:** If fill rate < 50% or WR < 80%, the strategy is not viable live

### Phase 2: Minimum Live (2-4 weeks)
4. Start with **1 symbol** (XRP — most robust to delay, best Sharpe)
5. **Minimum position size** ($100-500 per trade)
6. Compare live metrics to backtest — expect 50-70% of backtest returns
7. **Scale up only if:** fill rate > 60%, WR > 85%, avg PnL > +0.02%

### Phase 3: Scale
8. Add symbols one at a time
9. Prioritize mid-tier coins (ADA, LTC, NEAR) — higher per-trade quality, less competition
10. Target VIP fee tier to improve margins

### The Single Most Important Live Metric

**What % of your limit orders fill?**

| Fill Rate | Expected Return (% of backtest) | Verdict |
|-----------|--------------------------------|---------|
| 90%+ | ~90% of backtest | ✅ Excellent — scale up |
| 70-90% | ~70-90% of backtest | ✅ Good — continue |
| 50-70% | ~50-70% of backtest | ⚠️ Marginal — optimize |
| < 50% | < 50% of backtest | ❌ Not viable — rethink |

---

*Sources: `liq_stress_test.py`, `liq_cross_symbol_validation.py`, `liq_latency_analysis.py`*  
*Results: `results/liq_stress_test.txt`, `results/liq_cross_symbol_validation.txt`, `results/liq_latency_analysis.txt`*  
*Latency findings: `FINDINGS_latency_analysis.md`*
