# FINDINGS — Whipsaw Structure + Fade-First-Break on REG_OI_FUND

**Date:** 2026-03-03  
**Period:** 2026-01-01 → 2026-02-28  
**Coins:** 8 altcoins  
**Cooldown:** 60 min between signals

---

## 1) Phase 1: Whipsaw Structure — CONFIRMED

The oscillatory hypothesis is **strongly confirmed** by the data:

### Cross-symbol averages

| k | Break rate | touchP0_30m | Med retrace | Whipsaw rate |
|---|---:|---:|---:|---:|
| 0.3 | 85% | **82%** | **2.19 ATR** | **81%** |
| 0.5 | 95% | **80%** | **1.99 ATR** | **78%** |
| 0.7 | 99% | **79%** | **1.91 ATR** | **76%** |

**Key findings:**
- **~80% of first breaks retrace back to P0 within 30 minutes** — this is massive
- **Median retrace depth is ~2 ATR** — far more than the 0.4 ATR threshold
- **Median time to touch P0 is 1-6 minutes** — extremely fast reversal
- **MFE after break is 3-5 ATR** — the initial impulse IS real and large

### Structural criteria (§6) results

| Symbol | k=0.3 | k=0.5 | k=0.7 |
|--------|-------|-------|-------|
| 1000BONKUSDT | FAIL (tP0=60%) | FAIL (tP0=58%) | FAIL (tP0=64%) |
| 1000RATSUSDT | **PASS** | **PASS** | **PASS** |
| 1000TURBOUSDT | **PASS** | **PASS** | **PASS** |
| AIXBTUSDT | **PASS** | **PASS** | **PASS** |
| APTUSDT | **PASS** | **PASS** | **PASS** |
| ARBUSDT | **PASS** | **PASS** | **PASS** |
| ARCUSDT | **PASS** | **PASS** | **PASS** |
| ATOMUSDT | **PASS** | **PASS** | **PASS** |

**7 of 8 coins pass all structural criteria.** Only BONK narrowly fails on touchP0 rate.

### Per-coin detail (k=0.5)

| Symbol | N_breaks | Break% | tP0_5m | tP0_15m | tP0_30m | Med time | MFE (ATR) | Retrace (ATR) | WS% |
|--------|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000BONKUSDT | 12 | 86% | 42% | 58% | 58% | 1m | 4.98 | 0.47 | 58% |
| 1000RATSUSDT | 20 | 100% | 45% | 60% | 80% | 5m | 3.87 | 2.35 | 80% |
| 1000TURBOUSDT | 13 | 93% | 62% | 69% | 92% | 4m | 3.95 | 2.83 | 92% |
| AIXBTUSDT | 29 | 97% | 69% | 72% | 76% | 1m | 3.42 | 1.26 | 76% |
| APTUSDT | 53 | 98% | 64% | 77% | 87% | 2m | 3.63 | 2.16 | 83% |
| ARBUSDT | 50 | 98% | 56% | 70% | 84% | 2m | 3.63 | 1.91 | 78% |
| ARCUSDT | 58 | 95% | 55% | 71% | 81% | 3m | 3.17 | 2.09 | 78% |
| ATOMUSDT | 66 | 97% | 64% | 77% | 83% | 2m | 3.70 | 2.86 | 82% |

---

## 2) Phase 2: Fade-First-Break — FAILED

Despite the beautiful structural confirmation, **the fade strategy does not work.**

### Go/No-Go

| Symbol | Verdict | Mean net (bp) | PF | WR | N |
|--------|---------|---:|---:|---:|---:|
| 1000BONKUSDT | NO-GO | -17.0 | 0.03 | 10% | 10 |
| 1000RATSUSDT | NO-GO | -14.7 | 0.11 | 30% | 20 |
| 1000TURBOUSDT | NO-GO | -10.4 | 0.50 | 29% | 14 |
| AIXBTUSDT | NO-GO | -15.7 | 0.15 | 14% | 29 |
| APTUSDT | NO-GO | -18.6 | 0.08 | 13% | 53 |
| ARBUSDT | NO-GO | -17.1 | 0.05 | 5% | 41 |
| ARCUSDT | NO-GO | -12.1 | 0.40 | 52% | 52 |
| ATOMUSDT | NO-GO | -15.6 | 0.10 | 14% | 65 |

**ALL 8 coins are NO-GO.** Universal negative expectancy. Walk-forward confirms: every OOS test is negative.

---

## 3) Why the Paradox? (Structure ✓ but Trade ✗)

This is the critical insight:

### 3.1 The retrace IS real but the **path** kills the trade

- MFE after break: **3-5 ATR** in the break direction
- Retrace depth: **2 ATR** back toward P0

The problem: **price first extends 3-5 ATR in the break direction BEFORE retracing.** The fade entry (even with delay=3m) catches the initial continuation, not the reversal.

### 3.2 SL gets hit before TP

With fade SL at 0.7-1.3 ATR, and the initial impulse going 3-5 ATR, the fade gets stopped out long before the retrace begins:

- APTUSDT: 51% SL rate, 49% TP rate → but the SLs are bigger than the TPs
- ATOMUSDT: 40% SL rate with k_sl=0.7 → and those SLs are -35bp each
- Win rate 5-14% on most coins confirms the SL fires first

### 3.3 The whipsaw is real but the timing is wrong for a fixed-entry fade

The retrace happens on the **second phase** of the move. The sequence is:

```
Signal → Break → CONTINUATION (3-5 ATR) → PEAK → RETRACE to P0
```

A fade at the break catches the continuation phase and dies.
The retrace starts from the peak, which is 3-5 ATR away from entry.

---

## 4) Fork Decision: What Next?

The data points to a clear fork:

### ❌ Dead: Simple entry-at-break strategies (breakout or fade)

Both breakout (previous test) and fade (this test) fail because:
- Breakout: gets whipsawed after continuation
- Fade: gets stopped out during continuation

### ❓ Possible: Fade from the PEAK, not from the break

The data shows MFE of 3-5 ATR after break. If we could detect the peak (end of continuation) and fade from there, the 2 ATR retrace would be our profit.

**Problem:** Detecting the peak requires real-time signal (reversal candle, volume exhaustion, etc.) — not just a fixed delay or level.

### ❓ Possible: Wider entry threshold + patience

Instead of fading at k=0.3-0.7 ATR from P0, wait for price to extend to 2-3 ATR, THEN fade. The retrace from 3-5 ATR back to 2-3 ATR gives the profit.

**This is essentially a mean-reversion from an extreme within the regime window.**

### ✅ Most promising: Volatility sizing / risk management

REG_OI_FUND reliably predicts:
- Range expansion 1.5-3.9x
- Whipsaw with 80% probability
- Fast reversion (median 1-5 minutes)

**Use case:** When REG_OI_FUND fires, DO NOT enter new positions. Widen stops on existing positions by 2x. Or: reduce position size preemptively. This avoids the stop-hunt that kills most retail in these regimes.

---

## 5) Files

| File | Description |
|------|-------------|
| `flow_research/whipsaw_analysis.py` | Combined structure + fade analysis |
| `flow_research/output/regime/whipsaw_report.csv` | Whipsaw structure metrics |
| `flow_research/output/regime/fade_report.csv` | Fade trade results |
