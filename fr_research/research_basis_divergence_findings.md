# Futures Premium (Basis) Divergence: Binance vs Bybit

**Date:** 2026-02-24  
**Script:** `fr_research/research_basis_divergence.py`  
**Data:** Historical FR (200 days, Aug 2025 – Feb 2026) + real-time ticker (2 days)

---

## The Trade

When funding rate is **positive on exchange A** and **negative on exchange B**:

- **Short futures** on exchange A (collect positive FR)
- **Long futures** on exchange B (collect negative FR — shorts pay you)
- **Delta-neutral** — no spot leg, no directional exposure
- **Profit from:** (1) FR collection on both legs + (2) basis convergence

### Fee Structure (Futures-Only)

| Tier | Per-side | Round-trip (4 legs) |
|---|---|---|
| VIP-0 taker | 5.0 bps | **20 bps** |
| VIP-0 maker | 2.0 bps | **8 bps** |
| VIP-1 taker | 4.0 bps | 16 bps |
| VIP-1 maker | 1.6 bps | 6.4 bps |

---

## Key Findings

### 1. FR Sign Disagreements Are Common

| Metric | Value |
|---|---|
| Matched settlements (BN × BB) | 465,843 |
| Common symbols | 476 |
| Sign disagreements | 87,630 (**18.8%**) |
| BN+ & BB− | 17,072 (19%) |
| BN− & BB+ | 70,558 (81%) |

**Asymmetry:** BN−/BB+ is 4× more common — Binance FR skews more negative than Bybit.

### 2. Convergence Is Real — 94% Gross Win Rate

| Metric | Value |
|---|---|
| Total trades | 87,486 |
| Avg hold | 3.2 periods |
| Gross P&L | **+4.63 bps** |
| Gross Win Rate | **94.2%** |

53% of disagreements resolve within 1 settlement period. By t+6, 63% have re-aligned.

### 3. Gross P&L by Spread Bucket

| Bucket | Trades | Gross | WR | Net @20bps | Net @8bps | $/day @20 | $/day @8 |
|---|---|---|---|---|---|---|---|
| 0–5 bps | 82,870 | +3.7 | 94% | −16.3 | −4.3 | −$6,761 | −$1,787 |
| 5–10 bps | 3,505 | +15.3 | 98% | −4.7 | +7.3 | −$82 | +$129 |
| **10–20 bps** | **782** | **+30.7** | **98%** | **+10.7** | **+22.7** | **+$42** | **+$89** |
| **20–30 bps** | **176** | **+50.2** | **100%** | **+30.2** | **+42.2** | **+$27** | **+$37** |
| **30–50 bps** | **94** | **+70.6** | **99%** | **+50.6** | **+62.6** | **+$24** | **+$29** |
| **50–100 bps** | **45** | **+86.0** | **100%** | **+66.0** | **+78.0** | **+$15** | **+$18** |
| **100–500 bps** | **14** | **+178.3** | **100%** | **+158.3** | **+170.3** | **+$11** | **+$12** |

**Key insight:** Everything above 5 bps is profitable with maker fees. Everything above 10 bps is profitable even with taker fees.

### 4. Filtered Strategy Performance ($10K/leg)

| Filter | Trades | Per day | Gross | Net (taker) | Net (maker) | $/day (taker) | $/day (maker) |
|---|---|---|---|---|---|---|---|
| ≥10 bps | 1,111 | 5.6 | 41.3 | +21.3 | +33.3 | **$118** | **$185** |
| ≥15 bps | 522 | 2.6 | 57.9 | +37.9 | +49.9 | **$99** | **$130** |
| ≥20 bps | 329 | 1.6 | 66.3 | +46.3 | +58.3 | **$76** | **$96** |
| ≥30 bps | 153 | 0.8 | 85.0 | +65.0 | +77.0 | **$50** | **$59** |

**Best risk/reward: ≥10 bps filter with maker fees → $185/day on $10K/leg.**

### 5. Scaling Analysis (Maker Fees, 8 bps RT)

| Notional/leg | ≥10 bps | ≥20 bps |
|---|---|---|
| $10K | $185/day ($67K/yr) | $96/day ($35K/yr) |
| $25K | $462/day ($169K/yr) | $240/day ($88K/yr) |
| $50K | $924/day ($337K/yr) | $480/day ($175K/yr) |
| $100K | $1,848/day ($674K/yr) | $960/day ($350K/yr) |

### 6. Extreme Events (|spread| ≥ 30 bps)

153 events total over 200 days. Top examples show **hard snap-back within 1–2 periods**:

| Symbol | Date | Spread t0 | t+1 | t+2 |
|---|---|---|---|---|
| TUTUSDT | 2025-10-11 | −254 bps | 0 | 0 |
| LYNUSDT | 2025-10-06 | −201 bps | −56 | −47 |
| COTIUSDT | 2025-10-11 | +139 bps | +3 | +4 |
| FLOWUSDT | 2026-02-06 | +118 bps | −8 | −11 |

FLOWUSDT is a **serial diverger** (8 times in top-40). The 2025-10-11 cluster suggests a Binance-wide anomaly.

### 7. Real-Time Basis Analysis (1-min resolution, 2 days)

| Metric | Value |
|---|---|
| Basis spread mean (BN−BB) | −3.07 bps |
| Std | 20.00 bps |
| P1/P99 | −53 / +42 bps |

**Basis spread autocorrelation:**

| Lag | AC |
|---|---|
| 1 min | 0.774 |
| 5 min | 0.479 |
| 30 min | 0.274 |
| 60 min | 0.207 |

→ **Strong mean-reversion within 30–60 min.** Basis convergence is not just a FR-settlement phenomenon — it happens continuously.

---

## Verdict

**YES — this is a viable strategy**, particularly with maker fees.

| Aspect | Assessment |
|---|---|
| Edge exists? | ✅ 94% gross WR, convergence confirmed |
| Profitable after fees? | ✅ With maker fees (≥5 bps spread) or taker fees (≥10 bps) |
| Frequency? | ✅ 5.6 events/day at ≥10 bps filter |
| Scalable? | ✅ $185–$1,848/day depending on notional |
| Key risk | ⚠️ Basis risk — spread may widen before converging |

---

## Next Steps

- [ ] Build real-time monitor for FR sign divergence across all symbols
- [ ] Backtest with proper slippage model (are maker fills realistic on both exchanges simultaneously?)
- [ ] Analyze max adverse excursion — how much does basis widen before converging?
- [ ] Test combining with the single-exchange HOLD FR arb
- [ ] Deep-dive FLOWUSDT as a dedicated pair trade
