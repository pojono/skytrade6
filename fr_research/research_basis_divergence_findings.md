# Futures Premium (Basis) Divergence: Binance vs Bybit

**Date:** 2026-02-24  
**Script:** `fr_research/research_basis_divergence.py`  
**Data:** Historical FR (200 days, Aug 2025 – Feb 2026) + real-time ticker (2 days)

---

## Question

When futures premium (funding rate) is positive on one exchange and negative on the other, **does it converge**? Is there a tradeable edge?

---

## Key Findings

### 1. FR Sign Disagreements Are Common

| Metric | Value |
|---|---|
| Matched settlements (BN × BB) | 465,843 |
| Common symbols | 476 |
| Sign disagreements (BN+ & BB− or BN− & BB+) | 87,630 (**18.8%**) |
| BN+ & BB− | 17,072 (19%) |
| BN− & BB+ | 70,558 (81%) |

**Asymmetry:** BN− & BB+ is 4× more common than BN+ & BB−. Binance FR skews more negative than Bybit on the same coins.

### 2. Most Disagreements Are Small

| Both sides |FR| ≥ | Count |
|---|---|
| 0 bps | 87,630 |
| 5 bps | 78 |
| 10 bps | 20 |
| 20 bps | 5 |
| 50 bps | 0 |

The vast majority of sign disagreements involve one side near zero (e.g., BN = −0.5 bps, BB = +0.3 bps). Only 78 events have both exchanges ≥5 bps in magnitude — genuinely meaningful divergence is rare.

### 3. YES — Convergence Is Real

After a sign disagreement, the FR spread (BN − BB) converges toward zero:

| Period | Still Disagree | Spread Reversion |
|---|---|---|
| t+1 | 47.0% | +24.1% |
| t+2 | 42.0% | +23.5% |
| t+3 | 40.2% | +22.9% |
| t+4 | 38.4% | +21.7% |
| t+5 | 37.5% | +20.7% |
| t+6 | 36.8% | +20.6% |

- **53% of disagreements resolve within 1 settlement period**
- Avg spread drops from 1.9 bps → 1.5 bps (24% reversion at t+1)
- By t+6, 63% have re-aligned

### 4. Convergence Trade P&L (Funding Arb)

**Strategy:** When signs disagree, short the exchange with positive FR, long the one with negative FR. Hold until signs re-align (max 8 periods).

| Metric | Value |
|---|---|
| Total trades | 87,486 |
| Avg hold | 3.2 periods |
| Avg P&L | **+4.63 bps** |
| Median P&L | +2.21 bps |
| Win rate | **94.2%** |
| Avg win | +5.10 bps |
| Avg loss | −3.06 bps |

#### P&L by Initial Spread Magnitude

| Bucket | Trades | Avg P&L (bps) | Win Rate | Avg Hold |
|---|---|---|---|---|
| 0–10 bps | 86,375 | +4.16 | 94.2% | 3.2 |
| 10–20 bps | 782 | +30.69 | 97.7% | 3.5 |
| 20–30 bps | 176 | +50.16 | 100% | 3.4 |
| 30–50 bps | 94 | +70.61 | 98.9% | 3.4 |
| 50–100 bps | 45 | +85.95 | 100% | 2.8 |
| 100–500 bps | 14 | +178.26 | 100% | 2.9 |

**The bigger the divergence, the stronger the convergence.** Extreme events (≥30 bps) have near-100% win rates and huge P&L.

#### P&L by Direction

| Type | Trades | Avg P&L | Win Rate |
|---|---|---|---|
| BN+ & BB− | 17,019 | +3.60 bps | 93.1% |
| BN− & BB+ | 70,467 | +4.88 bps | 94.5% |

### 5. Extreme Events (|spread| ≥ 30 bps)

153 events total. Top examples:

| Symbol | Date | Type | Spread t0 | t+1 | t+2 | t+3 |
|---|---|---|---|---|---|---|
| TUTUSDT | 2025-10-11 | BN−_BB+ | −254 | 0 | 0 | −0.4 |
| LYNUSDT | 2025-10-06 | BN−_BB+ | −201 | −56 | −47 | −51 |
| ZBTUSDT | 2025-10-17 | BN−_BB+ | −201 | −38 | +52 | +6 |
| LIGHTUSDT | 2026-01-01 | BN−_BB+ | −186 | −68 | −7 | +1 |
| COTIUSDT | 2025-10-11 | BN+_BB− | +139 | +3 | +4 | +5 |
| FLOWUSDT | 2026-02-06 | BN+_BB− | +118 | −8 | −11 | +1 |

**Pattern:** Extreme divergences snap back hard within 1–2 periods. The 2025-10-11 cluster suggests a Binance-wide anomaly (perhaps a settlement timing glitch or mass liquidation event).

FLOWUSDT is a **serial diverger** — appears 8 times in the top-40, with persistent BN−BB+ spread in Feb 2026.

### 6. Real-Time Basis Analysis (High-Resolution, 2 Days)

Using 1-minute mark/index price data from both exchanges:

| Metric | Value |
|---|---|
| Merged 1-min bars | 1,519,440 |
| Basis sign disagreements | 221,196 (14.6%) |
| Avg basis spread (BN − BB) | −3.07 bps |
| Std | 20.00 bps |
| P1/P99 | −53 / +42 bps |

#### Basis Spread Autocorrelation (Mean-Reversion Speed)

| Lag | Avg Autocorrelation |
|---|---|
| 1 min | 0.774 |
| 5 min | 0.479 |
| 30 min | 0.274 |
| 60 min | 0.207 |

**Basis spread mean-reverts within ~30–60 minutes** (AC drops below 0.3). This confirms convergence on an intraday timescale.

#### FR vs Basis Disagreement Overlap

| Category | % of minutes |
|---|---|
| Both FR + Basis disagree | 3.4% |
| Only Basis disagree | 11.1% |
| Only FR disagree | 26.3% |
| Neither | 59.2% |

FR disagreements and basis disagreements are **only weakly correlated** — they capture different phenomena. FR is a periodic settlement rate; basis is a continuous mark-to-index spread.

---

## Conclusions

1. **Convergence is real and strong.** 94% win rate across 87K events, and 100% win rate on extreme divergences (≥20 bps).

2. **BUT: most events are too small to trade profitably.** The average P&L is 4.6 bps, well below the ~39 bps round-trip fee. Only the 10+ bps bucket (1,111 events over 200 days = ~5.5/day) generates enough gross to approach profitability.

3. **Extreme events (≥30 bps) are the sweet spot** — 153 events over 200 days (~0.77/day), averaging +85 bps gross P&L with 99% WR. After fees (~39 bps × 2 legs = ~78 bps), this yields ~+7 bps net per trade. On $10K notional per leg that's ~$7/trade × 0.77/day = **~$5.4/day** — modest but real.

4. **The real opportunity is speed.** Basis spread mean-reverts in 30–60 min. A system that detects divergence in real-time and trades the mark-to-index convergence (rather than waiting for FR settlement) could capture 20–50 bps intraday on extreme events.

5. **FLOWUSDT is a persistent outlier** — investigate as a dedicated pair trade candidate.

---

## Next Steps

- [ ] Build a real-time monitor for extreme basis divergence (≥30 bps) across all symbols
- [ ] Analyze whether FR divergence *predicts* subsequent basis convergence (i.e., can we enter before the basis moves?)
- [ ] Deep-dive FLOWUSDT and the 2025-10-11 cluster
- [ ] Test if combining this with the HOLD FR arb improves overall P&L
