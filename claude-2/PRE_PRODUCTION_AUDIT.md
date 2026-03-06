# Pre-Production Audit: Idea 4 — BTC Pump → Long Alts

**Date:** 2026-03-06 (v2 — without October 2025 outlier)
**Strategy:** When BTC pumps >150 bps in 3 minutes, market-buy altcoins, hold 4h.
**Data:** Bybit + Binance, 25 alts, 2024-01 → 2026-03 (26 months).

> **⚠️ All numbers in this audit EXCLUDE Oct 2025** (2 days, +6026 bps — a single
> unrepeatable event that inflated every metric by 2-3x). This is the honest view.

---

## Summary Table

| Check              | Status    | Risk       | Notes                                                        |
| ------------------ | --------- | ---------- | ------------------------------------------------------------ |
| Lookahead Bias     | YES       | LOW        | T+1 entry delay applied; signal uses only past 3 bars        |
| OOS Test           | PARTIAL   | **HIGH**   | OOS 2024 is +213 bps, but 2025 (no Oct) is **-3 bps**, 2026 is **-7 bps** |
| Overfitting        | PARTIAL   | **HIGH**   | Robust to threshold, but without best 3 days avg = **+26 bps** ≈ noise |
| Execution Modeling | PARTIAL   | **HIGH**   | Only 20 bps taker fees; spread + slippage not modeled (+10-30 bps) |
| PnL Stability      | NO        | **CRITICAL** | Only 2024 is profitable. 2025 (no Oct) and 2026 are flat/negative |

---

## 1. Lookahead Bias

**Status: YES (no lookahead) | Risk: LOW**

- **Signal:** `btc_ret_3m = (close / close.shift(3) - 1)` — uses only past 3 bars. ✓
- **Entry:** `dfwd_240m = fwd_240m.shift(-1)` — T+1 bar delay, entry at next bar's close. ✓
- **Declustering:** 30-bar minimum gap between signals prevents overlapping trades. ✓
- **Forward return:** calculated from T+1 bar, not from signal bar. ✓

**Minor issue:** T+1 entry uses next bar **close**, not **open**. In 1-minute bars the difference is ~0-5 bps — negligible.

---

## 2. Honest Out-of-Sample Test

**Status: PARTIAL | Risk: HIGH**

| Period | Role | Days | Avg net | Prof% |
|--------|------|------|---------|-------|
| 2024-01 → 2024-12 | **True OOS** | 15 | **+213 bps** | 67% |
| 2025-01 → 2025-12 (no Oct) | Walk-forward | 12 | **-3 bps** | 50% |
| 2026-01 → 2026-03 | Most recent | 5 | **-7 bps** | 60% |

**The edge existed in 2024 but appears DEAD in 2025-2026 once the October outlier is removed.**

- 2024 OOS was never used for parameter selection → legitimate. ✓
- But 2025 without October = breakeven → edge decayed or was never real outside 2024.
- 2026 confirms: 5 days, negative average.

---

## 3. Overfitting / Parameter Stability

**Status: PARTIAL | Risk: HIGH**

### Parameters (only 3):
1. **BTC threshold** (150 bps) — the only tuned parameter
2. **Hold period** (240 min) — not optimized, standard 4h horizon
3. **Decluster gap** (30 bars) — standard anti-clustering

### Threshold sensitivity (no October):

| Threshold | Days | Avg net | Prof% | No best day | No best 3 |
|-----------|------|---------|-------|-------------|-----------|
| 100 bps (-33%) | 103 | **+27** | 53% | +20 | +6 |
| 120 bps (-20%) | 72 | **+57** | 61% | +46 | +26 |
| **150 bps (base)** | **32** | **+105** | **59%** | **+78** | **+31** |
| 180 bps (+20%) | 24 | **+103** | 62% | +67 | +1 |
| 200 bps (+33%) | 17 | **+270** | 76% | +228 | +150 |

**Direction is consistent** (all positive at all thresholds) but **magnitudes are thin.**
At 100-150 bps threshold, removing 3 best days puts avg at +6 to +31 bps — **within noise and below realistic costs.**

### Hold period sensitivity (no October):

| Hold | Avg net | Prof% | No best day |
|------|---------|-------|-------------|
| 30m | **+37** | 59% | +17 |
| 60m | **+39** | 62% | +15 |
| 120m | **+35** | 59% | +16 |
| **240m** | **+105** | **59%** | **+78** |
| 360m | **+148** | 66% | +122 |
| 480m | **+193** | 75% | +169 |

Longer holds help, but short holds (30m-2h) are essentially at **+15-17 bps without best day — below any realistic execution cost.**

### Tick-level entry (no October):

| Window | Avg ret | Net (after 20 bps fees) | WR |
|--------|---------|------------------------|-----|
| +10s | +22 bps | **+2** | 85% |
| +30s | +69 bps | **+49** | 98% |
| +1m | +103 bps | **+83** | 98% |
| +5m | +112 bps | **+92** | 84% |
| +4h | +127 bps | **+107** | 67% |

**Tick data confirms smaller but still positive edge at +30s to +1m even without October.**

---

## 4. Realistic Execution (Fees + Slippage)

**Status: PARTIAL | Risk: HIGH**

### What IS modeled:
- **Taker fees:** 20 bps round-trip (0.10% per leg × 2). ✓

### What is NOT modeled:

| Cost | Estimate | Impact |
|------|----------|--------|
| **Bid-ask spread** | 3-10 bps per leg on alts | 6-20 bps RT |
| **Slippage** | 5-20 bps during volatile pump events | 5-20 bps |
| **Speed of execution** | Must enter within 30s-1m for 98% WR | Latency-dependent |
| **Simultaneous execution** | Buying 4-10 alts at once | Market impact |

### Realistic cost estimate (no October):

| Scenario | Total cost | Kline net (32 days) | Tick net (+1m) |
|----------|-----------|--------------------|--------------------|
| Modeled (current) | 20 bps | **+105 bps** | **+83 bps** |
| Conservative (+10 spread) | 30 bps | **+95 bps** | **+73 bps** |
| Realistic (+20 spread+slip) | 40 bps | **+85 bps** | **+63 bps** |
| Pessimistic (+30) | 50 bps | **+75 bps** | **+53 bps** |

At realistic costs, edge is **+63 to +85 bps** — still positive, but thin.
**Without best 3 days and realistic costs:** +31 - 20 = **+11 bps** ≈ zero.

### Maker vs Taker:

Maker orders (0.04% = 8 bps RT) are **unrealistic** — you need market orders within seconds. Limit orders during volatility spikes have very low fill rates.

---

## 5. PnL Stability

**Status: NO | Risk: CRITICAL**

### Monthly breakdown (no October):

| Month | Days | Avg net | Comment |
|-------|------|---------|---------|
| 2024-01 | 2 | +417 | ✓ |
| 2024-02 | 2 | -117 | ✗ |
| 2024-03 | 1 | +562 | ✓ |
| 2024-04 | 2 | +403 | ✓ |
| 2024-05 | 1 | +135 | ✓ |
| 2024-06 | 1 | +189 | ✓ |
| 2024-07 | 1 | -21 | ✗ |
| 2024-08 | 2 | +180 | ✓ |
| 2024-11 | 1 | -27 | ✗ |
| 2024-12 | 2 | +293 | ✓ |
| 2025-01 | 3 | -38 | ✗ |
| 2025-02 | 1 | +550 | ✓ |
| 2025-03 | 2 | +5 | ~ |
| 2025-04 | 4 | -9 | ✗ |
| 2025-07 | 1 | +196 | ✓ |
| 2025-12 | 1 | -648 | ✗ |
| 2026-01 | 2 | +84 | ✓ |
| 2026-02 | 3 | -68 | ✗ |

### By year (no October):

| Year | Days | Avg net | Prof% | Verdict |
|------|------|---------|-------|---------|
| **2024** | **15** | **+213** | **67%** | ✓ Real edge |
| **2025** | **12** | **-3** | **50%** | ❌ Breakeven |
| **2026** | **5** | **-7** | **60%** | ❌ Negative |

### Concentration risk:

- **Total PnL (no Oct):** +3116 bps across 32 days
- **Best single day:** +846 bps (27% of total)
- **Without best day:** avg drops to +73 bps
- **Without best 3 days:** avg drops to **+26 bps** — noise
- **8 of 18 months are negative** (44%)
- **No signals at all** in 7 months (Sep-Oct 2024, Jun 2025, Aug-Sep 2025, Nov 2025, Mar 2026)

### Verdict:
The strategy was profitable in 2024, but **the edge appears to have decayed in 2025-2026.** Without the October 2025 outlier, the most recent 14 months are essentially breakeven. PnL is driven by a handful of good days in 2024 and one in Feb 2025.

---

## Overall Assessment

### Probability that edge is real: **30-40%**

**Evidence FOR:**
- 2024 OOS is legitimately positive (+213 bps, 15 days, 67% profitable)
- Robust direction across all parameter variations (always positive)
- Structural explanation (alts lag BTC pumps) confirmed by tick data
- Tick-level entry at +30s still shows +49 bps net without October

**Evidence AGAINST:**
- **2025 (no Oct) = -3 bps, 2026 = -7 bps** — edge is dead in recent data
- Without top 3 days, avg = +26 bps — below realistic execution costs
- Only 32 signal days in 26 months — statistically meaningless
- Huge variance (std = 347 bps vs mean = 97 bps → Sharpe ~0.28)
- No spread/slippage modeling eats most of the thin remaining edge
- 44% of months are negative

### Key Risks:

1. **Edge decay** — 2024 worked, 2025-2026 does not. Market has likely adapted (CRITICAL)
2. **PnL concentration** — a few good 2024 days drive the entire result (CRITICAL)
3. **Execution during volatility** — spreads widen during BTC pumps, slippage eats edge (HIGH)
4. **Sample size** — 32 events, Sharpe 0.28 → not statistically significant (HIGH)
5. **Survivorship** — this was the only surviving signal out of 9 tested. Selection bias? (MEDIUM)

### Production readiness: **NOT READY — EDGE LIKELY DEAD**

**Recommendation:** Do NOT deploy. The October 2025 outlier was masking an edge that decayed after 2024. The honest numbers show:

- **2024:** +213 bps avg → worked
- **2025 (no Oct):** -3 bps avg → dead
- **2026:** -7 bps avg → still dead

Without that one extraordinary October day, this strategy has been flat-to-negative for the last 14 months. Deploying it now would be chasing a dead edge.
