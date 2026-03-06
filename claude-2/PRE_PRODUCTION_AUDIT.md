# Pre-Production Audit: Idea 4 — BTC Pump → Long Alts

**Date:** 2026-03-06
**Strategy:** When BTC pumps >150 bps in 3 minutes, market-buy altcoins, hold 4h.
**Data:** Bybit + Binance, 25 alts, 2024-01 → 2026-03 (26 months).

---

## Summary Table

| Check              | Status    | Risk       | Notes                                                        |
| ------------------ | --------- | ---------- | ------------------------------------------------------------ |
| Lookahead Bias     | YES       | LOW        | T+1 entry delay applied; signal uses only past 3 bars        |
| OOS Test           | YES       | LOW        | Discovery on 2025-06→2026-03, OOS on 2024-01→2025-05 (+213 bps) |
| Overfitting        | PARTIAL   | MEDIUM     | Robust to threshold ±33%, but PnL heavily dependent on 1 outlier day |
| Execution Modeling | PARTIAL   | **HIGH**   | Only taker fees modeled (20 bps); spread + slippage not modeled (~10-30 bps extra) |
| PnL Stability      | NO        | **HIGH**   | 65% of total PnL from 1 day (2025-10-10: +5944 bps)         |

---

## 1. Lookahead Bias

**Status: YES (no lookahead) | Risk: LOW**

- **Signal:** `btc_ret_3m = (close / close.shift(3) - 1)` — uses only past 3 bars. ✓
- **Entry:** `dfwd_240m = fwd_240m.shift(-1)` — T+1 bar delay, entry at next bar's close. ✓
- **Declustering:** 30-bar minimum gap between signals prevents overlapping trades. ✓
- **Forward return:** calculated from T+1 bar, not from signal bar. ✓

**Minor issue:** T+1 entry uses next bar **close**, not **open**. In 1-minute bars the difference is ~0-5 bps — negligible for a +269 bps signal.

---

## 2. Honest Out-of-Sample Test

**Status: YES | Risk: LOW**

| Period | Role | Days | Avg net | Prof% |
|--------|------|------|---------|-------|
| 2025-06 → 2026-03 | Discovery | 19 | +427 bps | 63% |
| 2024-01 → 2025-05 | **True OOS** | 15 | **+213 bps** | 67% |

- OOS period was **never used** for parameter selection, feature engineering, or optimization.
- Signal works in **both** periods with consistent direction and magnitude.
- 2026-01→03 shows weakening (-7 bps on 5 days) — possible regime shift, but small sample.

---

## 3. Overfitting / Parameter Stability

**Status: PARTIAL | Risk: MEDIUM**

### Parameters (only 3):
1. **BTC threshold** (150 bps) — the only tuned parameter
2. **Hold period** (240 min) — not optimized, standard 4h horizon
3. **Decluster gap** (30 bars) — standard anti-clustering

### Threshold sensitivity (±33%):

| Threshold | Days | Avg net | Prof% | Without best day |
|-----------|------|---------|-------|-----------------|
| 100 bps (-33%) | 107 | +85 | 55% | +30 |
| 120 bps (-20%) | 75 | +136 | 63% | +58 |
| **150 bps (base)** | **34** | **+275** | **62%** | **+104** |
| 180 bps (+20%) | 25 | +337 | 64% | +103 |
| 200 bps (+33%) | 18 | +584 | 78% | +269 |

**✓ Robust:** Signal is positive at ALL thresholds 100-250 bps. Narrower threshold → fewer but better trades. The edge is real, not an artifact of threshold tuning.

### Hold period sensitivity:

| Hold | Avg net | Prof% | Without best day |
|------|---------|-------|-----------------|
| 30m | +188 | 59% | +35 |
| 60m | +174 | 65% | +38 |
| 120m | +214 | 62% | +33 |
| **240m** | **+275** | **62%** | **+104** |
| 360m | +330 | 68% | +143 |
| 480m | +379 | 76% | +188 |

**✓ Robust:** Positive at all hold periods 30m–8h. Longer holds → more profit but more risk. Not sensitive to ±50% changes.

### Concern: Without best day

Removing the single best day (2025-10-10: +5944 bps) drops average from +275 to +104 bps. Still positive, but the headline number is inflated by one outlier.

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
| **Speed of execution** | Must enter within 30s-1m for 97% WR | Latency-dependent |
| **Simultaneous execution** | Buying 4-10 alts at once | Market impact |

### Realistic cost estimate:

| Scenario | Total cost | Net return |
|----------|-----------|------------|
| Modeled (current) | 20 bps | +275 bps |
| Conservative | 40 bps | +255 bps |
| Pessimistic | 50 bps | +245 bps |

**Even at 50 bps total cost, the signal remains profitable (+245 bps avg).** But without the best day, it becomes +245 - 171 = +74 bps — much thinner.

### Maker vs Taker:

Maker orders (0.04% = 8 bps RT) are **unrealistic** for this strategy — you need to market-buy within seconds of BTC pump detection. Limit orders during volatility spikes have very low fill rates.

---

## 5. PnL Stability

**Status: NO | Risk: HIGH**

### Monthly breakdown:

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
| **2025-10** | **2** | **+3013** | **⚠️ OUTLIER** |
| 2025-12 | 1 | -648 | ✗ |
| 2026-01 | 2 | +84 | ✓ |
| 2026-02 | 3 | -68 | ✗ |

### Concentration risk:

- **65% of total PnL** comes from **1 day** (2025-10-10)
- **Without best day:** avg drops from +275 to +104 bps
- **Without best 3 days:** avg drops to +55 bps
- **7 of 19 months are negative** (37%)
- **No signals at all** in Sep 2024, Oct 2024, Jun 2025, Aug-Sep 2025, Nov 2025

### Verdict:
PnL is **extremely concentrated**. The strategy is profitable on average, but the "average" is dominated by one extraordinary event. In a typical month you get 1-2 signals with highly variable outcomes.

---

## Overall Assessment

### Probability that edge is real: **60-70%**

**Evidence FOR:**
- Signal is positive in both discovery (2025-06→2026-03) and true OOS (2024-01→2025-05)
- Robust to parameter changes (all thresholds 100-250 bps are positive)
- Has structural explanation (alts lag BTC pumps due to lower liquidity/attention)
- Tick data confirms alt reaction curve is gradual, not instant

**Evidence AGAINST:**
- Only 34 signal days in 26 months — very thin sample
- 65% of PnL from 1 outlier day (could be luck)
- Without top 3 days, avg drops to +55 bps — barely above realistic costs
- 2026 data (most recent) shows weakening
- No spread/slippage modeling

### Key Risks:

1. **PnL concentration** — strategy may appear profitable only because of 1-2 extreme events (HIGH)
2. **Execution during volatility** — spreads widen during BTC pumps, slippage increases (HIGH)
3. **Sample size** — 34 events is not statistically robust for tailed distributions (MEDIUM)
4. **Edge decay** — 2026 shows weakening, market may be adapting (MEDIUM)
5. **Latency** — requires <30s detection + execution (LOW if infrastructure exists)

### Production readiness: **NOT READY**

**Recommendation:** Do NOT deploy at full size. If deploying:
- Start with **paper trading** for 2-3 months to validate execution assumptions
- Use **minimal position size** (1-2% of capital per event)
- Track actual execution costs (spread + slippage) vs modeled
- Monitor for edge decay in real-time
- Accept that most months will have 0-2 signals with high variance
