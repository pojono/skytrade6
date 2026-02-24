# Cross-Exchange FR Spread Arbitrage: Binance vs Bybit

**Date:** 2026-02-24 (v2: corrected to use absolute spread, not just opposite-sign)  
**Script:** `fr_research/research_basis_divergence.py`  
**Data:** Historical FR (200 days, Aug 2025 – Feb 2026) + real-time ticker (2 days)

---

## The Trade

When `|FR_Binance - FR_Bybit|` is large on a given symbol:

- **Short futures** on the exchange with the **higher** FR (collect more funding)
- **Long futures** on the exchange with the **lower** FR
- **Delta-neutral** — no spot leg, no directional exposure
- **Opposite signs NOT required** — same-sign spreads are 86–92% of all events!

### Fee Structure (Futures-Only, no spot)

| Tier | Per-side | Round-trip (open+close both exchanges) |
|---|---|---|
| VIP-0 taker | 5.0 bps | **20 bps** |
| VIP-0 maker | 2.0 bps | **8 bps** |
| VIP-1 maker | 1.6 bps | 6.4 bps |

---

## Key Findings

### 1. FR Spread Distribution

| Metric | Value |
|---|---|
| Matched settlements (BN × BB) | 465,843 |
| Common symbols | 476 |
| Data period | 200 days |
| Mean |spread| | 1.0 bps |
| P99 |spread| | 14.6 bps |
| P99.9 |spread| | 67.8 bps |

### 2. Event Frequency — Much Higher Than Previously Thought

| Threshold | Events | Per day | Same-sign | Opp-sign |
|---|---|---|---|---|
| ≥5 bps | 20,945 | **104.8** | 78% | 22% |
| ≥10 bps | 7,774 | **38.9** | 86% | 14% |
| ≥20 bps | 3,117 | **15.6** | 89% | 11% |
| ≥30 bps | 1,867 | **9.3** | 92% | 8% |
| ≥50 bps | 864 | **4.3** | 93% | 7% |

**Previous analysis only counted opposite-sign events (14% of total) — we were missing 86% of the opportunity.**

### 3. Convergence: P&L by Entry Threshold ($10K/leg)

| Filter | Trades | Per day | Gross | WR | Hold | Net @20bps | Net @8bps | **$/day taker** | **$/day maker** |
|---|---|---|---|---|---|---|---|---|---|
| ≥0 (all) | 465,367 | 2,328 | +2.9 | 60% | 2.1 | −17.1 | −5.1 | −$39,828 | −$11,894 |
| ≥5 bps | 20,907 | 105 | +41.2 | 96% | 3.6 | +21.2 | +33.2 | **+$2,214** | **+$3,469** |
| **≥10 bps** | **7,760** | **38.8** | **+84.2** | **96%** | **4.9** | **+64.2** | **+76.2** | **+$2,494** | **+$2,960** |
| ≥15 bps | 4,499 | 22.5 | +118.3 | 96% | 5.5 | +98.3 | +110.3 | +$2,212 | +$2,482 |
| **≥20 bps** | **3,114** | **15.6** | **+143.5** | **96%** | **5.8** | **+123.5** | **+135.5** | **+$1,923** | **+$2,110** |
| ≥30 bps | 1,865 | 9.3 | +177.7 | 96% | 6.1 | +157.7 | +169.7 | +$1,471 | +$1,583 |
| ≥50 bps | 862 | 4.3 | +218.0 | 96% | 6.1 | +198.0 | +210.0 | +$854 | +$905 |

**96% win rate across all filtered buckets.** Even with taker fees, every bucket ≥5 bps is profitable.

### 4. Net P&L by Fee Scenario (≥10 bps filter, $10K/leg)

| Scenario | RT Fee | Avg Net | Net WR | Total $ (200d) | $/day |
|---|---|---|---|---|---|
| VIP-0 taker | 20 bps | +64.2 | 71.5% | $498K | **$2,494** |
| VIP-0 maker | 8 bps | +76.2 | 93.4% | $592K | **$2,960** |
| VIP-1 maker | 6.4 bps | +77.8 | 94.2% | $604K | **$3,022** |

### 5. Scaling Analysis (Maker Fees, 8 bps RT)

| Notional/leg | ≥10 bps | ≥20 bps | ≥30 bps |
|---|---|---|---|
| $10K | **$2,960/day** | $2,110/day | $1,583/day |
| $25K | **$7,399/day** | $5,276/day | $3,958/day |
| $50K | **$14,798/day** | $10,552/day | $7,916/day |
| $100K | **$29,595/day** | $21,103/day | $15,833/day |

### 6. Direction Asymmetry

| Direction | Trades | Gross | WR |
|---|---|---|---|
| BN > BB (short BN, long BB) | 3,252 | +73.7 bps | 94% |
| BB > BN (short BB, long BN) | 4,508 | +91.8 bps | 97% |

BB > BN direction is more common and more profitable.

### 7. Spread Convergence Speed

For events with |spread| ≥ 20 bps (initial avg ~38 bps):

| Period | |Spread| | Reversion |
|---|---|---|
| t+0 | 38.1 bps | — |
| t+1 | 11.3 bps | +70% |
| t+2 | 8.7 bps | +77% |
| t+3 | 7.2 bps | +81% |

→ **70% of the spread reverts in just 1 settlement period.** Very fast convergence.

### 8. Real-Time Basis Analysis (1-min resolution, 2 days)

| Lag | Basis Spread Autocorrelation |
|---|---|
| 1 min | 0.774 |
| 5 min | 0.479 |
| 30 min | 0.274 |
| 60 min | 0.207 |

→ **Strong mean-reversion within 30–60 min** on an intraday timescale.

---

## Verdict

**This is a strong strategy.** The numbers are dramatically better than the initial (opposite-sign only) analysis.

| Aspect | Assessment |
|---|---|
| Edge exists? | ✅ 96% WR, 84 bps avg gross at ≥10 bps filter |
| Profitable after fees? | ✅ Even with taker fees at ≥5 bps filter |
| Frequency? | ✅ **38.9 events/day** at ≥10 bps (was 5.6 with opposite-sign only) |
| Daily P&L ($10K/leg) | ✅ **$2,960/day** maker fees, $2,494 taker |
| Scalable? | ✅ $29.6K/day at $100K/leg |
| Key risk | ⚠️ Basis risk (spread may widen before converging) |
| Key risk | ⚠️ Liquidity on altcoins — can we fill $50K+ without slippage? |

---

## Next Steps

- [ ] Build real-time monitor for FR spread across all common symbols
- [ ] Backtest with slippage model — are maker fills realistic on both exchanges simultaneously?
- [ ] Analyze max adverse excursion — how much does spread widen before converging?
- [ ] Check overlap with single-exchange HOLD strategy — are we double-counting?
- [ ] Add OKX as a third exchange for more arb opportunities
- [ ] Filter by liquidity/volume to avoid illiquid altcoins
