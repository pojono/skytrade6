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

## 9. Slippage Reality Check (Bybit ob200 Orderbook Data)

**Script:** `fr_research/research_ob200_slippage.py`

Downloaded and processed 200-level orderbook snapshots (ob200) for the top 18 FR-arb symbols to measure **actual** book depth and slippage.

### Orderbook Depth (Bybit futures, Feb 2026)

| Symbol | BA Spread | Slip $1K | Slip $5K | Slip $10K |
|---|---|---|---|---|
| PIPPINUSDT | 0.14 bps | 0.8 | 2.3 | 2.8 |
| ENSOUSDT | 0.51 bps | 1.7 | 4.5 | 7.0 |
| AXSUSDT | 0.77 bps | 1.0 | 2.1 | 2.8 |
| RIVERUSDT | 1.25 bps | 1.9 | 5.0 | 8.6 |
| 0GUSDT | 1.56 bps | 2.0 | 4.3 | 5.3 |
| LAUSDT | ~1.5 bps | 2.1 | 5.2 | 10.3 |
| COAIUSDT | 3.2–6.6 bps | 7.3 | 22.8 | 51+ |
| SOONUSDT | 9–11 bps | 9.0 | 20.2 | 54+ |
| MYXUSDT | 11–15 bps | 8.3 | 13.8 | 21.8 |

**Key finding:** The top arb symbols split into two groups:
- **Tradeable (BA < 3 bps):** PIPPINUSDT, AXSUSDT, ENSOUSDT, 0GUSDT, RIVERUSDT, LAUSDT — slippage $5K < 6 bps
- **Expensive (BA > 5 bps):** COAIUSDT, SOONUSDT, MYXUSDT — slippage eats most of the edge

### Realistic P&L with Per-Symbol Slippage

Using **maker entry + taker exit** (the realistic scenario — you place limits to enter, but need to cross spread to exit):

| Notional/leg | $/day | vs. ideal |
|---|---|---|
| $1K | **$258** | — |
| $2K | **$478** | — |
| $5K | **$1,023** | 35% of ideal |
| $10K | **$1,386** | 47% of ideal |

### Top Symbols by Realistic P&L ($5K/leg)

| Symbol | Trades | Gross | Slip/leg | Net | $/day |
|---|---|---|---|---|---|
| RIVERUSDT | 349 | 160.0 | 4.2 | 143.6 | **$125** |
| AXSUSDT | 144 | 207.2 | 2.7 | 193.8 | **$70** |
| ENSOUSDT | 147 | 195.4 | 4.5 | 178.4 | **$66** |
| 0GUSDT | 238 | 108.5 | 5.7 | 89.1 | **$53** |
| COAIUSDT | 249 | 135.7 | 24.3 | 79.2 | **$49** |
| PIPPINUSDT | 215 | 99.4 | 2.3 | 86.7 | **$47** |

204 out of 404 symbols are net profitable.

---

## Verdict

**The edge is real, but slippage on altcoins is the binding constraint.**

| Aspect | Ideal (all maker) | Realistic (maker+taker exit) |
|---|---|---|
| Frequency | 38.9/day | 38.9/day |
| $/day ($5K/leg) | $2,960 | **$1,023** |
| $/day ($10K/leg) | $5,920 | **$1,386** |
| Win rate | 93% | 58% |
| Profitable symbols | 404/404 | 204/404 |

| Risk | Assessment |
|---|---|
| Edge exists? | ✅ 96% gross WR, 84 bps avg gross |
| Profitable after slippage? | ✅ Yes, ~$1K/day at $5K/leg realistic |
| Scalable beyond $10K/leg? | ⚠️ Limited — book depth too thin on most altcoins |
| Key risk | ⚠️ Slippage on exit — taker fills eat 30–65% of gross edge |
| Key risk | ⚠️ Concentration — top 6 symbols drive 50%+ of P&L |

---

## Next Steps

- [ ] Build real-time monitor for FR spread across all common symbols
- [ ] Test limit-order-only execution (maker entry AND maker exit) — is it feasible?
- [ ] Analyze max adverse excursion — how much does spread widen before converging?
- [ ] Add OKX as third exchange for more arb pairs
- [ ] Download ob200 for more symbols to improve slippage coverage
- [ ] Research optimal position sizing per symbol based on book depth
