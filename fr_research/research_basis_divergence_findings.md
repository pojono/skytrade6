# Cross-Exchange FR Spread Arbitrage: Binance vs Bybit

**Date:** 2026-02-24 (v3: AUDITED — fixed lookahead bug + overlap double-counting)  
**Script:** `fr_research/research_basis_divergence.py`  
**Data:** Historical FR (200 days, Aug 2025 – Feb 2026) + real-time ticker (2 days)

> ⚠️ **AUDIT FINDINGS (v3):** The original analysis had two critical bugs:
> 1. **Lookahead in t=0:** Counting the entry-signal FR as P&L (31% of gross)
> 2. **Overlapping trades:** 74% of trades overlap with existing positions on same symbol
> 
> Corrected numbers are shown below. Original inflated numbers struck through where relevant.

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

> ⚠️ **CORRECTED (v3):** P&L now starts from t+1 (no lookahead). Original numbers included t=0 which inflated results by ~31%.

| Filter | Trades | Per day | Gross | WR | Hold | Net @20bps | Net @8bps | **$/day taker** | **$/day maker** |
|---|---|---|---|---|---|---|---|---|---|
| ≥5 bps | 20,907 | 105 | +27.3 | 80% | 2.6 | +7.3 | +19.3 | **+$761** | **+$2,016** |
| **≥10 bps** | **7,760** | **38.8** | **+58.3** | **81%** | **3.9** | **+38.3** | **+50.3** | **+$1,488** | **+$1,954** |
| ≥20 bps | 3,114 | 15.6 | +99.3 | 82% | 4.8 | +79.3 | +91.3 | +$1,235 | +$1,422 |
| ≥30 bps | 1,865 | 9.3 | +120.3 | 82% | 5.1 | +100.3 | +112.3 | +$935 | +$1,047 |
| ≥50 bps | 862 | 4.3 | +138.2 | 81% | 5.1 | +118.2 | +130.2 | +$510 | +$561 |

**81% win rate (was 96%).** Still profitable at ≥5 bps with maker fees.

### 4. Single-Position-Per-Symbol P&L (no overlap, ≥10 bps, $10K/leg)

> ⚠️ **CORRECTED (v3):** Original counted overlapping trades on the same symbol independently (74% overlap). With 1-position-per-symbol constraint:

| Metric | Original (buggy) | Fixed (no t=0, no overlap) |
|---|---|---|
| Trades | 7,760 | **2,562** |
| Avg gross | 84 bps | **28 bps** |
| Win rate | 96% | **74%** |
| $/day maker ($10K) | $2,960 | **$254** |
| $/day taker ($10K) | $2,494 | **$101** |

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

---

## 10. Audit Results

Three critical bugs were found and fixed:

### Bug 1: Lookahead in t=0 P&L (31% of gross)

The original code counted FR at t=0 as profit, but t=0 is the settlement we **observe** to decide entry. You can't collect t=0 FR unless you were positioned before seeing the spread. By construction, t=0 P&L = |spread| (always positive) — this is circular. **Fix: P&L starts from t+1.**

### Bug 2: Overlapping trades (74% of trades)

The original counted each settlement with |spread| ≥ 10 bps as an independent trade, even if the same symbol already had an open position. 74% of "trades" overlap. **Fix: 1 position per symbol at a time.**

### Bug 3: t=0 P&L always positive by construction

`pnl_t0 = sign_bn × FR_BN + sign_bb × FR_BB = |FR_BN - FR_BB|` — always equals the absolute spread. Selecting events with large |spread| and counting |spread| as profit is tautological.

### Impact

| Metric | Original (buggy) | Fixed (no t=0, no overlap) | Change |
|---|---|---|---|
| Trades (≥10 bps) | 7,760 | **2,562** | -67% |
| Avg gross | 84 bps | **28 bps** | -67% |
| Win rate | 96% | **74%** | -22 pp |
| $/day maker ($10K) | $2,960 | **$254** | **-91%** |
| $/day taker ($10K) | $2,494 | **$101** | -96% |

---

## Verdict

**Small edge exists, but far weaker than originally reported.**

| Aspect | Assessment |
|---|---|
| Edge exists? | ⚠️ Yes, but 74% WR and 28 bps gross (not 96%/84 bps) |
| Profitable after fees? | ⚠️ Marginal — $254/day maker, $101/day taker at $10K/leg |
| After slippage too? | ❌ Likely breakeven or negative for most symbols |
| Scalable? | ❌ Book depth too thin, overlap constraint limits trades |
| Key insight | The "spread convergence" is mostly regression to the mean — not a tradeable edge |

**This strategy is NOT worth pursuing as a standalone system.** The corrected numbers show ~$100–250/day on $10K/leg before slippage, which likely disappears with realistic execution.

---

## Next Steps

- [x] ~~Build real-time monitor~~ — not warranted given weak corrected edge
- [ ] Consider as a **secondary signal** in the single-exchange HOLD strategy (boost confidence when cross-exchange spread is large)
- [ ] Test whether the predicted/next FR (available before settlement) has enough signal to enter early and collect t=0 legitimately
