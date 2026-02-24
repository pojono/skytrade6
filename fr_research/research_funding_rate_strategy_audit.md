# Funding Rate Arbitrage — Strategy & Commission Audit

**Date:** 2026-02-24
**Data:** 2.1 days (2026-02-22 01:00 → 2026-02-24 03:00), Binance 1-min resolution
**Scripts:** `audit_strategy_comparison.py`, `backtest_settlement_arb_v2.py`
**Notional:** $10,000 per position

---

## 1. Fee Structure (Binance VIP 0 — Confirmed)

| Fee | Rate | bps |
|---|---|---|
| Spot taker | 0.100% | 10.0 |
| Futures taker | 0.055% | 5.5 |
| Slippage (est.) | — | 2.0/leg |

### Round-Trip Cost (delta-neutral: spot + futures, entry + exit)

```
ENTRY: Buy spot (10 bps) + Sell futures (5.5 bps) + slippage×2 (4 bps) = 19.5 bps
EXIT:  Sell spot (10 bps) + Buy futures (5.5 bps) + slippage×2 (4 bps) = 19.5 bps
ROUND TRIP = 39.0 bps = $39.00 on $10k
```

Break-even FR per settlement:
- 1 settlement: 39.0 bps
- 10 settlements: 3.9 bps avg
- 20 settlements: **1.9 bps avg** ← key insight

---

## 2. Three Approaches Evaluated

### Approach 1: Same-Exchange Spot + Futures (Binance)

Buy spot + sell futures on Binance. Collect full Binance 1h FR.

| Metric | Value |
|---|---|
| RT cost | 39.0 bps ($39) |
| FR income | Full Binance FR |
| Basis risk | None |
| Capital | 1 exchange |

### Approach 2: Cross-Exchange Futures + Futures (Binance + Bybit)

Long futures on Binance + short futures on Bybit. Collect FR spread (BN − BB).

| Metric | Value |
|---|---|
| RT cost | 30.0 bps ($30) |
| FR income | FR spread only (~30% of full FR) |
| Basis risk | **7–91 bps** (measured on target coins) |
| Capital | 2 exchanges |

**Why this is a trap:**
- 103 Bybit coins now have 1h funding (same as Binance) → FR spread is tiny
- Cross-exchange price basis on illiquid alts: 7–91 bps (AZTECUSDT: 91 bps)
- Basis paid on ENTRY and EXIT = up to 182 bps hidden cost
- Need margin on both exchanges

### Approach 3: Scalp (enter before settlement, exit after)

Same legs as Approach 1 but hold only ~2 minutes across one settlement.

| Metric | Value |
|---|---|
| RT cost | 39.0 bps ($39) |
| FR income | 1 settlement's FR |
| Sign-flip risk | Zero (FR visible before entry) |
| Break-even | FR > 39 bps that settlement |

---

## 3. Strategy Variants

### Strategy A: Scalp (Approach 3)

Enter 60s before settlement on the highest-FR coin, exit 60s after.

- 53 settlements analyzed
- FR signal error: mean 0.1 bps, max 1.3 bps (Binance FR is predictable)
- Only **13/53 (25%)** settlements had FR > 39 bps break-even
- Median best-coin FR: 21.6 bps (below break-even)

| Fee scenario | Profitable | Total P&L | Daily P&L |
|---|---|---|---|
| VIP 0 (RT: 39 bps) | 13/53 (25%) | +$350 | +$168 |
| Pessimistic (RT: 51 bps) | 13/53 (25%) | −$286 | −$137 |

**Verdict:** Marginally profitable at best. Negative under pessimistic assumptions.

### Strategy B: Hold Until FR Normalizes (Approach 1)

Enter when best-coin FR ≥ 20 bps. Hold across multiple settlements.
Exit when FR drops below 8 bps or FR sign flips.

| Fee scenario | Trades | Settlements | WR | Total P&L | Daily P&L |
|---|---|---|---|---|---|
| VIP 0 (RT: 39 bps) | 5 | 92 | 100% | **+$2,403** | **+$1,154** |
| Pessimistic (RT: 51 bps) | 5 | 92 | 100% | **+$2,343** | **+$1,125** |

**Why it wins:** 5 round trips × $39 = $195 total cost. Amortized to 2.1 bps/settlement.
Fee model barely matters ($60 difference between best and worst).

---

## 4. P&L Comparison (2.1 days, $10k notional)

| Strategy | Net P&L | Daily | Annual ROI (est.) |
|---|---|---|---|
| Approach 1 + Hold (spot+fut) | **+$2,403** | +$1,154 | ~4,200% |
| Approach 2 + Hold (fut+fut cross-ex) | +$130 | +$62 | ~230% |
| Approach 3 Scalp (selective) | +$566 | +$272 | ~990% |

---

## 5. Volatility & Execution Risk

### Price volatility during 2-min scalp window

| Metric | Value |
|---|---|
| Price range (high−low) | mean 92.5 bps, p95 229.6 bps, max 924 bps |
| Price move (entry−exit) | mean 61.0 bps, p95 165.6 bps |

**This does NOT affect delta-neutral P&L.** Both legs move together.

### Leg execution delay risk

| Delay between legs | Price move risk |
|---|---|
| ±1s | 0 bps |
| ±5s | mean 8.6 bps, p95 37.9 bps |
| ±10s | mean 18.4 bps, p95 57.4 bps |

With taker orders, both legs fill in <1s. The 2 bps slippage assumption is conservative.

---

## 6. FR Distribution (Best Coin Per Settlement)

| FR threshold | % of settlements | Implication |
|---|---|---|
| ≥ 10 bps | 92.5% | Almost always have a coin |
| ≥ 15 bps | 73.6% | Strategy B entry threshold viable |
| ≥ 20 bps | 52.8% | Strategy B sweet spot |
| ≥ 30 bps | 30.2% | Scalp break-even (optimistic fees) |
| ≥ 39 bps | ~25% | Scalp break-even (VIP 0 fees) |
| ≥ 50 bps | 24.5% | Scalp break-even (pessimistic) |

Mean: 45.6 bps | Median: 21.6 bps | p75: 32.4 bps

---

## 7. Lookahead Bias Audit (2026-02-24)

### Cross-checked dataminer data against official REST API endpoints

| Exchange | REST Endpoint | Match |
|---|---|---|
| Binance | `GET /fapi/v1/fundingRate` | **24/24 exact match** with `lastFundingRate` at settlement row |
| Bybit | `GET /v5/market/funding/history` | **0.31 bps avg prediction error** (dataminer `fundingRate` is a live prediction) |

### What each exchange's fields actually mean

| Field | Exchange | Meaning | Updates |
|---|---|---|---|
| `lastFundingRate` | Binance | **Last SETTLED rate** (historical) | Only at settlement, to new settled value |
| `fundingRate` | Bybit | **Live PREDICTED next rate** | Continuously; resets to ~0 after settlement |

### FR prediction visibility before settlement

| Exchange | Can see FR before settlement? | How | Accuracy |
|---|---|---|---|
| Binance | Partially | Previous hour's settled rate (autocorrelation proxy) | ~0.04 bps |
| Bybit | **Yes, directly** | `fundingRate` field IS the live prediction | ~0.31 bps avg, max 3.42 bps |

### Backtest data correctness

- **Binance backtest used `shift(1)`** = previous hour's settled rate. This is **stale** (not lookahead), causing ~0.04 bps avg conservative error. Fix: use `lastFundingRate` directly at settlement row.
- **Bybit backtest used `fundingRate` before settlement** = predicted rate. Correct approach for live trading. ~0.31 bps avg error vs actual.
- **No lookahead bias confirmed** on either exchange.

---

## 8. Four-Way Comparison — Official REST API FR (2026-02-24)

All results below use **official settled FR** fetched from exchange REST APIs.

### Binance: 1a HOLD vs 3a SCALP

| Option | Trades | Settlements | FR $ | Cost $ | Net P&L | Daily | Fee % of FR |
|---|---|---|---|---|---|---|---|
| **1a: BN HOLD** | 4 | 83 | $2,040 | $156 | **+$1,884** | **+$962** | **8%** |
| 3a: BN SCALP | 12 | 12 | $1,386 | $468 | +$918 | +$469 | 34% |

Top 1a trades: AZTECUSDT (43 settles, $625), AWEUSDT (11 settles, $750), OMUSDT (29 settles, $547)

### Bybit: 1b HOLD vs 3b SCALP

| Option | Trades | Settlements | FR $ | Cost $ | Net P&L | Daily | Fee % of FR |
|---|---|---|---|---|---|---|---|
| **1b: BB HOLD** | 5 | 56 | $1,552 | $195 | **+$1,357** | **+$693** | **13%** |
| 3b: BB SCALP | 16 | 16 | $957 | $624 | +$333 | +$170 | **65%** |

Top 1b trades: LAUSDT (17 settles, $804), AZTECUSDT (26 settles, $466), SENTUSDT (5 settles, $46)

### Why HOLD dominates SCALP

- **Cost amortization**: Hold pays $1.88–$3.48/settlement vs Scalp's $39.00/settlement
- **Settlement count**: Hold collects 56–83 FR payments vs Scalp's 12–16
- **Scalp per-settlement profit is higher** ($76 vs $23) but volume of settlements wins
- **3b Bybit Scalp loses 65% of FR income to fees** — barely viable

### Cross-exchange note

- Binance has higher extreme FR coins (AWEUSDT, POWERUSDT) → 1a > 1b
- Bybit has better FR prediction visibility → better entry timing for scalp
- Coin overlap (AZTECUSDT, LAUSDT on both) → hybrid must deduplicate

---

## 9. Key Data Findings

- **Binance `lastFundingRate`** = last settled rate, NOT a prediction. Updates at settlement. Verified via REST API 24/24.
- **Bybit `fundingRate`** = live predicted rate. Resets after settlement. Converges to actual (0.31 bps avg error).
- **103 Bybit coins now have 1h funding** (same as Binance). Eliminates 1h vs 8h mismatch edge.
- **OMUSDT is Binance-only** (not on Bybit futures).
- **Cross-exchange basis** on target altcoins: 7–91 bps (too wide for Approach 2).
- **No FR sign flips** occurred in this 2-day sample during held positions.
- **Binance Mark Price stream** (`r` field) has real-time predicted FR — we don't collect it yet but could.

---

## 10. Conclusions

1. **Approach 1 + Hold is the clear winner on both exchanges.** Fee amortization over many settlements is the key mechanic.

2. **1a (Binance Hold) is best single option:** +$962/day, 8% fee overhead.

3. **1b (Bybit Hold) is a solid second:** +$693/day, 13% fee overhead. Better FR visibility.

4. **Scalp is marginal on both exchanges.** 34-65% of FR goes to fees. Only viable on extremely high FR settlements.

5. **Approach 2 (cross-exchange) is a trap.** Basis risk (23-108 bps std) dwarfs the FR spread.

6. **Hybrid (1a+1b) could work** with deduplication, but need to avoid doubling up on same coin.

---

## 11. 200-Day Historical Validation (2025-08-08 → 2026-02-24)

**Data:** Official settled FR from REST APIs for ALL coins on both exchanges.
- Binance: 576,672 records, 580 symbols (28 with 1h funding)
- Bybit: 728,226 records, 552 symbols (86 with 1h funding)
- Script: `download_historical_fr.py`

### Extreme FR frequency (1h coins only)

| Threshold | Binance (28 coins) | Bybit (86 coins) |
|---|---|---|
| FR ≥ 10 bps | 60.8% of settlements | 79.3% |
| FR ≥ 20 bps (entry) | 40.6% | **61.3%** |
| FR ≥ 39 bps (scalp BE) | 21.8% | **39.5%** |
| FR ≥ 100 bps | 6.2% | 15.2% |

Bybit has 3× more 1h coins → more extreme FR opportunities per settlement.

### FR autocorrelation (crucial for HOLD)

| Lag | Binance mean r | Bybit mean r |
|---|---|---|
| 1h | **0.729** | 0.637 |
| 2h | **0.572** | 0.485 |
| 4h | **0.456** | 0.350 |
| 8h | **0.322** | 0.245 |
| 24h | 0.193 | 0.139 |

Binance FR is significantly more autocorrelated. 96.4% of BN 1h coins have lag-1 r > 0.5.

### HOLD strategy results (entry ≥ 20bps, exit < 8bps, max 3 positions)

| Metric | Binance | Bybit |
|---|---|---|
| Trades | 228 | 743 |
| **Win rate** | **75.4%** | 65.4% |
| Total net P&L | +$121,467 | +$202,858 |
| **Daily P&L** | +$607 | **+$1,015** |
| Avg winning trade | +$714 (20 settles) | +$431 (10 settles) |
| Avg losing trade | -$24 (1.7 settles) | -$25 (1.6 settles) |
| All 7 months profitable | **YES** | **YES** |

### FR persistence when extreme (Bybit 1h coins)

- 1,292 trades: entry ≥ 20bps, exit when < 8bps
- **77% profitable** (total FR collected > RT cost of 39 bps)
- Median trade: 3 settlements, 85 bps total FR collected
- Mean trade: 6 settlements, 256 bps total FR collected
- p95 streak: 18 settlements (1046 bps collected)
- Avg losing trade: -$11 (only 1.2 settlements before normalization)

### Monthly breakdown

| Month | BN trades | BN net | BB trades | BB net |
|---|---|---|---|---|
| 2025-08 | 16 | +$3,556 | 39 | +$8,275 |
| 2025-09 | 24 | +$7,667 | 71 | +$13,811 |
| 2025-10 | 40 | +$23,004 | 96 | +$29,123 |
| 2025-11 | 57 | +$27,827 | 143 | +$40,650 |
| 2025-12 | 34 | +$15,839 | 148 | +$41,265 |
| 2026-01 | 35 | +$28,633 | 128 | +$40,539 |
| 2026-02 | 22 | +$14,942 | 118 | +$29,194 |

---

## 12. Conclusions (Updated with 200-day data)

1. **HOLD strategy is validated over 200 days.** Profitable every month on both exchanges. Not a fluke.

2. **Bybit generates higher daily P&L** (+$1,015/day vs +$607/day) due to 3× more 1h coins.

3. **Binance has higher win rate** (75% vs 65%) due to stronger FR autocorrelation (r=0.73 vs 0.64).

4. **Losses are tiny and asymmetric**: avg loss -$25 (1.6 settlements) vs avg win +$431-714 (10-20 settlements).

5. **FR ≥ 20 bps exists 41-61% of all hours** — abundant opportunities, not dependent on rare events.

6. **Scalp is confirmed marginal**: only 22-40% of settlements beat the 39 bps RT cost.

7. **Hybrid (1a+1b) is the optimal approach**: run both exchanges for diversification and higher throughput.

---

## 13. Caveats & Next Steps

- **Capacity limits.** These are small-cap altcoins. $10k is likely near max per position.
- **Need to test:** What happens during regime changes (bull→bear, volatility spikes)?
- **Slippage model:** 2 bps/leg is an estimate. Live testing needed to validate.
- **Execution risk:** Spot + futures legs must execute within 1s to avoid basis drift.
- **Build live signal generator** using confirmed strategy parameters.
- **Paper trade** for 2-4 weeks before deploying real capital.
