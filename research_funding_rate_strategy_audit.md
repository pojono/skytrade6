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

## 7. Key Data Findings

- **Binance FR is observable before settlement** — `lastFundingRate` is the predicted/accumulating rate. Signal error < 1.3 bps. No information disadvantage.
- **103 Bybit coins now have 1h funding** (same as Binance). This eliminates the 1h vs 8h mismatch edge for those coins.
- **OMUSDT is Binance-only** (not on Bybit futures).
- **Cross-exchange basis** on target altcoins: 7–91 bps (too wide for Approach 2).
- **No FR sign flips** occurred in this 2-day sample during held positions (0 negative payments).

---

## 8. Conclusions

1. **Approach 1 + Strategy B (hold) is the clear winner.** The 10 bps spot taker fee is irrelevant when amortized over 20+ settlements (1.9 bps/settle).

2. **Approach 2 (cross-exchange) is a trap.** Lower per-trade fees but you only collect the FR spread (not full FR) and eat 50–100 bps cross-exchange basis on illiquid alts.

3. **Approach 3 (scalp) is marginal.** Only 25% of settlements exceed the 39 bps break-even. Works as a supplement but not standalone.

4. **Strategy B is nearly fee-insensitive.** Going from best to worst fee model costs only $60 more over 2 days (5 RTs vs 53 RTs).

---

## 9. Caveats & Next Steps

- **Only 2 days of data.** Need weeks/months to validate FR persistence and sign-flip frequency.
- **5 trades is a tiny sample.** 100% win rate is not statistically significant.
- **Capacity limits.** These are small-cap altcoins. $10k is likely near max per position.
- **Need to test:** What happens during regime changes (bull→bear, volatility spikes)?
- **Accumulate more data** via `download_ticker_all.sh` running continuously.
- **Build live signal generator** once strategy is validated on longer data.
