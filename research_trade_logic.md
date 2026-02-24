# FR Arbitrage: Trade Logic & Execution Rules

**Date:** 2026-02-24
**Based on:** 106-day backtest across Binance, Bybit, OKX (1h-settlement coins)

---

## 1. Core Strategy: HOLD

Short the funding rate via spot+futures hedge. Enter when FR is high, collect settlements, exit when FR normalizes.

- **Long spot** + **short futures** on the same exchange
- Collect funding rate every settlement (1h for target coins)
- Single round-trip cost: ~39 bps (spot taker 10 + futures taker 5.5 + slippage 4, × 2 legs)

---

## 2. Entry Rules

### When to enter

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Entry threshold** | FR ≥ 20–25 bps | Sweet spot: 25bps gives highest daily ($1,452), 20bps nearly identical ($1,443) with more trades |
| **Pick method** | Highest FR among available 1h coins | Simple greedy — take the juiciest rate |
| **Max positions** | 3 simultaneous | Diversification + enough capital deployed |

The entry threshold is **not very sensitive** — anything from 15–30 bps produces within 5% of optimal:

```
Entry ≥ 15bps: $1,448/day (more trades, lower WR)
Entry ≥ 20bps: $1,451/day (balanced)
Entry ≥ 25bps: $1,452/day (fewer trades, higher WR)
Entry ≥ 30bps: $1,441/day (starts missing opportunities)
Entry ≥ 50bps: $1,349/day (too selective, misses good trades)
```

### Breakeven speed

When entering at FR ≥ 20 bps, the average high-FR settlement is ~50+ bps. **You break even on the 39 bps RT cost in under 1 settlement** (< 1 hour). After that, every settlement is pure profit.

### What happens after entry (FR decay curve)

FR does NOT decay quickly. After entering at ≥ 20 bps:

```
Settle  1: 53% still > 20 bps, 79% still > 8 bps
Settle  5: 57% still > 20 bps, 85% still > 8 bps
Settle 10: 62% still > 20 bps, 88% still > 8 bps
Settle 15: 68% still > 20 bps, 92% still > 8 bps
Settle 20: 48% still > 20 bps, 86% still > 8 bps
```

**FR is highly persistent on 1h coins** (autocorrelation r = 0.62–0.73). Coins that have high FR tend to keep it for 10–20+ settlements. This is why HOLD works — you ride the wave.

---

## 3. Exit Rules

### When to exit

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Exit threshold** | FR < 5–8 bps | 5bps is marginally best ($1,451), 8bps very close ($1,443) and faster to act |

The exit threshold is also **not sensitive**:

```
Exit < 3 bps:  $1,421/day (holds too long, catches reversals)
Exit < 5 bps:  $1,451/day (sweet spot)
Exit < 8 bps:  $1,443/day (quicker exit, nearly same)
Exit < 10 bps: $1,404/day (exits too early, leaves money)
Exit < 15 bps: $1,286/day (clearly too aggressive)
```

### Average trade duration

With entry ≥ 20 bps and exit < 8 bps:
- **Average: 6.5 settlements** (6.5 hours for 1h coins)
- **Winners: ~10 settlements** (10 hours)
- **Losers: ~1.6 settlements** (exit quickly when FR drops immediately)

The win/loss asymmetry is massive: **avg winner +$382, avg loser -$26** (15:1 ratio).

---

## 4. Switching Logic

### Should you close a position to open a better coin?

**Rarely, and conservatively.** Switching means paying an extra 39 bps RT cost.

```
No switching:  $1,443/day (703 trades)
Switch at 3×:  $1,484/day (112 switches, +$41/day)
Switch at 5×:  $1,490/day (48 switches, +$47/day)
Switch at 2×:  $1,462/day (180 switches)
Switch at 1.5×: $1,445/day (241 switches, not worth the churn)
```

**Rule: Only switch if the new candidate's FR is ≥ 5× your worst position's current FR.** This adds ~$47/day (+3.3%) with only 48 switches over 106 days (once every 2 days). Not critical.

The reason switching barely helps: **your existing positions are likely still earning well** (FR is persistent), so the extra RT cost of closing + reopening usually isn't justified unless the difference is extreme.

---

## 5. Monitoring Frequency

### How often do you need to check?

```
Every  1h: $1,443/day (100% of optimal)
Every  2h: $1,357/day (94%)
Every  4h: $1,232/day (85%)
Every  6h: $1,240/day (86%)
Every  8h: $1,045/day (72%)
Every 12h: $1,019/day (71%)
Every 24h: $  898/day (62%)
```

**Checking every 4h captures 85% of the value.** Even checking just twice a day (12h) still captures 71%.

The reason: FR is persistent, so missing a few hours of entry/exit doesn't matter much. The main cost of infrequent monitoring is:
- **Late entry** — you miss the first few settlements of a high-FR episode
- **Late exit** — you sit through a few extra low-FR settlements before closing

Neither is catastrophic because the average trade lasts 6.5 settlements anyway.

### Practical recommendation

| Monitoring level | Frequency | Expected daily | Effort |
|-----------------|-----------|----------------|--------|
| **Automated bot** | Every settlement (1h) | $1,443 | Set and forget |
| **Active manual** | Every 4h (check 6× per day) | $1,232 | Moderate |
| **Lazy manual** | Every 12h (morning + evening) | $1,019 | Minimal |
| **Very lazy** | Once per day | $898 | Almost none |

---

## 6. Settlement Timing: When Exactly to Enter/Exit

### The hourly cycle (1h coins)

```
:00  ──── SETTLEMENT fires ──────────────────────────────────────
:05  You SEE the just-settled FR. Evaluate exits.
      If position FR < 8 bps → close position.
 ⋮   (positions are earning, nothing to do)
:50  Start scanning for new high-FR coins.
:55  ──── DECISION POINT ─────────────────────────────────────────
      ENTER new positions (FR ≥ 20 bps, slot available)
      Uses the :00 settled FR as signal (55 min old, still valid).
:58  Last safe moment to have orders filled.
:00  ──── NEXT SETTLEMENT ──── you collect FR ✓ ──────────────────
```

### Enter at :55, not :05

Both approaches use the **same information** (the last settled FR) and **collect the same next settlement**. The difference is price exposure:

| Enter at | Price exposure before next settlement | Risk |
|----------|--------------------------------------|------|
| :05 | 55 minutes | Higher — legs can diverge |
| :30 | 30 minutes | Medium |
| **:55** | **5 minutes** | **Lowest** |

You have 55 minutes after seeing the settled FR to analyze and decide. Then execute at :55 with minimal time exposed before settlement.

### Do NOT try to "predict" the next settlement and enter early

Entering based on the **previous** hour's FR to "capture one extra settlement" sounds attractive but is **6% worse** in practice:

- 13% of the time, FR collapses between settlements (previous ≥ 20bps but next < 8bps)
- Those bad entries cost ~$39 each in wasted round-trip fees
- The extra settlements captured don't compensate for the bad entries

```
Post-settle (see FR, enter at :55):  $1,416/day (baseline)
Pre-settle (predict, enter early):   $1,332/day (-6%)
```

### FR persistence validates the :55 approach

When current settled FR is high, the next settlement is almost always high too:

| Current FR | Next FR mean | Next still ≥ 20bps | Next < 8bps (bad) |
|------------|-------------|--------------------|--------------------|
| 20-30 bps  | 22.2 bps    | 44.5%              | 13.0%              |
| 30-50 bps  | 34.9 bps    | 67.4%              | 7.0%               |
| 50-100 bps | 58.7 bps    | 81.8%              | 5.0%               |
| ≥ 100 bps  | 123.1 bps   | 92.8%              | 3.4%               |

The signal (just-settled FR) is reliable. No need to rush — analyze at :05, execute at :55.

### Exit timing doesn't matter

When FR drops below 8 bps, the last settlement was worth ~$0.40-0.80. Over 700 trades that's $3-5/day. Close whenever convenient after seeing the drop.

---

## 7. Position Utilization

With 95 Bybit 1h coins and entry ≥ 20 bps:

```
0 positions:  4.1% of hours  (rare — almost always something to enter)
1 position:  18.2% of hours
2 positions: 29.0% of hours
3 positions: 48.7% of hours  (fully invested half the time)
Mean: 2.22 positions
```

You're running at ~74% capacity on average (2.22 / 3 slots).

---

## 8. Multi-Exchange Execution

Running the same logic across all 3 exchanges simultaneously:

| Exchange | 1h coins | Daily P&L | Role |
|----------|----------|-----------|------|
| Bybit    | 95       | $1,443    | Primary |
| Binance  | 32       | $820      | Secondary |
| OKX      | 20       | $752      | Tertiary |
| **Combined** | **147** | **$1,721** | **+19% over Bybit alone** |

The combined pool adds value because coins have **different FR cycles** — when one exchange's coins are quiet, another's may be spiking.

---

## 9. Multi-Interval Allocation: 1h vs 4h vs 8h

### Do NOT mix intervals in the same pool

When 1h and 4h coins compete for the same 3 position slots, 4h coins steal slots from the more profitable 1h coins:

| Pool (Bybit, 3 slots) | Daily |
|---|---|
| **1h only** | **$1,443** |
| 1h + 4h mixed | $1,034 ← worse! |

### Run separate allocations per interval instead

| Allocation (Bybit) | Daily |
|---|---|
| 1h × 3 slots | $1,443 |
| 4h × 3 slots (separate capital) | $287 |
| **1h + 4h = 6 slots** | **$1,730** |

### Why 1h coins are 5× more capital-efficient

| Metric | 1h coins | 4h coins | 8h coins |
|---|---|---|---|
| FR per settlement (held) | 39.5 bps | 30.0 bps | 27.6 bps |
| **Effective hourly FR** | **39.5 bps/hr** | **7.5 bps/hr** | **3.5 bps/hr** |
| Avg winner | +$382 | +$183 | +$178 |

A 4h coin settling at 30 bps only earns 7.5 bps per hour of capital locked. A 1h coin at 39.5 bps earns 5.3× more per hour.

### Different thresholds per interval

4h/8h coins settle less often, so each settlement matters more. They need higher entry bars and tighter exits:

| Interval | Entry | Exit | Best Daily (Bybit) |
|---|---|---|---|
| 1h | ≥ 20 bps | < 5 bps | $1,451 |
| 4h | ≥ 30 bps | < 3 bps | $337 |
| 8h | ≥ 30 bps | < 5 bps | $136 |

### Full multi-exchange, multi-interval potential

| Exchange | 1h (3 slots) | 4h (3 slots) | Total (6 slots) |
|---|---|---|---|
| Bybit | $1,443 | $287 | $1,730 |
| Binance | $820 | $615 | $1,435 |
| OKX | $752 | $483 | $1,235 |

Note: Binance 4h is strong ($615/day, 395 coins, 69% WR) because it has only 32 1h coins but 395 4h coins.

---

## 10. Summary: The Full Playbook

### Primary: 1h coins (3 slots per exchange)
```
:00  Settlement fires
:05  See settled FR → evaluate exits (close if FR < 8 bps)
 ⋮   Analyze candidates (55 min to decide)
:55  ENTER new positions (FR ≥ 20 bps, slot available)
:00  Next settlement → collect FR ✓
```

### Secondary: 4h coins (separate 3 slots per exchange)
```
Same logic at 4h settlement times (00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC)
ENTER:  FR ≥ 30 bps at :55 before settlement
EXIT:   FR drops below 3-5 bps
```

### Expected performance ($10k notional per slot)

| Config | Slots | Daily P&L |
|---|---|---|
| Bybit 1h only | 3 | $1,443 |
| All exchanges, 1h only | 9 | ~$2,500 |
| Bybit 1h + 4h | 6 | $1,730 |
| All exchanges, 1h + 4h | 18 | ~$4,000 |

- 60–70% win rate
- Winners: +$380 avg (10h hold) | Losers: -$26 avg (1.6h hold)
- Every month profitable over 106-day test
- 15:1 winner/loser ratio

---

## 11. Delta Neutral Execution: Futures Premium & Hedging

### What is the futures premium (basis)?

Perpetual futures trade at a slight premium or discount to spot. Measured from 2 days of Bybit tick data:

| Coin | Basis mean | Basis std | Futures spread |
|------|-----------|-----------|----------------|
| BTCUSDT | -5.1 bps | 0.7 bps | 0.02 bps |
| ETHUSDT | -5.6 bps | 0.7 bps | 0.05 bps |
| SOLUSDT | -6.8 bps | 1.7 bps | 1.2 bps |
| DOGEUSDT | -6.8 bps | 2.0 bps | 1.1 bps |
| ADAUSDT | -7.3 bps | 3.4 bps | 3.7 bps |

Negative basis = futures trading at discount (bearish period in sample). Altcoins have wider basis and spreads.

### Basis risk is negligible vs FR income

| Metric | Value |
|--------|-------|
| Basis drift over 6.5h hold | std ≈ 2 bps = **$2 on $10k** |
| FR income per trade (avg) | 368 bps = **$368 on $10k** |
| Basis risk as % of income | **0.5%** |

Even a 2-sigma basis move ($4) is < 2% of the average winning trade ($382). The basis is mean-reverting (it's what drives the funding rate mechanism), so over multiple settlements it averages out.

### Why basis doesn't break the hedge

1. **Perpetual futures converge to spot** via the funding rate — that's literally the mechanism we're harvesting
2. When FR is high (positive), basis tends to narrow (self-correcting)
3. Over multiple settlements, basis noise cancels out
4. Entry/exit spread cost is **already included** in our 39 bps RT estimate

### How to stay delta-neutral: execution rules

**CRITICAL: Match QUANTITIES, not dollar amounts.**

If you buy 150.0 SOL spot, short exactly 150.0 SOL futures. The dollar amounts will differ by the basis (~$0.50 per $10k), but the **delta** (exposure to price moves) is exactly zero.

```
ENTRY at :55:
  1. Get spot ask price and futures bid price
  2. Compute quantity: qty = $10,000 / spot_ask_price
  3. Round qty to exchange lot size
  4. Submit BOTH orders within <1 second:
     • BUY qty spot (market or IOC limit at ask)
     • SELL qty futures (market or IOC limit at bid)
  5. Verify: |spot_filled_qty - futures_filled_qty| < 1 lot
  6. If partial fill: immediately fill remainder or close excess

EXIT when FR < 8bps:
  1. Submit BOTH close orders within <1 second:
     • SELL spot (same qty as held)
     • BUY futures (close short, same qty)
  2. Same simultaneity requirement as entry
```

### Leg execution risk

5-second price moves (our data resolution):

| Coin | Mean move | P99 move | P99 on $10k |
|------|----------|----------|-------------|
| BTCUSDT | 0.9 bps | 7.1 bps | $7.07 |
| SOLUSDT | 1.5 bps | 10.2 bps | $10.22 |
| DOGEUSDT | 1.3 bps | 10.8 bps | $10.83 |

Submitting both legs within <1 second keeps leg risk to ~1 bps = $1 on average. Even at P99 it's ~$10 — one bad fill per 100 trades, trivially absorbed by $382 avg winner.

### Liquidation risk: zero

- **Spot leg**: fully paid, no leverage, cannot be liquidated
- **Futures leg**: set to 1× leverage (cross margin with enough balance) — the spot position covers any futures P&L
- **Net exposure**: zero — if price goes up, spot gains = futures losses, and vice versa
- Only risk: exchange counterparty risk (exchange goes down with your funds)

### Risk summary

| Risk | Magnitude | Impact on $10k | Mitigation |
|------|-----------|---------------|------------|
| Basis drift (6.5h) | 2 bps std | $2.00 | Negligible vs $382 avg trade |
| Leg execution gap | ~1 bps | $1.00 | Submit both within <1s |
| Size mismatch | <$20 | $1.00 at 5% move | Match quantities, not dollars |
| Bid-ask spread | 1-4 bps per leg | Already in 39 bps RT | Use liquid coins |
| Liquidation | N/A | $0 | Spot fully paid, futures 1× leverage |
| **Total hedge risk** | **~3-5 bps** | **$3-5 per trade** | **<1.5% of avg profit** |
