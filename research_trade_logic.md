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

## 6. Position Utilization

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

## 7. Multi-Exchange Execution

Running the same logic across all 3 exchanges simultaneously:

| Exchange | 1h coins | Daily P&L | Role |
|----------|----------|-----------|------|
| Bybit    | 95       | $1,443    | Primary |
| Binance  | 32       | $820      | Secondary |
| OKX      | 20       | $752      | Tertiary |
| **Combined** | **147** | **$1,721** | **+19% over Bybit alone** |

The combined pool adds value because coins have **different FR cycles** — when one exchange's coins are quiet, another's may be spiking.

---

## 8. Summary: The Simple Playbook

```
1. SCAN:  Every 1-4 hours, check FR on all 1h-settlement coins across BB/BN/OKX
2. ENTER: If FR ≥ 20 bps and position slots available → open spot long + futures short
3. HOLD:  Collect FR every hour. Do nothing while FR stays ≥ 8 bps.
4. EXIT:  When FR drops below 8 bps → close both legs.
5. SWITCH: Only if a new coin has FR ≥ 5× your worst position (rare).
```

**Expected performance (3 positions, $10k each):**
- ~$1,400–1,700/day
- 60–70% win rate
- Winners: +$380 avg (10h hold) | Losers: -$26 avg (1.6h hold)
- Every month profitable over 106-day test
- 15:1 winner/loser ratio
