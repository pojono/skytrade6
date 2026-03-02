# Stage-4.1: Event-Based Exit Feasibility — Final Verdict

## Summary

Tested whether the MR-leg (retrace to t0) can be captured with **event-based TP**
(touch of level) instead of fixed-horizon exits, using causal entry after Refill
detection and queue-realistic fills.

**Verdict: MR-class CLOSED at current fee structure.**

Even with optimal touch-TP, the retrace profit is consumed by fees.
The fundamental problem: after a stress event, the retrace to t0 yields
~0 bp gross for makers and ~-1 bp gross for takers (spread crossing cost).

## Data

- 8,364 events with Refill detected (96.5% of Stage-1)
- 133,824 trade simulations (16 scenario configs × events)
- 130,274 clean sims (|net_pnl| < 200bp, 97.3%)
- 30 days DOGEUSDT, September 2025

## Report A: Profitability Surface (clean data)

### Taker MR (fee = 20bp RT)

| Config | TP% | SL% | TO% | net_med | net_mean | p5 | EV/sig |
|--------|-----|-----|-----|---------|----------|----|--------|
| TP0 SL20 TO5s | 54% | 22% | 24% | -22.8 | -30.7 | -67 | -30.7 |
| TP0 SL20 TO10s | 58% | 29% | 13% | -23.2 | -32.9 | -71 | -32.9 |
| TP0 SL30 TO10s | 61% | 20% | 19% | -22.6 | -32.6 | -76 | -32.6 |
| TP0 SL30 TO30s | 65% | 30% | 6% | -22.9 | -36.3 | -81 | -36.3 |
| TP5 SL20 TO10s | 17% | 49% | 34% | -40.1 | -39.0 | -79 | -39.0 |
| TP10 SL30 TO30s | 13% | 62% | 25% | -53.7 | -49.7 | -90 | -49.7 |

**TP0 hits 54-65% of the time**, but net PnL is always ~-23 bp.
TP5/TP10 rarely hit and SL dominates.

### Maker MR (fee = 4bp RT)

| Config | Fill% | TP% | SL% | TO% | net_med | net_mean | p5 | EV/sig |
|--------|-------|-----|-----|-----|---------|----------|----|--------|
| L1 TP0 SL20 TO10s | 13% | 31% | 13% | 54% | -8.2 | -11.9 | -36 | -1.51 |
| L1 TP0 SL30 TO20s | 20% | 36% | 15% | 48% | -8.9 | -15.3 | -55 | -2.98 |
| L1 TP0 SL20 TO30s | 24% | 36% | 39% | 24% | -12.8 | -18.3 | -53 | -4.47 |
| L1 TP5 SL20 TO10s | 13% | 9% | 16% | 73% | -8.8 | -12.7 | -42 | -1.61 |
| MID TP0 SL20 TO10s | 13% | 32% | 14% | 52% | -8.7 | -12.5 | -36 | -1.68 |
| MID TP0 SL30 TO30s | 26% | 39% | 25% | 36% | -10.9 | -19.2 | -62 | -5.01 |

**Best config: MAKER L1 TP0 SL20 TO10s at -1.51 bp/signal.** Still negative.

Fill rates are 13-26%. TP rates when filled are 31-40%.
But TO (time-out with no TP) dominates at 50-70%.

## Report B: Time-to-Touch

### TP0 hits (touch of mid(t0))

| Scenario | n | Median | p75 | p90 | P(<1s) | P(<2s) | P(<5s) | P(<10s) |
|----------|---|--------|-----|-----|--------|--------|--------|---------|
| TAKER | 18,857 | 207ms | 1,040ms | 3,742ms | 74.3% | 82.5% | 93.5% | 98.4% |
| MAKER L1 | 1,645 | 2,492ms | 5,718ms | 10,043ms | 37.6% | 47.6% | 71.4% | 90.0% |

For taker: TP0 touch is FAST (median 207ms, 74% within 1s).
For maker: much slower (median 2.5s) due to fill delay eating into the window.

## TP0 Hit Analysis — Why It Doesn't Work

### Taker: gross profit at TP is ~0

When taker hits TP0 (touch of mid(t0)):
- **Gross PnL: median -1.1 bp** (mean -3.6 bp)
- Entry spread at signal: median 1.1 bp

The retrace takes price back to mid(t0), but the taker entered at ask/bid
(crossing the spread). The gross capture = distance from ask/bid to mid(t0)
≈ -spread/2 ≈ -0.5 bp. After 20bp fee → **-21 bp net**.

### Maker L1: gross profit is ~0 too

When maker L1 hits TP0:
- **Gross PnL: median +0.2 bp** (mean -0.9 bp)
- Net after 4bp fee: **median -3.8 bp**
- WR net: **5.8%**

The maker enters at bid, TP is at mid(t0). The distance is ~spread/2.
For DOGE with ~1bp spread, this gives ~0.5bp gross. After 4bp fee → **-3.5bp net**.

## Report C: Maker's Curse

| Config | Fill rate | net_med | TP% | SL% |
|--------|-----------|---------|-----|-----|
| L1 TP0 SL20 TO10s | 13% | -8.2 | 31% | 13% |
| L1 TP0 SL20 TO30s | 24% | -12.8 | 36% | 39% |

Longer windows → more fills → more SL hits → worse EV.
Maker's curse confirmed: fills cluster on adverse continuation events.

## Report D: Delay Sensitivity

### TAKER TP0 SL20 TO10s

| Delay | Fill | TP% | net_med | EV/sig |
|-------|------|-----|---------|--------|
| 0ms | 100% | 58% | -23.2 | -32.89 |
| 100ms | 83% | 57% | -23.6 | -27.77 |
| 300ms | 81% | 56% | -23.8 | -27.43 |

### MAKER L1 TP0 SL20 TO10s

| Delay | Fill | TP% | net_med | EV/sig |
|-------|------|-----|---------|--------|
| 0ms | 13% | 31% | -8.2 | -1.51 |
| 100ms | 9% | 29% | -8.2 | -1.16 |
| 300ms | 10% | 30% | -8.2 | -1.27 |

Delay has minimal impact. The problem isn't latency — it's fundamental:
the retrace doesn't produce enough gross profit to cover any fee.

## Root Cause Analysis

The MR-leg (retrace to t0) is structurally unprofitable because:

1. **Taker**: spread crossing cost (~1bp) + 20bp RT fee > retrace gross (~0bp)
2. **Maker**: limit at bid gets ~spread/2 gross (~0.5bp) < 4bp RT fee
3. **TP0 is mechanically a break-even target**: returning to mid(t0) means
   you capture exactly the distance from your entry to mid(t0), which is
   at most spread/2 for makers and negative for takers
4. **TP5/TP10 rarely hit**: the retrace overshoots t0 by >5bp only 17% of the time,
   and by then SL has already triggered in 49% of events

## Final Verdict

**MR-class: CLOSED.**

Not because the retrace doesn't exist — it does (TP0 hit rate 54-65%).
Not because detection is poor — it works (AUC 0.59-0.61).
But because **the retrace is a return to equilibrium**, not a profitable overshoot.

The TP target (mid(t0)) is mechanically ~0bp gross from any reasonable entry.
No fee structure makes this positive — you'd need negative fees (rebates > spread).

## What Would Reopen This Class

1. **Instruments with negative maker fees** (rebate > spread cost)
2. **Much wider spreads** where the retrace overshoots significantly
3. **Cross-venue latency arbitrage** where one venue's retrace leads another
4. **Using the state engine as a FILTER** (avoid entering other strategies during Shock/RetraceLeg)
