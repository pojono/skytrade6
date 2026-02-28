# Competitive Dynamics — Who Else Is Selling at Settlement?

**Date:** 2026-02-28  
**Dataset:** 150 settlements, ms-precision trade + OB.200 data  
**Script:** `research_competitive_dynamics.py`

---

## The Question

If other bots are already trading the settlement drop, what happens when we enter?
- Do they eat the bids before us?
- Does our selling make things worse or better?
- What does the orderbook actually look like at T+20ms (BB fill time) when we'd enter?

---

## The Timeline of a Settlement

```
T-0      Settlement happens. FR deducted. Orderbook intact.
         Price: 0.0 bps from mid

T+0-15ms DEAD ZONE. Almost nothing happens.
         Median sell volume: $6 (!!)
         Median trades: 1
         Bid depth consumed: 0.00%
         Price: -0.3 bps

T+15-25ms First wave of bots arrives.
         Sell volume: $306 median (100% sells, 0% buys)
         Price drops to -8.4 bps

T+25-50ms MAIN SELLING WAVE. 80 trades, $5,559 sell volume.
         This is where most bots execute.
         Price crashes to -29.7 bps

T+50-100ms Selling pressure fades. Buying starts.
         Sell ratio drops to 45% (near balanced)
         Price: -30.3 bps

T+100ms-1s Recovery/stabilization phase.
         Sell ratio ~52-61% (near balanced)
         Price: -33 to -37 bps

T+1-10s  Normal two-way trading resumes.
         Price bottoms around -35 bps median
```

---

## Key Finding #1: At T+15ms, the Book Is Untouched

| Metric | At T+15ms | At T+25ms |
|--------|-----------|-----------|
| Sell trades before us | 1 (median) | 32 |
| Sell volume before us | **$6** | **$1,487** |
| Bid depth consumed | **0.00%** | 1.15% |
| Price moved | **-0.3 bps** | -8.4 bps |

**At T+15ms, we are among the FIRST sellers.** The orderbook is essentially pristine. No one has eaten the bids yet. Our previous slippage analysis (which used the T-0 book) is almost exactly correct for T+15ms entry.

By T+25ms, ~$1,500 of selling has occurred and price has already dropped 8 bps. By T+50ms, $7,000+ of selling and price is down 30 bps. We want to be BEFORE this wave.

---

## Key Finding #2: Book Depletion Is Negligible at Our Entry

Entry slippage comparison — walking the original T-0 book vs the depleted book:

| Entry Time | Notional | Original Slip | Depleted Slip | Increase |
|-----------|----------|--------------|--------------|----------|
| **T+15ms** | $1,000 | 4.8 bps | 6.4 bps | **+0.0 bps** |
| **T+15ms** | $2,000 | 6.6 bps | 8.2 bps | **+0.0 bps** |
| **T+15ms** | $3,000 | 8.2 bps | 9.9 bps | **+0.0 bps** |
| T+25ms | $1,000 | 4.8 bps | 7.8 bps | +0.3 bps |
| T+25ms | $2,000 | 6.6 bps | 10.1 bps | +0.6 bps |
| T+25ms | $3,000 | 8.2 bps | 11.7 bps | +1.0 bps |
| T+25ms | $5,000 | 11.0 bps | 15.1 bps | +1.3 bps |

At T+15ms, the depletion impact is **zero** — the book is untouched. Even at T+25ms, it's only +0.3 to +1.3 bps extra. The competition impact is minimal.

---

## Key Finding #3: We Are Tiny Relative to Total Selling

| Our Notional | % of 1-Second Sell Volume |
|-------------|--------------------------|
| $1,000 | 4.9% |
| $2,000 | 9.7% |
| $3,000 | 14.6% |
| $5,000 | 24.3% |

At $2K, we are ~10% of the total selling that happens in the first second. We are not a dominant force — we're riding the wave, not creating it.

---

## Key Finding #4: Entry at T+15ms Gets Better Price Than T+25ms

| Entry Time | Price at Entry | Advantage |
|-----------|---------------|-----------|
| T+15ms | -0.3 bps | — |
| T+25ms | -8.4 bps | T+15ms is 8.1 bps better |
| T+50ms | -29.7 bps | T+15ms is 29.4 bps better |

By entering at T+15ms we catch the price BEFORE the main drop. This means:
- We enter near the top (only -0.3 bps from mid)
- Other bots sell AFTER us, pushing price down further
- We benefit from their selling pressure when we exit

---

## Key Finding #5: Cumulative Sell Pressure

| Period | Cumulative Sell $ | % of Bid Depth |
|--------|------------------|----------------|
| T+0 to T+5ms | $0 | 0.0% |
| T+0 to T+15ms | $6 | 0.0% |
| T+0 to T+25ms | $1,487 | 1.6% |
| T+0 to T+100ms | $12,182 | 13.1% |
| T+0 to T+1s | $18,906 | 20.3% |
| T+0 to T+5s | $29,952 | 32.1% |

In 5 seconds, ~$30K of selling consumes ~32% of bid depth. This is the collective selling of all bots + natural sellers. Our $2K is a small part of this.

---

## Entry Time Comparison (BB fill time)

### Market conditions at each entry time

| BB Fill Time | Sell Trades Before | Sell Volume Before | Price @ Entry | FR Safe? |
|-------------|-------------------|-------------------|--------------|----------|
| T+15ms | 1 | $6 | -0.3 bps | risky (BB created ~T+15ms, boundary T+18ms) |
| **T+20ms** | **8** | **$220** | **-5.3 bps** | **YES (BB created ~T+20ms > T+18ms)** |
| T+25ms | 32 | $1,487 | -8.4 bps | YES |
| T+30ms | 48 | $3,134 | -13.9 bps | YES |
| T+50ms | 128 | $8,877 | -29.7 bps | YES |

### Slippage & PnL at each entry time (depleted book, includes spread)

| BB Fill | Notional | Entry Slip | RT Slip | Net PnL | $ Profit | Win % |
|---------|----------|-----------|---------|---------|----------|-------|
| T+15ms | $1,000 | 6.4 bps | 11.0 | +12.6 | $1.26 | 92% |
| T+15ms | $2,000 | 8.2 bps | 14.3 | +9.3 | $1.87 | 81% |
| T+15ms | $3,000 | 9.9 bps | 17.0 | +6.6 | $1.99 | 69% |
| | | | | | | |
| **T+20ms** | **$1,000** | **7.1 bps** | **11.8** | **+11.8** | **$1.18** | **90%** |
| **T+20ms** | **$2,000** | **9.1 bps** | **15.3** | **+8.3** | **$1.66** | **78%** |
| **T+20ms** | **$3,000** | **10.9 bps** | **18.9** | **+4.7** | **$1.41** | **66%** |
| T+20ms | $5,000 | 14.2 bps | 24.4 | -0.8 | -$0.40 | 47% |
| | | | | | | |
| T+25ms | $1,000 | 7.8 bps | 12.4 | +11.2 | $1.12 | 85% |
| T+25ms | $2,000 | 10.1 bps | 16.2 | +7.4 | $1.49 | 75% |
| T+25ms | $3,000 | 11.7 bps | 19.6 | +4.0 | $1.19 | 62% |
| T+25ms | $5,000 | 15.1 bps | 25.6 | -2.0 | -$1.02 | 43% |

### T+20ms vs T+25ms

| Metric | T+20ms | T+25ms | Difference |
|--------|--------|--------|------------|
| Price at entry | -5.3 bps | -8.4 bps | **+3.1 bps better** |
| Entry slip ($2K) | 9.1 bps | 10.1 bps | **-1.0 bps better** |
| RT slip ($2K) | 15.3 bps | 16.2 bps | **-0.9 bps better** |
| Net PnL ($2K) | +8.3 bps | +7.4 bps | **+0.9 bps better** |
| Win rate ($2K) | 78% | 75% | +3% |
| FR payment | **NO** | **NO** | same |

T+20ms is better than T+25ms on every metric: better entry price, less book depletion, lower slippage.

---

## The Competitive Dynamics Story

```
BEFORE settlement:  Book is full. Bids stacked. Everyone waiting.

T+0ms:    FR deducted. Nothing visible yet in trades.

T+0-15ms: SILENCE. Almost no one trades. (median $6 sell volume)

T+15-20ms: First trickle of sells. $220 median.
           ┌─── YOU ENTER HERE (T+20ms BB fill) ───┐
           │ Book: ~99.8% intact                    │
           │ Price: -5.3 bps from mid               │
           │ FR: safely escaped (>T+18ms boundary)  │
           │ Competition: minimal (8 sell trades)    │
           └────────────────────────────────────────┘

T+20-50ms: MAIN SELLING WAVE. Other bots arrive.
           $5-9K sell volume. Price crashes to -30 bps.
           This is GOOD for you — you're already short.

T+50ms-1s: Selling fades. Price stabilizes ~-35 bps.

T+10-30s:  ML model says "exit now" (near bottom).
           You buy back, capturing the drop.
```

**Your selling ADDS to the pressure, making the drop slightly bigger. Other bots' selling ALSO adds to the drop. Everyone's selling is collectively creating the edge you're capturing.**

The question isn't "will competition hurt me?" — it's "am I early enough to get a good entry before the crowd?"

**Answer: At T+20ms (BB fill), yes. The main crowd arrives at T+25-50ms.**

---

## Key Takeaways

1. **T+20ms (BB fill) is the recommended entry.** Safely escapes FR payment (>T+18ms boundary), only 8 sell trades and $220 of selling before you, book ~99.8% intact.

2. **The selling wave is T+25-50ms.** That's when $5-9K of selling crashes the price 30 bps. You want to be positioned BEFORE this.

3. **Your $2K order is ~10% of total 1-second sell volume.** You're riding the wave, not creating it. Your marginal impact is small.

4. **Competition is GOOD for you** (once you're in). More selling after you = bigger drop = more profit when you exit.

5. **T+20ms vs T+25ms:** 3 bps better entry price, 1 bps less slippage, same FR safety. T+20ms is strictly better.

6. **T+15ms is tempting** (even better entry, untouched book) but **risky for FR** — BB created time ~T+15ms is too close to the T+18ms boundary. Network jitter could push you into the FR snapshot.

7. **The real risk isn't competition** — it's that the drop doesn't happen (low-FR settlements, or market absorbs the selling). The ML model handles this via signal quality.

---

## Production Entry Plan

```
1. EC2 sends market sell at T+~17ms (targeting BB fill at ~T+20ms)
2. BB fill time: T+20ms (safely past T+18ms FR boundary)
3. Notional: $1-2K based on OB depth at T-0
4. Book conditions: ~$220 of selling before us, 99.8% bids intact
5. Entry price: ~-5.3 bps from mid (vs -8.4 at T+25ms)
6. RT slippage ($2K): ~15.3 bps (spread + depth walking on depleted book)
7. Expected PnL: +8.3 bps median, $1.66/trade, 78% win rate
8. ML exit signal at T+10-30s: buy back near the bottom
```

---

## Files

| File | Purpose |
|------|---------|
| `research_competitive_dynamics.py` | Full analysis script |
| `FINDINGS_competitive_dynamics.md` | This document |
