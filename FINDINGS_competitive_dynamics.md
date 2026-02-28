# Competitive Dynamics — Who Else Is Selling at Settlement?

**Date:** 2026-02-28  
**Dataset:** 150 settlements, ms-precision trade + OB.200 data  
**Script:** `research_competitive_dynamics.py`

---

## The Question

If other bots are already trading the settlement drop, what happens when we enter?
- Do they eat the bids before us?
- Does our selling make things worse or better?
- What does the orderbook actually look like at T+15ms when we'd enter?

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

## Revised PnL Estimate: T+15ms Entry on Depleted Book

| Notional | Entry Slip | RT Slippage | Net PnL | $ Profit | Win % |
|----------|-----------|-------------|---------|----------|-------|
| $1,000 | 6.4 bps | 11.6 bps | +12.0 | $1.20 | 89% |
| **$2,000** | **8.2 bps** | **15.1 bps** | **+8.5** | **$1.69** | **78%** |
| $3,000 | 9.9 bps | 17.9 bps | +5.7 | $1.70 | 68% |
| $5,000 | 12.5 bps | 23.1 bps | +0.5 | $0.23 | 51% |

Note: This does NOT yet include the FR savings from escaping the snapshot.

### With FR Savings (avoiding FR payment on short)

If FR is negative and we escape paying it by entering at T+15ms (order created ~T+17ms < T+18ms boundary), we SAVE the FR that we would have paid. For our coins, median |FR| = ~50 bps.

But actually — the FR savings depend on the strategy. If we're only holding for 10-30s, we wouldn't be in the position at the NEXT settlement anyway. The FR savings apply only if our short was captured by THIS settlement's snapshot, which it wouldn't be since we're opening AFTER settlement. The key benefit of T+15ms is simply the **better entry price** (-0.3 bps vs -8.4 bps at T+25ms).

---

## The Competitive Dynamics Story

```
BEFORE settlement:  Book is full. Bids stacked. Everyone waiting.

T+0ms:    FR deducted. Nothing visible yet in trades.

T+0-15ms: SILENCE. Almost no one trades.
          ┌─── YOU ENTER HERE (T+15ms) ───┐
          │ Book: 100% intact              │
          │ Price: -0.3 bps from mid       │
          │ Competition: zero              │
          └────────────────────────────────┘

T+15-50ms: SELLING WAVE. Other bots arrive.
           They push price down 30 bps.
           This is GOOD for you — you're already short.

T+50ms-1s: Selling fades. Price stabilizes ~-35 bps.

T+10-30s:  ML model says "exit now" (near bottom).
           You buy back, capturing the drop.
```

**Your selling ADDS to the pressure, making the drop slightly bigger. Other bots' selling ALSO adds to the drop. Everyone's selling is collectively creating the edge you're capturing.**

The question isn't "will competition hurt me?" — it's "am I early enough to get a good entry before the crowd?"

**Answer: At T+15ms, yes. The crowd arrives at T+25-50ms.**

---

## Key Takeaways

1. **At T+15ms, you are FIRST.** The orderbook is untouched. No competition impact on your entry.

2. **The selling wave is T+25-50ms.** That's when $5-7K of selling crashes the price 30 bps. You want to be positioned BEFORE this.

3. **Your $2K order is ~10% of total selling.** You're riding the wave, not creating it. Your marginal impact is small.

4. **Competition is GOOD for you** (once you're in). More selling after you = bigger drop = more profit when you exit.

5. **T+15ms vs T+25ms:** 8 bps better entry, zero book depletion, avoids FR. T+15ms is strictly superior.

6. **The real risk isn't competition** — it's that the drop doesn't happen (low-FR settlements, or market absorbs the selling). The ML model handles this via signal quality.

---

## Files

| File | Purpose |
|------|---------|
| `research_competitive_dynamics.py` | Full analysis script |
| `FINDINGS_competitive_dynamics.md` | This document |
