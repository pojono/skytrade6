# Both-Legs Loser Analysis — When to Short-Only vs Short+Long

**Date:** 2026-03-01 | **Data:** 161 settlements, 33 symbols, 4 days

## TL;DR

- **Short leg: 0% losers** (127/127 win with depth≥$2K + spread≤8bps filters)
- **Long leg: 37% losers** (39/105 lose) — the long leg is the weak link
- **Best filter: bottom timing ≤ 15s** → 73% WR, **$114/day** (vs $100/day always-long)
- **Production rule:** If ML exit fires by T+15s → do 2x buy. If not → short-only exit.

## 4-Way Outcome Matrix

| Outcome | N | % | Avg Short $ | Avg Long $ | Avg Combined $ |
|---------|---|---|------------|-----------|---------------|
| **Both Win** | 66 | 63% | +$2.59 | +$2.62 | +$5.21 |
| **Short Win, Long Lose** | 39 | 37% | +$2.32 | -$1.51 | +$0.81 |
| Short Lose, Long Win | 0 | 0% | — | — | — |
| Both Lose | 0 | 0% | — | — | — |

**Key insight:** The short leg NEVER loses (with filters). The long leg is where all risk lives.

## Long Leg Loser Root Cause

Of 39 long losers:
- **38% (15)** — Negative recovery: price kept falling after "bottom"
- **59% (23)** — Low recovery 0–15 bps: barely bounced, fees ate the gain
- Avg recovery in losers: **-4.1 bps** vs winners: **+54.7 bps**

The losers are NOT random — they cluster on settlements where the bottom comes LATE.

## What Predicts Long Leg Failure?

### #1: Bottom Timing (STRONGEST signal)

| Bottom At | N | Long WR | Avg Recovery | $/trade |
|-----------|---|---------|-------------|---------|
| **1–5s** | 39 | **79%** | +54.2 bps | **$3.03** |
| **5–10s** | 15 | **80%** | +52.4 bps | **$3.09** |
| 10–15s | 11 | 45% | +20.3 bps | $0.41 |
| 15–20s | 8 | 62% | +21.4 bps | $0.39 |
| **20–30s** | 32 | **41%** | +4.8 bps | **-$0.84** |

**Pattern:** Early bottoms (≤10s) have strong recoveries. Late bottoms (>15s) barely bounce.
When the crash is sharp and fast, the recovery is strong. When price drifts down slowly, there's no bounce.

### #2: FR Magnitude

| FR Range | N | Long WR | $/trade |
|----------|---|---------|---------|
| <20 bps | 18 | 50% | $0.33 |
| 20–40 bps | 37 | 59% | $1.55 |
| **40–60 bps** | 18 | **72%** | **$2.78** |
| 60–100 bps | 21 | 76% | $1.46 |
| >100 bps | 11 | 55% | $0.11 |

**Sweet spot:** FR 40–100 bps. Below 40 bps, the drop is too shallow for a meaningful bounce.
Above 100 bps, extreme volatility makes recovery unpredictable.

### #3: Depth

| Depth | N | Long WR | $/trade |
|-------|---|---------|---------|
| $2–5K | 29 | 55% | $0.62 |
| $5–10K | 26 | 50% | $0.80 |
| **$10–25K** | 45 | **71%** | **$2.13** |
| **>$25K** | 5 | **100%** | **$2.06** |

Deeper books → more reliable recovery (market makers re-quote faster).

### #4: Drop Size (non-monotonic)

| Drop | N | Long WR | $/trade |
|------|---|---------|---------|
| <20 bps | 9 | 44% | $1.68 |
| 20–40 bps | 20 | 55% | $0.65 |
| 40–60 bps | 23 | 61% | $2.45 |
| 60–80 bps | 12 | 75% | $1.43 |
| **80–120 bps** | 21 | **86%** | **$1.69** |
| >120 bps | 19 | 47% | $0.13 |

**Goldilocks zone:** 40–120 bps drops have the best recovery.
Shallow drops (<40 bps) don't bounce enough. Extreme drops (>120 bps) are regime shifts with no bounce.

## Conditional Long Leg Strategies

| Strategy | N Long | Long WR | Long $/day | Combined $/day | vs Always |
|----------|--------|---------|-----------|---------------|-----------|
| Short-only (no long) | — | — | — | $72.5 | -$27.3 |
| Always long | 105 | 63% | $36.3 | $108.8* | baseline |
| **Long if bottom ≤ 15s** | **66** | **73%** | **$41.5** | **$114.0** | **+$5.2** |
| Long if bottom ≤ 10s | 54 | 80% | $41.1 | $113.6 | +$4.8 |
| Long if depth ≥ 10K | 50 | 74% | $26.6 | $99.0 | -$9.8 |
| Long if drop ≥ 40 & depth ≥ 5K | 54 | 74% | $27.8 | $100.3 | -$8.5 |

*Note: $108.8 vs earlier $114 due to slightly different short leg accounting.

**Winner: "Long if bottom ≤ 15s"** — takes fewer trades but avoids all the late-bottom losers.

## Production Implementation

The "bottom ≤ 15s" filter maps directly to ML exit timing:

```
IF ml_exit_signal fires at T ≤ 15s:
    → Buy 2x: close short (1x) + open long (1x)
    → Hold long for +20s, then limit sell exit
ELSE (ML hasn't fired by T+15s):
    → Buy 1x: just close the short (standard exit)
    → No long leg — late bottom = weak recovery
```

This is zero-cost to implement — the ML model already provides the exit time.

## Per-Symbol Breakdown (sorted by combined $/trade)

| Symbol | N | Short WR | Long WR | S$/tr | L$/tr | C$/tr |
|--------|---|----------|---------|-------|-------|-------|
| SAHARAUSDT | 34 | 100% | 76% | +$3.21 | +$2.57 | +$5.78 |
| BARDUSDT | 8 | 100% | 75% | +$3.30 | +$1.98 | +$5.27 |
| ENSOUSDT | 16 | 100% | 50% | +$3.77 | +$0.22 | +$3.99 |
| POWERUSDT | 17 | 100% | 53% | +$1.57 | +$1.10 | +$2.68 |
| STEEMUSDT | 7 | 100% | 43% | +$1.06 | +$0.02 | +$1.07 |
| MOVEUSDT | 1 | 100% | 0% | +$1.40 | -$3.21 | -$1.80 |

**Avoid long on:** MOVEUSDT, BIRBUSDT (negative combined). But N=1-2, too few to blacklist.

## Worst Combined Trades

Top 5 worst:
1. BIRBUSDT 0228_1600: S=+$0.87, L=-$4.50, C=-$3.63 (drop=+52, recovery=-72.7 bps)
2. SAHARAUSDT 0228_0000: S=+$3.85, L=-$6.21, C=-$2.36 (drop=+60, recovery=-45.6 bps)
3. POWERUSDT 0226_2000: S=+$1.74, L=-$4.09, C=-$2.35 (drop=+62, recovery=-23.6 bps)
4. SOLAYERUSDT 0228_0900: S=+$0.79, L=-$2.94, C=-$2.15 (drop=+274, recovery=-40.9 bps)
5. SAHARAUSDT 0227_1200: S=+$2.30, L=-$4.17, C=-$1.88 (drop=+317, recovery=-26.7 bps)

All worst trades have **negative recovery** — the price kept falling after the "bottom". This happens when the bottom was actually late (>15s), meaning there wasn't a sharp crash+bounce, but a slow grind down.

## Summary Revenue Stack

```
Short-only (15% cap, limit exit):              $72.5/day
 + Long ALWAYS (2x buy, +20s hold):            $108.8/day  (+50%)
 + Long CONDITIONAL (bottom ≤ 15s only):        $114.0/day  (+57%) ← RECOMMENDED
```
