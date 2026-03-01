# Independent Position Sizing — Short vs Long Leg

**Date:** 2026-03-01 | **Data:** 160 settlements, 4 days

## TL;DR

The short and long legs have different risk profiles and should be sized independently.
Removing the old `LONG_NOTIONAL_MAX = $1000` cap and letting the long leg size to its
own depth-based tiers jumped revenue from **$116 → $130/day** (+12%).

## Before vs After

| Config | Short $/day | Long $/day | **Total $/day** |
|--------|------------|-----------|----------------|
| Old (long capped at $1000, same as short) | $72.5 | $43.7 | $116.2 |
| **New (independent sizing, long up to $1500)** | **$72.5** | **$57.6** | **$130.1** |
| Delta | — | **+$13.9** | **+$13.9 (+12%)** |

## Why Different Sizes?

### Short leg: proven 100% WR, conservative sizing is fine
- Tiers: $500, $1000, $2000, $3000
- Cap: 15% of depth_20
- Edge is stable — slippage is the main concern

### Long leg: 65% WR, more variance, benefits from larger size on deep books
- Tiers: $250, $500, $750, $1000, $1500
- Cap: 15% of depth_20 (same rule, different tiers)
- On shallow books ($2-5K depth): $250-500 is enough
- On deep books ($10-25K depth): $1000-1500 is optimal with 76% WR

## Grid Search Results

| Short | Long | N | Short $/d | Long $/d | Total $/d | Long WR |
|-------|------|---|-----------|----------|-----------|---------|
| $1500 | $1500 | 55 | $41.3 | $41.5 | **$82.8** | 78% |
| $2000 | $2000 | 42 | $40.6 | $36.2 | $76.8 | 74% |
| $1000 | $1000 | 73 | $37.5 | $33.2 | $70.7 | 75% |
| $1000 | $1500 | 55 | $29.1 | $41.5 | $70.6 | 78% |

Best absolute combo is $1500/$1500 — but adaptive sizing (depth-dependent) beats any
fixed size because it trades MORE settlements.

## Long as Fraction of Short (with adaptive short)

| Ratio | N | Short $/d | Long $/d | Total $/d | Long WR |
|-------|---|-----------|----------|-----------|---------|
| 0% | 109 | $71.4 | $0.0 | $71.4 | — |
| 25% | 109 | $71.4 | $14.9 | $86.4 | 75% |
| 50% | 109 | $71.4 | $33.9 | $105.3 | 72% |
| 75% | 109 | $71.4 | $46.3 | $117.7 | 72% |
| **100%** | **109** | **$71.4** | **$59.7** | **$131.1** | **71%** |
| 150% | 109 | $71.4 | $60.4 | $131.9 | 71% |

Revenue plateaus at 100-150% of short size — WR stays stable (71%).
Going above 100% doesn't help much because the long leg's edge (recovery bps)
is the same regardless of size; only slippage changes.

## Depth-Dependent Long Sizing

| Depth Bucket | Optimal Long $ | Long WR | Rationale |
|-------------|---------------|---------|-----------|
| $2-5K | $250-500 | 56-67% | Shallow book, higher slip risk |
| $5-10K | $500-750 | 62% | Moderate depth |
| **$10-25K** | **$1000-1500** | **76%** | Deep book, low slip |
| >$25K | $1500 | 100% | Unlimited depth (N=5, small sample) |

## Production Rules

```python
# Short leg sizing (unchanged)
short_notional = adaptive_size(depth_20, cap=0.15, tiers=[500, 1000, 2000, 3000])

# Long leg sizing (INDEPENDENT)
long_notional = adaptive_size(depth_20, cap=0.15, tiers=[250, 500, 750, 1000, 1500])
```

Key change: the long leg uses its own tier list with finer granularity ($250, $750 steps)
and a higher effective max ($1500 vs old $1000 cap).

## Revenue Stack (updated)

```
Short-only:                                  $72.5/day
 + Long (fixed exit, old $1000 cap):         $109.5/day
 + Long (ML exit, old $1000 cap):            $116.2/day
 + Long (ML exit, independent sizing):       $130.1/day  ← CURRENT BEST
```
