# Stage-2: Vacuum Confirmation — Findings

## Summary

For each of the 8,669 Stage-1 events, we measured what happens in the 5s and 15s
**after** t0: did the liquidity vacuum persist? Did flow continue? Did price follow?

**Key finding:** VacuumScore (depth drop × flow continuation) cleanly separates
events into CONTINUATION (follow the flow) vs REVERSAL (fade the flow).

## Data Quality

- **7,527 clean events** (87%) with reliable mid-price
- **981 excluded** (11.3%) — spread blowout artifacts where spread=0 and mid=half_price
  (these are real extreme stress events but mid-price is unreliable for return measurement)

## VacuumScore Definition

```
VacuumScore = clip(1 - depth_drop_5s, 0, 1) × continuation_5s
```

Where:
- `depth_drop_5s` = min(TopDepth in [t0, t0+5s]) / TopDepth(t0)
  - <1 means depth dropped, >1 means depth recovered
- `continuation_5s` = same-direction notional / total notional in [t0, t0+5s]
  - 1.0 = all flow continued in trigger direction
  - 0.0 = flow reversed

High VacuumScore = depth vacuum opened AND flow continues pushing through it.

## VacuumScore → Returns (7,527 clean events)

| Quartile | VS Range       | n     | ret_5s_med | ret_5s_mean | WR    | Interpretation   |
|----------|---------------|-------|------------|-------------|-------|------------------|
| Q0       | [0, 0.025]    | 3,011 | +0.4 bp    | +0.7 bp     | 52.8% | MIXED (no signal)|
| Q1       | [0.025, 0.11] | 1,882 | -0.9 bp    | -1.2 bp     | 37.9% | **REVERSAL**     |
| Q2       | [0.11, 0.44]  | 1,505 | +0.3 bp    | +0.3 bp     | 52.4% | MIXED            |
| Q3       | [0.44, 1.00]  | 1,882 | **+1.2 bp**| **+2.0 bp** |**64.7%**| **CONTINUATION** |

### Interpretation

- **Q3 (HighVS):** Depth dropped, flow continues → price continues in flow direction.
  65% win rate, +1.2 bp median per event, 63 events/day. This is **real continuation**.
- **Q1 (LowVS):** Depth dropped slightly but flow reversed → price mean-reverts.
  Only 38% WR in flow direction → **62% WR if you FADE** (enter opposite direction).

## Stability Across Weeks (Q3 only)

| Week | n   | ret_5s median | ret_5s mean | WR  |
|------|-----|---------------|-------------|-----|
| 36   | 327 | +1.1 bp       | +2.2 bp     | 65% |
| 37   | 608 | +1.7 bp       | +2.8 bp     | 69% |
| 38   | 506 | +0.8 bp       | +1.7 bp     | 63% |
| 39   | 351 | +1.1 bp       | +1.5 bp     | 62% |
| 40   | 90  | +0.8 bp       | +0.0 bp     | 56% |

Signal is consistent across 4+ weeks (Week 40 partial).

## FlowImpact × VacuumScore (2D Analysis)

|           | FI [1-2) low | FI [2-5) mid | FI [5+) high |
|-----------|-------------|-------------|-------------|
| **VS Q0** | -0.6 bp (42%) | +0.2 bp (51%) | +0.9 bp (56%) |
| **VS Q1** | -0.4 bp (45%) | +0.1 bp (50%) | -0.4 bp (45%) |
| **VS Q2** | +0.6 bp (55%) | +0.4 bp (53%) | +1.4 bp (62%) |
| **VS Q3** | **+1.3 bp (66%)** | **+1.4 bp (67%)** | **+3.6 bp (71%)** |

The VS signal is stronger than FI alone. VS Q3 + FI high = **+3.6 bp median, 71% WR**.

## Top Decile VacuumScore Profile

- **753 events** (VS ≥ 0.627), ~25/day
- depth_drop_5s median: **0.17** (depth collapsed to 17% of pre-event level)
- continuation_5s median: **0.98** (98% of flow in same direction)
- ret_5s: **+1.4 bp median, +2.3 bp mean, 69% WR**
- ret_15s: **+1.5 bp median, +2.3 bp mean, 61% WR**

## Net After Fees (20 bps RT taker)

| Strategy           | n/day | Gross 5s      | Net 5s        | WR net |
|--------------------|-------|---------------|---------------|--------|
| All events         | 251   | +0.2/+0.5 bp  | -19.8/-19.5 bp| 3%     |
| HighVS (Q3)        | 63    | +1.2/+2.0 bp  | -18.8/-18.0 bp| 4%     |
| TopDecile VS       | 25    | +1.4/+2.3 bp  | -18.6/-17.7 bp| 4%     |
| HighVS+HighFI      | 9     | +1.2/+3.2 bp  | -18.8/-16.8 bp| 6%     |

**The 5s/15s horizons are NOT tradable at taker fees.**

The edge exists but is 1-3 bps — dwarfed by 20 bps round-trip costs.

## What This Means

### The signal IS real:
- VacuumScore cleanly separates continuation vs reversal
- 65-69% WR on direction prediction is genuine
- Stable across weeks

### But the magnitude is too small for 5s horizon:
- 1-3 bps gross on DOGE is $0.01-0.03 per $1000 notional
- 20 bps RT fee = $2.00 per $1000 notional

### Possible next directions:
1. **Longer horizons** — does the continuation grow at 30s, 60s, 5min?
2. **Larger coins** (BTC, ETH) — deeper book, might amplify the effect
3. **Use as filter** — combine with other signals (FR, regime) not standalone
4. **Maker entry** — reduce fee to 4 bps RT, then +1.2 bp gross might work
5. **Event quality filter** — the 981 "book wipe" events might have the biggest moves
