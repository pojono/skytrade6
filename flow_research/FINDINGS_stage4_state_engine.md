# Stage-4: Online State Engine — Execution Realism

## Summary

Built a 100ms-resolution online state engine for all 8,669 Stage-1 events.
Computed causal features (ExhaustionScore, RefillScore, OneSidedScore),
ran state machine, evaluated separation against outcomes, and simulated
both taker and maker trade scenarios.

**Bottom line: the retrace mechanism is real and detectable, but NOT tradeable.**
The retrace is transient — by any fixed exit horizon, continuation resumes
and the MR edge vanishes to ~0 bp median.

## Data

- 8,666 events processed (3 failed due to missing pre-shock OB)
- 7,751 clean events (|ret| < 200bp, 89.4% — rest are blown-spread contamination)
- Pre-shock depth baseline: valid-spread OB snapshot within 5s before t0
- Scores normalized to [0,1], all features strictly causal (data ≤ t_k only)

## Score Distributions

| Score | Median | p25 | p75 |
|-------|--------|-----|-----|
| exhaust_max | 0.749 | 0.697 | 0.845 |
| exhaust_mean | 0.384 | 0.316 | 0.457 |
| refill_max | 0.995 | 0.976 | 1.000 |
| refill_mean | 0.709 | 0.630 | 0.761 |
| onesided_max | 1.000 | 1.000 | 1.000 |

## State Transition Timing

| Transition | Detection rate | Median time | p25 | p75 |
|-----------|---------------|-------------|-----|-----|
| Exhaustion | 87.9% | 1.60s | 0.90s | 2.60s |
| Refill | 98.3% | 0.50s | 0.30s | 1.00s |

Final state at 5s: 78% RetraceLeg, 11% Continuation, 4% Refill, 4% Shock, 3% Exhaustion.

## Separation Analysis

### AUC for predicting TouchT0_60s

| Feature | AUC | p-value |
|---------|-----|---------|
| refill_mean | **0.607** | <0.0001 |
| depth_recovery_1s | **0.592** | <0.0001 |
| refill_max | 0.547 | <0.0001 |
| exhaust_max | 0.507 | 0.57 (NS) |
| flow_decay_1s | 0.502 | 0.85 (NS) |

**Verdict: AUC 0.59–0.61.** Exceeds the 0.58 threshold from the ТЗ, but barely.
Exhaustion score and flow_decay have NO separation power.

### Uplift: WR for MR conditioned on early refill

| Feature | High quartile | Low quartile | Uplift |
|---------|-------------|-------------|--------|
| refill_max | 87.4% | 69.8% | **+17.6%** |
| depth_recovery_1s | 89.8% | 72.9% | **+16.9%** |
| exhaust_max | 77.4% | 85.2% | -7.8% (inverted!) |

### MR move by refill timing

| Refill timing | n | MR_max median | P(MR>10bp) | P(MR>20bp) |
|--------------|------|-------------|-----------|-----------|
| Fast (0–1s) | 6,090 | +29.7 bp | 84.6% | 70.2% |
| Mid (1–2s) | 1,700 | +22.2 bp | 75.6% | 54.8% |
| Late (2–3s) | 461 | +19.3 bp | 70.1% | 48.2% |
| Very late (3–5s) | 267 | +20.9 bp | 70.0% | 52.1% |
| No refill | 143 | +16.7 bp | 65.7% | — |

**MR max is large (+30bp median for fast refill).** But...

## The Critical Finding: MR is Transient

### Final returns at t0 (limit order fills during retrace)

| Horizon | Median ret_opp | Mean | WR | WR net of 4bp fee |
|---------|---------------|------|-----|-------------------|
| 60s | **+0.0 bp** | +0.1 | 49.3% | 35.2% |
| 300s | **+0.8 bp** | +0.6 | 51.4% | 43.6% |

The retrace reaches +30bp MFE but then **continuation resumes**.
By any fixed exit horizon, the edge is ~0 bp.

### By refill timing (clean, touched t0)

| Refill time | n | RetOpp 60s | WR | RetOpp 300s | WR |
|-------------|------|----------|-----|-----------|-----|
| [0, 0.3)s | 110 | +3.2 | 57% | +3.6 | 56% |
| [0.3, 0.5)s | 3,121 | +0.4 | 51% | +1.6 | 53% |
| [0.5, 1.0)s | 2,117 | -0.4 | 47% | +0.6 | 51% |
| [1.0, 2.0)s | 1,477 | -0.6 | 48% | +0.0 | 50% |
| [2.0, 5.0)s | 595 | -0.2 | 49% | -0.4 | 49% |

Only the very fastest refill (<0.3s, n=110) shows any edge — and it's
too fast to act on causally.

### Best subgroup: Q4 refill + fast refill + touched t0

- n=1,868 (~62/day)
- ret_opp_300s: median +1.6 bp, WR 52.4%
- Net of 4bp maker fee: **median -2.4 bp, WR 43.4%**
- Tail risk: p5 = -34bp, 15% of events lose >20bp

## Trade Simulations

### Scenario A: Taker retrace entry (opposite to event after Refill)

| Horizon | n | Median PnL | WR |
|---------|------|----------|-----|
| 30s | 7,463 | **-20.0 bp** | 1.7% |
| 60s | 7,463 | **-20.0 bp** | 3.9% |

With any SL (20/30/40bp): median PnL ≈ -19 to -20 bp. **Dead.**
The 20bp taker fee exceeds the entire MR move.

### Scenario B: Maker MR (limit at t0 after Refill)

- Fill rate: 39.4% (queue-adjusted)
- Filled events: median PnL = **-10 bp** at all horizons
- **Maker's curse confirmed again**: fills cluster on adverse selection events

## Why the Edge Doesn't Exist

The mechanism is real:
```
Shock → Vacuum → Refill → Retrace → Continuation
```

But the retrace is **symmetric noise** around t0, not a tradeable move:
1. MR_max reaches +30bp (real retrace exists)
2. But continuation resumes after retrace
3. Final price at any horizon ≈ t0 price ± noise
4. Edge after fees: negative for taker, ~zero for maker

The retrace is a **liquidity rebalancing event**, not a mean-reversion opportunity.
It's the market's mechanism for transferring inventory, not a price inefficiency.

## What Was Learned

### Positive findings
- Online detection of Exhaustion/Refill works (AUC 0.59–0.61)
- Refill timing predicts MR magnitude (+17.6% uplift)
- The mechanism is structurally consistent across 30 days

### Negative findings
- No causal entry point has positive EV after fees
- VacuumScore edge was entirely lookahead bias
- Queue-based fills are adversely selected
- Exhaustion score has no predictive power for outcomes
- Flow decay (500ms/500ms) has no signal at all

## Outcome Classification (per ТЗ §8)

**→ Outcome 1 confirmed (most probable scenario):**

> Online detector works (separation exists),
> but taker EV is negative due to 20bps,
> maker EV is near zero with adverse fill selection.

## Files

- `stage4_state_engine.py` — full pipeline: features, scores, state machine, sims
- `output/DOGEUSDT/events_stage4.parquet` — per-event results
