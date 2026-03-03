# Multi-Coin Stress Event Analysis — Findings

## Objective

Test whether the stress-event → retrace → continuation mechanism discovered on DOGE
generalizes across 8 coins with diverse microstructure, and whether longer horizons
(15m–60m) reveal tradeable continuation or regime persistence.

## Coins Tested

| Category | Symbols |
|----------|---------|
| High-meme | 1000RATSUSDT, 1000BONKUSDT, 1000TURBOUSDT |
| Mid-cap | ARBUSDT, APTUSDT, ATOMUSDT |
| Experimental | ARCUSDT, AIXBTUSDT |

## Stage-1: Event Frequency

| Symbol | Stage-1 Events | Events/Day | Enriched | Spread med (bp) |
|--------|---------------|------------|----------|-----------------|
| 1000BONKUSDT | 5,079 | 169.3 | 2,482 | 1.6 |
| ARBUSDT | 4,512 | 150.4 | 1,498 | 16.6 |
| APTUSDT | 1,988 | 66.3 | 617 | 11.8 |
| ATOMUSDT | 1,100 | 36.7 | 391 | 14.9 |
| AIXBTUSDT | 120 | 4.0 | 73 | 3.1 |
| 1000RATSUSDT | 58 | 1.9 | 25 | 9.1 |
| ARCUSDT | 41 | 1.4 | 22 | 9.4 |
| 1000TURBOUSDT | 25 | 0.8 | 13 | 2.9 |

**Observations:**
- High-cap coins (BONK, ARB) produce 150+ events/day — mostly noise
- Low-cap (RATS, ARC, TURBO) produce 1–2/day — rare but potentially meaningful
- ~50% of Stage-1 events survive the enrichment quality filter (valid spread at entry, OB coverage for 1h horizon)
- Refill detection rate: 85–100% across all coins

## Cross-Coin Comparison: Direction-Signed Returns

Returns are signed so that **positive = continuation** (event direction), **negative = MR**.

| Symbol | N | 5s | 60s | 5m | 15m | 60m | Cont% 60m | Tail 60m | Regime 60m% |
|--------|---|-----|------|------|-------|-------|-----------|----------|-------------|
| **1000RATSUSDT** | **25** | **-2.5** | **-5.3** | **+14.1** | **+2.7** | **+28.3** | **56%** | **2.04x** | **50%** |
| 1000BONKUSDT | 2,482 | -0.2 | +0.2 | +0.5 | -1.5 | -0.1 | 50% | 1.02x | 49% |
| 1000TURBOUSDT | 13 | +9.0 | -8.3 | -28.2 | -45.6 | -13.4 | 38% | 0.63x | 12% |
| ARBUSDT | 1,498 | +0.0 | +1.0 | -1.0 | +2.2 | -5.1 | 47% | 1.05x | 47% |
| APTUSDT | 617 | +0.0 | +0.0 | -1.1 | +2.2 | +2.2 | 51% | 1.16x | 56% |
| ATOMUSDT | 391 | +0.0 | -1.1 | +0.0 | -4.5 | +4.3 | 54% | 1.31x | 52% |
| ARCUSDT | 22 | -7.0 | -29.8 | -28.5 | -20.7 | -33.7 | 50% | 0.49x | 67% |
| AIXBTUSDT | 73 | +0.4 | +0.0 | -3.9 | +0.5 | +8.1 | 53% | 1.21x | 62% |

## Key Findings

### Finding 1: High-volume coins show ZERO signal at all horizons

BONK (2,482 events), ARB (1,498), APT (617) all show median returns within ±5bp
at every horizon from 5s to 60m. Continuation rate ~50%. Tail asymmetry ~1.0x.

**These are pure noise.** The stress events on high-volume coins are just normal
volatility fluctuations that trigger the detector but carry no directional information.

### Finding 2: Low-volume coins show suggestive patterns but tiny samples

- **RATS** (25 events): +28bp median at 60m, tail asymmetry 2.04x — meets both criteria.
  But N=25 is too small for statistical significance (SE ≈ 40bp).
- **TURBO** (13 events): strong MR pattern (-45bp at 15m, only 12% regime persistence at 60m).
  But N=13 is meaningless.
- **ARC** (22 events): strong MR (-34bp at 60m). N=22.

### Finding 3: The mechanism is universal but carries no edge

Across all coins:
- **Touch t0 within 60s**: 92–100% (same as DOGE)
- **Refill detection**: 85–100%
- **MFE/MAE symmetric**: MFE ≈ MAE at 60s across all coins

The shock → vacuum → refill → retrace lifecycle exists everywhere.
But the retrace is a **return to equilibrium**, not a tradeable overshoot.

### Finding 4: Spread is the binding constraint

| Regime | Coins | Spread med | Signal |
|--------|-------|------------|--------|
| Tight (≤3bp) | BONK, TURBO, AIXBT | 1.6–3.1bp | None (BONK), tiny sample (TURBO) |
| Medium (5–15bp) | RATS, ARC, APT, ATOM | 9–15bp | ±0bp (APT, ATOM), tiny sample (RATS) |
| Wide (>15bp) | ARB | 16.6bp | None |

No spread regime produces a consistently tradeable signal at sufficient sample size.

## Pass/Fail Criteria

Criteria: AUC ≥ 0.58 on 1h target, or median ret_60m ≥ 25bp, or tail asymmetry > 2x.

| Symbol | med_ret≥25bp | tail≥2x | AUC≥0.58 | Verdict |
|--------|-------------|---------|----------|---------|
| 1000RATSUSDT | ✅ (+28bp) | ✅ (2.04x) | N too small | **PASS (conditionally)** |
| 1000BONKUSDT | ❌ (-0.1bp) | ❌ (1.02x) | N/A | **FAIL** |
| 1000TURBOUSDT | ❌ (-13bp) | ❌ (0.63x) | N too small | **FAIL** |
| ARBUSDT | ❌ (-5.1bp) | ❌ (1.05x) | N/A | **FAIL** |
| APTUSDT | ❌ (+2.2bp) | ❌ (1.16x) | N/A | **FAIL** |
| ATOMUSDT | ❌ (+4.3bp) | ❌ (1.31x) | N/A | **FAIL** |
| ARCUSDT | ❌ (-34bp) | ❌ (0.49x) | N too small | **FAIL** |
| AIXBTUSDT | ❌ (+8.1bp) | ❌ (1.21x) | N/A | **FAIL** |

**Only RATS passes**, and only because N=25 has huge variance. The +28bp median
could easily be noise (95% CI roughly [-50bp, +110bp]).

## Conclusion

### The answer: **it's the hypothesis, not the instrument**

The multi-coin test decisively shows that stress-event detection + causal retrace entry
produces zero edge across diverse microstructure regimes:

- **High-frequency coins** (BONK, ARB, APT): tons of events, all noise
- **Low-frequency coins** (RATS, ARC, TURBO): few events, unstable estimates
- **No coin** shows a robust, statistically significant continuation or MR signal at any horizon

The stress event detector works (it finds real microstructure events), but the events
carry no **directional information** beyond the immediate retrace to equilibrium.

### What this rules out

1. MR-based entry on retrace (closed in Stage-4.1)
2. Continuation-based entry on shock direction (no signal at 5s–60m)
3. Hour-horizon regime persistence (50% ± noise across all coins)
4. The idea that "memes are different" — BONK has 2,482 events and zero signal

### What remains possible

1. **Conditioning on external context** (funding rate, OI change, macro regime) might
   filter events into subsets with directional information — but this is a different
   hypothesis entirely
2. **Using the state engine as a filter/risk tool** — avoid entering other strategies
   during Shock states
3. **Cross-venue latency** — if the same shock propagates between exchanges with
   measurable delay
