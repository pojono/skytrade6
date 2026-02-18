# FINDINGS v28: Long-Horizon Liquidation Buildup Before Regime Switches

**Date**: February 18, 2026  
**Data**: BTCUSDT, May 12–18 2025 (7 days), 10.4M trades + 13.7K liquidations  
**Resolution**: 1-second, ±2 hours around 173 regime switches (min_gap=30min)  
**Script**: `research_v28_liq_buildup.py`

---

## Executive Summary

**There is NO gradual hours-long liquidation buildup before regime switches.** The v27c finding of "liquidations lead" is real but confined to the **last 30–60 seconds** before the switch. At longer timescales, the pre-switch environment is actually **quieter than average** — regime switches erupt from calm, not from gradually escalating stress.

### The Corrected Picture

| Timescale | Finding | Statistical Significance |
|-----------|---------|------------------------|
| **-2h to -5min** | Liq rate is **0.4–0.5x of random** (BELOW average) | p≈1.0 (significantly LESS) |
| **-5min to -1min** | Liq rate at baseline, no detectable signal | p>0.5 (not significant) |
| **-30s** | Liq_1min is **6.6x random** | p=0.0004*** |
| **-10s** | Liq_1min is **13.0x random** | p<0.001*** |
| **0s (switch)** | Liq_1min is **15.6x random** | p<0.001*** |

**The signal is real but SUDDEN — not gradual.**

---

## Analysis 1: Wide Profiles (±2h, 173 switches)

The raw profile shows what looks like a buildup in the `liq_1h` column:

```
Offset    vol_norm   liq_1h   l1h_norm
-2h       0.200      73.1     0.531
-50min    0.130      59.0     0.362
-40min    0.101      48.6     0.211
-20min    0.122      56.1     0.303
-10min    0.179      52.3     0.303
 -1min    0.029      55.0     0.326
  0s      0.867      56.6     0.410  ← SWITCH
 +5min    0.296      70.7     0.765
 +8min    1.000      76.3     1.000  ← PEAK
```

**This looks like a gradual rise from -50min.** But Analysis 5 reveals this is misleading — the "rise" is from an abnormally LOW baseline, not toward an abnormally HIGH level.

---

## Analysis 5: The Key Test — Switch vs Random (Mann-Whitney U)

For each time offset before the switch, we compare the liq level at that moment to the same liq level at random non-switch times (±30min exclusion zones).

### liq_1min (sharpest signal)

| Offset | Switch Mean | Random Mean | Ratio | p-value |
|--------|------------|-------------|-------|---------|
| -2h | 2.05 | 1.84 | 1.11x | 0.79 |
| -1h | 0.47 | 2.97 | 0.16x | 0.999 |
| -30min | 0.28 | 1.74 | 0.16x | 0.996 |
| -5min | 1.72 | 0.59 | 2.90x | 0.77 |
| -2min | 1.01 | 0.70 | 1.43x | 0.20 |
| **-30s** | **2.36** | **0.36** | **6.59x** | **0.0004\*\*\*** |
| **-10s** | **4.55** | **0.35** | **12.99x** | **<0.001\*\*\*** |
| **0s** | **5.87** | **0.38** | **15.60x** | **<0.001\*\*\*** |

### liq_1h (longest timescale)

| Offset | Switch Mean | Random Mean | Ratio | p-value |
|--------|------------|-------------|-------|---------|
| -2h | 73.1 | 101.9 | **0.72x** | 1.0 (LESS) |
| -1h | 59.0 | 150.6 | **0.39x** | 1.0 (LESS) |
| -30min | 54.7 | 143.4 | **0.38x** | 1.0 (LESS) |
| -10min | 56.1 | 114.2 | **0.49x** | 1.0 (LESS) |
| 0s | 60.4 | 94.7 | **0.64x** | 1.0 (LESS) |

**At every timescale from 30min to 1h, the pre-switch environment has LESS liquidation activity than random.** The switches happen when the market is calm, then suddenly erupts.

---

## Analysis 6: Raw Event Rate Confirms Sudden Onset

```
Time Bin         Liq/sec    $/sec       Pattern
-2h→-1h         0.016      $231        baseline
-1h→-30m        0.015      $255        baseline
-30m→-15m       0.020      $336        slight uptick
-15m→-10m       0.013      $150        back to baseline
-10m→-5m        0.014      $129        baseline
-5m→-2m         0.011      $171        baseline
-2m→-1m         0.016      $96         baseline
-1m→-30s        0.057      $854        ← 3.5x spike starts here
-30s→-10s       0.132      $1,548      ← 8x
-10s→0s         0.145      $2,896      ← 9x
0s→+10s         0.565      $7,793      ← 35x PEAK (cascade)
+10s→+30s       0.069      $876        ← rapid decay
+30s→+1m        0.079      $1,285
+1m→+2m         0.057      $1,065
+2m→+5m         0.055      $845
+5m→+10m        0.020      $290        back to baseline
```

**The liq rate is flat at ~0.015/sec from -2h to -1min.** Then it spikes 3.5x at -60s, 9x at -10s, and 35x at the switch. This is NOT a gradual buildup — it's a sudden cascade.

---

## Analysis 4: P(Switch | Liq Level) — P99 is the Only Signal

| Liq_15m Range | P(switch in 1min) | Lift | P(switch in 15min) | Lift |
|--------------|-------------------|------|-------------------|------|
| P0-P50 | 2.1% | 1.1x | 36.5% | 1.2x |
| P50-P75 | 2.2% | 1.1x | 28.3% | 0.9x |
| P75-P90 | 1.4% | 0.7x | 16.3% | **0.5x** |
| P90-P95 | 1.7% | 0.8x | 31.9% | 1.1x |
| P95-P99 | 1.8% | 0.9x | 28.2% | 0.9x |
| **P99-P100** | **3.5%** | **1.7x** | **46.5%** | **1.6x** |

Only the **P99 extreme** (top 1% of liq activity) has meaningful predictive lift. The P75-P90 range actually has **negative** lift — moderate liq activity is associated with FEWER switches.

---

## Why the v27c "Leading Indicator" Conclusion Was Partially Wrong

### What v27c got right:
- Liquidations DO spike 30–60 seconds before the vol threshold crossing
- The causal chain (liq → price impact → vol) is real at the second level

### What v27c got wrong:
- The "97% liq spikes first at P75" finding was an artifact of the P75 threshold being too low — most of the time, liq_count_60s P75 = 0, so ANY liquidation counts as "elevated"
- The "10-minute median lead" was measuring how long liq stays above a near-zero threshold, not a genuine early warning
- There is NO gradual buildup over minutes or hours

### The correct model:
1. Market is in a **quiet state** (below-average liq activity)
2. An **exogenous shock** hits (news, whale order, stop cascade)
3. **Within 30–60 seconds**: first liquidations fire, creating price impact
4. **Within 10 seconds**: vol threshold crossed, regime switch labeled
5. **Within 60 seconds**: full cascade — 35x normal liq rate
6. **Within 5 minutes**: cascade exhausts, rate returns to normal

The signal is **real but has ~30 seconds of lead time, not 10 minutes.**

---

## Practical Implications

### What works:
- **Real-time liq monitoring at 1-second resolution**: if liq_1min > 6x normal in the last 30 seconds, a regime switch is likely imminent
- **P99 extreme liq events**: 1.7x lift for 1-min prediction, 1.6x for 15-min
- **Post-switch cascade trading**: the 35x spike at 0s decays over 5 minutes — mean reversion strategies (v26c) exploit this

### What doesn't work:
- **Hours-ahead prediction from liq data**: the pre-switch environment is quieter than average
- **Gradual buildup detection**: there is no gradual buildup
- **Moderate liq levels as warning**: P75-P90 liq actually predicts FEWER switches

### Revised Architecture:

```
Layer 1: Regime Detection (vol features, HMM)
  → Standard approach, no change needed

Layer 1.5: Liq Spike Alert (NEW — 30-second horizon only)
  → Monitor liq_1min in real-time
  → If liq_1min > 6x rolling baseline: HIGH ALERT
  → Lead time: ~30 seconds (not minutes)
  → Use for: immediate position sizing, stop tightening

Layer 2: Post-Cascade Trading (v26c strategies)
  → Cascade Fade, ToD Fade
  → Triggered by the 35x spike, not the pre-spike
```

---

## Data & Code

- **Script**: `research_v28_liq_buildup.py`
- **Results**: `results/v28_liq_buildup.txt`
- **Reproducibility**: `python3 research_v28_liq_buildup.py 2>&1 | tee results/v28_liq_buildup.txt`

---

**Version**: v28  
**Status**: Complete ✅
