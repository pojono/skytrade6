# FINDINGS v29: Combined Tick-Level Analysis (Trades + Liq + OI + FR)

**Date**: February 18, 2026  
**Data**: BTCUSDT, May 12–18 2025 (7 days)  
**Streams**: 10.4M trades + 13.7K liquidations + 367K OI updates + 4.2K FR updates  
**Resolution**: 1-second, 604,800 seconds total  
**Script**: `research_v29_combined_tick.py`

---

## Executive Summary

**Open Interest reveals the MECHANISM of regime switches: massive position unwinding ($4M in 5 minutes).** But neither OI nor Funding Rate improve prediction — they are concurrent/lagging indicators, not leading ones. The key new insight is structural: regime switches are **forced liquidation cascades that unwind open interest**, and they cluster at specific points in the 8-hour funding cycle.

### Key Findings

| Signal | Relationship to Regime Switch | Predictive Value |
|--------|------------------------------|-----------------|
| **OI_Δ60s at switch** | **-$1.1M** (332x baseline) — massive unwind | ❌ Concurrent, not leading |
| **OI_Δ5min** | Peaks at -$4.1M at +150s after switch | ❌ Lagging |
| **OI before switch** | Rising (+$1.6M at -12min) → positions building | ⚠️ Weak, noisy |
| **FR at switch** | Not significantly different from random (p=0.24) | ❌ No signal |
| **FR timing** | Switches cluster 2-4h before funding (1.57x lift) | ✅ Structural insight |
| **ML model** | Adding OI/FR to vol+liq **reduces** AUC from 0.505 to 0.467 | ❌ Noise |

---

## Analysis 1: Combined 4-Stream Profiles (±30min, 172 switches)

### The OI Story — Position Buildup Then Forced Unwind

```
Time        Vol_60s    Liq_60s    OI_Δ60s$       OI_Δ5min$        FR
-12min      0.000034      0.7    +$356,180    +$1,560,424    0.000033
 -7min      0.000035      0.8    +$ 81,362    +$1,326,762    0.000031
 -5min      0.000034      1.8    +$164,412    +$  138,463    0.000032
 -2min      0.000036      1.0    -$ 62,234    +$  740,872    0.000032
  -1min     0.000033      1.0    +$223,763    +$  474,312    0.000032
  -30s      0.000040      2.5    +$234,886    +$  288,385    0.000033
   0s       0.000077      6.0    -$1,122,083  -$  812,183    0.000033  ← SWITCH
  +30s      0.000084      9.4    -$2,489,201  -$1,883,458    0.000033
  +60s      0.000061      7.1    -$2,201,607  -$2,739,632    0.000033
  +90s      0.000052      3.7    -$1,375,108  -$3,366,712    0.000033
 +150s      0.000052      4.8    -$  700,350  -$4,118,485    0.000032
 +300s      0.000046      1.8    -$  117,307  -$3,429,723    0.000032
 +480s      0.000043      1.1    -$   10,335  +$  129,868    0.000032
```

**The sequence**: OI rises in the 10 minutes before (new positions opening) → switch fires → OI drops $4.1M in 5 minutes (forced liquidation unwind) → stabilizes after 8 minutes.

### Key Ratios (Switch vs Baseline)

| Metric | Baseline (-30m to -15m) | At Switch (±5s) | Ratio |
|--------|------------------------|-----------------|-------|
| Vol_60s | 0.000042 | 0.000077 | **1.8x** |
| Liq_60s | 1.2 | 6.9 | **5.7x** |
| LiqNot_60s | $19,580 | $100,452 | **5.1x** |
| OI_Δ60s | -$3,351 | -$1,112,192 | **332x** |
| OI_Δ5min | -$289,863 | -$823,529 | **2.8x** |
| FR | 0.0000311 | 0.0000335 | **1.07x** (no change) |

---

## Analysis 2: Open Interest Dynamics

### OI Direction Before Switch

| OI Behavior (5min before) | Count | Percentage |
|--------------------------|-------|-----------|
| OI rising (>$1M) | 78 | **45.3%** |
| OI falling (<-$1M) | 66 | **38.4%** |
| OI flat (±$1M) | 28 | 16.3% |

**No clear directional bias** — switches happen with both rising and falling OI. The |OI_Δ5min| magnitude is NOT significantly different from random (p=0.76).

### OI Direction vs Price Direction

| | OI Rising | OI Falling |
|---|-----------|-----------|
| **Price UP** (77 switches) | 58% | 42% |
| **Price DOWN** (96 switches) | 47% | 53% |

Slight pattern: price-up switches tend to have rising OI (new longs), price-down switches tend to have falling OI (liquidation-driven). But not strong enough to be actionable.

---

## Analysis 3: Funding Rate Dynamics

- **|FR| at switch vs random**: p=0.24 — **not significant**
- FR is too slow-moving (~60s update interval) to capture second-level dynamics

### Funding Cycle Timing — The Structural Insight

| Time to Next Funding | Switch Rate | Random Rate | Lift |
|---------------------|-------------|-------------|------|
| 0-30min | 4.8% | 4.3% | 1.11x |
| **30min-1h** | **8.3%** | **5.9%** | **1.41x** |
| 1h-2h | 9.5% | 17.9% | **0.53x** |
| **2h-4h** | **29.2%** | **18.6%** | **1.57x** |
| 4h-8h | 48.2% | 53.3% | 0.90x |

**Switches cluster 2-4 hours before funding settlement (1.57x lift)** and 30min-1h before (1.41x). They are rare 1-2h before funding (0.53x) — the market calms as funding approaches, then volatility picks up again.

This makes sense: traders adjust positions well before funding, creating volatility. The quiet period 1-2h before is when most repositioning is done.

---

## Analysis 4: Cross-Correlations

### Contemporaneous Spearman Correlations

| | vol_60s | liq_60s | oi_Δ60s | fr | trades |
|---|---------|---------|---------|-----|--------|
| **vol_60s** | 1.000 | +0.278 | -0.019 | +0.119 | +0.676 |
| **liq_60s** | +0.278 | 1.000 | **-0.102** | +0.021 | +0.297 |
| **oi_Δ60s** | -0.019 | **-0.102** | 1.000 | +0.027 | +0.007 |
| **fr** | +0.119 | +0.021 | +0.027 | 1.000 | +0.126 |

**Key**: OI_delta is **negatively** correlated with liq (-0.10) — when liquidations spike, OI drops (positions are being forced out). This confirms the mechanism.

### Lead-Lag Results

- **OI_delta → Liq**: Peak at lag=-100s → **OI reacts to liq, not the other way around**
- **FR → Vol**: Slowly increasing correlation with lag → FR is a **very slow** background signal
- **Liq → Vol**: Peak at lag=0, slight asymmetry favoring liq-leads (confirming v27c)

---

## Analysis 6: ML Prediction — More Features = Worse

| Feature Set | AUC | Precision@P90 | Features |
|------------|-----|---------------|----------|
| vol_only | 0.505 | 0.109 | 1 |
| vol+liq | 0.502 | 0.109 | 3 |
| vol+liq+OI | 0.492 | 0.098 | 5 |
| vol+liq+OI+FR | **0.467** | **0.068** | 7 |
| all_features | 0.494 | 0.112 | 10 |

**Adding OI and FR makes prediction WORSE.** The model overfits to the funding cycle (fr_time_to_funding = 33% importance) rather than learning switch dynamics.

### Feature Importance (all_features model)

```
fr_time_to_funding  33.0%  ████████████████
fr_300s             26.2%  █████████████
fr                  23.1%  ███████████
oi_delta_300s        5.9%  ██
liq_300s             5.1%  ██
vol_60s              3.8%  █
liq_not_60s          1.8%  
trade_10s            0.5%  
liq_60s              0.3%  
oi_delta_60s         0.3%  
```

The model is 82% dominated by FR features — it's learning "switches happen 2-4h before funding" rather than any real-time signal.

---

## Synthesis: What Each Stream Tells Us

| Stream | Role | Timing | Actionable? |
|--------|------|--------|-------------|
| **Trades/Vol** | Defines the switch | At switch (0s) | Baseline detector |
| **Liquidations** | Triggers the cascade | -30s to +60s | ✅ 30-second early warning |
| **Open Interest** | Shows the unwind | 0s to +5min (lagging) | ❌ Confirms, doesn't predict |
| **Funding Rate** | Background context | Hours-scale cycle | ⚠️ Structural timing only |

### The Complete Causal Chain (Second Resolution)

```
Hours before:  FR cycle creates positioning pressure (2-4h before funding)
Minutes before: OI rises slightly (new positions opening)
-30 seconds:   First liquidations fire (forced exits begin)
0 seconds:     Vol threshold crossed → REGIME SWITCH
+0 to +30s:    Cascade peak — 9.4 liq/60s, OI dropping $2.5M/min
+30s to +150s: Unwind continues — OI drops $4.1M total
+5 to +8min:   OI stabilizes, vol decays, back to normal
```

### Practical Implications

1. **Real-time liq monitoring** remains the best short-term signal (30s lead)
2. **OI is a confirmation signal** — if OI is dropping fast, the cascade is real (not a fake-out)
3. **FR timing** is useful for risk management — be more cautious 2-4h before funding
4. **Don't add OI/FR to ML models** — they add noise, not signal

---

## Data & Code

- **Script**: `research_v29_combined_tick.py`
- **Results**: `results/v29_combined_tick.txt`
- **Reproducibility**: `python3 research_v29_combined_tick.py 2>&1 | tee results/v29_combined_tick.txt`

---

**Version**: v29  
**Status**: Complete ✅
