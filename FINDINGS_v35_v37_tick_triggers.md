# v35-v37: Tick-Level Entry Triggers & OOS Validation

## Overview

This document covers the evolution from blind temporal entries to smart tick-level triggers for the symmetric TP/SL strategy, culminating in a rigorous 59-day out-of-sample validation.

## v35: Smart Entry Triggers (60s Resolution)

### Hypothesis
Not every second is statistically equal for entry. Use microstructure signals to select the best moments.

### Signals Tested (TP=20/SL=10, 15m TL, weekday 13-18 UTC, 3-day in-sample)

**Trade count (TC) ratio** — ratio of recent trade count to longer-term average:
- SOL: TC>2.0 → +3.43 EV (vs +1.21 baseline) — **+183%**
- DOGE: TC>1.5 → +1.55 EV (vs +1.07 baseline) — **+45%**
- BTC: TC>1.5 → +1.23 EV (vs +0.42 baseline) — **+193%**

**Range z-score** — 60s price range vs 15m rolling norm:
- SOL: Range_z>1 → +1.71 EV — **+41%**
- XRP: Range_z>1 → +1.09 EV — **+65%**

**ETH anomaly**: Filtering HURTS ETH. TC>2.0 gives -0.40 EV. ETH mean-reverts during spikes (consistent with v31b findings). ETH needs smaller TP/SL (10/5) during normal activity.

### Discovery: Adaptive TP/SL

Testing TP/SL scaled to recent 60s range revealed a critical insight:

**BTC** (range P50 = 7.6 bps):
- Range <5 bps → TP=10/SL=5: **+1.01 EV, 73.2% WR** ← best
- Range 5-10 bps → TP=15/SL=7: **+1.03 EV, 67.3% WR** ← good
- Range 10-20 bps → TP=30/SL=15: **-1.33 EV, 51.9% WR** ← negative!
- Range 20-50 bps → TP=50/SL=25: **-7.21 EV, 36.7% WR** ← terrible!

**The edge is concentrated in low-to-moderate range environments.** When range is already high, the move is exhausting — wider TP/SL levels have negative EV. This maps to the v31 regime lifecycle: **Compression → early Breakout** is the money moment.

---

## v36: Tick-Level Triggers (1s Resolution, OI + Liquidations + Spread)

### Architecture Change
Moved from 60s rolling windows to raw 1-second resolution using actual websocket data streams:
- **Trades**: per-second count, notional, acceleration (5s vs 30s ratio)
- **Liquidations**: per-second count from hourly JSONL files (nested format)
- **Ticker**: bid/ask spread, OI changes at ~1.6s resolution

### Data Format
- Trades: `data/SYMBOL/bybit/futures/SYMBOL{date}.csv.gz`
- Liquidations: `data/SYMBOL/bybit/liquidations/liquidation_{date}_hr{HH}.jsonl.gz`
  - Nested: `result.data[].{T (ms), S (side), v (volume), p (price)}`
- Ticker: `data/SYMBOL/bybit/ticker/ticker_{date}_hr{HH}.jsonl.gz`
  - Fields: `result.data.{bid1Price, ask1Price, openInterestValue, ...}`

### In-Sample Results (3 days, May 12-14)

**SOL** (baseline +1.26):

| Trigger | N | EV | WR | Improvement |
|---------|---|-----|-----|-------------|
| **spread_z > 1** | 392 | **+2.58** | **75.3%** | +105% |
| **liq_30s>0 + tc_accel>2** | 95 | **+2.74** | **75.8%** | +118% |
| **tc_accel>2 + spread_z>1** | 46 | **+2.83** | **76.1%** | +125% |
| quiet: liq60==0 + tc_accel<1 | 2,459 | +1.42 | 71.4% | +13% |
| |oi_delta_60s| > 5 | 3,964 | +1.42 | 71.4% | +13% |

**DOGE** (baseline +1.28):

| Trigger | N | EV | WR |
|---------|---|-----|-----|
| **quiet: range10<3** | 351 | **+2.39** | **74.6%** |
| **quiet: liq60==0 + tc_accel<1** | 2,472 | **+1.69** | **72.3%** |
| spread_z > 1 | 216 | +1.67 | 72.2% |

**BTC** (baseline +0.11):

| Trigger | N | EV | WR |
|---------|---|-----|-----|
| **liq_10s > 0** | 315 | **+1.18** | **67.3%** |
| |oi_d60| > 5 | 2,718 | +0.57 | 64.8% |
| Spread signals: too few events (BTC spread ≈ 0.01 bps) |

### Key Discovery: The Bimodal Edge

Two distinct profitable regimes:

1. **Stress entries** (spread widening + liquidations + trade acceleration):
   - EV: +2.5-2.8 bps, rare (~1-2% of time)
   - Mechanism: MMs pulling liquidity → incoming cascade → fat tail

2. **Compression entries** (quiet, no liquidations, low activity):
   - EV: +1.4-1.7 bps, common (~45% of time)
   - Mechanism: positioned for next breakout → catch the expansion

3. **Mid-activity** (some liqs, moderate acceleration):
   - EV: +0.5-1.0 bps — the **worst** zone
   - The move is already partially played out

---

## v37: Out-of-Sample Validation (59 Days, 5 Symbols)

### Setup
- **In-sample**: May 12-14, 2025 (3 days)
- **OOS**: May 15 – Aug 7, 2025 (59 days for SOL, 11 stratified weekdays × 5 symbols)
- **Config**: TP=20/SL=10, 15m TL, 13-18 UTC, 1s resolution tick data
- **Coverage**: 57-62 days with all 3 data streams per symbol

### Cross-Symbol OOS Results

| Symbol | Baseline EV | spread_z>1 | liq_10s>0 | liq30+tc>2 | |oi_d60|>5 | Pos Days |
|--------|------------|-----------|----------|-----------|----------|----------|
| **SOL** | **+0.53** | **+1.53** | **+1.92** | +1.77 | +0.73 | **10/11** |
| **DOGE** | **+0.77** | +0.58 | **+1.54** | +0.56 | +0.79 | **8/10** |
| ETH | +0.35 | +0.00 | +0.70 | +0.61 | +0.55 | 8/10 |
| XRP | +0.17 | -1.23 | +0.23 | +0.84 | +0.48 | 7/10 |
| **BTC** | **-0.92** | **-4.09** | **-0.59** | **-0.43** | **-0.71** | **3/11** |

### Signal Robustness (In-Sample → OOS)

| Signal | SOL IS | SOL OOS | Degradation | Verdict |
|--------|--------|---------|-------------|---------|
| spread_z>1 | +2.58 | +1.53 | -41% | Overfitted |
| liq_10s>0 | +1.44 | +1.92 | +33% (improved!) | **Robust** |
| liq30+tc>2 | +2.74 | +1.77 | -35% | Moderate |
| |oi_d60|>5 | +1.42 | +0.73 | -49% | Weak |
| quiet:liq60=0 | +1.42 | +0.35 | -75% | Overfitted |

**`liq_10s>0` is the only signal that IMPROVED OOS** — it's genuinely robust.

### SOL Deep Dive: 59 Days

**Day of Week:**

| DOW | Days | Base EV | liq_10s>0 | Pos% |
|-----|------|---------|----------|------|
| Mon | 8 | **+1.14** | +1.44 | 88% |
| Tue | 8 | +0.54 | +1.10 | 75% |
| Wed | 9 | +0.60 | +0.45 | 67% |
| Thu | 10 | +0.61 | **+1.93** | 80% |
| Fri | 8 | +0.66 | +0.65 | 88% |
| Sat | 8 | +0.63 | +1.28 | 62% |
| Sun | 8 | +0.37 | +0.80 | 62% |

All days positive. Monday strongest. Weekday: +0.70, 79% pos. Weekend: +0.50, 62% pos.

**Week of Month:**

| Week | Days | Base EV | Pos% |
|------|------|---------|------|
| Wk1 | 17 | +0.47 | 65% |
| Wk2 | 7 | +0.43 | 71% |
| **Wk3** | 14 | **+0.84** | **79%** |
| **Wk4** | 14 | **+0.78** | **86%** |
| Wk5 | 7 | +0.64 | 71% |

Weeks 3-4 strongest. Consistent with v40 calendar anomalies.

**Month:**

| Month | Days | Base EV | Pos% |
|-------|------|---------|------|
| May | 17 | +0.76 | 88% |
| Jun | 30 | +0.64 | 70% |
| Jul | 5 | +0.54 | 80% |
| Aug | 7 | +0.47 | 57% |

Edge present in all months but slight decay over time.

### Fee Viability (OOS)

| Config | OOS EV | Net @ 0% maker | Net @ 0.005% (4 fills) | Net @ 0.01% |
|--------|--------|----------------|----------------------|-------------|
| SOL liq_10s>0 | +1.92 | **+1.92 ✅** | -0.08 ❌ | -2.08 ❌ |
| DOGE liq_10s>0 | +1.54 | **+1.54 ✅** | -0.46 ❌ | -2.46 ❌ |
| SOL baseline | +0.53 | **+0.53 ✅** | -1.47 ❌ | -3.47 ❌ |

### Fill Reduction Analysis

| Approach | Fills | Feasible? |
|----------|-------|-----------|
| Single direction | 2 | ❌ Direction unpredictable (AUC≈0.50) |
| Drop SL, hold loser | 3 | ⚠️ Unlimited risk |
| OCO/netting | 3 | ⚠️ Exchange-dependent |
| **Symmetric (current)** | **4** | **✅ Proven** |

---

## Conclusions

### What's Real
1. The symmetric TP/SL strategy has a **persistent, small edge** (+0.5-1.9 bps OOS)
2. **SOL and DOGE** are the best symbols; **BTC should be dropped**
3. **`liq_10s>0`** (liquidation in last 10 seconds) is the most robust entry trigger
4. The edge is **bimodal**: stress entries (high EV, rare) and compression entries (moderate EV, common)
5. Temporal filters help: weekday 13-18 UTC, weeks 3-4 of month

### The Fee Constraint
- Best OOS EV: +1.92 bps (SOL with liq trigger)
- 4-fill cost at 0.005% maker: 2.0 bps
- **Net: -0.08 bps** — does not survive any positive maker fee
- Only viable at **true 0% maker fees**

---

## v38: Combined Scoring Model (Continuous Probability)

### Motivation
In production, the code runs every second and computes P(win) — a continuous probability, not a binary trigger. We should combine all features into a single score and trade only when the score exceeds a threshold.

### Setup
- **Train**: First half of OOS weekdays (SOL: 21 days May 15 – Jun 12, DOGE: 20 days)
- **Test**: Second half (SOL: 22 days Jun 13 – Aug 7, DOGE: 21 days)
- **Features** (13 continuous): tc_5s, tc_accel, tn_accel, liq_5s, liq_10s, liq_30s, liq_60s, spread_z, |oi_d30|, |oi_d60|, range_10s, range_30s, spread_raw
- **Target**: Binary win/loss (pnl > 0)
- **Models**: Logistic Regression, LightGBM, Simple Weighted Score

### SOL Results

**LightGBM:**

| Threshold | Train N | Train EV | Test N | Test EV | Test WR |
|-----------|---------|----------|--------|---------|---------|
| 0.50 | 37,665 | +0.84 | 38,845 | +0.65 | 68.4% |
| 0.70 | 16,750 | +4.78 | 14,774 | +0.57 | 68.2% |
| 0.75 | 2,656 | +7.39 | 2,926 | **+1.08** | 69.8% |
| **0.80** | 182 | +8.68 | **232** | **+2.24** | **74.1%** |

Massive train-test gap at high thresholds = overfitting. The model memorizes train patterns.

**Logistic Regression:**

| Threshold | Test N | Test EV | Test WR |
|-----------|--------|---------|---------|
| 0.70 | 3,224 | +0.86 | 69.4% |
| **0.75** | **92** | **+1.85** | **72.8%** |

Less overfitting but very few trades.

**Simple Weighted Score** (features × PnL-correlation weights, no ML):

| Percentile | Test N | Test EV | Test WR |
|-----------|--------|---------|---------|
| P0 (all) | 39,600 | +0.63 | 68.3% |
| P75 | 10,669 | +0.60 | 68.5% |
| **P95** | **2,115** | **+1.11** | **70.3%** |

Most honest — minimal train/test gap. ~96 trades/day.

**Feature-PnL correlations** (SOL):
- liq_5s: +0.010 (strongest)
- liq_10s: +0.009
- tc_accel: +0.008
- range_30s: +0.008
- spread_z: +0.006

All correlations ≤ 0.01 — the signal is real but extremely weak per-feature.

### DOGE Results

**LightGBM:**

| Threshold | Train EV | Test N | Test EV |
|-----------|----------|--------|---------|
| 0.50 | +0.90 | 37,799 | +0.75 |
| 0.70 | +4.83 | 27,223 | +0.86 |
| 0.80 | +9.28 | 1,170 | +0.62 |

DOGE shows pure overfitting — test EV stays flat regardless of threshold.

**Simple Weighted Score:**

| Percentile | Test N | Test EV |
|-----------|--------|---------|
| P75 | 8,739 | **+1.18** |
| P90 | 3,410 | **+1.28** |
| P95 | 1,692 | **+1.32** |

Simple score works better than ML for DOGE. P90 gives +1.28 with ~162 trades/day.

### Key Findings

1. **ML models overfit badly** — LightGBM memorizes train data (train EV 5-9x, test EV 0.6-2.2x). The signal is too weak for tree models to generalize.

2. **Simple weighted score is most robust** — minimal train/test gap, works for both SOL and DOGE.

3. **Combining features helps marginally** — SOL P95 score gives +1.11 vs +0.63 baseline (+76%). DOGE P90 gives +1.28 vs +0.75 baseline (+71%).

4. **The improvement is real but modest** — from ~0.6-0.8 baseline to ~1.1-1.3 with scoring. Not enough to clear the 2.0 bps fee wall at 0.005% maker.

5. **SOL LightGBM at P(win)>0.80 is the only config that clears fees** — +2.24 test EV, but only 232 trades over 22 days (~10/day) and high overfitting risk.

### Fee Viability Summary (All Approaches)

| Approach | Symbol | OOS EV | Net @ 0% | Net @ 0.005% (4f) |
|----------|--------|--------|----------|-------------------|
| Binary: liq_10s>0 | SOL | +1.92 | **+1.92 ✅** | -0.08 ❌ |
| Simple score P95 | SOL | +1.11 | **+1.11 ✅** | -0.89 ❌ |
| Simple score P90 | DOGE | +1.28 | **+1.28 ✅** | -0.72 ❌ |
| LightGBM P>0.80 | SOL | +2.24 | **+2.24 ✅** | **+0.24 ✅** ⚠️ |

⚠️ The LightGBM result is the only one that survives 0.005% maker, but with only ~10 trades/day and high overfitting risk, it's not reliable.

### Production Architecture (if deployed)

```
Every 1 second during weekday 13:00-18:00 UTC:
  1. Update buffers from websocket (trades, liqs, ticker)
  2. Compute 13 features at tick resolution
  3. score = sum(feature_i * weight_i)  # simple weighted score
  4. If score > P90_threshold AND not in position:
     → Place symmetric TP/SL orders (20/10 bps, 15m TL)
  5. If in position → wait for resolution
  
Expected: ~100-160 trades/day, +1.1-1.3 bps gross EV
Requires: 0% maker fees to be profitable
```

---

## Overall Conclusions

### The Edge
The symmetric TP/SL strategy exploits **fat tails** (leptokurtosis) in crypto price distributions. The edge is:
- **Real**: Positive across 59 days, 4/5 symbols, multiple time periods
- **Small**: +0.5-1.9 bps depending on symbol and trigger
- **Persistent**: Present in all months (May-Aug 2025), all weekdays
- **Structural**: Driven by market microstructure, not a temporary anomaly

### The Constraint
The 4-fill structure (2 opens + 2 closes) creates a fee floor that the edge cannot clear at any exchange charging positive maker fees. Best OOS EV (+1.92 bps) vs 4-fill cost at 0.005% maker (2.0 bps) = net -0.08 bps.

### What Would Make This Work
1. **True 0% maker fees** — some exchanges offer this (promotional, high-volume tiers)
2. **Maker rebates** — would amplify the edge (net +1.92 + rebate)
3. **Fewer fills** — requires abandoning the symmetric design
4. **Larger edge** — would require directional prediction (proven impossible)
5. **Different asset class** — crypto options, funding rate arb, etc.
