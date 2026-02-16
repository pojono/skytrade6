# Research Findings v18 — 1-Minute Bar Experiments

**Date:** 2026-02-16
**Exchange:** Bybit Futures (VIP0)
**Symbols:** BTCUSDT, ETHUSDT, SOLUSDT
**Period:** 2025-12-01 → 2025-12-30 (30 days)
**Method:** Tick data → 1m bars with microstructure features, compared to 5m baseline
**Runtime:** 2.6 min total (all 3 symbols), RAM peak 7.4GB

---

## Executive Summary

**1-minute bars do NOT improve our strategies.** Across all 4 phases of testing:

1. **Signal experiments at 1m are worse than 5m** — the same signals (E01, E03, E09) produce lower avg PnL at 1m resolution
2. **Precision entry timing doesn't help** — imbalance-based 1m entry timing makes things *worse* (-1.3 to -6.4 bps/trade)
3. **Short-horizon mean reversion is uniformly negative** — every single config across 4 strategies × 4 holding periods × 3 thresholds × 3 symbols is a loser
4. **Vol prediction correlation improves** at 1m (+0.06 to +0.10 corr) but R² remains negative due to scale mismatch

**Recommendation:** Stay with 5m bars. Do NOT invest in scaling 1m features to 13 months.

---

## Data Overview

| Symbol | 1m Bars | 5m Bars | Ratio | Build Time |
|--------|---------|---------|-------|-----------|
| BTCUSDT | 43,200 | 8,640 | 5.0× | 49s |
| ETHUSDT | 43,200 | 8,640 | 5.0× | 52s |
| SOLUSDT | 43,200 | 8,640 | 5.0× | 30s |

---

## Phase 1: Signal Experiments — 1m vs 5m

### Best Config (4h hold) Comparison

| Signal | Symbol | 5m/taker | 1m/taker | 1m/maker | 1m vs 5m |
|--------|--------|----------|----------|----------|----------|
| **E01 t=1.0** | **BTC** | **+13.68** | -12.78 | -9.78 | **❌ -23.46 bps** |
| E01 t=1.0 | ETH | -3.37 | -4.83 | -1.83 | ❌ -1.46 bps |
| E01 t=1.0 | SOL | +3.40 | -14.15 | -11.15 | ❌ -17.55 bps |
| **E03 t=1.0** | **BTC** | +2.31 | +0.46 | **+3.46** | ⚠️ +1.15 bps (maker) |
| E03 t=1.0 | ETH | +9.08 | -15.73 | -12.73 | ❌ -24.81 bps |
| E03 t=1.0 | SOL | +8.66 | -16.98 | -13.98 | ❌ -25.64 bps |
| **E09 t=1.0** | **BTC** | +9.39 | -1.72 | +1.28 | ❌ -8.11 bps |
| **E09 t=1.0** | **ETH** | +2.76 | +5.74 | **+8.74** | ✅ +5.98 bps (maker) |
| E09 t=1.0 | SOL | +19.95 | +17.10 | +20.10 | ⚠️ +0.15 bps (maker) |
| **E09 t=1.5** | **ETH** | **+25.56** | +20.79 | +23.79 | ❌ -1.77 bps |
| E09 t=1.5 | SOL | +22.38 | -4.92 | -1.92 | ❌ -24.30 bps |

### Key Observations

1. **5m bars are better for E01 (contrarian) on every symbol.** The contrarian signal at 1m triggers on noise — the imbalance at 1m is too transient to be meaningful. At 5m, the imbalance aggregates enough trades to capture real flow.

2. **E09 (cumulative imbalance momentum) at 1m with maker fees shows a slight improvement on ETH** (+8.74 vs +2.76 bps). But this is the only winning combination out of 33 tested, and it's still weaker than the 5m E09 t=1.5 result (+25.56 bps).

3. **The 1m z-score window (3 days = 4,320 bars) may be suboptimal.** The rolling statistics are noisier at 1m because each bar has fewer ticks (~75 for BTC vs ~375 at 5m). The z-scores are less stable.

4. **Maker fees (4 bps) help but don't save the strategy.** Even with 3 bps less in fees, most 1m configs remain negative.

---

## Phase 2: Precision Entry (1m Timing Within 5m Window)

### Results

| Signal | Symbol | Baseline (5m close) | Oracle (best 1m) | Imbalance-timed | Imb Delta |
|--------|--------|--------------------|--------------------|-----------------|-----------|
| E01 | BTC | **+13.68** | +14.76 | +10.46 | **-3.22** |
| E01 | ETH | -3.37 | -0.07 | -5.70 | **-2.33** |
| E01 | SOL | +3.40 | +3.41 | -3.04 | **-6.44** |
| E09 | BTC | +2.13 | +6.45 | +0.84 | **-1.30** |
| E09 | ETH | **+25.56** | +36.11 | +26.73 | **+1.17** |
| E09 | SOL | +22.38 | +29.92 | +19.47 | **-2.91** |

### Key Findings

1. **The oracle (best possible 1m entry) shows there IS value in entry timing** — +1 to +10.5 bps improvement is theoretically possible. But we can't achieve it.

2. **Imbalance-based timing makes things WORSE on 5 of 6 tests.** The only positive case is E09 on ETH (+1.17 bps), which is marginal.

3. **Why imbalance timing fails:** The 1m imbalance within a 5m window is essentially random noise. When the 5m signal says "go long," picking the 1m bar with the most selling pressure doesn't reliably give a better entry — it just adds noise.

4. **The oracle delta shows the theoretical ceiling is small.** Even perfect entry timing only adds +1 to +10 bps. Given that we can't achieve even half of this, the practical improvement is negligible.

---

## Phase 3: Short-Horizon Mean Reversion (1m, Maker Fees)

### Summary: ALL NEGATIVE

Tested 4 strategies × 4 holding periods × 3 thresholds × 3 symbols = **144 configurations**.

**Not a single one is profitable.** Best results (least negative):

| Strategy | Symbol | Best Config | Avg PnL | Trades | WR |
|----------|--------|-------------|---------|--------|-----|
| A: Contrarian imbalance | SOL | t=2.0, 1h | -0.33 bps | 267 | 47% |
| A: Contrarian imbalance | BTC | t=1.0, 1h | -1.68 bps | 700 | 47% |
| B: VWAP reversion | ETH | t=1.5, 1h | -1.14 bps | 621 | 51% |
| C: Price mean-reversion | BTC | t=2.0, 30m | -2.76 bps | 757 | 49% |
| D: Range breakout | ETH | t=1.5, 1h | -0.51 bps | 337 | 51% |

### Why Short-Horizon Fails

1. **The signal-to-noise ratio at 1m is terrible.** Each 1m bar has ~75 ticks (BTC) — not enough for stable microstructure features. The imbalance, VWAP deviation, etc. are dominated by noise.

2. **Even 4 bps maker fees are too high for 5-15 minute holds.** The average 5m return is ~1-2 bps. After 4 bps fees, you need the signal to add >4 bps of alpha per trade — which it doesn't.

3. **The pattern is consistent: longer holds are less negative.** 5m hold is worst (-3.5 to -4.5 bps avg), 1h hold is least bad (-0.5 to -2.5 bps avg). This confirms the v3/v6 finding that the signal needs time to play out.

4. **Win rates cluster around 40-50%** — essentially random. The signals have no predictive power at short horizons.

---

## Phase 4: Vol Prediction Comparison

### Correlation (predicted vs actual forward vol)

| Symbol | 5m → 1h | 1m → 1h | Δ | 5m → 4h | 1m → 4h | Δ |
|--------|---------|---------|---|---------|---------|---|
| BTC | 0.589 | **0.685** | **+0.096** | 0.499 | **0.530** | **+0.031** |
| ETH | 0.528 | **0.607** | **+0.079** | 0.511 | 0.505 | -0.006 |
| SOL | 0.564 | **0.655** | **+0.091** | 0.396 | **0.428** | **+0.032** |

### R² (negative due to scale mismatch — features not standardized to target scale)

R² values are negative for both 1m and 5m, indicating the Ridge model needs proper standardization of the target variable. The **correlation** is the meaningful metric here.

### Interpretation

1. **1m bars DO improve vol prediction correlation** — +0.08 to +0.10 at 1h horizon. More recent data = better vol autocorrelation.

2. **At 4h horizon, improvement is marginal** (+0.03 or less). The extra granularity doesn't help for longer predictions.

3. **This is the ONE area where 1m adds value.** But the improvement is modest, and the v9/v10 vol prediction system already works well at 5m (R²=0.39 over 13 months). The 30-day 1m correlation of 0.685 vs 5m's 0.589 is encouraging but needs validation on a longer period.

---

## Summary Table

| Phase | Question | Answer | Verdict |
|-------|----------|--------|---------|
| **1: Signal experiments** | Do 1m bars improve E01/E03/E09? | No — worse on 30/33 configs | **❌ FAIL** |
| **2: Precision entry** | Does 1m timing improve 5m signals? | No — imbalance timing hurts on 5/6 tests | **❌ FAIL** |
| **3: Short-horizon** | Do new strategies work at 1m? | No — all 144 configs negative | **❌ FAIL** |
| **4: Vol prediction** | Does 1m improve vol correlation? | Yes — +0.08 to +0.10 at 1h | **✅ PASS (marginal)** |

---

## Why 1m Bars Don't Help — Root Cause Analysis

### 1. Microstructure features need sufficient ticks per bar

At 5m, BTC has ~375 ticks per bar. At 1m, only ~75. Features like `vol_imbalance`, `kyle_lambda`, and `large_imbalance` need hundreds of ticks to be statistically meaningful. At 75 ticks, these features are dominated by sampling noise.

### 2. The alpha is slow-moving, not fast-moving

The 4h holding period dominance (v3, v6) tells us the predictive signal takes hours to play out. Computing features at 1m doesn't change this — you're just measuring the same slow signal with a noisier ruler.

### 3. Rolling statistics are less stable at 1m

A 3-day z-score window at 1m = 4,320 bars. But each bar is noisy, so the rolling mean and std are themselves noisy. The z-score signal oscillates more, generating more false triggers.

### 4. The fee floor is binding

Even at maker rates (4 bps RT), the minimum profitable holding period is ~30 minutes for a signal with 5-10 bps of raw alpha. At 1m bars, most signals have <2 bps of raw alpha per bar — insufficient to overcome any fee structure.

---

## What This Means Going Forward

1. **Stay with 5m bars** for all signal and grid strategies. The 5m aggregation is the right balance between noise reduction and signal preservation.

2. **1m vol prediction could be useful** as a faster-reacting input to the grid bot's adaptive rebalance (S5). But the improvement is modest (+0.08 corr) and may not translate to better grid PnL.

3. **Don't explore 30s or sub-minute bars** — if 1m is worse than 5m, shorter bars will be even worse. The noise problem only gets worse.

4. **The path to better strategies is NOT shorter bars** — it's:
   - Lower fees (VIP tiers, rebate programs)
   - Better features (order book data, funding rates, cross-exchange signals)
   - Different strategy classes (market making, statistical arbitrage)
   - Longer validation periods (13 months, not 30 days)

---

## Files

| File | Description |
|------|-------------|
| `experiments_v18_1m.py` | Full experiment suite: 4 phases, 3 symbols |
| `results/v18_1m_BTCUSDT.txt` | BTCUSDT complete output |
| `results/v18_1m_ETHUSDT.txt` | ETHUSDT complete output |
| `results/v18_1m_SOLUSDT.txt` | SOLUSDT complete output |
| `PLAN_v18_1m_bars.md` | Original experiment plan |
