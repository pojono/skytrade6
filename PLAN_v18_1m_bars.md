# PLAN v18 — 1-Minute Bar Experiment

**Date:** 2026-02-16
**Goal:** Test whether 1m bars improve our best strategies vs 5m bars
**Scope:** Dec 2025 (30 days), Bybit futures only, BTC/ETH/SOL
**Constraints:** 15GB RAM, 4 cores — must process day-by-day, never load full month

---

## Phase 0: Build 1m Feature Pipeline

**Task:** Extend `build_features.py` to support 1m interval (already in INTERVAL_MAP).
Run for Dec 2025 only, bybit_futures source.

- 3.7M ticks/day for BTC → ~1,440 bars/day at 1m (vs 288 at 5m)
- Process 1 day at a time → peak RAM ~1.5GB per day (safe)
- Output: `parquet/{SYM}/features/1m/bybit_futures/{date}.parquet`

**Estimate:** ~10 min per symbol (30 days × ~20s/day), ~30 min total.

## Phase 1: Build Extended Features at 1m

**Task:** Compute the same rolling features used in v8-v12 experiments but at 1m resolution.
These include: realized vol, Parkinson vol, vol ratios, efficiency ratio, ADX, 
sign persistence, VPIN, entropy, etc.

Must be memory-safe: load 1m features day-by-day, compute rolling features with 
appropriate lookback warmup.

## Phase 2: Precision Entry Test

**Task:** Use existing 5m signals (E01, E03, E09) as the trigger, but refine entry 
timing using 1m features within the 5m window.

- When 5m signal fires at bar T, look at the five 1m bars within that window
- Use 1m microstructure features to pick the best 1m bar for entry
- Compare avg PnL vs baseline (entry at 5m close)

**Hypothesis:** Better entry timing adds 2-5 bps per trade.

## Phase 3: Short-Horizon Mean Reversion at 1m

**Task:** Test a new strategy class that only exists at 1m:
- Signal: 1m microstructure imbalance (contrarian)
- Hold: 5m, 15m, 30m (short holds that were unprofitable at 5m bars)
- Fee: 4 bps RT (maker-only, limit orders)
- Threshold: z-score > 1.5 on rolling 1h window

**Hypothesis:** At 1m resolution with maker fees (4 bps vs 7 bps taker), 
short-horizon mean reversion becomes viable.

## Phase 4: 1m Vol Prediction

**Task:** Test Ridge regression vol prediction at 1m bars.
- Same 45+ features as v9/v10 but computed at 1m
- Predict 15m, 1h, 4h forward vol
- Compare R² and correlation vs 5m baseline

## Phase 5: Document & Compare

**Task:** Write FINDINGS_v18_1m_bars.md with:
- 1m vs 5m comparison tables
- Whether precision entry helps
- Whether short-horizon strategies become viable
- Whether vol prediction improves
- Recommendation: worth scaling to 13 months or not

---

## Memory Safety Rules

1. Never load more than 1 day of tick data at a time (~3.7M rows, ~1.5GB)
2. For rolling features, keep a sliding window buffer of max 1440 bars (1 day of 1m)
3. Delete DataFrames with `del df` after use
4. Print memory usage periodically
5. Process symbols sequentially, not in parallel
