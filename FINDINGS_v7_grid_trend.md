# Research Findings v7 — Grid + Trend Combined Strategies

**Date:** 2026-02-16
**Exchange:** Bybit Futures (VIP0)
**Symbol tested:** ETHUSDT (14-day validation: Dec 1–14, 2025)
**Grid fee:** 2 bps maker per fill (4 bps RT)
**Trend fee:** 5.5 bps taker per side (11 bps RT)

---

## Motivation

FINDINGS v6 showed that:
- **Grid strategies** earn reliably on completed trades (100% WR) but suffer catastrophic unrealized losses from inventory accumulation in trending markets.
- **Signal strategies** have weak edge (~5–12 bps avg) with poor Sharpe ratios.

The hypothesis: **combining grid (profits in range) with trend-following (profits in trends)** should produce a smoother equity curve than either alone, since they are fundamentally opposite strategies that complement each other.

## Modes Tested

| Mode | Description |
|------|-------------|
| **Baseline** | Pure grid, no trend overlay |
| **Mode 1: Regime Switch** | Detect trend → pause grid, activate trend follower. Re-center grid when returning to range. |
| **Mode 2: Parallel** | Grid and trend follower run independently, always on. No interference. |
| **Mode 3: Grid + Hedge** | Grid always runs. When trend detected AND grid has adverse inventory, open hedge position. |
| **Mode 4: Adaptive Grid** | Grid always runs. Re-center grid when trend detected (close positions, rebuild at new price). |

Trend detection: EMA crossover (fast vs slow) on subsampled tick prices.
Three EMA speeds tested: fast (500/2000), medium (2000/8000), slow (5000/20000).

## 14-Day Results (ETHUSDT, Dec 1–14, 2025)

### Comparison Table (sorted by combined net PnL)

| Mode | EMA | Grid Trades | Grid Net | Trend Trades | Trend Net | Recenters | **Combined** |
|------|-----|-------------|----------|--------------|-----------|-----------|-------------|
| **Parallel** | **fast** | 56 | +910.5 | 25 | **+1,444.8** | 0 | **+2,355.3** |
| Adaptive | medium | 207 | +1,722.1 | 0 | +0.0 | — | +1,722.1 |
| Baseline | — | 56 | +910.5 | 0 | +0.0 | 0 | +910.5 |
| Parallel | medium | 56 | +910.5 | 4 | -42.3 | 0 | +868.2 |
| Hedge | fast | 56 | +910.5 | 24 | -137.0 | 24 | +773.5 |
| Hedge | medium | 56 | +910.5 | 4 | -634.1 | 4 | +276.4 |
| Hedge | slow | 56 | +910.5 | 2 | -1,092.6 | 2 | -182.1 |
| Regime Switch | medium | 19 | -97.0 | 4 | -127.6 | 4 | -224.5 |
| Parallel | slow | 56 | +910.5 | 1 | -1,226.5 | 0 | -316.0 |
| Adaptive | slow | 157 | -714.0 | 0 | +0.0 | — | -714.0 |
| Regime Switch | fast | 87 | -1,728.9 | 39 | +677.4 | 39 | -1,051.4 |
| Regime Switch | slow | 13 | -1,388.7 | 1 | -1,081.6 | 1 | -2,470.3 |
| Adaptive | fast | 367 | -3,178.0 | 0 | +0.0 | — | -3,178.0 |

## Key Findings

### 1. Parallel + Fast EMA is the clear winner (+2,355 vs +911 baseline)

The parallel mode works because **grid and trend don't interfere with each other**:
- Grid earns its usual +910 bps from oscillations (56 completed trades, 100% WR)
- Fast trend follower independently captures directional moves (+1,445 bps from 25 trades, 36% WR)
- The trend follower's winners are large enough to overcome its 64% loss rate

### 2. Regime Switch is the worst mode

Closing grid positions to switch modes **realizes the unrealized loss** that would otherwise recover if price returns. The recenter cost destroys all value. This is the key insight: **don't try to switch between strategies — run both simultaneously.**

### 3. Adaptive Grid is mixed

Re-centering helps with medium EMA (+1,722) because it generates more grid trades (207 vs 56). But fast EMA re-centers too often (destroying value), and slow EMA re-centers too rarely (not helping).

### 4. Hedge mode doesn't help

The hedge positions are opened too late (after trend is already established) and closed too early (when EMA spread narrows). The hedge itself whipsaws.

### 5. EMA speed matters enormously

- **Fast EMA (500/2000):** Best for parallel mode — captures more moves, accepts more whipsaw
- **Medium EMA (2000/8000):** Best for adaptive mode — fewer false signals
- **Slow EMA (5000/20000):** Too slow for all modes — misses most of the action

## Limitations & Next Steps (Paused)

### What we didn't test:
- Stop-loss and trailing stop on trend positions (implemented but not yet validated)
- Multiple grid cell sizes in combination
- Longer time periods (only 14 days tested)
- Other symbols (BTC, SOL)

### The fundamental problem: regime detection

All four modes depend on **detecting when the market is trending vs ranging**. Our EMA crossover is a crude proxy. The results show:
- Too fast → whipsaw, false signals
- Too slow → misses transitions, reacts too late
- No speed is "right" because **regime transitions are not clean EMA crossovers**

**This is why we are pausing grid+trend research to focus on regime detection as a standalone research problem.** Better regime detection would improve ALL strategies — grid, trend, signal, and combined.

## Files

| File | Description |
|------|-------------|
| `grid_trend_backtest.py` | Grid + Trend combined backtester (4 modes, parameterized sweep) |
| `run_grid_trend.sh` | Batch runner for all symbols |

---

*Research paused. Next: regime detection experiments (v8).*
