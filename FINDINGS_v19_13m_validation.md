# FINDINGS v19: 13-Month Validation of Signal & Novel Microstructure Experiments

## Overview

**Objective:** Replicate the v6 findings (Signal E01–E15 and Novel N01–N16 experiments) over a much longer period to test whether any edges survive extended out-of-sample validation.

| Parameter | v6 (Original) | v19 (This Study) |
|-----------|---------------|-------------------|
| Period | Nov 2025 – Jan 2026 (92 days) | Jan 2025 – Jan 2026 (396 days) |
| Symbols | BTCUSDT, ETHUSDT, SOLUSDT | BTCUSDT, ETHUSDT, SOLUSDT |
| Bar size | 5-minute | 5-minute |
| Fee model | 7 bps round-trip (Bybit VIP0) | 7 bps round-trip (Bybit VIP0) |
| Signal suite | E01–E15 (15 experiments) | E01–E15 (15 experiments) |
| Novel suite | N01–N16 (16 experiments) | N01–N16 (16 experiments) |
| Total configs | ~93 (signal) + ~96 (novel) | 45 (signal best-per-exp) + 48 (novel best-per-exp) |

---

## 1. Signal Experiments (E01–E15) — 13-Month Results

### Summary

| Metric | v6 (3 months) | v19 (13 months) |
|--------|---------------|-----------------|
| Winners (avg > 0, trades ≥ 10) | 18/45 (40%) | **2/45 (4%)** |
| Best avg PnL | +10.36 bps | **+1.26 bps** |
| Worst avg PnL | ~-8 bps | -8.76 bps |

### Top 10 Results (Best of Each Experiment × Symbol)

| Rank | Experiment | Symbol | Thresh | Hold | Trades | Avg PnL | Total PnL | WR | Sharpe |
|------|-----------|--------|--------|------|--------|---------|-----------|-----|--------|
| 1 | E06 Volume surge direction | ETHUSDT | 1.5 | 4h | 1,235 | **+1.26** | +1,552 | 46% | +0.01 |
| 2 | E05 VWAP reversion | SOLUSDT | 1.0 | 4h | 2,172 | **+0.23** | +496 | 49% | +0.00 |
| 3 | E01 Contrarian imbalance | SOLUSDT | 1.5 | 4h | 2,064 | -0.43 | -895 | 50% | -0.00 |
| 4 | E13 Momentum 1h | ETHUSDT | 2.0 | 4h | 798 | -0.88 | -700 | 45% | -0.00 |
| 5 | E06 Volume surge direction | SOLUSDT | 1.5 | 2h | 1,793 | -0.99 | -1,780 | 46% | -0.01 |
| 6 | E13 Momentum 1h | SOLUSDT | 1.0 | 4h | 1,779 | -1.07 | -1,900 | 48% | -0.01 |
| 7 | E12 Vol regime contrarian | BTCUSDT | 1.0 | 4h | 1,584 | -1.23 | -1,949 | 48% | -0.01 |
| 8 | E03 Vol breakout | SOLUSDT | 1.5 | 4h | 1,366 | -1.24 | -1,699 | 47% | -0.01 |
| 9 | E13 Momentum 1h | BTCUSDT | 2.0 | 4h | 759 | -1.68 | -1,271 | 46% | -0.01 |
| 10 | E12 Vol regime contrarian | ETHUSDT | 1.5 | 2h | 1,860 | -1.99 | -3,700 | 49% | -0.02 |

### Signal Suite Observations

1. **Catastrophic degradation:** From 40% winners (v6, 3 months) to **4% winners** (v19, 13 months). Only 2 of 45 best-per-experiment configs are net-positive.
2. **Best signal is barely positive:** E06 (volume surge direction) on ETH at +1.26 bps — this is economically insignificant after accounting for slippage.
3. **All experiments net-negative on BTC:** Zero winning configs for BTCUSDT across all 15 signal experiments.
4. **4h holding period still dominates:** The few near-zero configs all use 4h holds, confirming that shorter horizons are pure noise at 5-min frequency.
5. **Sharpe ratios universally terrible:** Best is +0.01, indicating no risk-adjusted edge whatsoever.

---

## 2. Novel Microstructure Experiments (N01–N16) — 13-Month Results

### Summary

| Metric | v6 (3 months) | v19 (13 months) |
|--------|---------------|-----------------|
| Winners (avg > 0, trades ≥ 10) | 18/48 (38%) | **3/48 (6%)** |
| Best avg PnL | +10.36 bps | **+4.60 bps** |
| Worst avg PnL | ~-8 bps | -7.27 bps |

### Top 10 Results (Best of Each Experiment × Symbol)

| Rank | Experiment | Symbol | Thresh | Hold | Trades | Avg PnL | Total PnL | WR | Sharpe |
|------|-----------|--------|--------|------|--------|---------|-----------|-----|--------|
| 1 | N12 VW momentum | SOLUSDT | 1.0 | 4h_m | 2,111 | **+4.60** | +9,710 | 50% | +0.03 |
| 2 | N14 Vol speed informed | ETHUSDT | 1.5 | 4h | 1,234 | **+0.37** | +461 | 46% | +0.00 |
| 3 | N06 Toxic flow cumulative | SOLUSDT | 1.5 | 4h_c | 1,587 | **+0.01** | +17 | 49% | +0.00 |
| 4 | N12 VW momentum | ETHUSDT | 1.5 | 4h_m | 1,803 | -0.41 | -733 | 48% | -0.00 |
| 5 | N09 Aggressive flow | ETHUSDT | 1.5 | 4h | 2,019 | -1.29 | -2,610 | 48% | -0.01 |
| 6 | N07 Illiquidity shock | ETHUSDT | 1.5 | 4h_m | 2,030 | -1.55 | -3,154 | 48% | -0.01 |
| 7 | N02 Hurst regime | ETHUSDT | 1.5 | 2h | 2,939 | -2.16 | -6,349 | 46% | -0.02 |
| 8 | N14 Vol speed informed | SOLUSDT | 1.5 | 2h | 1,800 | -2.24 | -4,031 | 46% | -0.01 |
| 9 | N15 Composite informed | ETHUSDT | 1.0 | 4h_m | 2,264 | -2.51 | -5,686 | 49% | -0.02 |
| 10 | N04 Info decay | BTCUSDT | 1.5 | 4h | 1,615 | -2.61 | -4,220 | 48% | -0.03 |

### Novel Suite Observations

1. **Same collapse as signal suite:** From 38% winners (v6) to **6% winners** (v19). Academic features offer no durable edge.
2. **N12 (VW momentum) on SOL is the sole meaningful winner:** +4.60 bps avg over 2,111 trades with Sharpe +0.03. This is the only result with any economic significance, but the Sharpe is still very low.
3. **N15 (Composite informed) collapsed:** Was the #1 novel signal in v6 (+10.36 bps on SOL). Over 13 months: -4.56 bps on BTC, -2.51 bps on ETH, -3.98 bps on SOL. Complete failure.
4. **N10 (Herding runs) collapsed:** Was #3 in v6 (+7.42 bps on SOL). Over 13 months: -3.74 bps on BTC, -4.47 bps on ETH, -5.51 bps on SOL. Complete failure.
5. **BTC is worst again:** Zero winning configs across all 16 novel experiments for BTCUSDT.
6. **VPIN (N01) still disappoints:** Net-negative across all symbols, confirming v6 finding.
7. **Hurst (N02) still fails:** Net-negative across all symbols, confirming v6 finding.

---

## 3. Cross-Suite Comparison

### Winners by Asset

| Asset | Signal Winners | Novel Winners | Total Winners |
|-------|---------------|---------------|---------------|
| BTCUSDT | 0/15 | 0/16 | **0/31** |
| ETHUSDT | 1/15 | 1/16 | **2/31** |
| SOLUSDT | 1/15 | 1/16 | **2/31** |

### v6 vs v19 Head-to-Head: Top v6 Winners Over 13 Months

| v6 Winner | v6 Avg PnL | v19 Avg PnL | Survived? |
|-----------|-----------|-------------|-----------|
| N15 Composite informed (SOL) | +10.36 | -3.98 | ❌ |
| E09 Cumulative imbalance (ETH) | +8.87 | -4.84 | ❌ |
| N10 Herding runs (SOL) | +7.42 | -5.51 | ❌ |
| N07 Illiquidity shock (SOL) | +5.27 | -4.15 | ❌ |
| N10 Herding runs (BTC) | +3.92 | -3.74 | ❌ |
| E01 Contrarian imbalance (BTC) | +3.75 | -4.19 | ❌ |
| E06 Volume surge (ETH) | +3.41 | +1.26 | ⚠️ Marginal |
| N12 VW momentum (SOL) | +2.99 | +4.60 | ✅ Only survivor |

---

## 4. Key Conclusions

### 4.1 The v6 Results Were Regime-Specific, Not Durable

The 3-month v6 period (Nov 2025 – Jan 2026) was a specific market regime. When extended to 13 months covering multiple regimes (ranging, trending up, trending down, high-vol, low-vol), **virtually all edges evaporate**. This is the textbook definition of overfitting to a specific market condition.

### 4.2 Microstructure Signals at 5-Min Frequency Have No Durable Edge

- **Signal suite:** 2/45 winners (4%), best +1.26 bps
- **Novel suite:** 3/48 winners (6%), best +4.60 bps
- **Combined:** 5/93 winners (5%), with only 1 economically meaningful result

At retail fee levels (7 bps RT), 5-minute microstructure signals from tick data do not produce tradeable edges over meaningful time horizons.

### 4.3 Academic Literature Does Not Translate to Retail Crypto Trading

Features inspired by Easley/Lopez de Prado (VPIN), Lillo/Farmer (Hurst), Bouchaud (autocorrelation), Hasbrouck (info decay), Shannon (entropy), Amihud (illiquidity), Kaufman (efficiency), Kyle (lambda), and Mandelbrot (multifractal) — **none produce durable edges** at 5-minute frequency with 7 bps fees. The academic research describes real phenomena, but the signal-to-noise ratio is too low for profitable trading at retail fee levels.

### 4.4 BTC Is Untradeable with Microstructure Signals

Zero winning configurations across 31 experiments (15 signal + 16 novel) for BTCUSDT. BTC's microstructure is too efficient for these approaches.

### 4.5 SOL Remains the Most Tradeable Asset

The only meaningful winner (N12 VW momentum, +4.60 bps) is on SOLUSDT. SOL's higher volatility and lower microstructure efficiency provide the best (though still marginal) opportunity.

### 4.6 The N12 VW Momentum Signal on SOL Deserves Further Investigation

This is the sole survivor: volume-weighted momentum on SOL with contrarian direction at 4h holding period. +4.60 bps avg over 2,111 trades (13 months) is notable. However:
- Sharpe is only +0.03 — very low risk-adjusted return
- Win rate is 50% — essentially a coin flip with slightly positive skew
- Needs walk-forward validation before any confidence

---

## 5. Recommendations

1. **Abandon 5-min microstructure signal strategies at retail fee levels.** The evidence from 13 months across 3 assets is conclusive: there is no durable edge.
2. **If pursuing microstructure signals, focus exclusively on SOL** — the only asset showing any residual edge.
3. **Investigate N12 (VW momentum) on SOL further** — the only signal that survived 13-month validation. Consider:
   - Walk-forward cross-validation
   - Fee sensitivity analysis (does it survive at 5 bps? 3 bps?)
   - Regime conditioning (does it work better in specific vol regimes?)
4. **Grid trading (from v14–v17) remains the more promising direction** — mechanical grid profits are more robust than directional signals.
5. **Volatility prediction (from v8–v11) is the strongest ML capability** — use it for position sizing and risk management, not directional trading.

---

## 6. Runtime Statistics

| Suite | Symbol | Time (s) | Bars |
|-------|--------|----------|------|
| Signal | BTCUSDT | 520 | ~114k |
| Signal | ETHUSDT | 508 | ~114k |
| Signal | SOLUSDT | 324 | ~114k |
| Signal | **Total** | **1,352** | |
| Novel | BTCUSDT | 653 | ~114k |
| Novel | ETHUSDT | 659 | ~114k |
| Novel | SOLUSDT | 460 | ~114k |
| Novel | **Total** | **1,772** | |
| **Grand Total** | | **3,124s (~52 min)** | |

Peak RAM: ~7.7 GB. All processing memory-safe (day-by-day tick loading).
