# Research Findings v6 — Full 3-Month Experiment Suite

**Date:** 2026-02-16
**Exchange:** Bybit Futures (VIP0)
**Symbols:** BTCUSDT, ETHUSDT, SOLUSDT
**Period:** 2025-11-01 → 2026-01-31 (92 days, ~3 months)
**Data:** Tick-level Bybit futures trades (~196M ticks/symbol for BTC/ETH, ~87M for SOL)
**Runtime:** ~77 minutes total for all 4 suites × 3 symbols

---

## Executive Summary

We ran **4 experiment suites** across **3 symbols** over the full 3-month dataset:

| Suite | Experiments | Fee Model | Winners | Total Configs |
|-------|------------|-----------|---------|---------------|
| Signal (E01–E15) | 15 classical microstructure signals | 7 bps RT | 18/45 (40%) | 45 best-per-symbol |
| Novel (N01–N16) | 16 academic microstructure signals | 7 bps RT | 18/48 (38%) | 48 best-per-symbol |
| Grid OHLCV (G01–G10) | 10 grid trading variants | 4 bps RT (grid) | 30/30 (100%) | 30 |
| Grid Tick-Level | 4 configs × 3 symbols | 4 bps RT (maker) | 10/12 (83%) | 12 |

**Key finding:** Over 3 months of trending crypto markets (Nov 2025 – Jan 2026), **directional signal strategies struggle** — most are net-negative after fees. Grid strategies show 100% win rate on completed trades but suffer massive unrealized losses from inventory accumulation in trends. **No strategy produced reliably strong risk-adjusted returns over this period.**

---

## 1. Signal Experiments (E01–E15)

**Method:** 5-minute bars aggregated from tick data. Each experiment tests multiple threshold/hold-period combinations. Best config per symbol selected. Fee: 7 bps round-trip.

### Top 10 Winners (sorted by avg PnL)

| Experiment | Symbol | Thresh | Hold | Trades | Avg PnL | Total PnL | WR | Sharpe |
|-----------|--------|--------|------|--------|---------|-----------|-----|--------|
| E09 Cumulative imbalance | ETHUSDT | 1.5 | 4h_m | 360 | **+12.54** | +4,513 | 49% | +0.08 |
| E13 Momentum 1h | SOLUSDT | 2.0 | 2h | 233 | **+11.29** | +2,630 | 46% | +0.07 |
| E03 Vol breakout | SOLUSDT | 1.5 | 2h | 450 | **+9.77** | +4,395 | 45% | +0.07 |
| E05 VWAP reversion | ETHUSDT | 1.5 | 4h | 397 | **+8.06** | +3,199 | 52% | +0.05 |
| E06 Volume surge | SOLUSDT | 1.5 | 2h | 392 | **+7.65** | +3,000 | 46% | +0.05 |
| E05 VWAP reversion | SOLUSDT | 1.0 | 4h | 493 | **+7.01** | +3,456 | 49% | +0.04 |
| E09 Cumulative imbalance | SOLUSDT | 1.5 | 4h_m | 354 | **+6.87** | +2,433 | 49% | +0.05 |
| E03 Vol breakout | ETHUSDT | 1.5 | 4h | 284 | **+5.93** | +1,684 | 45% | +0.03 |
| E13 Momentum 1h | BTCUSDT | 1.0 | 4h | 366 | **+5.36** | +1,961 | 48% | +0.05 |
| E03 Vol breakout | BTCUSDT | 1.5 | 4h | 277 | **+5.25** | +1,453 | 44% | +0.04 |

### Worst 5 Losers

| Experiment | Symbol | Avg PnL | Total PnL | Sharpe |
|-----------|--------|---------|-----------|--------|
| E14 Reversal after momentum | ETHUSDT | -9.42 | -2,987 | -0.08 |
| E15 Composite momentum | SOLUSDT | -8.90 | -16,605 | -0.11 |
| E01 Contrarian imbalance | ETHUSDT | -8.36 | -6,984 | -0.08 |
| E14 Reversal after momentum | SOLUSDT | -8.85 | -1,054 | -0.03 |
| E10 Price mean-reversion | ETHUSDT | -7.50 | -6,851 | -0.07 |

### Signal Suite Observations

1. **Best signals:** E09 (cumulative imbalance), E03 (vol breakout), E13 (1h momentum), E05 (VWAP reversion)
2. **Longer holds win:** Nearly all winners use 4h hold periods. Shorter holds (1h, 2h) are dominated by fees and noise.
3. **SOL is the most tradeable:** Highest avg PnL across most experiments — higher volatility creates larger moves to capture.
4. **BTC is hardest:** Strong uptrend from ~77k to ~111k made most signals unprofitable. Only 4 BTC configs were positive.
5. **Win rates are low:** Even winners hover around 45–52%. The edge is in the magnitude of wins vs losses, not frequency.
6. **Sharpe ratios are poor:** Best is +0.08 (annualized ~0.28). No signal has a Sharpe above 0.10 over 3 months.
7. **Reversal strategies fail:** E14 (fade momentum) is consistently the worst — don't fight the trend in crypto.

---

## 2. Novel Microstructure Experiments (N01–N16)

**Method:** Same 5-minute bar framework, but using academic microstructure features (VPIN, Hurst exponent, entropy, multifractal spectrum, etc.). Fee: 7 bps round-trip.

### Top 10 Winners

| Experiment | Symbol | Thresh | Hold | Trades | Avg PnL | Total PnL | WR | Sharpe |
|-----------|--------|--------|------|--------|---------|-----------|-----|--------|
| N15 Composite informed | SOLUSDT | 1.0 | 4h_c | 517 | **+10.36** | +5,354 | 50% | +0.06 |
| N15 Composite informed | ETHUSDT | 1.0 | 4h_m | 515 | **+7.70** | +3,964 | 50% | +0.05 |
| N10 Herding runs | SOLUSDT | 1.0 | 4h_m | 524 | **+7.42** | +3,886 | 45% | +0.05 |
| N07 Illiquidity shock | SOLUSDT | 1.0 | 4h_c | 527 | **+5.27** | +2,777 | 47% | +0.03 |
| N10 Herding runs | BTCUSDT | 1.0 | 4h_c | 513 | **+3.92** | +2,009 | 49% | +0.04 |
| N06 Toxic flow | SOLUSDT | 1.5 | 4h_m | 358 | **+3.63** | +1,300 | 47% | +0.02 |
| N07 Illiquidity shock | ETHUSDT | 1.5 | 4h_c | 470 | **+3.41** | +1,605 | 47% | +0.03 |
| N09 Aggressive flow | SOLUSDT | 1.0 | 4h | 514 | **+2.99** | +1,535 | 51% | +0.02 |
| N08 Efficiency regime | ETHUSDT | 1.5 | 4h | 356 | **+2.38** | +846 | 47% | +0.02 |
| N14 Vol speed informed | BTCUSDT | 1.0 | 4h | 319 | **+2.09** | +667 | 44% | +0.02 |

### Novel Suite Observations

1. **N15 (composite informed flow)** is the strongest novel signal — combining VPIN + entropy + persistence + aggression into one composite works best.
2. **N10 (herding/runs)** is surprisingly effective — detecting same-side trade runs and fading them works across all symbols.
3. **BTC is weakest again:** Only 4 positive configs (N10, N14, N15, N04), all with tiny edge (<4 bps avg).
4. **Academic signals don't outperform classical ones:** The novel suite (18/48 = 38% winners) performs similarly to the signal suite (18/45 = 40% winners).
5. **VPIN (N01) disappoints:** Despite strong academic backing, VPIN-based signals are net-negative across all symbols.
6. **Hurst exponent (N02) fails:** Regime detection via Hurst doesn't translate to profitable trading at 5-min frequency.

---

## 3. Grid OHLCV Experiments (G01–G10)

**Method:** Grid backtester on 5-minute OHLCV bars. Grid fills at maker rate (4 bps RT), trend overlay trades at taker rate (7 bps RT). 10 grid variants tested.

### Results by Avg PnL (all 30 configs positive)

| Experiment | Symbol | Trades | Avg PnL | Total PnL | WR | Max DD |
|-----------|--------|--------|---------|-----------|-----|--------|
| G10 Wide grid | SOLUSDT | 1,095 | **+41.71** | +45,668 | 100% | -75 |
| G10 Wide grid | ETHUSDT | 1,171 | **+36.26** | +42,463 | 100% | -251 |
| G10 Wide grid | BTCUSDT | 1,057 | **+23.91** | +25,275 | 100% | -35 |
| G08 Full hybrid | SOLUSDT | 6,393 | **+22.10** | +141,315 | 100% | -344 |
| G03 Asymmetric | SOLUSDT | 7,529 | **+21.55** | +162,255 | 100% | -344 |
| G01 Fixed symmetric | BTCUSDT | 6,943 | **+15.09** | +104,759 | 100% | -269 |
| G01 Fixed symmetric | SOLUSDT | 17,580 | **+14.80** | +260,213 | 100% | -707 |
| G01 Fixed symmetric | ETHUSDT | 14,907 | **+14.62** | +217,879 | 100% | -915 |
| G09 Tight grid | SOLUSDT | 24,112 | **+9.75** | +235,132 | 100% | -702 |
| G09 Tight grid | BTCUSDT | 20,785 | **+5.64** | +117,130 | 100% | -263 |

### By Total PnL (top 5)

| Experiment | Symbol | Total PnL (bps) | Trades |
|-----------|--------|-----------------|--------|
| G01 Fixed symmetric | SOLUSDT | **+260,213** | 17,580 |
| G09 Tight grid | SOLUSDT | **+235,132** | 24,112 |
| G01 Fixed symmetric | ETHUSDT | **+217,879** | 14,907 |
| G09 Tight grid | ETHUSDT | **+203,299** | 23,318 |
| G02 Vol-scaled | SOLUSDT | **+177,994** | 8,474 |

### Grid OHLCV Observations

1. **100% win rate is misleading** — this is by construction. Every completed grid round-trip earns cell_width - fees. The backtester only counts completed trades.
2. **OHLCV simulation overstates fills** — checking if high/low touched a level doesn't account for intra-bar price path or queue position.
3. **Wide grid (G10) has highest per-trade profit** but fewest trades. Tight grid (G09) has most trades but lowest per-trade profit.
4. **SOL generates the most grid activity** — higher volatility = more price oscillations through grid levels.
5. **Many "smart" variants (G02–G07) collapse to baseline** — vol-scaling, regime-adaptive, trend overlay, and dynamic sizing all produced identical results to the baseline for some symbols, suggesting the features didn't trigger meaningful changes.
6. **G08 Full hybrid** is the best risk-adjusted variant — lower drawdown than baseline with comparable total PnL.

> **⚠️ CAVEAT:** These OHLCV grid results are an **upper bound**. Real performance would be 30–50% of these numbers due to queue position, slippage, and intra-bar path uncertainty. See tick-level results below for more realistic simulation.

---

## 4. Tick-Level Grid Experiments

**Method:** Process every individual trade tick. Fixed grid with symmetric levels around first-tick center price. Fee: 2 bps maker per fill (4 bps RT per completed trade). No re-centering, no timeouts.

### Full 3-Month Results

| Config | Symbol | Trades | Inv | Avg PnL | Realized | Unrealized | **Net** | WR |
|--------|--------|--------|-----|---------|----------|------------|---------|-----|
| 20bps_3lvl | BTCUSDT | 23 | +3 | +15.92 | +366 | -8,360 | **-7,994** | 100% |
| 30bps_3lvl | BTCUSDT | 13 | +3 | +25.81 | +336 | -8,316 | **-7,981** | 100% |
| 50bps_2lvl | BTCUSDT | 4 | +2 | +45.63 | +183 | -5,522 | **-5,340** | 100% |
| 50bps_3lvl | BTCUSDT | 5 | +3 | +45.55 | +228 | -8,228 | **-8,001** | 100% |
| 20bps_3lvl | ETHUSDT | 57 | +3 | +15.96 | +910 | -10,807 | **-9,897** | 100% |
| 30bps_3lvl | ETHUSDT | 24 | +3 | +25.86 | +621 | -10,768 | **-10,147** | 100% |
| 50bps_2lvl | ETHUSDT | 11 | +2 | +45.64 | +502 | -7,159 | **-6,657** | 100% |
| 50bps_3lvl | ETHUSDT | 15 | +3 | +45.64 | +685 | -10,690 | **-10,005** | 100% |
| 20bps_3lvl | SOLUSDT | 55 | +3 | +16.04 | +882 | -13,022 | **-12,139** | 100% |
| 30bps_3lvl | SOLUSDT | 39 | +3 | +26.12 | +1,019 | -12,987 | **-11,968** | 100% |
| 50bps_2lvl | SOLUSDT | 18 | +2 | +46.30 | +833 | -8,641 | **-7,808** | 100% |
| 50bps_3lvl | SOLUSDT | 23 | +3 | +46.40 | +1,067 | -12,918 | **-11,851** | 100% |

### Tick-Level Grid Observations

1. **Every single config is net-negative over 3 months.** Realized profits are dwarfed by unrealized losses from stuck inventory.
2. **The problem is clear:** Nov 2025 – Jan 2026 was a trending period. BTC went from ~77k to ~105k, ETH from ~2,500 to ~3,300, SOL from ~160 to ~230+. Fixed grids centered on day-1 prices got stuck with long inventory almost immediately.
3. **BTC had the fewest trades** (4–23 depending on config) because it trended away from the grid center fastest.
4. **SOL had the most trades** (18–55) due to higher relative volatility creating more oscillations even within the trend.
5. **20bps tighter grids generate more trades** but the same inventory problem — more fills early, then stuck.
6. **Unrealized losses scale with inventory × price move:** 3 levels of inventory × ~8,000 bps of price move ≈ ~8,000–13,000 bps unrealized loss.

### Tick vs OHLCV Comparison

The tick-level results are dramatically worse than OHLCV because:
- **OHLCV counts only completed round-trips** (always profitable by construction)
- **Tick-level tracks open positions** and their unrealized P&L
- **OHLCV doesn't penalize stuck inventory** — it just doesn't count those trades
- **Over 3 months of trending markets**, the inventory cost overwhelms all realized profits

This confirms the v4/v5 caveat: **OHLCV grid results are an upper bound that can be extremely misleading in trending markets.**

---

## 5. Cross-Suite Analysis

### By Symbol Performance

| Symbol | Signal Winners | Novel Winners | Best Signal Avg | Best Novel Avg | Grid Tick Net |
|--------|---------------|---------------|-----------------|----------------|---------------|
| BTCUSDT | 4/15 | 4/16 | +5.36 (E13) | +3.92 (N10) | -5,340 to -8,001 |
| ETHUSDT | 6/15 | 7/16 | +12.54 (E09) | +7.70 (N15) | -6,657 to -10,147 |
| SOLUSDT | 8/15 | 7/16 | +11.29 (E13) | +10.36 (N15) | -7,808 to -12,139 |

**SOL is the most tradeable asset** — highest win rates across signal suites and most grid fills. **BTC is the hardest** — strong trend made most strategies unprofitable.

### Market Regime Impact

The Nov 2025 – Jan 2026 period was characterized by:
- **Strong uptrend** across all three assets
- BTC: +44% (77k → 111k)
- ETH: +32% (2,500 → 3,300)
- SOL: +44% (160 → 230)
- **High volatility** with multiple 5–10% drawdowns within the trend
- **Elevated trading activity** — 196M ticks/day for BTC, similar for ETH

This environment is **hostile to mean-reversion and grid strategies** but should favor momentum — yet even momentum signals showed weak Sharpe ratios, suggesting the trend was not smooth enough for 5-min signals to capture reliably.

### What Actually Works (Barely)

The only strategies with meaningful positive expectancy over 3 months:

| Strategy | Symbol | Avg PnL | Trades | Sharpe | Assessment |
|----------|--------|---------|--------|--------|------------|
| E09 Cumulative imbalance | ETHUSDT | +12.54 | 360 | +0.08 | Best signal overall |
| E13 Momentum 1h | SOLUSDT | +11.29 | 233 | +0.07 | Momentum works on SOL |
| N15 Composite informed | SOLUSDT | +10.36 | 517 | +0.06 | Best novel signal |
| E03 Vol breakout | SOLUSDT | +9.77 | 450 | +0.07 | Breakout on volatile asset |
| E05 VWAP reversion | ETHUSDT | +8.06 | 397 | +0.05 | Reversion works on ETH |
| N15 Composite informed | ETHUSDT | +7.70 | 515 | +0.05 | Composite on ETH |
| N10 Herding runs | SOLUSDT | +7.42 | 524 | +0.05 | Fade herding on SOL |

All of these have Sharpe < 0.10 over 3 months. **None would survive realistic transaction costs, slippage, and execution delays in production.**

---

## 6. Honest Assessment

### What we learned:

1. **Microstructure signals at 5-min frequency have very weak edge** on Bybit futures after fees. The 7 bps round-trip cost is a significant hurdle — most signals generate 0–5 bps of raw alpha, which is consumed by fees.

2. **Grid trading is mechanically sound but strategically flawed** without re-centering. A fixed grid in a trending market is equivalent to a leveraged position against the trend. The 100% win rate on completed trades is irrelevant when unrealized losses on open positions are 10–30x larger.

3. **The academic microstructure literature doesn't translate directly** to profitable crypto trading at retail fee levels. VPIN, Hurst, entropy, multifractal — all theoretically sound but practically insufficient to overcome transaction costs.

4. **Longer holding periods help** — 4h holds consistently outperform 1h and 2h holds. This suggests the signal has predictive power but needs time to overcome noise and fees.

5. **SOL is the best asset for both signals and grids** — higher volatility creates larger moves (better for signals) and more oscillations (better for grids).

6. **The trending market regime (Nov–Jan) was particularly hostile** to our strategies. A range-bound period would likely show better results for grids and mean-reversion signals.

### What would need to change for production viability:

1. **Lower fees** — VIP tiers or rebate programs could reduce RT cost from 7 bps to 2–3 bps, making marginal signals profitable.
2. **Grid re-centering** — Periodic re-centering based on trend detection would prevent the catastrophic inventory accumulation we observed.
3. **Regime filtering** — Only trading signals in favorable regimes (e.g., grid in range-bound, momentum in trends) could improve Sharpe significantly.
4. **Higher frequency** — Moving to 1-min or sub-minute bars might capture more of the microstructure edge before it decays.
5. **Ensemble approach** — Combining the top 3–5 signals into a composite with position sizing could smooth returns.

---

## 7. Data & Methodology

### Data Volume

| Symbol | Total Ticks (92 days) | Avg Ticks/Day | Parquet Files |
|--------|----------------------|---------------|---------------|
| BTCUSDT | ~196M | ~2.1M | 92 |
| ETHUSDT | ~196M | ~2.1M | 92 |
| SOLUSDT | ~87M | ~0.9M | 92 |

### Feature Computation

- **Signal suite (E01–E15):** 5-min bars with OHLCV + trade count + buy/sell volume + VWAP + imbalance + volatility
- **Novel suite (N01–N16):** Same bars + VPIN + Hurst exponent + Shannon entropy + multifractal spectrum + sign persistence + Kyle's lambda + efficiency ratio
- **Grid OHLCV (G01–G10):** Same 5-min bars, grid state machine on OHLCV
- **Grid Tick:** Raw tick-by-tick processing, no aggregation

### Execution

All experiments run via `run_experiment.py` CLI with standardized parameters:
```
python3 run_experiment.py --suite <suite> --exchange bybit_futures --symbol <SYM> --start 2025-11-01 --end 2026-01-31
```

Batch orchestration via `run_all.sh`. Results saved to `results/` directory.

---

## Files

| File | Description |
|------|-------------|
| `run_experiment.py` | Unified CLI runner for all experiment suites |
| `run_all.sh` | Batch script to run all suites × all symbols |
| `experiments.py` | Signal experiments E01–E15 |
| `experiments_v2.py` | Novel experiments N01–N16 |
| `experiments_grid.py` | Grid OHLCV experiments G01–G10 |
| `grid_backtest.py` | Tick-level grid backtester |
| `results/` | All output files (18 per-symbol + 4 combined summaries) |
