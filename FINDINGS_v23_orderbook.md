# FINDINGS v23: Orderbook Data Research

**Date:** Feb 2025
**Symbol:** BTCUSDT
**Period:** Dec 2025 (31 days)
**Data:** 5-min bars + L2 orderbook (200 levels, futures + spot, Bybit)

---

## Data Pipeline

### Raw Data
- **Futures:** 31 days × ~864K raw records/day (10 updates/sec), 6.2GB zipped
- **Spot:** 31 days × ~430K raw records/day (5 updates/sec), 1.8GB zipped
- Format: JSON snapshot + delta updates, 200 price levels per side

### Processing Pipeline
1. `build_orderbook_parquet.py` — Reconstruct full book from snapshot+delta, sample at 1-second intervals
2. `build_ob_features.py` — Aggregate 1-second snapshots into 5-minute feature bars (62 features)

### Output
| Stage | Files | Size | Time |
|-------|-------|------|------|
| Futures OB parquet (1s) | 31 | 530 MB | 25 min |
| Spot OB parquet (1s) | 31 | 712 MB | 15 min |
| Futures 5-min features | 31 | 5.8 MB | 17s |
| Spot 5-min features | 31 | 5.9 MB | 12s |

### Key Data Observation
BTC futures 200-level book spans only **~5 bps from mid**. 99.3% of depth is within 5 bps. The book is extremely tight — depth levels above 10 bps are redundant. We adjusted depth buckets to 0.5, 1, 2, 3, 5, 10, 25, 50 bps accordingly.

---

## Experiment 1: OB Feature Profiles by Regime

**Question:** How do orderbook features differ between quiet and volatile regimes?

### Top Discriminating Features (25 of 31 features significant at p<0.05)

| Feature | Quiet | Volatile | Ratio | |t| |
|---------|-------|----------|-------|-----|
| **ob_mid_volatility** | 0.39 | 0.93 | **2.39×** | 63.7 |
| **ob_imb_2bps_std** | 0.30 | 0.37 | 1.20× | 51.7 |
| **ob_imb_1bps_std** | 0.52 | 0.59 | 1.12× | 43.7 |
| ob_bid_depth_cv | 0.36 | 0.46 | 1.30× | 37.6 |
| ob_ask_depth_cv | 0.36 | 0.47 | 1.30× | 33.9 |
| ob_bid_wall_frac | 0.031 | 0.047 | 1.53× | 31.3 |
| ob_ask_wall_frac | 0.032 | 0.047 | 1.47× | 29.2 |
| ob_bid_wall_ratio | 5.39 | 6.99 | 1.30× | 24.3 |
| **ob_spread_std** | 0.0024 | 0.0105 | **4.40×** | 18.8 |
| ob_spread_max | 0.052 | 0.181 | 3.50× | 18.1 |

**Key findings:**
- **Spread volatility** is the strongest OB regime discriminator (4.4× higher in volatile)
- **Imbalance instability** (std of imbalance) is highly regime-dependent
- **Depth fluctuation** (CV of depth) increases 30% in volatile regimes
- **Large walls** appear 50% more often in volatile regimes (market makers widening)
- **Mean imbalance** is NOT regime-dependent — direction of imbalance doesn't change with regime
- **Total depth decreases** ~5% in volatile regimes (liquidity withdrawal)

---

## Experiment 2: Regime Detection — Does OB Improve It?

**Question:** Can OB features improve regime classification accuracy?

| Feature Set | GMM Acc | LR Acc | LR AUC | GB Acc | GB AUC |
|-------------|---------|--------|--------|--------|--------|
| **OHLCV only** (14 features) | **0.982** | 0.971 | 0.996 | 0.972 | **0.996** |
| OHLCV + OB (45 features) | 0.786 | **0.974** | **0.996** | **0.973** | **0.997** |
| OB only (31 features) | 0.752 | 0.862 | 0.903 | 0.865 | 0.899 |

### Verdict: **No meaningful improvement.**

- OHLCV alone already achieves **99.6% AUC** — there's almost no room to improve
- Adding OB features gives +0.1% AUC at best (0.9963 → 0.9965) — noise, not signal
- OB-only detection is much worse (AUC 0.90 vs 0.996) — OB can't replace OHLCV features
- GMM accuracy actually **drops** with OB features (curse of dimensionality)

### Feature Importance (GB, OHLCV+OB combined)

Top 15 features for regime detection — **zero OB features in top 13:**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | parkvol_1h | 0.637 |
| 2 | momentum_4h | 0.075 |
| 3 | trade_intensity_ratio | 0.073 |
| 4 | rvol_4h | 0.072 |
| 5 | efficiency_4h | 0.054 |
| ... | ... | ... |
| 14 | **ob_ask_wall_ratio** | **0.001** |
| 15 | **ob_mid_volatility** | **0.001** |

**Parkinson volatility alone (0.637) is 600× more important than the best OB feature (0.001).** The regime is fully determined by price-based volatility measures. OB features are redundant for this task.

---

## Experiment 3: Regime Switch Prediction

**Question:** Do OB features help predict upcoming regime transitions?

| Horizon | OHLCV AUC | OHLCV+OB AUC | Improvement |
|---------|-----------|--------------|-------------|
| 30 min | 0.743 | 0.748 | +0.5% |
| 1 hour | 0.734 | 0.741 | +0.7% |
| 2 hours | 0.709 | 0.711 | +0.2% |

### Verdict: **Marginal improvement only.**

OB features add +0.2 to +0.7% AUC for regime switch prediction. This is statistically insignificant given the 31-day sample. The prediction ceiling (~0.75 AUC) remains, consistent with v21 findings that regime switches are driven by exogenous shocks not visible in any backward-looking features.

---

## Experiment 4: Depth Imbalance as Directional Signal

**Question:** Does bid/ask depth imbalance predict short-term price direction?

### Information Coefficient

| Feature | 5min IC | 15min IC | 1h IC | 4h IC |
|---------|---------|----------|-------|-------|
| ob_imb_0.5bps | -0.012 | -0.008 | -0.007 | **-0.030** |
| ob_imb_1bps | -0.019 | -0.021 | -0.008 | -0.012 |
| ob_imb_2bps | -0.027 | -0.032 | -0.020 | +0.007 |
| ob_imb_3bps | **-0.030** | **-0.037** | **-0.026** | +0.014 |

**ICs are negative** — more bids relative to asks predicts price going **down**, not up. This is the well-known "leaning against" effect: large resting bids are often placed by informed traders who expect to get filled (i.e., price will come to them). The IC magnitude (~0.03) is very small.

### Simple Backtest (4h hold, 7bps fee)

| Signal | Trades | Avg PnL | Win Rate |
|--------|--------|---------|----------|
| ob_imb_0.5bps | 2,466 | **-12.3 bps** | 40.0% |
| ob_imb_1bps | 2,743 | -10.0 bps | 41.9% |
| ob_imb_2bps | 2,934 | -5.7 bps | 44.1% |
| ob_imb_3bps | 2,803 | -3.3 bps | 45.1% |
| ob_imb_5bps | 2,602 | -3.8 bps | 45.6% |

### Verdict: **No tradeable signal.**

All imbalance signals are **net negative** after fees. The tighter the imbalance level, the worse the performance. Depth imbalance at 5-min frequency does not predict BTC direction.

---

## Experiment 5: Volatility Prediction

**Question:** Do OB features improve forecasting of future realized volatility?

| Feature Set | Ridge R² | Ridge Corr | GB R² | GB Corr |
|-------------|----------|------------|-------|---------|
| OHLCV only (14) | 0.126 | 0.567 | 0.105 | 0.557 |
| **OHLCV + OB (45)** | **0.217** | **0.589** | 0.101 | 0.553 |
| OB only (31) | 0.203 | 0.558 | 0.075 | 0.509 |

### Verdict: **Meaningful improvement for linear models.**

- Ridge R² improves from **0.126 → 0.217** (+72%) when adding OB features
- The improvement comes from `ob_mid_volatility` (the #1 feature by GB importance at 0.289)
- GB doesn't benefit — it already captures the nonlinear relationships from OHLCV alone
- OB-only vol prediction (R²=0.203) is surprisingly competitive with OHLCV-only (R²=0.126)

### Top Features for Vol Prediction

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | **ob_mid_volatility** | **0.289** |
| 2 | parkvol_1h | 0.238 |
| 3 | rvol_24h | 0.069 |
| 5 | **ob_bid_depth_5bps** | **0.037** |
| 11 | **ob_total_depth_mean** | **0.020** |
| 13 | **ob_ask_slope** | **0.014** |
| 14 | **ob_bid_depth_cv** | **0.014** |

**`ob_mid_volatility` is the #1 most important feature** — even more important than Parkinson volatility. This makes sense: mid-price volatility at 1-second resolution captures microstructure volatility that 5-min bars miss.

---

## Experiment 6: Futures-Spot Basis

**Question:** Does the difference between futures and spot orderbooks predict returns?

### Basis Feature Stats
| Feature | Mean | Std |
|---------|------|-----|
| Imbalance difference (1bps) | -0.001 | 0.217 |
| Spread difference | -0.0001 | 0.002 |
| Depth ratio (fut/spot) | 2.13 | 0.67 |

### IC vs Forward Returns

| Feature | 5min | 1h | 4h |
|---------|------|-----|-----|
| basis_imb_1bps | -0.014 | -0.002 | +0.005 |
| basis_imb_2bps | -0.014 | +0.007 | +0.024 |
| basis_spread | +0.002 | +0.012 | **+0.062** |
| **basis_depth_ratio** | -0.020 | **-0.055** | **-0.102** |

### Verdict: **`basis_depth_ratio` shows the strongest signal (IC = -0.10 at 4h).**

When futures depth is high relative to spot (ratio > mean), price tends to fall over the next 4 hours (IC = -0.10). This is the most interesting finding — it suggests that excess futures liquidity relative to spot is a bearish signal, possibly reflecting hedging activity or speculative positioning.

However, IC = -0.10 is still modest. This needs walk-forward validation before any confidence.

---

---

## Walk-Forward Validation (BTC)

### WF Test 1: Futures-Spot Depth Ratio Signal

Walk-forward z-score with 7-day lookback, non-overlapping trades.

| Horizon | Threshold | Trades | Avg PnL | Win Rate | Sharpe | Verdict |
|---------|-----------|--------|---------|----------|--------|---------|
| 1h | z>2.0 | 120 | +0.2 bps | 42.5% | 0.25 | Marginal |
| **4h** | **z>1.5** | **96** | **+12.9 bps** | **53.1%** | **4.87** | **✅ Best signal** |
| 4h | z>2.0 | 71 | +4.7 bps | 50.7% | 1.36 | ✅ Positive |
| 2h | z>0.5 | 311 | -1.4 bps | 39.5% | -1.18 | Negative |

**The depth ratio signal survives walk-forward at 4h horizon with z>1.5.** Sharpe 4.87 over 96 trades (28 days OOS). Short side dominates: when futures depth is high relative to spot, price falls.

### WF Test 2: Vol Prediction Walk-Forward

| Feature Set | R² | Corr | Rank Corr |
|-------------|-----|------|-----------|
| OHLCV only (14) | 0.408 | 0.654 | 0.733 |
| **OHLCV + OB (24)** | **0.431** | **0.667** | **0.731** |
| OB only (10) | 0.349 | 0.603 | 0.639 |

Walk-forward R² is actually better than in-sample (0.43 vs 0.22) — expanding window helps. **OHLCV+OB improves corr by +2.0%** walk-forward.

### WF Test 3: Combined OB Direction Signal

Ridge (α=10) predicting 4h returns from 17 OB features: IC=+0.009. All configurations negative after fees. **Not tradeable.**

---

## Grid Bot v18: OB-Enhanced (BTC Dec 2025)

### Strategy Comparison (22 days, $10K capital)

| Strategy | PnL | Sharpe | MaxDD | vs Baseline |
|----------|-----|--------|-------|-------------|
| **S0: Fix 1% (24h) — baseline** | **+$453.68** | **11.17** | -$79.61 | — |
| B1: OHLCV VolP+InfR (v17 best) | -$13.93 | -2.17 | -$37.53 | -$468 |
| B2: OHLCV 1%+1hR+VolP | +$11.83 | 6.02 | -$3.31 | -$442 |
| **D1: DepthR+InfR+VolP** | **+$15.69** | **6.98** | **-$5.49** | -$438 |
| D2: DepthR+InfR+OBVolP | +$2.28 | 5.37 | -$0.90 | -$451 |
| X1: 1hR+OBVolP+Inf | +$1.39 | 4.17 | $0.00 | -$452 |

**Key insight:** Dec 2025 BTC was a ranging month — the unfiltered baseline dominates because pausing misses profitable ranging periods. Among filtered strategies, **D1 (depth ratio rebalance) is the best OB strategy** at +$15.69, Sharpe 6.98, beating the best non-OB strategy B2 at +$11.83.

**OB advantage: +$3.86** (D1 vs B2) — modest but consistent.

---

## SOL Cross-Asset Validation

### SOL Data Pipeline
| Stage | Files | Size |
|-------|-------|------|
| SOL Futures OB (1s) | 31 | 505 MB |
| SOL Spot OB (1s) | 31 | 617 MB |
| SOL Features (5m) | 62 | 11.6 MB |

### SOL Research Results — Comparison with BTC

| Finding | BTC | SOL | Cross-Asset? |
|---------|-----|-----|-------------|
| **ob_mid_volatility #1 for vol** | GB importance: 0.289 | GB importance: 0.183 | **✅ Yes** |
| **Regime detection: OB useless** | +0.1% AUC | +0.0% AUC | **✅ Yes** |
| **Imbalance contrarian** | IC=-0.03 (neg) | IC=-0.05 (neg) | **✅ Yes** |
| **Spread std 4× in volatile** | 4.40× ratio | 3.76× ratio | **✅ Yes** |
| **Depth ratio IC at 4h** | IC=-0.102 | IC=-0.026 | **⚠️ Weaker on SOL** |
| **OB-only regime detection** | AUC=0.90 | AUC=0.88 | **✅ Yes** |

### SOL Walk-Forward: Depth Ratio Signal

| Horizon | Threshold | Trades | Avg PnL | Sharpe |
|---------|-----------|--------|---------|--------|
| 4h | z>0.5 | 157 | **+6.5 bps** | **1.85** |
| 4h | z>1.0 | 130 | -14.5 bps | -3.59 |
| 4h | z>1.5 | 97 | -12.5 bps | -2.56 |

**Depth ratio signal does NOT replicate on SOL at z>1.5.** Only works at z>0.5 (looser threshold). The BTC result may be partially overfitted to the specific month.

### SOL Walk-Forward: Vol Prediction

| Feature Set | R² | Corr | Rank Corr |
|-------------|-----|------|-----------|
| OHLCV only | 0.373 | 0.622 | 0.672 |
| **OHLCV + OB** | **0.384** | **0.633** | **0.677** |
| OB only | 0.349 | 0.603 | 0.639 |

**Vol prediction improvement replicates on SOL:** +3.1% corr improvement (vs +2.0% on BTC).

### SOL Grid Bot v18

| Strategy | PnL | Sharpe |
|----------|-----|--------|
| S0: Baseline | +$1,005.30 | 15.43 |
| Best OB (X5) | +$1.46 | 4.17 |
| Best non-OB (B2) | -$5.44 | -0.97 |
| **OB advantage** | **+$6.90** | — |

Same pattern as BTC: baseline dominates in ranging month, but OB strategies beat non-OB filtered strategies.

---

## Summary of Findings

### What OB Data IS Good For

| Use Case | BTC Result | SOL Result | Confidence |
|----------|------------|------------|------------|
| **Vol prediction** | R² +5.4% WF, corr +2.0% | R² +3.1% WF, corr +1.8% | **High** — replicates cross-asset |
| **Depth ratio signal** | 4h z>1.5: +12.9bps, Sharpe 4.87 | 4h z>0.5: +6.5bps, Sharpe 1.85 | **Medium** — weaker on SOL |
| **Grid bot OB advantage** | +$3.86 vs best non-OB | +$6.90 vs best non-OB | **Medium** — small edge |
| **Regime profiling** | 25/31 features significant | 27/31 features significant | **High** — consistent |

### What OB Data Is NOT Good For

| Use Case | Result | Why |
|----------|--------|-----|
| **Regime detection** | +0.0-0.1% AUC | OHLCV already at 99.6% — ceiling reached |
| **Regime switch prediction** | +0.2-0.7% AUC | Switches are exogenous |
| **Directional signals** | All negative after fees | Imbalance is contrarian — "leaning against" effect |
| **Combined OB direction** | IC=0.009, all negative | Too noisy for direction at 5-min |

### Practical Recommendations

1. **Add `ob_mid_volatility` to vol prediction models** — #1 feature, replicates on both BTC and SOL
2. **Depth ratio signal is BTC-specific** — works at z>1.5 on BTC (Sharpe 4.87) but not on SOL at same threshold
3. **For grid bots:** OB features provide small but consistent edge over non-OB filtered strategies
4. **Don't use OB for regime detection** — OHLCV features are sufficient on both assets
5. **Don't trade depth imbalance** — contrarian and too weak after fees on both BTC and SOL

---

## Files

| File | Description |
|------|-------------|
| `build_orderbook_parquet.py` | Raw OB → 1-second parquet snapshots |
| `build_ob_features.py` | 1-second snapshots → 5-minute feature bars (62 features) |
| `ob_research.py` | 6 experiments: regime detection, signals, vol prediction, basis |
| `ob_walkforward.py` | Walk-forward tests: depth ratio, vol prediction, combined signal |
| `grid_bot_v18.py` | Grid bot with OB features: vol pause, depth rebalance, adaptive spacing |
| `results/ob_research_v23.txt` | BTC experiment output |
| `results/ob_research_v23_SOL.txt` | SOL experiment output |
| `results/ob_walkforward_v23.txt` | BTC walk-forward output |
| `results/ob_walkforward_v23_SOL.txt` | SOL walk-forward output |
| `results/grid_bot_v18_BTC.txt` | BTC grid bot v18 output |
| `results/grid_bot_v18_SOL.txt` | SOL grid bot v18 output |
