# Research Findings v8 — Regime Detection (5 Symbols, 13 Months)

**Date:** 2026-02-16
**Exchange:** Bybit Futures (VIP0)
**Symbols:** BTCUSDT, ETHUSDT, SOLUSDT, DOGEUSDT, XRPUSDT
**Period:** 2025-01-01 → 2026-01-31 (396 days, ~114K bars per symbol)
**Method:** 5-minute bars from tick data, forward-looking regime labels, 60 backward-looking features
**Runtime:** ~36 min total (5 symbols sequentially, first run builds cache)

---

## Why Regime Detection?

Our previous research (v1–v7) revealed a fundamental problem: **every strategy we tested depends on market regime, but none can detect it reliably.**

- Grid strategies earn in range-bound markets but bleed in trends (v5, v6)
- Signal strategies have weak edge everywhere (v6)
- Grid+trend combinations fail because the regime switch is too slow or too noisy (v7)

**Regime detection is the bottleneck.** If we can detect regime transitions early, we can:
1. Turn grids on/off at the right time
2. Adjust position sizing based on volatility regime
3. Select the right signal strategy for the current regime
4. Avoid the most dangerous periods entirely

## What Can We Actually Predict?

From first principles, we hypothesized:
- **Volatility is highly predictable** — vol clusters, mean-reverts, and has strong autocorrelation
- **Trend vs range is moderately predictable** — efficiency ratio and return autocorrelation carry information
- **Direction is essentially unpredictable** — confirmed by v6 (best Sharpe 0.08)

**Our 13-month, 5-symbol experiments confirm the first hypothesis strongly. The second is weaker than expected.**

## Experimental Design

### Ground Truth (forward-looking labels)

We define regimes using a **48-bar (4-hour) forward window**:

| Regime | Definition | Actionable Meaning |
|--------|-----------|-------------------|
| **quiet_range** | Low vol + ranging | Best for tight grids |
| **active_range** | Normal/high vol + ranging | Best for wide grids |
| **smooth_trend** | Low/normal vol + trending | Best for trend following |
| **volatile_trend** | High vol + trending | Reduce size, hedge |
| **chaos** | High vol + mixed | Stay out |

Volatility threshold: 0.6× and 1.5× of 3-day rolling median realized vol.
Trend threshold: efficiency ratio > 0.4 (trending) or < 0.25 (ranging).

### Features (backward-looking only)

60 features computed from past data, grouped into:
- **Volatility:** realized vol (5 windows), Parkinson vol (5 windows), vol ratios, vol acceleration
- **Trend:** efficiency ratio (4 windows), return autocorrelation (4 windows), ADX, sign persistence
- **Microstructure:** trade imbalance, large trade fraction, ITI clustering, bar efficiency
- **Price structure:** momentum, price vs SMA, VWAP deviation

### Evaluation Criteria

| Metric | Pass Threshold | Meaning |
|--------|---------------|---------|
| Accuracy | >60% (binary) | Better than naive |
| F1 Score | >0.30 (vol), >0.10 (trend) | Balanced precision/recall |
| Avg Run Length | >12 bars | Not flipping every few minutes |
| Correlation | >0.20 at lag 1 | Feature actually leads the regime |

---

## Result 1: Volatility Is Highly Predictable (Confirmed at Scale)

### Correlation with Future Volatility (13-month, per symbol)

| Feature | BTC | ETH | SOL | DOGE | XRP |
|---------|-----|-----|-----|------|-----|
| parkvol_1h (lag 1) | **+0.58** | **+0.54** | **+0.54** | +0.36 | +0.37 |
| parkvol_2h (lag 1) | **+0.58** | **+0.54** | **+0.54** | +0.36 | +0.37 |
| parkvol_4h (lag 1) | **+0.57** | **+0.54** | **+0.55** | +0.37 | +0.38 |
| rvol_2h (lag 1) | **+0.53** | **+0.50** | **+0.50** | +0.34 | +0.36 |
| bar_eff_4h (lag 1) | **-0.43** | **-0.40** | **-0.40** | -0.30 | -0.31 |

**Key findings:**
- Correlations are **remarkably consistent** across BTC, ETH, and SOL (r = 0.53–0.58)
- DOGE and XRP show **weaker but still significant** correlations (r = 0.34–0.38) — likely due to different microstructure (more retail-driven, less institutional flow)
- Parkinson volatility (high-low based) remains the single best predictor across all assets
- Correlations persist out to **48 bars (4 hours)** at r > 0.27 — long-lasting signal

### Cross-Symbol Feature Consistency (passing >60% accuracy on all 5 symbols)

| Feature | BTC (acc/F1) | ETH (acc/F1) | SOL (acc/F1) | DOGE (acc/F1) | XRP (acc/F1) |
|---------|-------------|-------------|-------------|--------------|-------------|
| **parkvol_1h** | 0.64/0.42 | 0.63/0.38 | 0.70/0.36 | 0.69/0.35 | 0.70/0.38 |
| **trade_intensity** | 0.64/0.43 | 0.63/0.39 | 0.69/0.35 | 0.69/0.35 | 0.63/0.37 |
| **rvol_1h** | 0.63/0.41 | 0.63/0.37 | 0.69/0.35 | 0.63/0.35 | 0.70/0.37 |
| **vol_ratio_2h_24h** | 0.63/0.41 | 0.63/0.37 | 0.69/0.35 | 0.76/0.35 | 0.70/0.36 |
| **vol_ratio_1h_24h** | 0.63/0.41 | 0.69/0.37 | 0.69/0.34 | 0.69/0.35 | 0.69/0.36 |
| **vol_ratio_bar** | 0.64/0.42 | 0.62/0.37 | 0.62/0.34 | 0.62/0.34 | 0.63/0.36 |

**6 features pass on ALL 5 symbols.** This is the most robust finding — these features are universally predictive of high-volatility regimes.

### Best Vol Detector Per Symbol

| Symbol | Best Feature | Accuracy | F1 | Precision | Recall |
|--------|-------------|----------|-----|-----------|--------|
| BTCUSDT | trade_intensity_ratio | **64.4%** | 0.426 | 0.330 | 0.599 |
| ETHUSDT | trade_intensity_ratio | **63.4%** | 0.388 | 0.289 | 0.588 |
| SOLUSDT | parkvol_1h | **70.0%** | 0.364 | 0.286 | 0.501 |
| DOGEUSDT | parkvol_1h | **69.3%** | 0.354 | 0.281 | 0.480 |
| XRPUSDT | parkvol_1h | **70.1%** | 0.375 | 0.299 | 0.503 |

**Verdict: ✅ Volatility detection WORKS across all 5 symbols and 13 months.** 63–70% accuracy, F1 0.35–0.43. Consistent and robust.

### Early Warning for High-Vol Transitions (averaged across all 5 symbols)

Before transitions **→ high_vol** (~900–1170 transitions per symbol):

| Feature | 12 bars before | 6 bars before | 3 bars before | 1 bar before |
|---------|---------------|---------------|---------------|-------------|
| vol_ratio_1h_24h | **+0.39** | +0.42 | +0.44 | +0.47 |
| vol_ratio_2h_24h | **+0.39** | +0.42 | +0.44 | +0.46 |
| trade_intensity_ratio | **+0.31** | +0.35 | +0.38 | +0.45 |
| vol_accel_4h | **+0.31** | +0.33 | +0.35 | +0.36 |
| parkvol_1h | **+0.28** | +0.30 | +0.32 | +0.34 |

**Key insight:** Vol ratios show elevated z-scores **12 bars (1 hour) before** the regime transitions to high-vol. The signal builds gradually — not a sudden spike. This gives **actionable lead time** for position reduction.

---

## Result 2: Trend Detection Fails with Simple Thresholds

### Binary Trend Detection (best per symbol)

| Symbol | Best Feature | Accuracy | F1 | Note |
|--------|-------------|----------|-----|------|
| BTCUSDT | imbalance_persistence | 88.5% | 0.054 | Predicts "not trending" |
| ETHUSDT | bar_eff_4h | 78.8% | 0.055 | |
| SOLUSDT | ret_autocorr_4h | 88.4% | 0.055 | |
| DOGEUSDT | bar_eff_4h | 88.5% | 0.066 | |
| XRPUSDT | bar_eff_4h | 79.4% | 0.047 | |

**Verdict: ❌ Trend detection FAILS.** F1 scores are 0.05–0.07 across all symbols. The detectors achieve high accuracy by predicting "not trending" almost always — since trending is only **1.1–2.5% of bars** over 13 months, this is essentially the naive baseline.

**Why trending is so rare in 13 months vs 3 months:** With a longer dataset, the proportion of trending bars drops because most of the year is range-bound. The 3-month window (Nov–Jan) happened to include more volatile trending periods.

### Correlation with Future Trend

Correlations with future efficiency ratio are weak across all symbols (max r ≈ 0.09). No single feature reliably predicts trend onset.

---

## Result 3: Regime Structure (13 Months)

### Regime Distribution

| Regime | BTC | ETH | SOL | DOGE | XRP |
|--------|-----|-----|-----|------|-----|
| **active_range** | 81.3% | 84.0% | 88.6% | 87.4% | 87.0% |
| **quiet_range** | 12.3% | 9.6% | 5.8% | 6.9% | 7.2% |
| **chaos** | 4.2% | 3.9% | 3.2% | 3.4% | 3.5% |
| **smooth_trend** | 1.1% | 1.1% | 1.5% | 1.2% | 1.2% |
| **volatile_trend** | 1.0% | 1.4% | 0.9% | 1.1% | 1.1% |
| **high_vol (binary)** | 22.0% | 19.7% | 17.1% | 17.5% | 17.8% |
| **trending (binary)** | 2.1% | 2.5% | 2.3% | 2.3% | 2.3% |

**Key observations:**
1. Markets are **ranging 85–94% of the time** over 13 months — grids have a massive structural advantage
2. **Trending is extremely rare** (1–2.5%) — this is why trend-following F1 is so low
3. **BTC has the most quiet_range** (12.3%) — it's the "calmest" of the 5
4. **SOL has the most active_range** (88.6%) — consistently active but not trending
5. **High-vol occurs 17–22% of the time** — frequent enough to matter for risk management

### Transition Patterns (consistent across all symbols)

Most common transitions:
1. **active_range ↔ quiet_range** — the dominant cycle
2. **active_range ↔ chaos** — vol spikes without direction
3. **chaos ↔ volatile_trend** — chaos sometimes resolves into trend
4. **active_range → smooth_trend** — rare but important breakouts

---

## Practical Implications

### What we CAN do now (high confidence):

1. **Vol-based position sizing** — reduce exposure when `parkvol_1h` or `vol_ratio_2h_24h` is elevated. Works on ALL 5 symbols. This alone would prevent the worst grid losses.

2. **Early warning system** — `vol_ratio_1h_24h` rising above normal gives ~1 hour lead time before high-vol regime. Start reducing grid levels and tightening stops.

3. **Chaos detection → pause** — when multiple vol features are elevated AND bar efficiency is low, we're in chaos. Pause all strategies.

### What we CANNOT do reliably:

1. **Predict trend onset** — trending is only 1–2.5% of bars. No feature predicts it with useful F1.

2. **Distinguish between symbols** — the same features work on all 5 symbols. No symbol-specific tuning needed (a good thing for generalization).

3. **Predict regime duration** — transitions happen frequently (~900–2400 per symbol per year).

### Recommended next steps:

1. **Build a vol-adaptive position sizer** using `parkvol_1h` — highest-confidence, universal finding
2. **Test regime-filtered grid** — grid ON only when vol features are below threshold
3. **Try ML classifiers** (RF, XGBoost) on the 60-feature set — might find non-linear combinations for trend detection
4. **Explore longer forward windows** (96, 288 bars) — trend might be more predictable at daily scale
5. **Test on out-of-sample period** — use Jan 2025–Oct 2025 for training, Nov 2025–Jan 2026 for validation

---

## Success/Failure Criteria Applied

| Experiment | Target | Result | Verdict |
|-----------|--------|--------|---------|
| Vol detection (single feature, 5 sym) | is_high_vol | 63–70% acc, F1 0.35–0.43 | **✅ PASS** |
| Vol correlation (parkvol_1h, 5 sym) | fwd_vol | r=0.37–0.58 at lag 1 | **✅ STRONG PASS** |
| Vol early warning (5 sym) | → high_vol | z=+0.28–0.39 at 12 bars | **✅ PASS** |
| Feature consistency (5 sym) | is_high_vol | 6 features pass all 5 symbols | **✅ STRONG PASS** |
| Trend detection (single feature) | is_trending | F1 = 0.05–0.07 | **❌ FAIL** |
| Trend correlation | fwd_efficiency | r < 0.09 | **❌ FAIL** |
| Composite detectors | both | Worse than single features | **❌ FAIL** |

**Bottom line:** Volatility regime detection is robust, universal, and actionable across all 5 crypto assets over 13 months. Trend detection does not work with simple threshold-based detectors — trending is too rare (1–2.5% of bars) for binary classification to be useful.

---

## Files

| File | Description |
|------|-------------|
| `regime_detection.py` | Full experiment suite with caching, multi-symbol mode, cross-symbol comparison |
| `results/regime_ALL_5sym_13months.txt` | Complete output for all 5 symbols |
| `results/regime_BTCUSDT_*.txt` | BTC 3-month results (earlier run) |
| `results/regime_ETHUSDT_*.txt` | ETH 3-month results (earlier run) |
| `results/regime_SOLUSDT_*.txt` | SOL 3-month results (earlier run) |
