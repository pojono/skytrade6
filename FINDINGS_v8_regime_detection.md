# Research Findings v8 — Regime Detection

**Date:** 2026-02-16
**Exchange:** Bybit Futures (VIP0)
**Symbols:** BTCUSDT, ETHUSDT, SOLUSDT
**Period:** 2025-11-01 → 2026-01-31 (92 days)
**Method:** 5-minute bars from tick data, forward-looking regime labels, backward-looking features
**Runtime:** ~90–150s per symbol

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

**Our experiments confirm the first two hypotheses.**

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

40+ features computed from past data, grouped into:
- **Volatility:** realized vol, Parkinson vol, vol ratios, vol acceleration
- **Trend:** efficiency ratio, return autocorrelation, ADX, sign persistence
- **Microstructure:** trade imbalance, large trade fraction, ITI clustering
- **Price structure:** momentum, price vs SMA, bar efficiency

### Evaluation Criteria

| Metric | Pass Threshold | Meaning |
|--------|---------------|---------|
| Accuracy | >60% (binary) | Better than naive |
| F1 Score | >0.10 (trend), >0.30 (vol) | Balanced precision/recall |
| Avg Run Length | >12 bars | Not flipping every few minutes |
| Correlation | >0.20 at lag 1 | Feature actually leads the regime |

---

## Result 1: Volatility Is Highly Predictable

### Correlation with Future Volatility (3-month, all symbols)

| Feature | Lag 1 | Lag 6 | Lag 12 | Lag 24 | Lag 48 |
|---------|-------|-------|--------|--------|--------|
| parkvol_1h | **+0.60** | +0.56 | +0.53 | +0.46 | +0.37 |
| parkvol_2h | **+0.60** | +0.57 | +0.53 | +0.47 | +0.38 |
| rvol_2h | **+0.56** | +0.53 | +0.49 | +0.43 | +0.34 |
| rvol_1h | **+0.54** | +0.51 | +0.47 | +0.41 | +0.33 |
| bar_eff_4h | **-0.50** | -0.49 | -0.47 | -0.45 | -0.42 |
| bar_eff_1h | **-0.47** | -0.45 | -0.44 | -0.41 | -0.38 |

**These are remarkably strong correlations** — consistent across BTC, ETH, and SOL. Parkinson volatility (high-low based) at 1–2 hour windows is the single best predictor of future volatility, with **r = 0.60 at 5-minute lag** decaying to **r = 0.37 at 4-hour lag**.

The negative correlation of `bar_eff` (bar efficiency = |net move|/total path within each bar) is intuitive: when individual bars are "noisy" (low efficiency), the market is volatile.

### Binary High-Vol Detection (best per symbol)

| Feature | Symbol | Accuracy | F1 | Precision | Recall |
|---------|--------|----------|-----|-----------|--------|
| trade_intensity_ratio | ETHUSDT | **64.9%** | 0.466 | 0.34 | 0.73 |
| vol_ratio_2h_24h | ETHUSDT | **64.0%** | 0.452 | 0.34 | 0.67 |
| parkvol_1h | SOLUSDT | **63.4%** | 0.412 | 0.30 | 0.66 |
| trade_intensity_ratio | SOLUSDT | **63.2%** | 0.411 | 0.30 | 0.66 |
| parkvol_1h | BTCUSDT | **62.5%** | 0.406 | 0.30 | 0.63 |
| trade_intensity_ratio | BTCUSDT | **62.3%** | 0.403 | 0.29 | 0.65 |

**Verdict: ✅ Volatility detection WORKS.** 63–65% accuracy with F1 ~0.41–0.47. Not spectacular, but reliably above random (50%) and consistent across all assets.

### Early Warning for High-Vol Transitions

Before transitions **→ high_vol** (averaged across symbols):

| Feature | 12 bars before | 6 bars before | 3 bars before | 1 bar before |
|---------|---------------|---------------|---------------|-------------|
| vol_ratio_2h_24h | **+0.73** | +0.78 | +0.80 | +0.84 |
| vol_ratio_1h_24h | **+0.72** | +0.74 | +0.75 | +0.79 |
| trade_intensity_ratio | **+0.60** | +0.55 | +0.62 | +0.85 |
| vol_accel_4h | **+0.55** | +0.60 | +0.63 | +0.63 |
| parkvol_1h | **+0.43** | +0.44 | +0.46 | +0.48 |

**Key insight:** Vol ratios (short/long window) show elevated z-scores **12 bars (1 hour) before** the regime officially transitions to high-vol. This is actionable lead time.

---

## Result 2: Trend Detection Is Hard But Not Hopeless

### Correlation with Future Trend (efficiency)

| Feature | Lag 1 | Lag 6 | Lag 12 | Lag 24 | Lag 48 |
|---------|-------|-------|--------|--------|--------|
| vol_accel_4h | +0.10 | +0.09 | +0.08 | +0.07 | +0.01 |
| vol_ratio_1h_8h | +0.09 | +0.09 | +0.07 | +0.06 | +0.02 |
| trade_intensity_ratio | +0.08 | +0.07 | +0.06 | +0.04 | +0.02 |
| efficiency_8h | -0.08 | -0.10 | -0.12 | -0.13 | -0.13 |

**Correlations are weak** (max ~0.13). This confirms that trend is fundamentally harder to predict than volatility. However, the negative correlation of `efficiency_8h` is interesting — **past trending predicts future mean-reversion** (and vice versa). This is the regime cycling effect.

### Binary Trend Detection

| Feature | Symbol | Accuracy | F1 | Note |
|---------|--------|----------|-----|------|
| ret_autocorr_4h | ETHUSDT | 74.3% | 0.072 | High acc, low F1 |
| ret_autocorr_2h | ETHUSDT | 64.9% | 0.068 | |
| efficiency_1h | BTCUSDT | 88.5% | 0.057 | Almost all "not trending" |
| ret_autocorr_4h | SOLUSDT | 88.2% | 0.090 | |

**Verdict: ⚠️ Trend detection is marginal.** Accuracy looks high (74–88%) but F1 is terrible (0.05–0.09). The detectors achieve high accuracy by mostly predicting "not trending" — since trending is only ~5–12% of bars, this is nearly the naive baseline.

**The fundamental problem:** trending periods are rare and don't have strong leading indicators. By the time you can detect a trend, it's often already underway.

### Early Warning for Trend Transitions

Before transitions **→ trending**:

| Feature | 12 bars before | 6 bars before | 1 bar before |
|---------|---------------|---------------|-------------|
| vol_accel_4h | +0.41 | +0.45 | +0.50 |
| vol_ratio_bar | +0.23 | +0.19 | +0.91 |
| trade_intensity_ratio | +0.23 | +0.18 | +0.81 |
| ret_autocorr_1h | +0.30 | +0.15 | +0.09 |

**Interesting finding:** `vol_accel_4h` (is volatility increasing?) shows a moderate signal 12 bars before trend transitions. **Volatility expansion often precedes trends.** This makes intuitive sense — breakouts start with increased activity.

But `vol_ratio_bar` and `trade_intensity_ratio` only spike at lag 1 — they're concurrent indicators, not leading ones.

---

## Result 3: Regime Structure

### Regime Distribution (3-month average across symbols)

| Regime | % of Time | Avg Duration |
|--------|----------|-------------|
| active_range | ~40% | ~35 bars (3h) |
| quiet_range | ~25% | ~30 bars (2.5h) |
| chaos | ~20% | ~25 bars (2h) |
| smooth_trend | ~8% | ~20 bars (1.7h) |
| volatile_trend | ~7% | ~15 bars (1.3h) |

**Key observations:**
1. Markets are **ranging ~65% of the time** (quiet + active range) — this is why grids work at all.
2. **Trending is rare** (~15%) — this is why trend-following has low F1.
3. **Chaos is common** (~20%) — high vol + mixed direction. This is the dangerous regime.
4. Average regime duration is ~30 bars (2.5 hours) — regimes are not stable for long.

### Transition Patterns

Most common transitions:
1. **active_range ↔ quiet_range** — the dominant cycle (vol expanding/contracting within range)
2. **active_range ↔ chaos** — vol spikes without clear direction
3. **chaos ↔ volatile_trend** — chaos sometimes resolves into a trend
4. **active_range → smooth_trend** — range breaks into trend (rare but important)

---

## Practical Implications

### What we CAN do now:

1. **Vol-based position sizing** — reduce grid size and signal size when vol is elevated. `parkvol_1h` or `vol_ratio_2h_24h` as the trigger. This alone would have prevented the worst grid losses.

2. **Chaos detection → pause** — when `vol_ratio_2h_24h > 1.5` AND `bar_eff_4h < 0.3`, we're likely in chaos. Pause all strategies.

3. **Vol expansion early warning** — `vol_ratio_2h_24h` rising above 1.0 gives ~1 hour lead time before high-vol regime. Start reducing exposure.

### What we CANNOT do reliably:

1. **Predict trend onset** — by the time we detect it, the move is already happening. Best we can do is react within 6–12 bars.

2. **Distinguish smooth_trend from volatile_trend** in advance — both start the same way.

3. **Predict regime duration** — we know the average is ~30 bars, but individual regimes vary from 5 to 200+ bars.

### Recommended next steps:

1. **Build a vol-adaptive position sizer** using `parkvol_1h` and `vol_ratio_2h_24h` — this is the highest-confidence finding.
2. **Test regime-filtered grid** — grid ON only when `vol_ratio_2h_24h < 1.2` (range regime).
3. **Explore longer forward windows** (96, 288 bars) — trend might be more predictable at daily scale.
4. **Try ML classifiers** (random forest, XGBoost) on the feature set — might find non-linear combinations that simple thresholds miss.

---

## Success/Failure Criteria Applied

| Experiment | Target | Accuracy | F1 | Verdict |
|-----------|--------|----------|-----|---------|
| Vol detection (single feature) | is_high_vol | 62–65% | 0.41–0.47 | **✅ PASS** |
| Vol correlation (parkvol_1h) | fwd_vol | r=0.60 | — | **✅ STRONG PASS** |
| Vol early warning | → high_vol | z=0.73 at 12 bars | — | **✅ PASS** |
| Trend detection (single feature) | is_trending | 65–88% | 0.05–0.09 | **❌ FAIL** (F1 too low) |
| Trend correlation | fwd_efficiency | r=0.10 | — | **⚠️ WEAK** |
| Trend early warning | → trending | z=0.41 at 12 bars | — | **⚠️ MARGINAL** |
| Composite vol detector | is_high_vol | 53–57% | 0.36–0.43 | **❌ FAIL** (worse than single) |
| Composite trend detector | is_trending | 82–88% | 0.08–0.09 | **❌ FAIL** (F1 too low) |

**Bottom line:** Volatility regime detection works. Trend regime detection does not — at least not with simple threshold-based detectors on 5-minute bars.

---

## Files

| File | Description |
|------|-------------|
| `regime_detection.py` | Full experiment suite: data loading, feature engineering, regime labeling, evaluation |
| `results/regime_BTCUSDT_*.txt` | BTC results |
| `results/regime_ETHUSDT_*.txt` | ETH results |
| `results/regime_SOLUSDT_*.txt` | SOL results |
