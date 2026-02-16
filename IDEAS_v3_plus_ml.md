# Combination Ideas: v3 Directional Signals + v9-12 ML Predictions

**Date:** 2026-02-16
**Core insight:** v3 has **direction** (which way), v9-12 has **magnitude** (how much). Neither alone is complete.

---

## Idea #1: Vol-Adaptive Position Sizing for v3 Signals

**Priority: HIGH | Complexity: LOW | Expected Impact: HIGH**

### Concept
v3 signals use fixed position sizes. But volatility varies 3-5× across regimes. Scale position inversely to predicted vol so each trade carries equal dollar risk.

### Formula
```
position_size = base_size × (target_vol / predicted_vol)
target_vol = median predicted vol across dataset
```

### Implementation Plan
1. Load data with BOTH pipelines: v3 features (tick-based) + v9 regime features (OHLCV-based)
2. Align timestamps (both are 5m bars)
3. For each v3 signal entry, look up the current predicted vol from Ridge model
4. Scale PnL by position_size multiplier
5. Compare Sharpe ratio: fixed sizing vs vol-adaptive sizing

### Test Plan
- **Small:** BTCUSDT, 30 days (Dec 2025), E01 + E09 signals only
- **Expand if promising:** All 3 symbols, 90+ days, all winning signals

### Success Criteria
- Sharpe ratio improvement > 20%
- Max drawdown reduction
- Same or better total PnL

---

## Idea #2: Vol-Regime Filtered Signals

**Priority: MEDIUM | Complexity: LOW | Expected Impact: MEDIUM**

### Concept
Some signals work better in specific vol regimes. E03 (vol breakout) should only fire when vol is expanding. E01 (contrarian) may work better in calm markets.

### Implementation Plan
1. Compute predicted vol and vol regime (low/medium/high) from v9 Ridge model
2. For each signal, test 3 variants:
   - **All regimes** (baseline, same as v3)
   - **Low-vol only** (predicted vol < median)
   - **High-vol only** (predicted vol > median)
   - **Rising vol only** (vol_accel > 0)
3. Compare avg PnL and trade count per variant

### Test Plan
- **Small:** BTCUSDT, 30 days, E01 + E03 + E09
- **Expand if promising:** All symbols, 90+ days

### Success Criteria
- At least one regime filter improves avg PnL by > 5 bps
- Trade count doesn't drop below 50% of baseline

---

## Idea #3: Dynamic Hold Period & Take-Profit from Predicted Range

**Priority: HIGH | Complexity: LOW-MEDIUM | Expected Impact: HIGH**

### Concept
v3 uses fixed 4h holding. But v11 predicts the expected range for the next 1h/4h. Use predicted range as a dynamic take-profit level, and exit early if TP is hit.

### Formula
```
TP_level = entry_price ± predicted_range_P75 × trade_direction
exit when: price hits TP OR holding period expires (whichever first)
```

### Implementation Plan
1. For each v3 signal entry, compute predicted range (P50 and P75) from Ridge model
2. Simulate bar-by-bar: check if price hits TP before holding period expires
3. Compare: fixed 4h hold vs dynamic TP at P50 vs dynamic TP at P75
4. Also test: exit early if predicted vol drops (signal exhaustion)

### Test Plan
- **Small:** BTCUSDT, 30 days, E09 (strongest signal)
- **Expand if promising:** All symbols, all winning signals

### Success Criteria
- Higher avg PnL per trade (capture profits earlier, avoid giveback)
- Higher win rate (TP locks in gains)
- Similar or better total PnL despite potentially fewer holding bars

---

## Idea #4: Breakout Confirmation Filter

**Priority: MEDIUM | Complexity: LOW | Expected Impact: LOW-MEDIUM**

### Concept
v12 breakout model (AUC 0.69) is too weak alone. But as a confirmation filter for v3 momentum signals (E03, E09), it could reduce false positives.

### Implementation Plan
1. Compute breakout probability from LGBM model at each bar
2. For momentum signals (E03, E09): only enter if breakout_prob > threshold
3. For contrarian signals (E01): only enter if breakout_prob < threshold
4. Test thresholds: 0.3, 0.4, 0.5, 0.6

### Test Plan
- **Small:** BTCUSDT, 30 days, E03 + E09
- **Expand if promising:** All symbols

### Success Criteria
- Precision improvement > 10% (fewer false signals)
- Avg PnL improvement > 3 bps
- Acceptable trade count reduction (< 50% drop)

---

## Idea #5: Full Stack Trade Chain

**Priority: MEDIUM | Complexity: HIGH | Expected Impact: POTENTIALLY HIGH**

### Concept
Layer all predictions into a single decision chain:
1. **Vol prediction** → "Is this a good time to trade?" (regime filter)
2. **Breakout model** → "Is a big move likely?" (confirmation)
3. **v3 signal** → "Which direction?" (entry)
4. **Range quantile** → "How far will it go?" (TP/SL)
5. **Vol prediction** → "How much to risk?" (position sizing)

### Implementation Plan
1. Compute all predictions at each bar
2. Define entry rules: vol regime = active AND breakout_prob > 0.5 AND v3 signal fires
3. Define exit rules: hit P75 range TP OR vol drops below threshold OR max hold 4h
4. Define sizing: inverse vol
5. Backtest the full chain vs each component alone

### Test Plan
- **Small:** BTCUSDT, 30 days, E09 signal only
- **Expand if promising:** All symbols, multiple signals

### Success Criteria
- Sharpe ratio > 1.5 (vs ~0.5-1.0 for individual components)
- Positive PnL after fees on out-of-sample data
- Fewer but higher-quality trades

---

## Idea #6: Contrarian Signal + Vol Compression Setup

**Priority: MEDIUM | Complexity: MEDIUM | Expected Impact: SPECULATIVE**

### Concept
During vol compression (tight range), retail traders accumulate positions. When the range breaks, it snaps against them. Combine:
- v12 consolidation features detect compression
- v3 E01 contrarian imbalance detects retail positioning
- Entry: compression + strong retail imbalance → trade opposite to retail

### Implementation Plan
1. Compute consolidation_2h_vs_24h from v12 features
2. Compute E01 contrarian signal from v3
3. Entry: consolidation < 0.5 AND |E01 signal| > threshold
4. Direction: opposite to retail imbalance (contrarian)
5. TP: predicted range P75 (the compression release)

### Test Plan
- **Small:** BTCUSDT, 30 days
- **Expand if promising:** All symbols

### Success Criteria
- Higher avg PnL than E01 alone (> 15 bps vs 13.68 bps)
- Reasonable trade count (> 30/month)

---

## Idea #7: Grid Bot + Directional Bias

**Priority: MEDIUM | Complexity: HIGH | Expected Impact: HIGH (if it works)**

### Concept
Combine vol-adaptive grid bot (v9/v11) with directional signals (v3):
- Grid runs continuously with vol-adaptive spacing
- When a v3 signal fires, shift grid center in signal direction (asymmetric grid)
- When no signal, symmetric grid around current price

### Implementation Plan
1. Build a simple grid bot simulator with configurable:
   - Grid width (from v11 P50/P90 range prediction)
   - Grid center offset (from v3 signal direction)
   - Number of grid levels
2. Simulate on historical data:
   - Baseline: symmetric grid, fixed width
   - v11 only: symmetric grid, adaptive width
   - v3+v11: asymmetric grid, adaptive width
3. Track: total fills, PnL, max drawdown, capital efficiency

### Test Plan
- **Small:** BTCUSDT, 30 days, E09 signal for bias
- **Expand if promising:** All symbols, longer periods

### Success Criteria
- Higher total PnL than symmetric adaptive grid
- Lower max drawdown
- More fills on the "right" side of the grid

---

## Execution Order & Rationale

| Order | Idea | Why This Order |
|-------|------|---------------|
| 1 | **#1 Vol-adaptive sizing** | Simplest, highest expected impact, validates the combination concept |
| 2 | **#3 Dynamic TP from range** | Second simplest, directly uses v11 range prediction |
| 3 | **#2 Vol-regime filter** | Quick to test, may reveal regime-specific signal behavior |
| 4 | **#4 Breakout confirmation** | Tests whether v12 adds value as a filter |
| 5 | **#6 Contrarian + compression** | Creative combination, tests a specific market hypothesis |
| 6 | **#5 Full stack chain** | Most complex, builds on learnings from #1-#4 |
| 7 | **#7 Grid bot + bias** | Most ambitious, requires grid simulator, do last |

**Rule:** If an idea shows no improvement on the small test, skip expansion and move to the next idea. Don't waste time on dead ends.

---

## Data Pipeline Notes

v3 and v9-12 use different data pipelines:
- **v3:** `experiments.py` → `load_features()` → tick-level features (vol_imbalance, kyle_lambda, etc.)
- **v9-12:** `regime_detection.py` → `load_bars()` + `compute_regime_features()` → OHLCV + rolling features

Both produce 5m bars with aligned timestamps. The combination script must:
1. Load v3 features (tick-based microstructure)
2. Load v9 features (OHLCV-based regime features) OR compute them from v3 OHLCV columns
3. Merge on timestamp
4. Run v3 signals with v9-12 enhancements
