# ML System Audit — Settlement Trading Pipeline

**Date:** 2026-03-01  
**Scope:** Full audit of all ML models, training methodology, feature engineering, backtest assumptions, and data integrity.

---

## Executive Summary

| Area | Verdict | Severity |
|------|---------|----------|
| **Short exit ML** | Trained but UNUSED in backtest | 🔴 Critical |
| **Long exit ML** | Working, decent precision at threshold 0.6 | 🟢 OK |
| **Long entry decision** | Rule-based (bottom_t ≤ 15s) — uses look-ahead | 🔴 Critical |
| **Train/test split** | Alphabetical by symbol, not temporal | 🟡 Misleading |
| **LOSO validation** | Correct methodology, honest estimate | 🟢 OK |
| **Bottom detection** | Perfect hindsight — look-ahead bias | 🔴 Critical |
| **Short PnL model** | Fixed 23.6 bps constant, no per-trade prediction | 🟡 Simplification |
| **OB reconstruction** | Working, 100% success rate | 🟢 OK |
| **Sample size** | 160 settlements, 4 days — very thin | 🔴 Critical |
| **Price bins** | Last trade per 100ms, not VWAP | 🟡 Optimistic |
| **Feature NaN rate** | 0% — clean | 🟢 OK |
| **Production model** | Trained on ALL data (no holdout) | 🟡 Accepted risk |

---

## 1. Models Inventory

### 1.1 Short Exit ML (`short_exit_logreg`, `short_exit_hgbc`)

| Property | Value |
|----------|-------|
| **Purpose** | Predict when price is near the crash bottom |
| **Target** | `target_near_bottom_10` — within 10 bps of eventual min |
| **Features** | 56 tick-level features at 100ms resolution |
| **Models** | LogisticRegression (C=0.1) + HGBC (max_depth=6) |
| **Training data** | 93,446 ticks from 160 settlements |
| **Target positive rate** | 39.6% |
| **Saved to** | `models/short_exit_logreg.pkl`, `models/short_exit_hgbc.pkl` |
| **Used in backtest?** | **NO** — backtest uses fixed 23.6 bps constant |

**🔴 CRITICAL ISSUE:** These models are trained and saved but **never loaded or used** in `simulate_settlement()`. The short leg PnL is computed as:
```python
short_net = GROSS_PNL_BPS - rt_slip + fee_save  # 23.6 bps constant
```
No ML inference happens for the short leg at runtime. The model's LOSO AUC was used to derive the 23.6 bps constant, but the model itself is not used for per-trade exit timing.

### 1.2 Long Entry Decision (`should_go_long`)

| Property | Value |
|----------|-------|
| **Type** | Rule-based (not ML) |
| **Rule** | `bottom_t ≤ 15s` |
| **Eligible** | 83/127 filtered settlements (65%) |
| **Bottom timing** | Median 9.3s, P25=3.5s, P75=21.1s |

**🔴 CRITICAL ISSUE:** `bottom_t` comes from `find_bottom()`, which scans ALL price bins from 1s to 30s to find the **global minimum**. This is perfect hindsight — in production you cannot know the bottom until after it has passed. The long entry decision therefore benefits from look-ahead bias.

### 1.3 Long Exit ML (`long_exit_logreg`)

| Property | Value |
|----------|-------|
| **Purpose** | Predict when recovery is near its peak |
| **Target** | `target_near_peak_10` — within 10 bps of future max |
| **Features** | 28 recovery tick features |
| **Model** | LogisticRegression (C=0.1) via Pipeline (Imputer→Scaler→LR) |
| **Training data** | 37,674 recovery ticks from 126 settlements |
| **Target positive rate** | 46.7% |
| **Threshold** | p(near_peak) ≥ 0.6 |
| **Used in backtest?** | **YES** — actively drives exit timing |

**Threshold precision:**

| Threshold | Ticks above | % of all ticks | Precision |
|-----------|------------|----------------|-----------|
| 0.3 | 25,969 | 68.9% | 0.594 |
| 0.4 | 21,023 | 55.8% | 0.646 |
| 0.5 | 16,645 | 44.2% | 0.697 |
| **0.6** | **12,466** | **33.1%** | **0.754** |
| 0.7 | 8,413 | 22.3% | 0.816 |
| 0.8 | 4,248 | 11.3% | 0.907 |

The 0.6 threshold fires on ~1/3 of ticks with 75.4% precision. This is reasonable.

---

## 2. Training Methodology

### 2.1 "Temporal" Split Is Actually Alphabetical

The 70/30 split sorts `settle_id` values (e.g., `ACEUSDT_20260227_180000`) and takes the first 70%. Since settlement IDs start with the symbol name, this splits **alphabetically by symbol**, not by time.

- **Train:** ACE → SAHARA (23 symbols)
- **Test:** SAHARA → ZKC (10 symbols)
- **Symbol overlap:** Only 1 (SAHARAUSDT)
- **Date overlap:** ALL 4 dates appear in both train and test

**Consequence:** This is a **symbol holdout** test (9/10 test symbols are completely unseen), which tests cross-symbol generalization. It does **not** test temporal generalization (performance on future dates).

### 2.2 LOSO Cross-Validation ✓

Leave-One-Symbol-Out with `groups=symbols` is correctly implemented and provides the most honest generalization estimate. The 23.6 bps GROSS_PNL is derived from LOSO, which is sound.

### 2.3 Production Model Trained on All Data

After evaluation, the production LogReg is retrained on 100% of the data. This is standard practice but means:
- The evaluation AUC (from 70/30 or LOSO) is an **estimate** of production performance
- There is no independent test set for the deployed model
- If the model overfits to the 4 days of data, there is no safety net

---

## 3. Critical Issues — Look-Ahead Bias

### 3.1 `find_bottom()` Uses Future Data

```python
def find_bottom(sd, t_min_ms=1000, t_max_ms=30000):
    for t_ms in sorted(sd.price_bins.keys()):
        if t_ms < t_min_ms or t_ms > t_max_ms:
            continue
        p = sd.price_bins[t_ms]
        if bottom_bps is None or p < bottom_bps:
            bottom_bps = p
            bottom_t = t_ms
    return bottom_bps, bottom_t
```

This scans **all** price bins (1s–30s) to find the global minimum. In production, at any given moment, you don't know if the current price is the bottom or if it will go lower.

**Impact on backtest:**
1. **Long entry decision** uses `bottom_t` from `find_bottom()` — knows the exact bottom time
2. **Long entry price** is `price_bins[bottom_t]` — gets the exact bottom price
3. **Long recovery** is measured from this perfect bottom — maximum possible recovery
4. **Long notional** is sized from OB at the exact bottom time

**In production**, the short exit ML model would signal "near bottom" with ~10 bps error. The actual entry would be worse than the perfect bottom, and the recovery measurement would be reduced.

### 3.2 Short PnL Is a Constant Average

The short leg does not simulate per-trade ML exit timing. Instead it applies a fixed 23.6 bps average derived from LOSO. This means:
- No variance in short leg outcomes (all trades get the same gross edge)
- Short WR = 100% for filtered coins (because 23.6 bps > max RT slippage)
- In reality, individual short trades would have variable PnL (some would lose)

---

## 4. Data Quality

### 4.1 Sample Size — 🔴 Critical

| Metric | Value |
|--------|-------|
| Days of data | **4** (Feb 26–Mar 1, 2026) |
| Total settlements | 160 |
| After filters | 127 |
| Long trades | 83 |
| Unique symbols | 32 |

With N=83 long trades at 60% WR, the 95% confidence interval is **±10.5 percentage points** (49.5%–70.5%). The strategy could be a coin flip. Four days provides zero visibility into:
- Different market regimes (trending, ranging, high-vol, low-vol)
- Exchange-level changes (fee changes, market maker behavior)
- Seasonal patterns (time-of-day effects beyond 3 settlement times)

### 4.2 OB200 Reconstruction ✓

100% success rate on reconstruction. The depth and spread at bottom_t show meaningful variation from T-0:

| Metric | Median | Mean | P10 | P90 |
|--------|--------|------|-----|-----|
| Depth@bottom / Depth@T-0 | 1.02x | 1.34x | 0.66x | 2.33x |
| Spread@bottom / Spread@T-0 | 1.00x | 4.55x | 0.24x | 3.21x |

Spread can widen to 3x+ in the crash phase (mean is skewed by outliers at 4.55x). The reconstruction is adding real value.

### 4.3 Price Bins — Last Trade Bias

`price_bins[t]` stores the **last trade** in each 100ms window, not a VWAP. This means:
- Bottom detection picks the last trade at the lowest bin, not the actual trade low
- Recovery measurement uses last-trade prices, not achievable fill prices
- Actual market orders would fill at VWAP (worse than last-trade in thin books)

---

## 5. Feature Engineering

### 5.1 Short Exit Features (56 features)

**Categories:**
- **Price dynamics:** velocity (500ms, 1s, 2s), acceleration, running min, distance from low
- **Trade flow:** sell ratio, trade rate, vol rate, avg size, large trade % (4 windows)
- **Orderbook:** L1 spread, imbalance, bid/ask qty changes
- **Sequence:** bounce count, consecutive new lows, reversals, price range/std
- **Static context:** FR, time elapsed, phase indicator

**Quality:** 0% NaN rate — all features are well-constructed. No missing data issues.

### 5.2 Long Exit Features (28 features)

**Categories:**
- **Recovery dynamics:** recovery bps, % of drop recovered, running max, distance from high
- **Velocity/acceleration:** 500ms, 1s, 2s windows
- **Trade flow:** buy ratio (not sell), vol rate, trade count (3 windows)
- **Buy pressure momentum:** change in buy ratio between 2 windows
- **Orderbook:** spread, imbalance
- **Static context:** drop magnitude, bottom timing, FR

**Quality:** Clean, no NaN issues.

---

## 6. Backtest Methodology

### 6.1 What's Honest ✓

- **OB-based slippage** from reconstructed book at entry/exit time
- **Adaptive position sizing** from actual depth
- **Fee structure** with maker/taker blended rates
- **Filter skipping** for thin/wide-spread coins

### 6.2 What's Optimistic

| Assumption | Reality |
|------------|---------|
| Perfect bottom detection | ML model has ~10 bps error |
| Entry at exact bottom price | Would enter when "near bottom" signal fires, not at the actual minimum |
| Constant 23.6 bps short edge | Per-trade PnL varies; some shorts would lose |
| 100ms price bin resolution | Real execution has latency + partial fills |
| No queue priority modeling | Limit orders may not fill at target price |
| No adverse selection | Market makers may pull liquidity when we enter |

---

## 7. Risk Assessment

### 7.1 What Could Break in Production

1. **Bottom detection error:** Without perfect hindsight, long entry would be ~10 bps worse on average, reducing the $53.1/day long leg revenue by ~$5–15/day.

2. **Regime change:** 4 days of data in a single market regime. If volatility structure changes, both the short edge and recovery pattern could disappear.

3. **Latency:** The 100ms tick resolution assumes we can observe + compute + execute within 100ms. Real systems have 50–200ms additional latency.

4. **Model staleness:** LogReg trained on 4 days of data will likely degrade as market microstructure shifts. No retraining schedule defined.

5. **Concurrent trading:** If multiple participants exploit the same settlement pattern, the edge will shrink as liquidity gets consumed.

### 7.2 Conservative Revenue Estimate

| Scenario | Daily Revenue |
|----------|--------------|
| **Backtest (optimistic)** | $125.6 |
| After bottom detection error (−$10) | $115.6 |
| After short PnL variance (−$5) | $110.6 |
| After latency degradation (−$10) | $100.6 |
| After regime uncertainty (−30%) | **$70** |

**Conservative production estimate: $50–$80/day**, depending on how well the system adapts to real-world conditions.

---

## 8. Recommendations

### Immediate (Before Going Live)

1. **Replace `find_bottom()` with ML-detected bottom in the backtest** — use the short exit model's "near_bottom" signal to trigger long entry, measuring recovery from the ML exit point instead of the perfect hindsight bottom.

2. **Simulate per-trade short PnL variance** — replace the 23.6 bps constant with per-settlement ML predictions to get realistic short leg WR and variance.

3. **Collect more data** — 4 days is dangerously thin. Minimum 2–4 weeks (30–60 settlement cycles) before trusting any revenue estimate.

### Medium-Term

4. **Implement true temporal split** — sort by date, train on first N days, test on last M days. This tests the question "does yesterday's model work today?"

5. **Use VWAP for price bins** — switch from last-trade to volume-weighted average price per bin for more realistic fill simulation.

6. **Add model staleness detection** — monitor AUC/precision on recent data and trigger retraining when performance drops.

### Architecture

7. **Wire short exit ML into the backtest** — currently the models are trained and saved but never used. Either use them or remove the training step.

8. **Add latency simulation** — add configurable delay between signal and execution.

---

## Files Audited

| File | Lines | Purpose |
|------|-------|---------|
| `pipeline/config.py` | 75 | Constants and sizing functions |
| `pipeline/data.py` | 367 | JSONL parser, OB reconstruction |
| `pipeline/features.py` | 396 | Short exit + long exit feature engineering |
| `pipeline/models.py` | 320 | Training, persistence, inference |
| `pipeline/backtest.py` | 410 | Per-settlement simulation + comparison |
| `pipeline/report.py` | 209 | Markdown report generation |
| `pipeline/run.py` | 182 | CLI orchestrator |
| `research_position_sizing.py` | 527 | Slippage computation |
