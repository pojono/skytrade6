# ML System Audit — Settlement Trading Pipeline

**Date:** 2026-03-01  
**Scope:** Full audit of all ML models, training methodology, feature engineering, backtest assumptions, and data integrity.

---

## Executive Summary

| Area | Verdict | Severity |
|------|---------|----------|
| **Short exit ML** | ✅ FIXED — now drives per-trade PnL + exit timing | � Fixed |
| **Long exit ML** | Working, decent precision at threshold 0.6 | 🟢 OK |
| **Long entry decision** | ✅ FIXED — uses ML exit time, no look-ahead | � Fixed |
| **Train/test split** | Alphabetical by symbol, not temporal | 🟡 Misleading |
| **LOSO validation** | Correct methodology, honest estimate | 🟢 OK |
| **Bottom detection** | ✅ FIXED — ML-detected bottom replaces find_bottom() | � Fixed |
| **Short PnL model** | ✅ FIXED — per-trade variable PnL (76% WR, not 100%) | � Fixed |
| **OB reconstruction** | Working, 100% success rate | 🟢 OK |
| **Sample size** | 160 settlements, 4 days — very thin | 🔴 Critical |
| **Price bins** | Last trade per 100ms, not VWAP | 🟡 Optimistic |
| **Feature NaN rate** | 0% — clean | 🟢 OK |
| **Production model** | Trained on ALL data (no holdout) — short PnL inflated | 🟡 Accepted risk |
| **Long leg profitability** | ⚠️ UNPROFITABLE without look-ahead (19–23% WR) | 🔴 Critical discovery |

### Post-Fix Honest Results

| Strategy | Short $/d | Short WR | Long $/d | Long WR | Total $/d |
|----------|----------|----------|---------|---------|----------|
| **short_only** | **$137.7†** | **76%** | — | — | **$137.7†** |
| fixed_exit | $137.7† | 76% | −$22.3 | 19% | $115.4† |
| ml_exit | $137.7† | 76% | −$17.4 | 23% | $120.2† |

**†** Short PnL is inflated (in-sample). The production model was trained on all data; LOSO gave 23.6 bps average. Conservative short estimate: **$72.5/day** (LOSO-based).

### Key Discovery: Long Leg Requires Look-Ahead to Be Profitable

The long leg was previously showing $53.1/day (60% WR) because `find_bottom()` used **perfect hindsight** to pick the exact crash minimum. With the ML-detected exit point as the long entry (no look-ahead), the long leg becomes **unprofitable** (−$17 to −$22/day, 19–23% WR). The recovery measured from a non-optimal entry point is insufficient to cover fees + slippage.

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

### 6.2 What's Optimistic (remaining after fixes)

| Assumption | Reality |
|------------|---------|
| ~~Perfect bottom detection~~ | ✅ FIXED — ML exit used as bottom |
| ~~Constant 23.6 bps short edge~~ | ✅ FIXED — per-trade ML PnL (76% WR) |
| Short PnL in-sample | Production model trained on all data; LOSO gave 23.6 bps avg |
| 100ms price bin resolution | Real execution has latency + partial fills |
| No queue priority modeling | Limit orders may not fill at target price |
| No adverse selection | Market makers may pull liquidity when we enter |

---

## 7. Risk Assessment

### 7.1 What We Now Know (Post-Fix)

1. **Short-only is the reliable strategy.** Short leg: 76% WR, $137.7/day (in-sample) or ~$72.5/day (LOSO-based conservative). The short edge is real.

2. **Long leg is UNPROFITABLE without look-ahead.** When entering at the ML-detected exit point (not the perfect bottom), the recovery is insufficient to cover fees + slippage. Long WR drops from 60% (look-ahead) to 19–23% (honest).

3. **The old $125.6/day combined figure was an illusion.** $53/day of "long leg profit" came from perfect hindsight on the bottom. Honest combined revenue is lower than short-only.

4. **Regime change:** 4 days of data in a single market regime. If volatility structure changes, even the short edge could disappear.

5. **Model staleness:** LogReg trained on 4 days will degrade. No retraining schedule defined.

6. **Concurrent trading:** If multiple participants exploit the same settlement pattern, the edge will shrink.

### 7.2 Revised Revenue Estimate

| Scenario | Daily Revenue |
|----------|--------------|
| Short-only (in-sample, inflated) | $137.7 |
| Short-only (LOSO-based conservative) | $72.5 |
| Short+Long ML exit (honest, no look-ahead) | $120.2† |
| **Recommended: short-only** | **$50–$75/day** |

**†** Combined figure still uses inflated in-sample short PnL.

**Production recommendation: SHORT-ONLY at $50–$75/day.** The long leg should NOT be deployed until a method exists to profitably enter without hindsight bias.

---

## 8. Recommendations

### Completed ✅

1. ~~**Replace `find_bottom()` with ML-detected bottom**~~ — ✅ Done. Short exit ML now drives both exit timing and long entry point. No look-ahead.

2. ~~**Simulate per-trade short PnL variance**~~ — ✅ Done. Short WR now 76% (not 100%). Per-trade PnL varies.

### Immediate (Before Going Live)

3. **Deploy short-only strategy.** The long leg is unprofitable without look-ahead. Remove it from production plans.

4. **Collect more data** — 4 days is dangerously thin. Minimum 2–4 weeks (30–60 settlement cycles) before trusting any revenue estimate.

5. **Implement true temporal split** — sort by date, train on first N days, test on last M days.

### Research (Long Leg Improvement)

6. **Better bottom detection for long entry** — the current "near_bottom_10" target has ~10 bps error. A tighter model or ensemble may reduce entry error enough to make the long leg viable.

7. **Alternative long entry signals** — instead of "enter at ML bottom", consider entering on specific recovery patterns (e.g., first 5 bps bounce after crash).

8. **Use VWAP for price bins** — switch from last-trade to VWAP for more realistic fill simulation.

### Architecture

9. **Add latency simulation** — add configurable delay between signal and execution.

10. **Add model staleness detection** — monitor AUC/precision on recent data and trigger retraining.

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
