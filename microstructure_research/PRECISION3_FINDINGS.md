# Precision3 Strategy — Findings

**Date:** 2026-02-23
**Strategy:** Precision3 (5 models, no meta-stacking)
**Version:** v2 — strict no-lookahead (feature selection inside each training split)

---

## Design Philosophy

Instead of 15-19 base models → meta-model stacking, Precision3 uses **5 purpose-built models** with direct outputs driving trade decisions:

| Model | Target | Type | Role |
|-------|--------|------|------|
| Direction | `tgt_ret_1` | Regressor | Predict 1-bar return → long/short/flat |
| Vol Sizing | `tgt_realized_vol_5` | Regressor | Inverse-vol position sizing |
| Stop Loss | `tgt_max_drawdown_long_3` | Regressor | Adaptive stop-loss placement |
| Filter 1 | `tgt_consolidation_3` | Classifier | Skip low-opportunity bars |
| Filter 2 | `tgt_crash_10` | Classifier | Halve size before crashes |

**Key difference from Strategy 1/2:** No meta-model. Each model has one job. Simpler = more robust OOS.

---

## Lookahead Fix (v1 → v2)

v1 used pre-computed feature lists from `predictable_targets.json`, which were derived from
a full-dataset analysis that included test periods. This is feature-selection lookahead.

**v2 fix:** All feature selection is done fresh each period via Spearman correlation on
training data only. Inner CV re-selects features on the CV-train split (never sees CV-val
targets). No pre-computed feature lists are used.

---

## Results — Cross-Coin Comparison (v2, strict no-lookahead)

| Metric | SOLUSDT | DOGEUSDT |
|--------|---------|----------|
| **Net Return** | **+400.9%** | **+292.0%** |
| **Compound Return** | **+2,784%** | **+1,338%** |
| **Sharpe (ann)** | **7.33** | **4.64** |
| **Profit Factor** | **3.07** | **2.01** |
| **Win Rate** | 67.4% | 61.8% |
| **Max Drawdown** | **-12.7%** | **-15.8%** |
| **Periods Positive** | **12/12** | **12/12** |
| **Total Trades** | 321 | 329 |
| **Avg Trade Return** | +1.088% | +0.873% |
| **Buy & Hold** | -17.7% | -58.9% |
| **Models** | 5 | 5 |

### vs Strategy 1/2 (SOLUSDT)

| Metric | Precision3 v2 | Strategy 1 (best) | Strategy 2 (best) |
|--------|--------------|-------------------|-------------------|
| **Net Return** | **+400.9%** | +102.4% | +122.4% |
| **Sharpe** | **7.33** | 4.05 | 3.81 |
| **Profit Factor** | **3.07** | 1.80 | 1.74 |
| **Max Drawdown** | **-12.7%** | -19.7% | -18.6% |
| **Periods Positive** | **12/12** | 11/12 | 10/12 |
| **Models** | **5** | 19 | 19 |

### Per-Period Breakdown — SOLUSDT

| Period | Dates | Net % | B&H % | Trades | Win Rate |
|--------|-------|-------|-------|--------|----------|
| P1 | Dec 29 → Jan 27 | +23.45% | +20.57% | 25 | 64.0% |
| P2 | Jan 28 → Feb 26 | +29.85% | -42.80% | 27 | 63.0% |
| P3 | Feb 27 → Mar 28 | +36.82% | -6.60% | 25 | 72.0% |
| P4 | Mar 29 → Apr 27 | +38.58% | +14.87% | 30 | 73.3% |
| P5 | Apr 28 → May 27 | +7.99% | +19.05% | 30 | 46.7% |
| P6 | May 28 → Jun 26 | +29.38% | -20.59% | 25 | 72.0% |
| P7 | Jun 27 → Jul 26 | +39.43% | +30.86% | 28 | 75.0% |
| P8 | Jul 27 → Aug 25 | +16.73% | -0.07% | 26 | 61.5% |
| P9 | Aug 26 → Sep 24 | +17.14% | +13.21% | 24 | 66.7% |
| P10 | Sep 25 → Oct 24 | +53.52% | -6.61% | 26 | 69.2% |
| P11 | Oct 25 → Nov 23 | +49.09% | -32.52% | 29 | 69.0% |
| P12 | Nov 24 → Dec 23 | +58.91% | -7.06% | 26 | 76.9% |

### Per-Period Breakdown — DOGEUSDT

| Period | Dates | Net % | B&H % | Trades | Win Rate |
|--------|-------|-------|-------|--------|----------|
| P1 | Dec 29 → Jan 27 | +33.14% | +2.52% | 28 | 71.4% |
| P2 | Jan 28 → Feb 26 | +37.51% | -39.20% | 30 | 70.0% |
| P3 | Feb 27 → Mar 28 | +44.71% | -12.67% | 28 | 57.1% |
| P4 | Mar 29 → Apr 27 | +18.88% | +0.36% | 27 | 48.1% |
| P5 | Apr 28 → May 27 | +25.01% | +26.39% | 24 | 62.5% |
| P6 | May 28 → Jun 26 | +24.22% | -28.79% | 27 | 59.3% |
| P7 | Jun 27 → Jul 26 | +13.29% | +45.22% | 28 | 57.1% |
| P8 | Jul 27 → Aug 25 | +14.36% | -12.43% | 31 | 54.8% |
| P9 | Aug 26 → Sep 24 | +20.56% | +14.92% | 28 | 64.3% |
| P10 | Sep 25 → Oct 24 | +23.27% | -16.23% | 27 | 59.3% |
| P11 | Oct 25 → Nov 23 | +32.98% | -26.72% | 28 | 67.9% |
| P12 | Nov 24 → Dec 23 | +4.05% | -12.27% | 23 | 69.6% |

---

## Why It Works Better

### 1. Direct Return Prediction vs Binary Classification
Strategy 1/2 predict binary targets (profitable_long/short) and lose magnitude information. Precision3 predicts the actual return value, so it naturally takes bigger positions on stronger signals.

### 2. Fewer Models = Less Overfitting
19 base models → meta-model has many degrees of freedom to overfit the inner validation set. 5 direct models with clear roles have far less room to overfit.

### 3. Inverse-Vol Sizing
Instead of confidence-based sizing (which depends on meta-model calibration), Precision3 sizes inversely to predicted volatility. This is a well-established risk management technique that doesn't depend on model confidence being well-calibrated.

### 4. Adaptive Stops
Predicted drawdown-based stops (from `tgt_max_drawdown_long_3`) cut losers early when the model expects large adverse moves. Only 15/321 trades (4.7%) hit stops on SOL — the model is selective about when to set tight stops.

### 5. Regime Filters Are Multiplicative
The consolidation filter blocks ~5-10% of signals (low-opportunity bars). The crash filter doesn't block trades but halves position size. This preserves trade count while reducing risk.

---

## Anti-Lookahead Verification (v2)

- [x] Expanding window: each period trains on all data before purge boundary
- [x] 3-day purge gap between training end and trade start
- [x] **Feature selection via Spearman on training data only — NO pre-computed lists**
- [x] **Inner CV re-selects features on CV-train split (never sees CV-val targets)**
- [x] Direction threshold calibrated via inner 3-fold CV on training data only
- [x] All 5 models retrained each period
- [x] Entry at next-bar open after signal (signal at bar close → enter bar+1 open)
- [x] Stop-loss evaluated at bar close (conservative)
- [x] No future data in any feature (all features use `.shift()` or past-only windows)

---

## Concerns / Caveats

1. **Results improved after removing lookahead** — this is unusual and warrants scrutiny. Possible explanation: the JSON feature lists were suboptimal (selected from a different pipeline) and fresh per-period selection finds better features.
2. **Single timeframe (4h)** — needs multi-timeframe validation.
3. **Two assets only** — needs more coins (ETH, BTC, XRP) for full validation.
4. **Direction threshold always calibrates to 40-50 bps** — model only trades when it predicts >0.4-0.5% moves. Good for quality but limits trade count.

---

## Next Steps

1. **More cross-coin validation** — run on XRPUSDT, ETHUSDT, BTCUSDT
2. **Drawdown reduction** — test convex sizing (size = confidence^2) to reduce max DD
3. **Robustness test** — vary threshold, hold period, stop buffer to check sensitivity
4. **Live paper trading** — deploy on Bybit testnet with real-time feature pipeline
