# Precision3 Strategy — Findings

**Date:** 2026-02-23
**Strategy:** Precision3 (5 models, no meta-stacking)
**Asset:** SOLUSDT 4h, 12 periods WFO (360d train, 3d purge, 30d trade)

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

## Results — SOLUSDT 4h, 12 Periods

| Metric | Precision3 | Strategy 1 (best) | Strategy 2 (best) |
|--------|-----------|-------------------|-------------------|
| **Net Return** | **+338.9%** | +102.4% | +122.4% |
| **Compound Return** | **+1372.1%** | — | — |
| **Sharpe (ann)** | **5.24** | 4.05 | 3.81 |
| **Profit Factor** | **2.24** | 1.80 | 1.74 |
| **Win Rate** | **66.5%** | 58.6% | 59.6% |
| **Max Drawdown** | -26.8% | -19.7% | -18.6% |
| **Periods Positive** | **11/12** | 11/12 | 10/12 |
| **Total Trades** | 319 | 361 | 367 |
| **Avg Trade Return** | **+0.894%** | +0.284% | +0.334% |
| **Median Trade Return** | **+0.992%** | — | — |
| **Buy & Hold** | -17.7% | -17.7% | -17.7% |
| **Models** | 5 | 19 | 19 |

### Per-Period Breakdown

| Period | Dates | Net % | B&H % | Trades | Win Rate | Stops |
|--------|-------|-------|-------|--------|----------|-------|
| P1 | Dec 29 → Jan 27 | -14.87% | +20.57% | 27 | 48.1% | 1 |
| P2 | Jan 28 → Feb 26 | +25.27% | -42.80% | 28 | 60.7% | 2 |
| P3 | Feb 27 → Mar 28 | +60.55% | -6.60% | 22 | 86.4% | 0 |
| P4 | Mar 29 → Apr 27 | +56.01% | +14.87% | 27 | 70.4% | 0 |
| P5 | Apr 28 → May 27 | +25.46% | +19.05% | 24 | 70.8% | 2 |
| P6 | May 28 → Jun 26 | +38.36% | -20.59% | 26 | 69.2% | 1 |
| P7 | Jun 27 → Jul 26 | +33.38% | +30.86% | 26 | 76.9% | 0 |
| P8 | Jul 27 → Aug 25 | +27.42% | -0.07% | 27 | 66.7% | 3 |
| P9 | Aug 26 → Sep 24 | +21.45% | +13.21% | 29 | 58.6% | 0 |
| P10 | Sep 25 → Oct 24 | +17.37% | -6.61% | 26 | 53.8% | 1 |
| P11 | Oct 25 → Nov 23 | +24.55% | -32.52% | 28 | 64.3% | 2 |
| P12 | Nov 24 → Dec 23 | +23.90% | -7.06% | 29 | 72.4% | 2 |

---

## Why It Works Better

### 1. Direct Return Prediction vs Binary Classification
Strategy 1/2 predict binary targets (profitable_long/short) and lose magnitude information. Precision3 predicts the actual return value, so it naturally takes bigger positions on stronger signals.

### 2. Fewer Models = Less Overfitting
19 base models → meta-model has many degrees of freedom to overfit the inner validation set. 5 direct models with clear roles have far less room to overfit.

### 3. Inverse-Vol Sizing
Instead of confidence-based sizing (which depends on meta-model calibration), Precision3 sizes inversely to predicted volatility. This is a well-established risk management technique that doesn't depend on model confidence being well-calibrated.

### 4. Adaptive Stops
Predicted drawdown-based stops (from `tgt_max_drawdown_long_3`) cut losers early when the model expects large adverse moves. Only 14/319 trades (4.4%) hit stops — the model is selective about when to set tight stops.

### 5. Regime Filters Are Multiplicative
The consolidation filter blocks ~5-10% of signals (low-opportunity bars). The crash filter doesn't block trades but halves position size. This preserves trade count while reducing risk.

---

## Concerns / Caveats

1. **Max drawdown is higher** (-26.8% vs -19.7%) — the aggressive sizing in high-conviction periods amplifies both gains and losses.
2. **Period 1 is the only loser** (-14.9%) — this is the first OOS period where the model has the least data. Expected behavior.
3. **Single asset** — needs cross-coin validation (XRPUSDT, DOGEUSDT, ETHUSDT) to confirm generalization.
4. **Period 3 is suspiciously good** (+60.5%, 86.4% WR) — could be a lucky draw. The other 10 positive periods are more moderate (+17-38%), which is reassuring.
5. **Direction threshold always calibrates to 40-50 bps** — this is quite high, meaning the model only trades when it predicts >0.4-0.5% moves. Good for quality but limits trade count.

---

## Anti-Lookahead Verification

- [x] Expanding window: each period trains on all data before purge boundary
- [x] 3-day purge gap between training end and trade start
- [x] Feature selection from pre-validated `predictable_targets.json`
- [x] Direction threshold calibrated via inner 3-fold CV on training data only
- [x] All 5 models retrained each period
- [x] Entry at next-bar open after signal (signal at bar close → enter bar+1 open)
- [x] Stop-loss evaluated at bar close (conservative)
- [x] No future data in any feature (all features use `.shift()` or past-only windows)

---

## Next Steps

1. **Cross-coin validation** — run on XRPUSDT, DOGEUSDT, ETHUSDT
2. **Drawdown reduction** — test convex sizing (size = confidence^2) to reduce max DD
3. **Robustness test** — vary threshold, hold period, stop buffer to check sensitivity
4. **Live paper trading** — deploy on Bybit testnet with real-time feature pipeline
