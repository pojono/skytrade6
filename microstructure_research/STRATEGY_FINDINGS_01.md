# Strategy Findings 01: Initial ML Screening

**Date:** 2026-02-21
**Data:** BTCUSDT 15m, 2024-01-01 to 2024-03-31 (91 days, 8736 candles)
**Features:** 733 after cleaning (from 763 raw), 120 targets

---

## Phase 1: Quick Screening (14-day dev set)

Only 1 fold possible. One positive result: D_reg_h5_min2.0 (+7.71 bps, 46 trades).
All others negative. Too little data to conclude anything.

## Phase 2: Proper WFO (90-day, 5 folds)

### Results Summary

| Strategy | Description | Avg bps | WR | PF | Sharpe | Prof Folds |
|---|---|---|---|---|---|---|
| A_pnl_reg_h5_p70 | P&L regression, 5-bar, 70th pct | -2.68 | 47.2% | 0.91 | -1.68 | 3/5 |
| A_pnl_reg_h5_p75 | P&L regression, 5-bar, 75th pct | -1.77 | 47.8% | 0.94 | -1.08 | 3/5 |
| A_pnl_reg_h5_p80 | P&L regression, 5-bar, 80th pct | -2.51 | 48.0% | 0.92 | -1.41 | 3/5 |
| **B_dual_h5_p0.55** | Dual classifier, 5-bar, 55% conf | **+1.44** | **52.8%** | **1.07** | **1.08** | 2/5 |
| B_dual_h5_p0.6 | Dual classifier, 5-bar, 60% conf | -3.74 | 52.2% | 0.87 | -1.45 | 1/5 |
| B_dual_h5_p0.65 | Dual classifier, 5-bar, 65% conf | -13.55 | 50.0% | 0.52 | -2.78 | 1/5 |
| C_ensemble_h5_v2 | Ensemble 4 targets, 2 votes | -6.50 | 45.9% | 0.77 | -4.26 | 1/5 |
| C_ensemble_h5_v3 | Ensemble 4 targets, 3 votes | +24.09 | 40.0% | 1.48 | 2.37 | 1/5 (5 trades) |
| **D_longhold_h10_p75** | 10-bar hold, 75th pct | **+2.34** | **51.5%** | **1.07** | **0.97** | 2/5 |
| D_longhold_h10_p85 | 10-bar hold, 85th pct | +1.56 | 51.2% | 1.04 | 0.51 | 2/5 |
| E_multihorizon_min3.0 | Multi-horizon cum return | -10.02 | 43.7% | 0.72 | -6.62 | 1/5 |
| E_multihorizon_min5.0 | Multi-horizon cum return | -12.32 | 45.9% | 0.67 | -7.74 | 1/5 |

### Key Observations

1. **Signal is very weak**: Best strategies show +1-2 bps avg — barely above noise
2. **Dual classifier (B) slightly positive**: +1.44 bps, 52.8% WR, but only 2/5 folds profitable
3. **Longer hold (D) slightly positive**: +2.34 bps at 10-bar hold, consistent with prior finding that longer holds work better
4. **Regression approaches (A) nearly breakeven**: 3/5 folds profitable but avg is slightly negative
5. **Ensemble too selective**: 3-vote threshold produces almost no trades
6. **Multi-horizon (E) clearly negative**: Overtrading, poor signal quality
7. **Higher confidence thresholds reduce trades but don't improve quality**

### Fold-Level Analysis

Best performing fold across all strategies: **Fold 2 (Feb 23 - Mar 4)** — this was the BTC rally period.
Worst: **Fold 0 (Feb 3-13)** — ranging/declining market.

This suggests **regime dependence**: strategies work better in trending markets.

### Lessons Learned

1. **700+ features is too many** — LightGBM likely overfitting despite regularization
2. **15m candles may be too noisy** — signal-to-noise ratio too low
3. **Need regime filtering** — only trade in favorable conditions
4. **Feature selection per fold is unstable** — different features selected each time
5. **Base rates matter**: profitable_long_5 base rate ~45-55%, hard to beat significantly

---

## Next Steps

1. **Try 1h candles** — less noise, stronger signal per candle
2. **Regime-conditional trading** — only trade when vol regime is favorable
3. **Fewer, curated features** — hand-pick 20-30 most meaningful features
4. **Stacked models** — use model predictions as features for a meta-model
5. **Different ML approach** — try linear models (Ridge/Lasso) which overfit less
6. **Longer training windows** — 60+ days instead of 30
