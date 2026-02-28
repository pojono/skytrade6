# Exit ML Feature Experiments Plan

**Objective:** Systematically test feature additions to improve bottom prediction.  
**Baseline:** v2 model — 47 features, LogReg AUC=0.765, HGBC AUC=0.786, LOSO=0.736  
**Data:** 140 settlements, 81,511 ticks (100ms), 30 symbols

---

## Methodology

- **One feature group at a time** vs baseline
- **Same validation:** 70/30 temporal split + LOSO (symbol)
- **Same models:** LogReg (captures linear signal) + HGBC (captures nonlinear)
- **Metric:** Test AUC on near_bottom_10 (primary), LOSO AUC (honest)
- **Overfit check:** Train AUC vs Test AUC gap (must be <0.15 for LogReg)
- **Backtest:** Single exit per settlement, LOSO predictions

## Experiments

| # | Feature Group | Features Added | Hypothesis |
|---|--------------|----------------|-----------|
| 0 | Baseline (v2) | 47 | Current best |
| 1 | OB depth (L5-L50) | +8-12 | Multi-level imbalance detects buyer absorption |
| 2 | CVD (cumulative volume delta) | +4-6 | Running buy-sell imbalance shows exhaustion |
| 3 | Sequence features | +5-8 | Bounce count, consecutive lows, price range |
| 4 | Ensemble (stack) | 0 (method change) | Combine LogReg + HGBC predictions |
| 5 | FR regime interactions | +6-10 | FR × feature interactions capture regime-specific dynamics |
| 6 | Higher resolution (50ms) | 0 (param change) | More data density in critical first 5s |

## Anti-Overfit Rules

1. Never tune hyperparameters on test set
2. LOSO AUC must improve (not just test AUC)
3. LogReg overfit gap must remain < 0.05
4. Feature must have permutation importance > 0.001 to keep
5. Backtest must use LOSO predictions (not in-sample)
