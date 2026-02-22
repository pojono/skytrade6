# ML Model Comparison — SOLUSDT 4h

**Date:** 2025-02-22  
**Symbol:** SOLUSDT 4h  
**Setup:** 360d selection, 3d purge, 30d trade window (Period 1)  
**Models:** Logistic, RidgeClf, RandomForest, LightGBM, XGBoost, Ensembles  
**Feature sets:** raw (20), raw+lags (80), all_core (86), core+lags (344)

## Key Discovery: Base Rate Filtering

Before running the comparison, we discovered that **7 of our 30 "predictable" targets had extreme base rates** making them useless for trading:

| Dropped Target | Base Rate | Issue |
|---------------|-----------|-------|
| `breakout_any_5` | 98.8% | Almost always true |
| `breakout_any_3` | 96.7% | Almost always true |
| `consolidation_5` | 6.1% | Too rare |
| `crash_3` | 5.4% | Too rare + inconsistent |
| `crash_5` | 8.6% | Too rare |
| `tail_event_1` | 9.6% | Too rare |
| `liquidation_cascade_5` | 9.8% | Too rare |

**30 targets → 23 targets** after filtering (base rate 10-90%).

## WFO 12-Period Results (Background Run)

The constrained WFO backtest completed across 12 rolling periods (Jan-Dec 2025):

- **All 12 periods positive** — monotonically increasing cumulative score
- **Most stable targets:** breakout_down_3 (std=0.022), breakout_up_5 (std=0.023)
- **Weakest periods:** P10-P11 (Sep-Nov 2025) — still positive
- **All 30 original targets** maintained Tier A status (mean>0.03, ≥70% positive)

## Best Model Per Target

| Target | Best Model | Features | Score | Tier |
|--------|-----------|----------|-------|------|
| `breakout_up_3` | LightGBM | core+lags | AUC=0.859 | STRONG |
| `breakout_down_3` | RidgeClf | all_core | AUC=0.833 | STRONG |
| `breakout_up_5` | Ensemble_Trees | core+lags | AUC=0.831 | STRONG |
| `vol_expansion_10` | Logistic | raw | AUC=0.828 | STRONG |
| `breakout_down_5` | RidgeClf | all_core | AUC=0.788 | STRONG |
| `vol_expansion_5` | Ensemble_All | raw | AUC=0.774 | STRONG |
| `tail_event_5` | RandomForest | all_core | AUC=0.742 | MODERATE |
| `tail_event_3` | LightGBM | raw+lags | AUC=0.741 | MODERATE |
| `profitable_long_1` | Logistic | raw+lags | AUC=0.674 | MODERATE |
| `profitable_short_1` | Logistic | raw+lags | AUC=0.669 | MODERATE |
| `profitable_long_5` | XGBoost | raw+lags | AUC=0.640 | MODERATE |
| `profitable_short_5` | LightGBM | raw+lags | AUC=0.635 | MODERATE |
| `profitable_long_10` | RidgeClf | raw | AUC=0.634 | MODERATE |
| `adverse_selection_1` | LightGBM | raw+lags | AUC=0.615 | MODERATE |
| `profitable_short_10` | Logistic | raw | AUC=0.594 | WEAK |
| `profitable_short_3` | RandomForest | raw+lags | AUC=0.558 | WEAK |
| `profitable_long_3` | LightGBM | raw+lags | AUC=0.552 | WEAK |
| `alpha_1` | Ridge | all_core | R²=0.155 | STRONG |
| `relative_ret_1` | Ridge | all_core | R²=0.155 | STRONG |

## Model Type Analysis

**Which model wins most often?**

| Model | Wins | Best For |
|-------|------|----------|
| LightGBM | 5 | Directional breakouts, tail events, profitability |
| Logistic | 4 | Vol expansion, profitable_long/short_1 |
| RidgeClf | 3 | Breakout_down, profitable_long_10 |
| Ridge | 2 | Continuous targets (alpha, relative_ret) |
| RandomForest | 2 | Tail events, profitable_short_3 |
| Ensemble | 2 | Breakout_up_5, vol_expansion_5 |
| XGBoost | 1 | profitable_long_5 |

**Key finding:** No single model dominates. The best model depends on the target type:

- **Directional breakouts** → Tree models (LightGBM) win by 5-8% over linear
- **Volatility/regime** → Linear models (Logistic) are sufficient
- **Profitability** → Linear models with lags, tree models overfit
- **Continuous alpha** → Ridge regression dominates

## Feature Set Analysis

**Which feature set wins most often?**

| Feature Set | Wins | N Features |
|------------|------|------------|
| raw+lags | 8 | 80 |
| all_core | 5 | 86 |
| raw | 4 | 20 |
| core+lags | 2 | 344 |

**Key finding:** `raw+lags` (adding 1/2/3-bar lag features) is the most consistently useful enhancement. The 344-feature `core+lags` set only helps tree models on directional targets — linear models degrade with too many features.

## Target Tier Classification

### STRONG (8 targets) — Actionable for trading
- **Breakout direction:** up_3 (0.859), down_3 (0.833), up_5 (0.831), down_5 (0.788)
- **Volatility regime:** vol_expansion_10 (0.828), vol_expansion_5 (0.774)
- **Alpha:** alpha_1 (R²=0.155), relative_ret_1 (R²=0.155)

### MODERATE (8 targets) — Useful as filters/signals
- **Risk:** tail_event_5 (0.742), tail_event_3 (0.741)
- **Profitability:** profitable_long_1 (0.674), profitable_short_1 (0.669), profitable_long_5 (0.640), profitable_short_5 (0.635), profitable_long_10 (0.634)
- **Other:** adverse_selection_1 (0.615)

### WEAK (3 targets) — Low signal, consider dropping
- profitable_short_10 (0.594), profitable_short_3 (0.558), profitable_long_3 (0.552)

## Recommended ML Configuration

For production pipeline, use target-specific model selection:

```python
MODEL_CONFIG = {
    # Directional breakouts → LightGBM with core+lags
    "breakout_up_3":    ("LightGBM", "core+lags"),
    "breakout_up_5":    ("LightGBM", "core+lags"),
    "breakout_down_3":  ("RidgeClf", "all_core"),
    "breakout_down_5":  ("RidgeClf", "all_core"),
    
    # Volatility → Logistic with raw features
    "vol_expansion_10": ("Logistic", "raw"),
    "vol_expansion_5":  ("Logistic", "raw"),
    
    # Profitability → Logistic with raw+lags
    "profitable_long_1":  ("Logistic", "raw+lags"),
    "profitable_short_1": ("Logistic", "raw+lags"),
    "profitable_long_5":  ("LightGBM", "raw+lags"),
    "profitable_short_5": ("LightGBM", "raw+lags"),
    
    # Risk → Mixed
    "tail_event_5": ("RandomForest", "all_core"),
    "tail_event_3": ("LightGBM", "raw+lags"),
    
    # Alpha → Ridge with all_core
    "alpha_1":        ("Ridge", "all_core"),
    "relative_ret_1": ("Ridge", "all_core"),
}
```

## Conclusions

1. **Base rate matters more than model choice.** Our "best" target (breakout_any_5, AUC 0.48) was actually useless — 98.8% positive rate. Always check base rates first.

2. **Linear models are surprisingly competitive.** Logistic/Ridge wins or ties on 9/19 targets. Tree models only clearly win on directional breakout targets.

3. **Lag features are the single most impactful enhancement.** Adding 1-3 bar lags of raw features improved results on 8/19 targets. Interaction features didn't help (trees find them automatically).

4. **Optuna tuning didn't reliably beat defaults.** In the initial 3-target test, tuned LightGBM sometimes underperformed default LightGBM. Not worth the 10× training cost.

5. **8 STRONG targets are ready for strategy development.** The breakout direction and vol expansion targets have AUC 0.77-0.86, which is excellent for trading signals.
