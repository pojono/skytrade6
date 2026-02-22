# Feature-Target Analysis Methodology

## Overview

With ~418 raw features, ~345 z-score features, and ~120 targets across 5 timeframes,
we need a systematic approach to identify which features genuinely predict which targets.

The core principle: **a weak but stable signal beats a strong but unstable one**.
A feature with Spearman=0.03 that holds its sign 90% of the time across multiple coins
is far more valuable than one with Spearman=0.15 that flips sign every month.

---

## Tier 1: Univariate Signal Scan (fast, scan everything)

**Goal**: Rank all feature-target pairs by raw predictive strength.

**Metrics**:
- **Spearman rank correlation** — nonlinear-friendly, robust to outliers.
  Best for continuous targets (`tgt_ret_*`, `tgt_cum_ret_*`, `tgt_sharpe_*`).
- **AUC (Area Under ROC Curve)** — for binary targets (`tgt_profitable_long_*`,
  `tgt_profitable_short_*`, `tgt_direction_*`). Measures standalone discriminative
  power of a single feature. AUC=0.5 is random, >0.52 is interesting, >0.55 is strong.
- **Mutual Information (MI)** — captures nonlinear relationships that correlation misses.
  Noisier, needs careful binning. Use as a secondary check when Spearman is low but
  you suspect nonlinear structure.

**Output**: A matrix of (feature × target) scores. Heatmap visualization.

**Thresholds**:
- Spearman |r| > 0.02 → worth investigating
- AUC > 0.52 or < 0.48 → worth investigating
- MI > baseline (shuffled) at p < 0.01 → worth investigating

---

## Tier 2: Temporal Stability (critical for trading)

**Goal**: Filter out features whose signal is not stable over time.

**Metrics**:
- **Rolling correlation stability** — compute Spearman in rolling windows (e.g., 30-day),
  then measure the std of those correlations. Low std = stable signal.
- **Sign consistency** — what % of rolling windows have the same sign of correlation?
  Target: >70% sign consistency.
- **Regime robustness** — does the signal hold in both trending and ranging markets?
  Split data by volatility regime and check.

**Output**: For each feature-target pair that passed Tier 1, a stability score.

**Filter**: Keep only pairs where:
- Sign consistency > 70%
- Rolling correlation std < 0.5 × |mean correlation|

---

## Tier 3: Cross-Asset Consistency

**Goal**: Ensure the signal is not coin-specific noise.

**Metrics**:
- **Cross-symbol correlation agreement** — does the same feature-target relationship
  hold on DOGE, SOL, BTC, ETH? Compute Tier 1 metrics per symbol, then check agreement.
- **Rank correlation of feature rankings** — if feature X is the #3 predictor of
  `tgt_ret_5` on DOGE, is it also top-10 on SOL?

**Output**: For each feature-target pair, a cross-symbol consistency score.

**Filter**: Keep only pairs where:
- Same sign on ≥ 3 out of 4+ symbols
- Rank correlation of feature importance across symbols > 0.3

---

## Tier 4: Redundancy & Clustering

**Goal**: Eliminate redundant features that carry the same information.

**Metrics**:
- **Feature-feature Spearman correlation** — cluster features with |r| > 0.8.
- **Hierarchical clustering** — group features into clusters, pick the best
  representative from each cluster (highest Tier 1 score × Tier 2 stability).
- **Variance Inflation Factor (VIF)** — for linear models, identify multicollinearity.

**Output**: A reduced feature set with one representative per cluster.

**Target**: Reduce from ~760 features to ~30-50 uncorrelated representatives.

---

## Tier 5: Walk-Forward Predictive Power

**Goal**: Final validation with realistic time-series methodology.

**Metrics**:
- **Single-feature WFO Ridge/Logistic** — for each surviving feature, fit a
  univariate model in walk-forward fashion (train 30d, test 15d, expanding window).
  Measure OOS R² (regression) or OOS AUC (classification).
- **Marginal improvement** — add features one-by-one to a multivariate model,
  measure marginal lift. This catches features that are redundant with others
  already in the model.
- **Feature importance stability** — in a multivariate WFO model, does the feature
  maintain consistent importance across folds?

**Output**: Final ranked list of features per target, with realistic OOS performance.

---

## Execution Plan

1. **Tier 1**: Run on DOGE + SOL (1yr, all 5 timeframes). ~30 min compute.
2. **Tier 2**: Run on Tier 1 survivors. ~10 min.
3. **Tier 3**: Cross-check DOGE vs SOL. ~5 min.
4. **Tier 4**: Cluster surviving features. ~2 min.
5. **Tier 5**: WFO validation on shortlist. ~30 min.

Total: ~1-2 hours for a complete feature audit.

## Key Targets of Interest

**Binary (directly tradeable)**:
- `tgt_profitable_long_3`, `tgt_profitable_long_5` — can we predict profitable longs?
- `tgt_profitable_short_3`, `tgt_profitable_short_5` — can we predict profitable shorts?

**Continuous (flexible)**:
- `tgt_ret_1`, `tgt_ret_3`, `tgt_ret_5` — raw forward returns
- `tgt_cum_ret_5`, `tgt_cum_ret_10` — cumulative returns over longer horizons
- `tgt_sharpe_5`, `tgt_sharpe_10` — risk-adjusted forward returns

**Regime**:
- `tgt_vol_regime_5`, `tgt_vol_regime_10` — predicting volatility regime helps with position sizing
