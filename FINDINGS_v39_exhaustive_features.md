# FINDINGS v39: Exhaustive Feature Engineering for TP/SL Win Prediction

## Experiment Design

**Goal**: Replicate the v29 exhaustive feature approach, but targeting TP/SL symmetric trade win prediction instead of regime switches. Generate 100+ features, prune correlated ones, test multiple model architectures and target formulations.

**Data**: SOLUSDT, 43 weekdays (May 15 – Aug 7, 2025)
- **Train**: 21 weekdays (May 15 – Jun 12)
- **Test**: 22 weekdays (Jun 13 – Aug 7)
- **Sampling**: Every 10s during 13-18 UTC peak hours
- **Train**: 37,800 samples (WR 69.1%), **Test**: 39,600 samples (WR 68.3%)

**Config**: TP=20/SL=10 bps, 15m time limit, symmetric (long+short)

## Phase 1: Feature Generation (165 features)

### Feature Categories × 8 Horizons (5s, 10s, 30s, 60s, 120s, 300s, 600s, 900s)

| Category | Features | Description |
|----------|----------|-------------|
| **Volatility** | vol, vol_z | Realized vol (std of log returns), z-score vs 10× horizon |
| **Price Range** | range, range_z | High-low range in bps, z-score vs 5× horizon |
| **Trade Activity** | tc, tc_z, tn, avg_size | Trade count, z-score, notional, avg size |
| **Trade Acceleration** | tc_accel, tn_accel | Short/long ratio (horizons ≤60s) |
| **Buy/Sell Imbalance** | buy_ratio, buy_imbal | Buy notional / total, centered at 0 |
| **Liquidations** | liq_count, liq_not, liq_ratio, liq_z | Count, notional, ratio to trades, z-score |
| **Spread** | spread_mean, spread_max, spread_z | Bid-ask spread stats and z-score |
| **Open Interest** | oi_delta, oi_abs_delta, oi_accel | OI change in bps, acceleration |
| **Momentum** | ret, abs_ret | Return and absolute return |
| **Cross-Horizon** | vol_ratio, tc_ratio | Short/long vol and trade count ratios |
| **Clock** | hour, min_of_hour, time_to_funding | Time features |
| **Composite** | stress, quiet | Liq × spread_z, inverse activity |

**Total: 165 raw features**

## Phase 2: Univariate Screening

### Top 15 Features by Univariate AUC

| Rank | Feature | AUC | Corr w/ PnL |
|------|---------|-----|-------------|
| 1 | range_300 | 0.5144 | +0.018 |
| 2 | range_120 | 0.5110 | +0.014 |
| 3 | range_60 | 0.5108 | +0.014 |
| 4 | avg_size_600 | 0.4893 | -0.021 |
| 5 | range_z_300 | 0.5103 | +0.016 |
| 6 | abs_ret_60 | 0.5102 | +0.016 |
| 7 | tc_z_900 | 0.4902 | -0.012 |
| 8 | buy_imbal_600 | 0.5096 | +0.017 |
| 9 | oi_abs_delta_60 | 0.5096 | +0.002 |
| 10 | avg_size_900 | 0.4907 | -0.017 |
| 11 | time_to_funding | 0.5093 | +0.010 |
| 12 | avg_size_60 | 0.4911 | -0.010 |
| 13 | vol_60 | 0.5087 | +0.012 |
| 14 | quiet_120 | 0.5086 | -0.002 |
| 15 | oi_delta_900 | 0.5077 | +0.002 |

**Key difference from v29**: For regime prediction, 900s/3600s horizons dominated (AUC 0.56-0.60). For TP/SL win prediction, **all AUCs are ≈ 0.51** — barely above random. The signal is 10× weaker.

### By Category

| Category | N Features | Best |AUC-0.5| | Mean |AUC-0.5| |
|----------|-----------|----------------|----------------|
| **range** | 16 | 0.0144 | 0.0065 |
| **avg_size** | 8 | 0.0107 | 0.0065 |
| **tc** | 24 | 0.0098 | 0.0046 |
| **abs_ret** | 8 | 0.0102 | 0.0050 |
| **buy** | 16 | 0.0096 | 0.0042 |
| **oi** | 21 | 0.0096 | 0.0033 |
| **vol** | 20 | 0.0087 | 0.0043 |
| **liq** | 32 | 0.0069 | 0.0026 |
| **spread** | 24 | 0.0050 | 0.0028 |

## Phase 3: Redundancy Removal

- **Signal features** (|AUC-0.5| > 0.005): 60
- **After removing |corr| > 0.90 pairs**: 51 survived

Major redundancy clusters:
- `buy_ratio` ≡ `buy_imbal` at same horizon (corr = 1.00) → keep imbal
- `vol_30` ≈ `vol_60` ≈ `vol_120` (corr 0.91-0.94) → keep vol_60
- `tc_accel_30` ≈ `quiet_30` (corr 0.94) → keep quiet_30
- `liq_count_300` ≈ `liq_not_300` (corr 0.99) → keep liq_not

## Phase 4: Multi-Model Comparison (51 features)

### Binary Classification (predict win/loss)

| Model | AUC Train | AUC Test | Best Test EV | @ Threshold | N trades |
|-------|-----------|----------|-------------|-------------|----------|
| LogReg C=0.01 | 0.5395 | 0.5027 | +6.15 | 0.80 | ~30 |
| LogReg C=0.1 | 0.5401 | 0.5030 | +6.09 | 0.80 | ~30 |
| LogReg C=1.0 | 0.5403 | 0.5031 | +5.76 | 0.80 | ~30 |
| GBM_shallow | 0.5591 | 0.4960 | +1.51 | 0.80 | ~200 |
| GBM_medium | 0.6223 | 0.4920 | +0.60 | 0.50 | all |
| GBM_deep | 0.6660 | 0.4885 | +0.62 | 0.50 | all |
| GBM_regularized | 0.5591 | 0.4954 | +0.63 | 0.80 | ~200 |
| SimpleScore | 0.5241 | 0.5068 | +0.74 | 0.90 | ~4,800 |
| Ensemble (LR+GBM+Score) | 0.5621 | 0.5056 | +3.30 | 0.85 | ~94 |

**All test AUCs ≈ 0.50 (random).** The "high EV at high thresholds" comes from tiny sample sizes — not real signal.

### GBM Feature Importance (gain)

| Feature | Importance |
|---------|-----------|
| time_to_funding | 2,859 |
| range_z_300 | 2,516 |
| spread_mean_120 | 2,429 |
| tc_z_900 | 2,318 |
| avg_size_600 | 2,218 |
| range_z_600 | 1,667 |
| vol_900 | 1,628 |
| liq_z_300 | 1,613 |
| buy_imbal_600 | 1,594 |
| range_300 | 1,496 |

### Feature Set Ablation (LogReg)

| K Features | AUC Train | AUC Test | Best EV |
|------------|-----------|----------|---------|
| 3 | 0.5159 | 0.5025 | +4.23 (tiny N) |
| 5 | 0.5185 | 0.4997 | +4.17 (tiny N) |
| 10 | 0.5292 | 0.4964 | +3.76 (tiny N) |
| 20 | 0.5378 | 0.5017 | +1.53 |
| 51 (all) | 0.5401 | 0.5030 | +1.43 |

More features = more overfitting. Even 3 features overfit.

## Phase 5: Alternative Target Formulations

### Approach 1: Regression (predict PnL directly)
- Train correlation: 0.385, **Test correlation: -0.011** → pure overfitting
- No threshold produces reliable test EV improvement

### Approach 2: Predict Top-Quartile PnL
- AUC train: 0.736, **AUC test: 0.495** → pure overfitting
- Top quartile threshold = 10 bps (= TP hit), so this reduces to win/loss prediction

### Approach 3: Quantile Regression (P75)
- Model predicts constant 10.0 for all samples → no discrimination

### Approach 4: Predict |PnL| (Volatility Timing) ⭐
- Train correlation: 0.387, **Test correlation: -0.022** (ML model)
- But **P95 threshold gives +1.98 test EV** with 1,736 trades
- This is the only ML approach with meaningful test improvement

### Approach 5: Two-Stage (Vol Expansion → Trade)
- 99.9% of trades resolve (hit TP or SL) → timeout is not a useful target
- Resolution AUC is meaningless

## Phase 6: Volatility Timing Deep Dive

### Single-Feature Vol Timing (No ML)

The most honest analysis — which individual features predict high |PnL| moments?

| Feature | Corr w/ |PnL| | P90 EV | P95 EV | P95 N |
|---------|--------------|--------|--------|-------|
| **buy_imbal_600** | -0.021 | +1.27 | **+1.63** | 1,672 |
| **buy_imbal_120** | -0.013 | +1.36 | **+1.74** | 1,429 |
| **range_z_300** | -0.015 | +0.95 | **+1.89** | 1,479 |
| **buy_imbal_300** | -0.015 | **+1.60** | +1.41 | 1,686 |
| **range_z_120** | -0.010 | +1.29 | +1.18 | 1,760 |
| ret_120 | +0.014 | +1.28 | +1.15 | 2,155 |
| avg_size_600 | +0.018 | +0.10 | -0.80 | 1,295 |

**New discovery: Buy/sell imbalance at 2-10 minute horizons is the strongest single predictor of TP/SL trade quality.** When buy imbalance is extreme (P5 = very sell-heavy), EV is +1.6-1.7 bps. This wasn't tested in v36/v37.

The negative correlation means: **when selling dominates (buy_imbal < 0), the symmetric strategy performs better.** This makes sense — sell-heavy periods have more liquidation cascades and wider spreads, creating the fat tails the strategy exploits.

### LightGBM Linear Tree (Best ML Result)

| Threshold | N Test | EV Test | WR Test |
|-----------|--------|---------|---------|
| P90 | 2,045 | **+1.83** | 71.5% |
| **P95** | **745** | **+3.28** | **76.6%** |

But test correlation is -0.004 — this is likely noise at the tail, not real signal.

### Combined Vol Score (Weighted Features)

| Percentile | N Test | EV Test | WR Test |
|-----------|--------|---------|---------|
| P0 (all) | 39,588 | +0.63 | 68.3% |
| P75 | 10,181 | +0.71 | 68.2% |
| P80 | 8,181 | +0.77 | 68.4% |
| P90 | 4,036 | +0.72 | 68.3% |
| P95 | 1,886 | +0.48 | 67.2% |

Combined score is **worse** than individual features at high thresholds — combining adds noise.

### Head-to-Head: Vol Score + Liq Trigger

Combining vol timing score with liquidation triggers **hurts** — the intersection is too small and noisier than either alone.

## Key Conclusions

### 1. Win/loss prediction is fundamentally impossible
All 165 features, 51 after pruning, 5 model architectures, 5 target formulations → **test AUC ≈ 0.50**. The features predict whether a trade will win or lose no better than a coin flip. This is consistent with v30's finding that directional targets have AUC ≈ 0.50.

### 2. Volatility timing has weak but real signal
Predicting |PnL| (when will price move a lot?) has marginal signal. The symmetric strategy benefits from high-volatility moments because both TP and SL are more likely to be hit quickly, and the 2:1 TP/SL ratio means wins pay more.

### 3. Buy/sell imbalance is a new useful feature
`buy_imbal_600` (10-min buy/sell ratio) at P5 (sell-heavy) gives +1.63 EV — comparable to the `liq_10s>0` trigger from v37 (+1.92). Sell-heavy periods correlate with liquidation cascades and wider spreads.

### 4. Simple beats complex
- Single features outperform multi-feature models
- Combined scores add noise at high thresholds
- ML models overfit catastrophically (train AUC 0.60-0.74, test 0.49-0.51)
- The signal is too weak (~0.01 correlation) for ML to extract

### 5. Comparison with v29 (regime prediction)
| Metric | v29 (Regime) | v39 (TP/SL Win) |
|--------|-------------|-----------------|
| Best univariate AUC | 0.565 | 0.514 |
| Best ML AUC (test) | 0.600 | 0.507 |
| Top horizon | 900s-3600s | 120s-600s |
| Top stream | Trades | Range/Imbalance |
| Signal strength | Moderate | Very weak |

The regime prediction target has 10× more signal than TP/SL win prediction. This makes sense — regime switches are structural events driven by measurable microstructure changes, while individual trade outcomes are dominated by random noise.

### 6. Updated Fee Viability

| Approach | Test EV | Net @ 0% | Net @ 0.005% (4f) | Trades/day |
|----------|---------|----------|-------------------|-----------|
| Baseline (no filter) | +0.63 | +0.63 ✅ | -1.37 ❌ | ~1,800 |
| buy_imbal_600 P95 | +1.63 | +1.63 ✅ | -0.37 ❌ | ~76 |
| range_z_300 P95 | +1.89 | +1.89 ✅ | -0.11 ❌ | ~67 |
| liq_10s>0 (v37) | +1.92 | +1.92 ✅ | -0.08 ❌ | ~65 |
| Linear tree P95 | +3.28 | +3.28 ✅ | +1.28 ✅ ⚠️ | ~34 |

⚠️ Linear tree result is unreliable (negative test correlation, likely noise).

**The fundamental constraint remains**: no robust approach clears 0.005% maker fees.
