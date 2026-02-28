# Plan: Improve ML Settlement Prediction Accuracy

**Goal:** Reduce MAE from ±38-43 bps to ±20-25 bps using all available data

---

## What We Have (and aren't using)

### Data Inventory per JSONL file:
- **orderbook.1** (10ms): ~7,300 snapshots → bid/ask price + qty at BBO
- **orderbook.50** (20ms): ~5,900 snapshots → top 50 levels
- **orderbook.200** (100ms): ~1,200 snapshots → full depth 200 levels
- **publicTrade**: ~6,600 trades → price, qty, side, notional
- **tickers**: ~1,200 snapshots → FR, OI, mark, index, vol24h, turnover, prevPrice1h
- **allLiquidation**: ~5-6 events

### Currently using: 17 features
- FR (2), spread (3), OB imbalance (4), depth (4), trade flow (4)

### NOT using (massive untapped signal):
1. **OI change** (Δ$+87k in one example)
2. **Basis spread** (mark-index: -182 → -300 bps, WIDENING before settlement)
3. **Spread/imbalance TRENDS** (direction of change, not just average)
4. **Trade size distribution** (P90, P99, max, large trade %)
5. **Trade arrival rate** and acceleration
6. **Buy pressure surge** in last 1s (imbalance +0.01 → +0.41!)
7. **Orderbook shape** (concentration, depth at different levels)
8. **Liquidation data** (count, direction)
9. **Volume24h, turnover24h** (liquidity context)
10. **prevPrice1h** (recent momentum)

---

## Phase 1: Deep Feature Engineering (~60 new features)

### 1A. Time-Series Dynamics (trends, acceleration)

Extract at 3 windows: T-10s, T-5s, T-2s → compare trends

```
Spread features:
- spread_trend_10s       # Linear slope over last 10s
- spread_trend_5s        # Slope over last 5s
- spread_trend_2s        # Slope over last 2s (acceleration)
- spread_widening        # Binary: is spread increasing?
- spread_last_vs_mean    # Last value vs 10s average (anticipation)

Imbalance features:
- qty_imb_trend_10s      # Is imbalance shifting?
- qty_imb_trend_5s
- qty_imb_last_1s        # Imbalance in LAST SECOND (very predictive!)
- qty_imb_acceleration   # Is the shift accelerating?

Depth features:
- depth_imb_trend_10s
- bid_depth_trend        # Are bids building or draining?
- ask_depth_trend        # Are asks building or draining?
- depth_drain_rate       # How fast is liquidity disappearing?
```

### 1B. Ticker-Derived Features (OI, basis, context)

```
Open Interest:
- oi_value_usd           # Current OI in USD
- oi_change_60s          # OI change in last 60s (position building)
- oi_change_pct_60s      # % change

Basis (mark - index):
- basis_bps              # Current mark-index spread
- basis_trend            # Is basis widening? (anticipation signal)
- basis_abs_bps          # Absolute basis

Market context:
- volume_24h             # 24h volume (liquidity proxy)
- turnover_24h           # 24h turnover USD
- price_change_1h_bps    # 1h momentum from prevPrice1h
- price_change_24h_pct   # 24h change from price24hPcnt
```

### 1C. Trade Microstructure Features

```
Trade size distribution:
- trade_size_median      # Median trade size in USD
- trade_size_p90         # 90th percentile
- trade_size_p99         # 99th percentile (whale activity)
- trade_size_max         # Max single trade
- trade_size_skew        # Skewness of trade size distribution
- large_trade_count      # Trades > $500
- large_trade_pct        # % of trades that are large
- large_trade_imb        # Imbalance among large trades only

Trade arrival dynamics:
- trade_rate_10s         # Trades per second (last 10s)
- trade_rate_2s          # Trades per second (last 2s)
- trade_rate_acceleration # Is trading speeding up?
- trade_vol_acceleration  # Is volume per second increasing?

Buy pressure:
- buy_imb_last_1s        # Buy imbalance in LAST 1 SECOND
- buy_imb_last_2s        # Buy imbalance in last 2s
- buy_pressure_surge     # last_1s - last_10s (sudden buying?)
- vwap_vs_mid_bps        # VWAP above/below mid? (directional pressure)
```

### 1D. Orderbook Shape Features

```
Depth concentration:
- bid_concentration      # bid10 / total_bid (is liquidity concentrated?)
- ask_concentration      # ask10 / total_ask
- concentration_ratio    # bid_concentration / ask_concentration

Orderbook layers:
- depth_at_1pct          # $ available within 1% of mid
- depth_at_50bps         # $ available within 50bps
- depth_at_10bps         # $ available within 10bps (immediate liquidity)
- thin_side_depth        # Min(bid_depth, ask_depth) - weakest side

Orderbook "pulling":
- bbo_level_changes      # How often best bid/ask level changes
- bid_refresh_rate       # How often bid quantity refreshes
- ask_refresh_rate       # How often ask quantity refreshes
```

### 1E. Liquidation Features

```
- liq_count_pre          # Liquidation events before settlement
- liq_volume_usd         # Total liquidation volume
- liq_direction          # Net long vs short liquidations
```

### 1F. Interaction Features (non-linear)

```
- fr_x_depth             # FR × total_depth (high FR + thin = massive)
- fr_x_spread            # FR × spread (high FR + wide = fast)
- fr_x_buy_pressure      # FR × buy_imb_last_1s
- imb_x_volume           # Imbalance × volume (strong signal when busy)
- spread_x_depth         # Spread × depth (wide + thin = danger)
```

**Total new features: ~55-60**

---

## Phase 2: Better Targets

### 2A. Classification Targets (more stable)

```
# Binary: Will this trade be profitable?
target_profitable = 1 if drop_min_bps < -40 else 0  # Beat 20bps fees

# Multi-class: How big will the drop be?
target_drop_class = {
    0: drop > -40 bps     (SKIP)
    1: -80 < drop <= -40  (MARGINAL)
    2: -120 < drop <= -80 (GOOD)
    3: drop <= -120        (EXCELLENT)
}

# Binary: Fast or slow drop?
target_fast = 1 if time_to_bottom < 500 else 0
```

### 2B. Better Regression Targets

```
# Use max drop (more stable than "final price")
target_max_drop_bps = min price in T+0 to T+5s

# Recovery strength
target_recovery_bps = price_5s - min_price

# Sell intensity
target_sell_wave_imbalance = sell_volume / total_volume in T+0 to T+1s
```

### 2C. Multi-Output Prediction

Predict jointly:
1. Drop magnitude (regression)
2. Profitable? (binary classification)
3. Drop category 0-3 (multi-class)
4. Fast drop? (binary classification)

---

## Phase 3: Modeling Strategy

### 3A. Handle Small N (25 complete, 64 total)

```
Approach 1: Use ALL 64 samples
- Use HistGradientBoosting (handles NaN natively)
- Missing OB.1/OB.50 features → model handles it

Approach 2: Two-tier model
- Tier 1: 64 samples with FR + OB.200 + ticker + trade features
- Tier 2: 25 samples with full OB.1/OB.50 features
- Ensemble: Tier 1 prediction as input to Tier 2

Approach 3: Imputation
- For 39 files without OB.1/OB.50, estimate from OB.200
- Use KNN imputation or median imputation
```

### 3B. Feature Selection (critical for N=25)

```
Step 1: Remove highly correlated features (r > 0.95)
Step 2: Recursive Feature Elimination (RFE) with LightGBM
Step 3: Permutation importance on CV folds
Step 4: Keep only top 10-15 features per target
         (Rule of thumb: N/10 = 2-6 features for N=25)
```

### 3C. Regularization

```
- LightGBM: min_child_samples=5, max_depth=4, reg_alpha=1.0
- Ridge regression as baseline (strong regularization)
- Elastic Net for sparse feature selection
- Cross-validation: Leave-One-Out (for small N)
```

### 3D. Model Comparison

```
Models to try:
1. LightGBM (current)
2. Ridge Regression (strong baseline for small N)
3. ElasticNet (feature selection built-in)
4. XGBoost (different regularization)
5. Random Forest (robust to overfitting)
6. Stacking ensemble (Ridge + LGBM + RF)

For classification:
1. LightGBM Classifier
2. Logistic Regression
3. Random Forest Classifier
4. SVM (good for small N)
```

---

## Phase 4: Validation Strategy

### 4A. Cross-Validation

```
- Leave-One-Out CV (LOOCV) for N=25 (most reliable for small N)
- Stratified K-Fold for classification
- Report MAE, R², AUC, precision, recall
```

### 4B. Metrics to Track

```
Regression:
- MAE (primary)
- MAE/Std ratio (how much better than baseline?)
- R² on CV (NOT on training set!)

Classification:
- AUC-ROC (overall discrimination)
- Precision @ 80% recall (minimize false positives)
- F1 score
- Confusion matrix
```

### 4C. Overfitting Detection

```
- Compare train MAE vs CV MAE (gap should be < 2x)
- Learning curve: plot performance vs N
- Feature importance stability across CV folds
```

---

## Phase 5: Production Integration

### 5A. Real-Time Feature Calculator
- Processes live WebSocket data
- Maintains rolling buffers for all features
- Outputs feature vector at T-1s

### 5B. Ensemble Predictor
- Runs both classification and regression models
- Outputs: {should_trade, position_size, exit_time, expected_drop}

### 5C. Monitoring
- Log predictions vs actuals
- Alert if prediction error exceeds 2x CV MAE
- Weekly model retraining with new data

---

## Expected Accuracy Improvement

| Phase | Features | MAE (bps) | AUC | Notes |
|-------|----------|-----------|-----|-------|
| Current | 17 | ±38-43 | N/A | FR + basic OB |
| Phase 1 | ~75 | ±25-30 | N/A | +60 new features |
| Phase 2 | ~75 | N/A | 0.85+ | Classification targets |
| Phase 3 | 10-15 | ±20-25 | 0.88+ | Feature selection + regularization |
| Phase 4 | 10-15 | ±20-25 | 0.88+ | Validated, production-ready |

**Target: ±20-25 bps MAE, 85%+ AUC on profitable vs unprofitable**

---

## Implementation Order

1. Build deep feature extractor (Phase 1) → `analyse_settlement_v2.py`
2. Generate feature CSV with all 64 settlements
3. Add better targets (Phase 2)
4. Train + compare models (Phase 3)
5. Feature selection + regularization (Phase 3B/3C)
6. Validate thoroughly (Phase 4)
7. Build production predictor (Phase 5)

**Estimated effort:** 2-3 hours for Phases 1-4, then ongoing for Phase 5

---

## Key Insight

The data we have is **extremely rich** - 22,000+ messages per settlement with millisecond precision across 6 data streams. We're currently using ~20% of the available signal. The biggest untapped signals are:

1. **OI change** (position building before settlement)
2. **Basis widening** (mark-index divergence = anticipation)
3. **Buy pressure surge in last 1s** (imb jumped +0.01 → +0.41!)
4. **Trade size distribution** (whale activity)
5. **Orderbook dynamics** (trends, not just averages)

These features combined with proper feature selection and regularization should push us from ±38 bps to ±20-25 bps.
