# FOLLOW Strategy - Production Plan

**Date:** March 2, 2026  
**Goal:** Make FOLLOW strategy production-ready with thick edge and stable execution

---

## Current Status

**Stage-1 Detector:** ✅ Ready (21.2 events/day)  
**Regime Classifier:** ✅ Complete (FOLLOW only)  
**FOLLOW Performance:** +6.21 bps net (maker), 66.7% WR, 2.4 events/day  
**FADE Strategy:** ❌ Frozen (gross edge too small)

**Problem:** FOLLOW edge is thin (+6.21 bps net with maker fees), sensitive to execution

**Solution:** Add Liquidity Vacuum filter + pullback entry timing

---

## Step-by-Step Plan

### Step 1: Freeze Baseline & Create Event Dataset ✅

**Goal:** Stable dataset for iterations

**Parameters to freeze:**
- Stage-1: FlowImpact > 70, imbalance > 0.6, min_trades 20
- Classifier: Global thresholds (vol_15m_q30, range_10s_q80, drift_2m_q70)

**Event dataset fields:**
```python
{
    'ts_event': timestamp,
    'side': 'Buy' or 'Sell',
    'flow_impact': float,
    'imbalance': float,
    'max_run': float,
    'agg_trades': int,
    'spread': float,
    'depth_l1_l5': [float, float, float, float, float],
    'depth_drop': float,
    'spread_ratio': float,
    'pre_vol_15m': float,
    'pre_range_15m': float,
    'pre_range_10s': float,
    'pre_vol_30s': float,
    'pre_drift_2m': float,
    'label': 'FOLLOW' or 'NO_TRADE'
}
```

**Status:** In progress

---

### Step 2: Add Liquidity Vacuum Score

**Goal:** Measure market emptiness after forced flow

**Components:**

1. **Depth Collapse**
   ```python
   DepthDropNow = TopDepthNow / median(TopDepth, 15m)
   # Strong vacuum: DepthDropNow < 0.6-0.7
   ```

2. **Spread Expansion**
   ```python
   SpreadRatioNow = SpreadNow / median(Spread, 15m)
   # Vacuum: SpreadRatioNow > 1.5
   ```

3. **Price Impact per Notional**
   ```python
   ImpactPerFlow = |return_2s| / AggNotional_2s
   # High impact = vacuum
   ```

**Combined Score:**
```python
VacuumScore = w1*(1 - DepthDropNow) + w2*(SpreadRatioNow - 1) + w3*ImpactPerFlow_norm
# Start with equal weights, tune later
```

**Measurement window:** 1-10 seconds after event

**Status:** Pending

---

### Step 3: Filter FOLLOW by Vacuum

**Goal:** Increase gross edge from ~14 bps to 30-60 bps

**Method:**
1. Calculate VacuumScore for all FOLLOW events
2. Split by quantiles (Q50, Q75, Q90)
3. Measure gross return by quantile
4. Find threshold where gross > 30 bps consistently

**Expected result:**
```
VacuumScore Quantile | Count | Gross | Net (maker)
Q0-Q50              | High  | ~10   | ~2 bps
Q50-Q75             | Med   | ~20   | ~12 bps
Q75-Q90             | Low   | ~40   | ~32 bps  ← Target
Q90-Q100            | VLow  | ~60   | ~52 bps
```

**New FOLLOW filter:**
```python
FOLLOW = (regime == 'FOLLOW') & (VacuumScore > threshold)
```

**Trade-off:** Lower frequency, higher edge per trade

**Status:** Pending

---

### Step 4: Pullback Entry Timing

**Goal:** Get maker fills, avoid buying highs

**Two-phase entry:**

#### Phase 1: Confirm "still follow" (3-10s after event)
```python
# Check context hasn't changed
still_valid = (
    vol_15m_low and  # Stable base
    short_expansion_present and  # Still expanding
    aggression_not_dead  # Still has momentum
)
```

#### Phase 2: Wait for pullback (maker entry)
```python
# For BUY follow (price went up):
1. Wait for pullback down X bps (5-15 bps, tune this)
2. Place limit BUY at pullback level (maker)
3. TP: further in trend direction

# For SELL follow (price went down):
1. Wait for bounce up X bps
2. Place limit SELL at bounce level (maker)
3. TP: further down
```

**Parameters to tune:**
- `pullback_bps`: 5-15 bps (start with 10)
- `max_wait_time`: 30-60s (if no pullback, skip)
- `limit_offset`: 0-2 bps inside pullback level

**Expected improvement:**
- Fill rate: 60-80% (vs 100% taker)
- Fees: 8 bps (vs 20 bps taker)
- Adverse selection: Lower (not buying tops)

**Status:** Pending

---

### Step 5: Fill Model

**Goal:** Understand execution reality

**Track for each signal:**
```python
{
    'signal_ts': timestamp,
    'limit_price': float,
    'filled': bool,
    'fill_ts': timestamp or None,
    'time_to_fill': float or None,  # seconds
    'adverse_move': float,  # price move against us before fill
    'missed_move': float or None,  # if not filled, what we missed
}
```

**Metrics:**
```python
FillRate = filled_count / total_signals
AvgTimeToFill = mean(time_to_fill | filled)
AvgAdverseMove = mean(adverse_move | filled)
AvgMissedMove = mean(missed_move | not filled)

# True EV accounting for fills
EV_true = FillRate * EV_filled + (1 - FillRate) * EV_missed
# where EV_missed often = 0 (opportunity cost)
```

**Status:** Pending

---

### Step 6: OOS Validation with All Improvements

**Goal:** Validate production readiness

**Test on 3 samples:**
1. Baseline FOLLOW (current)
2. FOLLOW + Vacuum filter
3. FOLLOW + Vacuum + pullback entry

**Metrics to track:**
```
Strategy          | Trades | Gross | Net (maker) | WR  | Stable?
Baseline          | 51     | +14   | +6          | 67% | 1/3 ❌
+ Vacuum filter   | ?      | ?     | ?           | ?   | ?
+ Pullback entry  | ?      | ?     | ?           | ?   | ?
```

**Success criteria:**
- Net EV > +10 bps (maker fees)
- Positive in 2/3 samples minimum
- Frequency: 1-5 trades/day acceptable (quality > quantity)

**Status:** Pending

---

### Step 7: FADE Frozen

**Status:** ✅ Complete

**Reason:** Gross edge +2.31 bps too small, cannot be fixed with filters

**Future research lane:**
- Different features (not current multi-scale)
- Different targets (t+60s, t+120s for mean reversion)
- Different event types (not forced flow)

---

## Implementation Order

### Week 1: Data Collection
- [ ] Build event dataset with all features
- [ ] Add orderbook snapshots (depth, spread)
- [ ] Calculate VacuumScore components
- [ ] Save to parquet for fast iteration

### Week 2: Vacuum Filter
- [ ] Analyze VacuumScore distribution
- [ ] Test quantile filters on OOS
- [ ] Find optimal threshold
- [ ] Measure gross edge improvement

### Week 3: Entry Timing
- [ ] Implement pullback detection
- [ ] Simulate limit order fills
- [ ] Build fill model
- [ ] Measure true EV with fills

### Week 4: Production
- [ ] Final OOS validation
- [ ] Deploy to paper trading
- [ ] Monitor live performance
- [ ] Iterate on parameters

---

## Expected Final Performance

**Conservative estimate:**
- Signals: 1-3/day (after vacuum filter)
- Gross edge: 30-50 bps
- Net edge (maker): 22-42 bps
- Fill rate: 70%
- True EV: ~20-30 bps per signal
- Daily EV: 20-90 bps = **0.2-0.9% daily**

**With multi-symbol (SOL + BTC + ETH):**
- Signals: 3-9/day
- Daily EV: 60-270 bps = **0.6-2.7% daily**

---

## Files Structure

```
flow_shock_research/
├── PRODUCTION_PLAN.md (this file)
├── build_event_dataset.py (Step 1)
├── calculate_vacuum_score.py (Step 2)
├── filter_by_vacuum.py (Step 3)
├── simulate_pullback_entry.py (Step 4)
├── build_fill_model.py (Step 5)
├── validate_production.py (Step 6)
└── results/
    ├── events_baseline.parquet
    ├── events_with_vacuum.parquet
    ├── vacuum_analysis.csv
    ├── fill_simulation.csv
    └── production_validation.csv
```

---

**Next:** Start Step 1 - Build baseline event dataset
