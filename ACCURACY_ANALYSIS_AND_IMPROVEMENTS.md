# Prediction Accuracy Analysis & Improvement Strategy

**Date:** 2026-02-27  
**Current Status:** Models trained, but accuracy needs improvement

---

## Critical Assessment: What We Actually Found

### Current Prediction Accuracy

| Target | CV MAE | Baseline Std | MAE/Std Ratio | Assessment |
|--------|--------|--------------|---------------|------------|
| **Time to bottom** | ±1,755 ms | 1,624 ms | **1.08** | ⚠️ POOR - barely better than guessing |
| **Sell volume** | ±$78,635 | $195,273 | **0.40** | ✅ GOOD - 60% better than baseline |
| **Price @ T+100ms** | ±50 bps | 64.3 bps | **0.78** | ⚠️ MARGINAL - only 22% better |
| **Price @ T+500ms** | ±47 bps | 62.3 bps | **0.75** | ⚠️ MARGINAL - only 25% better |
| **Price @ T+1s** | ±57 bps | 69.8 bps | **0.81** | ⚠️ MARGINAL - only 19% better |
| **Price @ T+5s** | ±60 bps | 64.8 bps | **0.93** | ❌ VERY POOR - barely better than mean |

### Reality Check

**You're absolutely right - ±60 bps is HUGE variance!**

**Why this matters:**
- Fees are 20 bps round-trip
- Need >40 bps profit to break even
- ±60 bps error means:
  - Predicted -100 bps could be -40 bps (break-even) or -160 bps (great)
  - **Can't reliably distinguish profitable from unprofitable trades**

**High R² is misleading:**
- R² = 0.98 just means model fits training data well
- But MAE/Std ratio shows we're only 20-25% better than guessing the mean
- **Overfitting is likely** (N=25 is very small)

---

## What We CAN Actually Predict

### ✅ Sell Wave Volume (GOOD)
- **MAE:** ±$78k on mean of $183k (43% error)
- **MAE/Std:** 0.40 (60% better than baseline)
- **Useful for:** Position sizing based on liquidity

### ⚠️ Price Drops (MARGINAL)
- **MAE:** ±47-60 bps on mean of -60 to -80 bps
- **MAE/Std:** 0.75-0.93 (only 7-25% better than baseline)
- **Problem:** Error is larger than the signal we're trying to capture
- **Not reliable enough** for precise exit timing

### ❌ Time to Bottom (POOR)
- **MAE:** ±1,755 ms on mean of 1,746 ms
- **MAE/Std:** 1.08 (WORSE than just guessing the mean!)
- **Completely unreliable** - don't use for exit timing

---

## Why Accuracy Is Poor

### 1. Insufficient Training Data (CRITICAL)

**Current:** N = 25 settlements with complete features

**Problem:**
- Need 10-20 samples per feature for reliable ML
- We have 15 features → need 150-300 samples
- **We're 6-12x short of minimum data**

**Evidence of overfitting:**
- Train MAE = 5-6 bps (perfect fit)
- CV MAE = 47-60 bps (poor generalization)
- **10x gap between train and CV error**

### 2. Missing Critical Features

**What we DON'T have:**

1. **FR Magnitude** ⭐ CRITICAL
   - Not captured in ticker stream
   - Likely the STRONGEST predictor
   - Analysis shows |FR| > 100 bps → -104 bps drop
   - Missing this is like predicting rain without knowing humidity

2. **Orderbook Dynamics** (time-series features)
   - Spread trend (widening vs tightening)
   - Depth trend (building vs draining)
   - Imbalance trend (shifting bid/ask)
   - We only use static averages, not trends

3. **Trade Flow Momentum**
   - Buy/sell pressure acceleration
   - Large trade detection (>$1k notional)
   - Trade size distribution
   - We only use simple imbalance ratio

4. **Market Microstructure**
   - Bid-ask bounce rate
   - Effective spread (vs quoted spread)
   - Price impact of recent trades
   - Order arrival rate

5. **Cross-Asset Signals**
   - BTC correlation (risk-on/risk-off)
   - Funding rate on BTC/ETH (market sentiment)
   - Volume on major pairs

6. **Temporal Features**
   - Hour of day (Asian/EU/US session)
   - Day of week
   - Time since last settlement
   - Recent volatility regime

### 3. Wrong Targets

**Current targets may not be optimal:**

**Time to bottom** is noisy:
- Depends on random large trades
- High variance (std = 1,624 ms)
- Better target: **probability of fast drop** (< 500ms vs > 1s)

**Final price** is noisy:
- Recovery is unpredictable
- Better target: **maximum drop** (worst price reached)

### 4. Feature Engineering Issues

**Current features are too simple:**
- Static averages (mean, std)
- No time-series dynamics
- No interaction terms
- No non-linear transformations

**Better features:**
```python
# Instead of: spread_mean_bps
# Use:
- spread_trend (linear regression slope)
- spread_acceleration (2nd derivative)
- spread_percentile_90 (tail behavior)
- spread_regime (high/medium/low based on history)

# Instead of: qty_imb_mean
# Use:
- qty_imb_trend (is it getting more bid-heavy?)
- qty_imb_volatility (how stable is it?)
- qty_imb_extreme_count (how often does it spike?)
```

---

## Missing Features Analysis

### Priority 1: MUST HAVE (Expected +30-50% accuracy improvement)

1. **Funding Rate Magnitude** ⭐⭐⭐
   ```python
   'fr_bps': abs(funding_rate) * 10000
   'fr_sign': 1 if funding_rate < 0 else -1
   'fr_extreme': 1 if abs(fr_bps) > 100 else 0
   ```
   **Why critical:** Historical analysis shows |FR| is strongest predictor
   - |FR| > 100 bps → -104 bps drop (very consistent)
   - |FR| < 40 bps → -30 bps drop (not worth trading)

2. **Orderbook Trend Features** ⭐⭐⭐
   ```python
   # Calculate over last 10s
   'spread_trend_bps_per_sec': linear_regression_slope(spread_timeseries)
   'depth_trend_usd_per_sec': slope(total_depth_timeseries)
   'imbalance_trend': slope(qty_imbalance_timeseries)
   ```
   **Why critical:** Direction of change matters more than static level
   - Widening spread → anticipation of volatility
   - Draining depth → liquidity providers pulling out

3. **Large Trade Detection** ⭐⭐
   ```python
   'large_trade_count': count(trades > $1000 in last 10s)
   'large_trade_imbalance': (large_buys - large_sells) / total_large
   'max_trade_size_usd': max(trade_notional in last 10s)
   ```
   **Why critical:** Whales moving the market
   - Large sells before settlement → more selling after
   - Institutional positioning

### Priority 2: SHOULD HAVE (Expected +10-20% improvement)

4. **Microstructure Features** ⭐⭐
   ```python
   'bid_ask_bounce_rate': count(price crosses spread) / total_trades
   'effective_spread_bps': mean(|trade_price - mid_price|) * 2
   'trade_arrival_rate': trades_per_second
   'price_impact_bps': mean(|price_change_per_$1k_traded|)
   ```

5. **Temporal/Regime Features** ⭐
   ```python
   'hour_of_day': 0-23
   'is_asian_session': 1 if 0-8 UTC else 0
   'is_us_session': 1 if 13-21 UTC else 0
   'recent_volatility_regime': 'high' if vol > 90th percentile else 'low'
   ```

6. **Cross-Asset Signals** ⭐
   ```python
   'btc_funding_rate_bps': BTC FR (market sentiment)
   'btc_price_change_1h_bps': BTC momentum
   'correlation_with_btc': rolling correlation
   ```

### Priority 3: NICE TO HAVE (Expected +5-10% improvement)

7. **Liquidation Signals**
   ```python
   'liquidation_count_10s': count(liquidations in last 10s)
   'liquidation_volume_usd': sum(liq notional)
   'liquidation_imbalance': (long_liqs - short_liqs) / total
   ```

8. **Order Book Shape**
   ```python
   'depth_concentration': bid10 / total_bid (how concentrated?)
   'spread_at_depth_10': price_diff at 10th level
   'orderbook_skew': (bid_levels - ask_levels) / total_levels
   ```

---

## Better Target Definitions

### Current Problems

**Time to bottom:**
- High variance (±1,755 ms)
- Noisy (depends on random trades)
- Not actionable (can't time exit precisely anyway)

**Price at T+5s:**
- Includes recovery (unpredictable)
- High variance (±60 bps)

### Proposed Better Targets

1. **Maximum Drop (instead of time to bottom)**
   ```python
   'max_drop_bps': min(price_0_to_5s) - ref_price
   ```
   **Why better:** What we actually care about
   - More stable than timing
   - Directly actionable (expected profit)

2. **Drop Probability Buckets (instead of exact price)**
   ```python
   'drop_category': 
     0 if drop < 40 bps (skip - won't beat fees)
     1 if 40 <= drop < 80 bps (marginal)
     2 if 80 <= drop < 120 bps (good)
     3 if drop >= 120 bps (excellent)
   ```
   **Why better:** Classification is easier than regression
   - Only need to distinguish profitable from unprofitable
   - Lower variance

3. **Fast Drop Indicator (instead of exact timing)**
   ```python
   'is_fast_drop': 1 if time_to_bottom < 500ms else 0
   ```
   **Why better:** Binary classification
   - Easier to predict than exact time
   - Actionable (exit at T+500ms if fast, T+2s if slow)

4. **Recovery Strength (new target)**
   ```python
   'recovery_bps': price_5s - min_price
   ```
   **Why useful:** Know when to exit early
   - Strong recovery → exit at T+1s
   - Weak recovery → can hold to T+5s

---

## Accuracy Improvement Strategy

### Phase 1: Add Critical Features (This Week)

**Goal:** Reduce MAE from ±60 bps to ±30 bps

1. **Add FR magnitude** (MUST DO FIRST)
   - Fetch from REST API at analysis time
   - Or fix ticker stream to capture it
   - Expected improvement: **-20 bps MAE**

2. **Add trend features**
   - Spread trend, depth trend, imbalance trend
   - Expected improvement: **-10 bps MAE**

3. **Add large trade detection**
   - Count and imbalance of >$1k trades
   - Expected improvement: **-5 bps MAE**

**Expected result:** MAE = 25-35 bps (usable for trading)

### Phase 2: Collect More Data (Next 2 Weeks)

**Goal:** Increase training set from N=25 to N=100+

1. **Continue recording** with all orderbook depths
2. **Retrain weekly** as data accumulates
3. **Monitor overfitting** (train vs CV gap)

**Expected result:** 
- More stable predictions
- Better generalization
- MAE = 20-30 bps

### Phase 3: Better Targets (Week 3)

**Goal:** Predict what actually matters

1. **Switch to classification** (drop category 0-3)
2. **Add fast drop indicator** (binary)
3. **Add recovery strength** (regression)

**Expected result:**
- 80-85% accuracy on profitable vs unprofitable
- Better trading decisions

### Phase 4: Advanced Features (Week 4+)

1. Microstructure features
2. Cross-asset signals
3. Temporal features
4. Feature interactions

**Expected result:** MAE = 15-25 bps

---

## Realistic Expectations

### What We CAN Achieve

**With N=100+ samples and critical features:**
- **Drop magnitude:** ±20-30 bps (vs current ±60 bps)
- **Profitable vs unprofitable:** 80-85% accuracy (vs current ~70%)
- **Fast vs slow drop:** 75-80% accuracy (vs current unusable)

**This is GOOD ENOUGH for trading:**
- Can distinguish >80 bps drops from <40 bps drops
- Can size positions appropriately
- Can time exits reasonably well

### What We CANNOT Achieve

**With any amount of data:**
- **Exact price prediction:** Market is stochastic, ±15-20 bps is the limit
- **Exact timing:** Too much randomness, ±500ms is the limit
- **100% win rate:** Impossible, 80-85% is realistic maximum

**Why:**
- Random large trades create noise
- Market microstructure is chaotic
- Other traders' actions are unpredictable

---

## Immediate Action Plan

### Step 1: Add FR Magnitude (TODAY)

```python
# In analyse_settlement_predictability.py
# Add FR fetching from REST API or ticker

def get_funding_rate(symbol, timestamp):
    # Fetch from Bybit API or parse from ticker
    return fr_bps

# Add to features
features['fr_bps'] = get_funding_rate(symbol, settle_time)
features['fr_abs_bps'] = abs(features['fr_bps'])
```

### Step 2: Add Trend Features (TODAY)

```python
# Calculate trends over 10s window
def calculate_trend(timeseries):
    x = np.arange(len(timeseries))
    slope, _ = np.polyfit(x, timeseries, 1)
    return slope

features['spread_trend'] = calculate_trend(spread_timeseries)
features['depth_trend'] = calculate_trend(depth_timeseries)
features['imbalance_trend'] = calculate_trend(imbalance_timeseries)
```

### Step 3: Retrain Models (TODAY)

```bash
# Re-run analysis with new features
python3 analyse_settlement_predictability.py charts_settlement/*.jsonl

# Retrain ML models
python3 ml_settlement_predictor.py --train settlement_predictability_analysis.csv

# Compare old vs new MAE
```

### Step 4: Collect More Data (ONGOING)

- Keep recorder running 24/7
- Target: 100+ settlements with complete features
- Retrain weekly

---

## Expected Timeline

| Week | Action | Expected MAE |
|------|--------|--------------|
| **Now** | Current models (N=25, basic features) | ±60 bps |
| **Week 1** | Add FR + trends (N=25) | ±35 bps |
| **Week 2** | Collect data (N=50) | ±30 bps |
| **Week 3** | Better targets + more data (N=75) | ±25 bps |
| **Week 4** | Advanced features (N=100) | ±20 bps |

**Target:** ±20-25 bps MAE = **usable for production trading**

---

## Conclusion

### What We Found

✅ **Sell volume prediction works** (±$79k, 60% better than baseline)  
⚠️ **Price prediction is marginal** (±60 bps, only 20% better than baseline)  
❌ **Time prediction doesn't work** (±1.8s, worse than guessing)

### Why Accuracy Is Poor

1. **Too little data** (N=25 vs need 150-300)
2. **Missing FR magnitude** (strongest predictor)
3. **No trend features** (static averages only)
4. **Wrong targets** (noisy regression vs stable classification)

### How to Improve

**Priority 1:** Add FR magnitude + trend features → **±35 bps**  
**Priority 2:** Collect 100+ samples → **±25 bps**  
**Priority 3:** Switch to classification → **80% accuracy**  
**Priority 4:** Advanced features → **±20 bps**

**Bottom line:** Current models are NOT production-ready. Need more data and better features to achieve ±20-25 bps accuracy (usable for trading).
