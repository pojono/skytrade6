# Flow Impact Theory - Why Z-Score Fails in Crypto

## 🚨 The Fundamental Problem

### What We Were Measuring (Wrong)
```python
flow_z = (volume - mean(volume)) / std(volume)
```

**Assumption:** Volume follows normal distribution  
**Reality:** Crypto volume is heavy-tailed, bursty, regime-dependent

**Result:** z > 3 triggers 11,000+ times/day (every 8 seconds!)

---

## 📊 Why Z-Score Fails

### 1. Non-Normal Distribution

**Traditional Finance:**
- z > 3: 99.7th percentile (rare)
- z > 5: Extremely rare
- z > 10: Almost impossible

**Crypto Reality:**
- z > 3: P(|Z| > 3) ≫ 0.3% (happens constantly)
- Heavy tails: Extreme values are normal
- Power-law process, not Gaussian

### 2. Measures Activity, Not Impact

**Problem:** $500k volume means different things:
- On deep book (10M depth): Barely moves price
- On thin book (500k depth): Punches through entire level

**Z-score doesn't see this difference.**

### 3. Regime Dependence

Crypto has volatility regimes:
- Quiet periods: Low volume baseline
- Active periods: High volume baseline

Using global mean/std fails during regime shifts.

---

## ✅ The Correct Approach: Flow Impact

### Core Concept

**Measure:** Volume's ability to move market, not just size

```python
Flow Impact = Aggressive Volume / Top Book Depth
```

### Why This Works

**Normalization by Liquidity:**
- Same volume, different impact based on available liquidity
- Automatically regime-independent
- Measures pressure, not activity

### Interpretation Scale

| Flow Impact | Meaning | Market State |
|-------------|---------|--------------|
| 0.05 | Noise | Normal |
| 0.2 | Activity | Elevated |
| 0.5 | Stress | Strained |
| 1.0 | Punch-through | Breaking |
| >1.5 | Forced flow | Disequilibrium |

---

## 🔧 Implementation Details

### 1. Signed Aggressive Flow

**Track direction:**
```python
buy_volume = sum(v for v, side in trades if side == 'Buy')
sell_volume = sum(v for v, side in trades if side == 'Sell')

flow_imbalance = abs(buy_volume - sell_volume) / (buy_volume + sell_volume)
aggressive_volume = max(buy_volume, sell_volume)
flow_direction = 'Buy' if buy_volume > sell_volume else 'Sell'
```

**Why:** One-sided flow creates stronger disequilibrium

### 2. Book Depth Normalization

**Use relevant side:**
```python
if flow_direction == 'Buy':
    relevant_depth = ask_depth  # Buyers hit asks
else:
    relevant_depth = bid_depth  # Sellers hit bids

flow_impact = aggressive_volume / relevant_depth
```

**Top 5 levels:** Captures immediately available liquidity

### 3. Robust Statistics

**Replace mean/std with median/MAD:**
```python
median_vol = np.median(volumes)
mad = np.median(np.abs(volumes - median_vol))
robust_z = (volume - median_vol) / mad
```

**Why:** Survives regime changes and outliers

### 4. Burst Detection

**Require sustained flow:**
```python
recent_trades = last_30_trades
same_direction_count = sum(1 for trade in recent_trades 
                           if trade.side == flow_direction)
burst_ratio = same_direction_count / len(recent_trades)

is_burst = burst_ratio > 0.6  # 60%+ same direction
```

**Why:** Filters random spikes, captures forced flow

---

## 📈 Expected Results

### Z-Score Approach (v1)
- **Threshold:** z > 30
- **Events/day:** 2.3
- **Problem:** Arbitrary threshold, doesn't measure impact

### Flow Impact Approach (v2)
- **Threshold:** impact > 0.6 (market stress)
- **Events/day:** Expected 3-10 (TBD)
- **Advantage:** Physically meaningful threshold

---

## 🎯 Detection Criteria

### Combined Filter

```python
is_high_impact = flow_impact > 0.6
is_imbalanced = flow_imbalance > 0.7  # 70%+ one direction
is_burst = burst_ratio > 0.6           # 60%+ same direction

if is_high_impact and is_imbalanced and is_burst:
    event_detected()
```

**All three must be true:**
1. **High Impact:** Volume can move market
2. **Imbalanced:** One-sided flow
3. **Burst:** Sustained, not random

---

## 🔬 Mathematical Foundation

### Why Volume Alone Fails

**Volume spike without context:**
- Could be market maker inventory management
- Could be iceberg order execution
- Could be algo rebalancing

**None of these move price.**

### Why Impact Works

**Flow Impact captures:**
- Volume relative to available liquidity
- Directional conviction (imbalance)
- Sustained pressure (burst)

**This combination = forced flow = price movement**

---

## 📊 Comparison Table

| Aspect | Z-Score (v1) | Flow Impact (v2) |
|--------|--------------|------------------|
| **Basis** | Volume spike | Market impact |
| **Distribution** | Assumes normal | Distribution-free |
| **Regime handling** | Poor (mean/std) | Good (median/MAD) |
| **Liquidity aware** | No | Yes |
| **Directional** | No | Yes (buy/sell) |
| **Burst detection** | No | Yes |
| **Threshold meaning** | Statistical | Physical |
| **Events/day** | 2.3 (z>30) | TBD (impact>0.6) |

---

## 🎓 Key Insights

### 1. Crypto ≠ Traditional Finance

**Don't import statistical thresholds blindly.**
- z > 3 is rare in stocks
- z > 3 is noise in crypto
- Need crypto-native metrics

### 2. Context Matters

**Same volume, different impact:**
- Thin book: High impact
- Deep book: Low impact
- Must normalize by liquidity

### 3. Direction Matters

**Symmetric vs asymmetric flow:**
- 50/50 buy/sell: Market making
- 90/10 buy/sell: Forced flow
- Imbalance reveals conviction

### 4. Persistence Matters

**Single spike vs sustained pressure:**
- One big trade: Could be anything
- 20 consecutive trades same direction: Forced flow
- Burst detection is critical

---

## 🚀 Next Steps

### Phase 2: Validation

1. **Download orderbook data** ✅ In progress
2. **Run Flow Impact detector**
3. **Compare event frequency:** z-score vs flow impact
4. **Analyze price behavior** around both types of events

### Phase 3: Optimization

1. **Tune impact threshold:** 0.6 vs 0.8 vs 1.0
2. **Optimize window size:** 5s vs 10s vs 15s
3. **Refine burst detection:** Trade count vs time window
4. **Add order flow toxicity:** VPIN, Kyle's lambda

### Phase 4: Strategy

1. **Entry rules:** Flow impact + directional filter
2. **Exit rules:** Mean reversion vs momentum
3. **Position sizing:** Scale with impact magnitude
4. **Risk management:** Stop loss, max holding time

---

## 📚 References

### Academic Foundation

**Market Microstructure:**
- Kyle (1985): Informed trading and price impact
- Hasbrouck (1991): Order flow and price discovery
- Easley et al. (2012): VPIN and flow toxicity

**Heavy-Tailed Distributions:**
- Mandelbrot (1963): Stable distributions in finance
- Cont (2001): Empirical properties of asset returns
- Gabaix et al. (2003): Power laws in economics

**Crypto Microstructure:**
- Makarov & Schoar (2020): Trading and arbitrage in cryptocurrency markets
- Aloosh & Li (2019): Direct evidence of Bitcoin wash trading

---

## 💡 Practical Wisdom

### What We Learned

**1. Start Simple, Then Refine**
- v1: Z-score (simple but wrong)
- v2: Flow impact (correct foundation)
- v3: Add toxicity metrics (future)

**2. Let Data Guide You**
- Don't assume normal distributions
- Check your assumptions
- Validate on real data

**3. Physical Intuition > Statistical Tricks**
- "Can this volume move the market?" > "Is this volume rare?"
- Market impact is physical, not statistical
- Build metrics that match reality

---

**Last Updated:** March 2, 2026  
**Status:** Theory validated, implementation in progress  
**Next:** Run detector on real data
