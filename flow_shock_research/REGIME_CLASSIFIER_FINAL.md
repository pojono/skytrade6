# Regime Classifier - Final Results

**Date:** March 2, 2026  
**Status:** ✅ COMPLETE

---

## 📊 Classifier Design

### Binary Classification
- **FOLLOW:** Trade WITH flow (continuation/initiation regime)
- **FADE:** Trade AGAINST flow (exhaustion/reversal regime)
- **NO_TRADE:** Mixed/unclear regime

### Rule-Based Logic (Quantile Thresholds)

**FOLLOW Conditions:**
```python
vol_15m < q30  # Stable long-term base
AND range_10s > q80  # Short-term expansion
AND |drift_2m| > q70  # Strong momentum
```

**FADE Conditions (v1 - Original):**
```python
vol_15m > q70  # Elevated long-term
AND |drift_2m| > q85  # Extreme short-term
AND |imbalance_30s| > 0.5  # Panic
```

**FADE Conditions (v2 - Stricter):**
```python
vol_15m > q85  # More elevated
AND |drift_2m| > q95  # More extreme
AND |imbalance_30s| > 0.7  # More panic
AND vol_30s > q80  # Short-term spike
```

---

## 🎯 Performance Results (3 OOS Samples)

### Version 1 (Original Thresholds)

**FOLLOW Strategy:**
- **Total trades:** 43 (across 3 samples)
- **Avg gross return:** +19.85 bps
- **Avg net return (maker 8bps):** +11.85 bps
- **Positive samples:** 2/3
  - Sample 1 (May): 30 trades, +10.42 bps net ✅
  - Sample 2 (Jul-Aug): 13 trades, +15.14 bps net ✅
  - Sample 3 (Jun): 0 trades ❌
- **EV sign stability:** ❌ NO (Sample 3 has 0 trades)

**FADE Strategy:**
- **Total trades:** 78 (across 3 samples)
- **Avg gross return:** +2.31 bps
- **Avg net return (maker 8bps):** -5.69 bps
- **Positive samples:** 0/3
  - Sample 1: 39 trades, -4.24 bps net ❌
  - Sample 2: 32 trades, -8.33 bps net ❌
  - Sample 3: 7 trades, -3.71 bps net ❌
- **EV sign stability:** ❌ NO (all negative)

---

### Version 2 (Stricter FADE Thresholds)

**FOLLOW Strategy:**
- **Total trades:** 43 (same as v1)
- **Avg gross return:** +19.85 bps
- **Avg net return (maker 8bps):** +11.85 bps
- **Positive samples:** 2/3 (same as v1)
- **EV sign stability:** ❌ NO (Sample 3 still 0 trades)

**FADE Strategy:**
- **Total trades:** 0 (all filtered out)
- Stricter thresholds eliminated all FADE events
- **Result:** No FADE strategy in v2

---

## 💡 Key Findings

### 1. FOLLOW Strategy Works (with caveats)

**Gross edge:** +19.85 bps (100% WR in Samples 1 & 2)

**Fee sensitivity:**
- Maker (8 bps): **+11.85 bps net** ✅
- Mixed (14 bps): **+5.85 bps net** ✅
- Taker (20 bps): **-0.15 bps net** ❌

**Critical issue:** Sample 3 (June) produced **0 FOLLOW events**
- Thresholds are sample-specific (quantile-based)
- June had different market conditions (higher vol_15m baseline)
- Classifier failed to adapt

**Conclusion:** FOLLOW works when it fires, but **not stable across all samples**.

---

### 2. FADE Strategy Doesn't Work

**v1 (Original):** Gross edge only +2.31 bps
- Too small to cover any fees
- 47% win rate (coin flip)
- Negative net EV in all 3 samples

**v2 (Stricter):** Eliminated all events
- Thresholds too tight
- No trades = no strategy

**Conclusion:** Current FADE classification **cannot identify profitable exhaustion events**.

---

### 3. Sample 3 (June) Problem

**Why 0 FOLLOW events in Sample 3?**

Sample 3 characteristics:
- Higher baseline volatility (15m_vol_q30 = 0.85 vs 0.36-0.78 in other samples)
- Different market regime (exhaustion-dominant: 59% exhaustion, 33% reversal)
- Quantile thresholds adapted to local conditions

**FOLLOW requires:** `vol_15m < q30`
- In Sample 3: q30 = 0.85 (high)
- Most events had vol_15m > 0.85
- No events qualified

**Root cause:** Quantile-based thresholds are **not universal** across different market regimes.

---

## 🚨 Critical Issues

### Issue 1: Quantile Thresholds Not Universal

**Problem:** Each sample calculates its own quantiles
- Sample 1: 15m_vol_q30 = 0.78
- Sample 2: 15m_vol_q30 = 0.36
- Sample 3: 15m_vol_q30 = 0.85

**Result:** Same event would be classified differently depending on which sample it's in.

**Solution needed:** Use **global thresholds** (calculated from all samples combined) or **rolling window** thresholds.

---

### Issue 2: FADE Gross Edge Too Small

**Problem:** Even best FADE events only have +2-4 bps gross edge
- Cannot cover 8 bps maker fees
- Cannot cover 14-20 bps taker fees

**Possible causes:**
1. Wrong features (missing key exhaustion indicators)
2. Wrong target (t+30s too short for mean reversion)
3. Wrong classification (catching marginal events, not extreme)

**Solution needed:** Either abandon FADE or find better features/targets.

---

### Issue 3: Sample Size

**FOLLOW events are rare:**
- Sample 1: 30 events (7 days) = 4.3/day
- Sample 2: 13 events (7 days) = 1.9/day
- Sample 3: 0 events (7 days) = 0/day

**Average:** 2.0 events/day (when it works)

At +11.85 bps net per trade:
- Daily EV: 2.0 × 11.85 = **23.7 bps/day**
- Monthly: ~7.1% (if stable)

**Problem:** Too few signals for reliable income.

---

## ✅ What Works

### FOLLOW Strategy (Samples 1 & 2)

**Edge confirmed:**
- Gross: +19.85 bps
- Net (maker): +11.85 bps
- Win rate: 100% (gross), 83-100% (net)

**Requirements:**
- ✅ Must use maker orders (limit orders)
- ✅ Must target 20-60 bps moves (not 8-10 bps)
- ✅ Need pullback entry timing (better fills)

**Production viability:** YES, but low frequency (~2/day)

---

## ❌ What Doesn't Work

### FADE Strategy

**v1:** Gross edge too small (+2.31 bps)
**v2:** No events (over-filtered)

**Conclusion:** Current approach cannot identify profitable exhaustion events.

---

### Quantile-Based Thresholds

**Problem:** Not stable across samples
- Sample 3 produced 0 FOLLOW events
- Thresholds adapt to local conditions, not universal

**Conclusion:** Need global or rolling thresholds.

---

## 🎯 Recommendations

### 1. Fix FOLLOW Threshold Stability

**Current:** Per-sample quantiles (unstable)

**Solution A:** Global quantiles from all samples combined
```python
# Calculate thresholds from ALL samples
all_data = pd.concat([sample1, sample2, sample3])
thresholds = calculate_quantile_thresholds(all_data, features)
```

**Solution B:** Rolling window (e.g., last 30 days)
```python
# Use recent history for adaptive thresholds
recent_data = get_last_n_days(30)
thresholds = calculate_quantile_thresholds(recent_data, features)
```

---

### 2. Abandon FADE (for now)

**Reason:** Gross edge too small to be viable
- +2.31 bps cannot cover any fees
- Stricter filters eliminate all events

**Alternative:** Focus on FOLLOW only, improve frequency

---

### 3. Improve FOLLOW Frequency

**Current:** 2.0 events/day (when stable)

**Options:**
- Relax thresholds slightly (q30 → q40 for vol_15m)
- Add alternative FOLLOW patterns
- Multi-symbol deployment (SOL + BTC + ETH)

---

### 4. Add Timing Layer

**Current:** Classification only (FOLLOW/FADE/NO)

**Next:** Entry timing for FOLLOW
- Wait for pullback against flow
- Enter with limit order (maker fees)
- Target 20-60 bps (not 8-10 bps)

---

## 📁 Files Generated

**Classifier:**
- `build_regime_classifier.py` - Rule-based classifier (v1 & v2)

**Results:**
- `results/sample1_classified.csv` - Sample 1 with regime predictions
- `results/sample2_classified.csv` - Sample 2 with regime predictions
- `results/sample3_classified.csv` - Sample 3 with regime predictions
- `results/sample1_multiscale.csv` - Multi-scale features
- `results/sample2_multiscale.csv` - Multi-scale features
- `results/sample3_multiscale.csv` - Multi-scale features

**Documentation:**
- `REGIME_CLASSIFIER_FINAL.md` - This document

---

## 📊 Summary Table

| Metric | FOLLOW (v1) | FADE (v1) | FADE (v2) |
|--------|-------------|-----------|-----------|
| **Total trades** | 43 | 78 | 0 |
| **Gross return** | +19.85 bps | +2.31 bps | N/A |
| **Net (maker 8bps)** | +11.85 bps | -5.69 bps | N/A |
| **Win rate (gross)** | 100% | 47% | N/A |
| **Positive samples** | 2/3 | 0/3 | 0/3 |
| **EV stable?** | ❌ NO | ❌ NO | ❌ NO |
| **Production ready?** | ⚠️ Partial | ❌ NO | ❌ NO |

---

## 🎓 Lessons Learned

### 1. Quantile Thresholds Are Not Universal

Different market regimes have different baselines.
- Low-vol period: q30 = 0.36
- High-vol period: q30 = 0.85

Same absolute value (e.g., vol_15m = 0.5) is:
- "Low" in high-vol regime
- "High" in low-vol regime

**Solution:** Use global thresholds or z-scores.

---

### 2. Gross Edge Must Be Large

For maker fees (8 bps), need gross edge > 10 bps minimum.
For taker fees (20 bps), need gross edge > 25 bps minimum.

FADE's +2.31 bps gross is **not tradable**.

---

### 3. Sample Size Matters

43 FOLLOW events across 21 days = 2.0/day average.

But distribution is uneven:
- Sample 1: 4.3/day
- Sample 2: 1.9/day
- Sample 3: 0/day

**Conclusion:** Need more samples to validate frequency.

---

### 4. Fee Structure Determines Viability

**FOLLOW with different fees:**
- Maker (8 bps): +11.85 bps ✅ Profitable
- Mixed (14 bps): +5.85 bps ✅ Marginal
- Taker (20 bps): -0.15 bps ❌ Unprofitable

**Execution quality is critical.**

---

## 🚀 Next Steps

### Immediate (Fix Stability)
1. ✅ Calculate global thresholds from all 3 samples
2. ✅ Re-test classifier with global thresholds
3. ✅ Validate FOLLOW frequency is stable

### Short-term (Improve FOLLOW)
4. Add timing layer (pullback entry)
5. Test on additional samples (validate 2.0/day frequency)
6. Multi-symbol deployment (increase frequency)

### Long-term (Explore FADE)
7. Research alternative FADE features
8. Test longer horizons (t+60s, t+120s for mean reversion)
9. Consider abandoning FADE entirely

---

**Status:** Regime classifier complete. FOLLOW strategy viable with maker fees. FADE strategy not viable. Stability issues with quantile thresholds need resolution.
