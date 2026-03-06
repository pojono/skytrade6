# WFO Monthly Breakdown - Verified Coins Analysis

**Date:** 2026-03-06  
**Methodology:** Walk-Forward Out-of-Sample with Monthly Consistency Checks  
**Coins Tested:** DOGEUSDT, IPUSDT, SPXUSDT  
**Parameters Tested:** Baseline (5d, 1%) vs Optimized (10d, 0.5%)

---

## Executive Summary

**Only 1 coin passes all 3 robustness criteria:** SPXUSDT with optimized parameters (10-day lookback, 0.5% threshold)

| Coin | Params | OOS Robust | Monthly Robust | Outlier Free | Status |
|------|--------|------------|----------------|--------------|--------|
| SPX | 10d, 0.5% | ✓ | ✓ | ✓ | **★★★ TRULY ROBUST** |
| SPX | 5d, 1% | ✓ | ✓ | ✗ | ★★ OOS+Monthly (outliers) |
| IP | 5d, 1% | ✓ | ✓ | ✗ | ★★ OOS+Monthly (outliers) |
| IP | 10d, 0.5% | ✓ | ✓ | ✗ | ★★ OOS+Monthly (outliers) |
| DOGE | Both | ✗ | ✗ | ✗ | ✗ Failed |

**Critical Finding:** Rigorous WFO + monthly validation eliminates coins that appeared robust in simpler tests. Only SPX with optimized parameters survives.

---

## Methodology

### Walk-Forward Splits
1. **S1_Early:** First 50% of data
2. **S2_Mid:** Middle 50% (25%-75%)
3. **S3_Late:** Last 50% of data

### Robustness Criteria
1. **OOS Robust:** Profitable in ALL 3 splits
2. **Monthly Robust:** ≥50% profitable months in EACH split
3. **Outlier Free:** No single month contributes >50% of total PnL in ANY split

---

## Detailed Results

### SPXUSDT - Optimized (10-day, 0.5%) ⭐ TRULY ROBUST

| Split | PnL | Return | Trades | Win Rate | Profitable Months | Best Month % |
|-------|-----|--------|--------|----------|-------------------|--------------|
| S1_Early | +$13.13 | +1.3% | 16 | 69% | 100% | 40% |
| S2_Mid | +$14.88 | +1.5% | 18 | 78% | 100% | 35% |
| S3_Late | +$9.46 | +0.9% | 12 | 75% | 80% | 43% |
| **Avg** | - | **+1.2%** | 15 | **74%** | **93%** | **39%** |

**Key Metrics:**
- Max Drawdown: 0.2% (extremely low)
- All splits profitable
- All splits have ≥80% profitable months
- No outlier months (all <50%)

**Assessment:** Passes all criteria. Consistent performance across time with excellent monthly distribution.

---

### SPXUSDT - Baseline (5-day, 1%)

| Split | PnL | Return | Trades | Win Rate | Profitable Months | Best Month % |
|-------|-----|--------|--------|----------|-------------------|--------------|
| S1_Early | +$8.73 | +0.9% | 21 | 52% | 67% | **60%** |
| S2_Mid | +$14.70 | +1.5% | 23 | 65% | 86% | **40%** |
| S3_Late | +$9.88 | +1.0% | 15 | 60% | 67% | **59%** |

**Issue:** S1_Early has 60% concentration (best month >50% threshold)
- Total PnL: $8.73, Best month likely ~$5.24 (60%)
- Fails outlier-free criterion

**Assessment:** OOS robust but monthly concentration issues.

---

### IPUSDT - Baseline (5-day, 1%)

| Split | PnL | Return | Trades | Win Rate | Profitable Months | Best Month % |
|-------|-----|--------|--------|----------|-------------------|--------------|
| S1_Early | +$16.06 | +1.6% | 13 | 54% | 67% | **79%** |
| S2_Mid | +$2.67 | +0.3% | 13 | 46% | 50% | **169%** |
| S3_Late | +$14.91 | +1.5% | 12 | 58% | 80% | **77%** |

**Issues:**
- S1: 79% concentration (high)
- S2: 169% concentration (extreme - best month > total PnL!)
- Sample size small (12-13 trades per split)

**Assessment:** OOS robust but severe outlier dependency.

---

### IPUSDT - Optimized (10-day, 0.5%)

| Split | PnL | Return | Trades | Win Rate | Profitable Months | Best Month % |
|-------|-----|--------|--------|----------|-------------------|--------------|
| S1_Early | +$2.77 | +0.3% | 4 | 50% | 67% | **184%** |
| S2_Mid | +$5.06 | +0.5% | 8 | 38% | 67% | **101%** |
| S3_Late | +$11.05 | +1.1% | 8 | 50% | 100% | **55%** |

**Issues:**
- Very low trade count (4-8 trades per split)
- S1: 184% concentration (extreme outlier)
- S2: 101% concentration (best month > total PnL)

**Assessment:** OOS robust but very few trades + outlier issues.

---

### DOGEUSDT - Both Configurations

**Baseline (5d, 1%):**
| Split | PnL | Return | Trades | Profitable Months |
|-------|-----|--------|--------|-------------------|
| S1_Early | **-$17.46** | -1.7% | 32 | 14% |
| S2_Mid | +$11.82 | +1.2% | 51 | 47% |
| S3_Late | +$21.09 | +2.1% | 68 | 65% |

**Optimized (10d, 0.5%):**
| Split | PnL | Return | Trades | Profitable Months |
|-------|-----|--------|--------|-------------------|
| S1_Early | **-$12.44** | -1.2% | 24 | 30% |
| S2_Mid | +$14.37 | +1.4% | 42 | 62% |
| S3_Late | +$22.51 | +2.3% | 55 | 67% |

**Issue:** S1_Early is unprofitable in both configurations
- Early period (2023-mid 2024) shows negative performance
- Strategy fails in this regime
- Only becomes profitable in later periods (S2, S3)

**Assessment:** Fails OOS robustness (not profitable in all splits).

---

## Key Insights

### 1. Parameter Optimization Matters

| Metric | SPX Baseline | SPX Optimized | Improvement |
|--------|--------------|---------------|-------------|
| Win Rate | 59% | **74%** | +25% |
| Monthly Consistency | 73% | **93%** | +27% |
| Max Drawdown | 0.3% | **0.2%** | -33% |
| Outlier Free | ✗ | ✓ | Critical |

**10-day lookback + 0.5% threshold significantly improves robustness.**

---

### 2. Temporal Regime Sensitivity

**DOGE fails because of early period underperformance:**
- S1 (Early 50%): -1.2% to -1.7%
- S2 (Mid 50%): +1.2% to +1.4%
- S3 (Late 50%): +2.1% to +2.3%

**Pattern:** Strategy improves over time for DOGE
- Possible explanation: Higher volatility regime in 2024-2025
- Trend/momentum environment more favorable

**Implication:** DOGE may work NOW but not historically - regime dependency.

---

### 3. Outlier Detection is Critical

Without checking for outlier months, IP and baseline SPX appeared robust:
- **IP (baseline):** 169% concentration in S2 (best month > total PnL!)
- **SPX (baseline):** 60% concentration in S1

These are **not robust** - results driven by single months.

---

### 4. Sample Size Concerns

| Coin | Avg Trades/Split (10d) | Assessment |
|------|------------------------|------------|
| SPX | 15 | Adequate |
| IP | 7 | **Low** |
| DOGE | 40 | Good |

IP has only 4-8 trades per split with optimized params - too few for reliable statistics.

---

## Comparison to Previous Analyses

### Previous Finding (No WFO)
| Coin | Status |
|------|--------|
| DOGE | ✓ Robust |
| IP | ✓ Robust |
| SPX | ✓ Robust |

### Current Finding (WFO + Monthly)
| Coin | Status |
|------|--------|
| DOGE | ✗ Failed (S1 unprofitable) |
| IP | ✗ Failed (outliers) |
| SPX | ✓ Truly Robust (optimized only) |

**Adding WFO eliminated 2 of 3 "robust" coins.**

---

## Recommendations

### For Production Trading

**Only SPXUSDT with optimized parameters (10d, 0.5%) is truly robust.**

**Configuration:**
- Symbol: SPXUSDT
- Lookback: 10 days
- Threshold: 0.5%
- Risk per trade: 2-10% (based on risk tolerance)

### Risk Considerations

1. **Single coin concentration:** Only 1 coin passes - portfolio risk
2. **Limited track record:** SPX data starts ~2024 - short history
3. **Regime risk:** Early period underperformance in DOGE suggests possible regime shifts

### Suggested Position Sizing

Given only 1 truly robust coin:
- **Conservative:** 5% risk per trade on SPX only
- **Moderate:** 10% risk per trade on SPX only
- **Aggressive:** 20% risk per trade on SPX only + monitor for additional coins

---

## Validation Checklist

| Criterion | DOGE Baseline | DOGE Opt | IP Baseline | IP Opt | SPX Baseline | SPX Opt |
|-----------|---------------|----------|-------------|--------|--------------|---------|
| OOS Robust (all splits +) | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| Monthly Robust (≥50%) | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| Outlier Free (<50%) | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| **TRULY ROBUST** | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |

---

## Conclusion

**Rigorous WFO + monthly validation dramatically changes conclusions:**

1. **Previous analysis** (aggregate only): 3 robust coins
2. **WFO analysis** (temporal splits): 2-3 robust coins depending on params
3. **WFO + Monthly** (full rigor): **1 truly robust coin**

**SPXUSDT with 10-day lookback and 0.5% threshold is the only coin that:**
- Is profitable in ALL 3 temporal splits
- Has ≥50% profitable months in ALL splits  
- Has NO outlier months driving results

**This demonstrates the critical importance of:**
- Walk-forward validation (not just aggregate backtests)
- Monthly breakdowns (detecting outlier months)
- Parameter sensitivity analysis (baseline vs optimized)

---

## Files Generated

- `wfo_monthly_verified.py` - Analysis script
- `WFO_MONTHLY_VERIFIED.csv` - Detailed results per split

---

*Analysis complete. Only SPXUSDT with optimized parameters passes all robustness criteria.*
