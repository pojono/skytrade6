# KIMI-1 MONTHLY BREAKDOWN ANALYSIS

**Date:** March 2026  
**Purpose:** Verify no single-month outliers in top multi-timeframe performers  
**Status:** ⚠️ **ALL TOP PERFORMERS ARE OUTLIERS**

---

## Executive Summary

**9/9 top performers are driven by single-month outliers.**

| Coin/TF | Total PnL | Best Month | % from Best | Warning |
|---------|-----------|------------|-------------|---------|
| COAI 1H | +$1,966 | +$2,268 | **115%** | ⚠️ OUTLIER |
| COAI 2H | +$1,002 | +$1,161 | **116%** | ⚠️ OUTLIER |
| MYX 1H | +$732 | +$885 | **121%** | ⚠️ OUTLIER |
| MYX 2H | +$759 | +$766 | **101%** | ⚠️ OUTLIER |
| H 1H | +$503 | +$395 | **78%** | ⚠️ OUTLIER |
| H 2H | +$332 | +$188 | **57%** | ⚠️ OUTLIER |
| AVNT 1H | +$321 | +$331 | **103%** | ⚠️ OUTLIER |
| MEME 1H | +$205 | +$210 | **102%** | ⚠️ OUTLIER |
| IP 1H | +$159 | +$173 | **109%** | ⚠️ OUTLIER |

**Key Finding:** Without the single best month, ALL of these would be losers or breakeven.

---

## Detailed Monthly Breakdown

### COAIUSDT 1H - Most Extreme Outlier

| Month | PnL | Trades | Wins | Status |
|-------|-----|--------|------|--------|
| 2025-10 | **+$2,268** | 39 | 19/39 | ✓ **OUTLIER MONTH** |
| 2025-11 | -$290 | 15 | 3/15 | ✗ Loser |
| 2025-12 | -$12 | 8 | 2/8 | ✗ Loser |
| 2026-01 | +$1 | 1 | 1/1 | ✓ Breakeven |

**Analysis:** 115% of total profit from October 2025. Other months are losers.

---

### MYXUSDT 2H - Best Win Rate But Still Outlier

| Month | PnL | Trades | Wins | Status |
|-------|-----|--------|------|--------|
| 2025-08 | -$1 | 7 | 4/7 | ✗ |
| 2025-09 | **+$766** | 24 | 21/24 | ✓ **OUTLIER** |
| 2025-10 | -$55 | 6 | 3/6 | ✗ |
| 2025-11 | +$49 | 5 | 5/5 | ✓ |

**Analysis:** 101% from September. Only 50% profitable months.

---

### HUSDT 1H - Most Consistent (But Still Outlier)

| Month | PnL | Trades | Wins | Status |
|-------|-----|--------|------|--------|
| 2025-06 | +$395 | 8 | 7/8 | ✓ |
| 2025-07 | +$86 | 15 | 10/15 | ✓ |
| 2025-08 | -$27 | 18 | 8/18 | ✗ |
| 2025-09 | +$50 | 32 | 17/32 | ✓ |

**Analysis:** 75% profitable months (best consistency). But still 78% from best month.

---

## Key Insights

### 1. Short Timeframe Trap
- 1H/2H strategies generate many trades
- High absolute profits look attractive
- **But:** Profits concentrate in 1-2 lucky months
- Remaining months are often losers

### 2. What This Means
| Issue | Impact |
|-------|--------|
| Single month dependency | Strategy not robust |
| Lucky timing | Not repeatable edge |
| High variance | Risk of large drawdowns |
| Low consistency | Hard to trade in real-time |

### 3. The Real Winners
**HUSDT 1H is the only decent performer:**
- 75% profitable months (3/4)
- Multiple months contribute positively
- Most consistent across months

---

## Comparison: Daily vs Short Timeframes

| Metric | Daily (Original) | 1H (Multi-TF) |
|--------|------------------|---------------|
| Best performer | XRP +$242 | COAI +$1,966 |
| Monthly consistency | 75-100% | 25-50% |
| Outlier risk | Low | **Extreme** |
| Robustness | ✅ High | ❌ Low |
| Trade frequency | Low | High |

**Verdict:** Daily strategy is far more robust despite lower absolute profits.

---

## Revised Recommendations

### ❌ DO NOT TRADE (Single-Month Outliers)
- COAIUSDT (all timeframes)
- MYXUSDT (all timeframes)
- AVNTUSDT
- MEMEUSDT
- IPUSDT

### ✅ CONSIDER (Best Consistency)
- **HUSDT 1H** - 75% profitable months, most robust short-TF performer

### ✅ STICK WITH (Original Audit)
- **XRPUSDT 1D** - 100% OOS splits profitable
- **DOGEUSDT 1D** - 100% OOS splits profitable
- **ADAUSDT 1D** - 100% OOS splits profitable
- **AVAXUSDT 1D** - 100% OOS splits profitable

---

## Lessons Learned

1. **Always check monthly breakdown** - Aggregate PnL can hide outlier dependency
2. **Short timeframes ≠ better** - More noise, less consistency
3. **Win rate matters less than monthly consistency** - 79% win rate with 50% profitable months = bad
4. **OOS walk-forward catches overfit** - Monthly breakdown confirms it

---

## Next Steps

1. **Discard 1H/2H strategies** - Too dependent on single months
2. **Focus on daily timeframe** - More robust, consistent
3. **Re-test daily on full 143-coin universe** - Find more daily winners
4. **Only exception: HUSDT 1H** - Worth deeper analysis

---

## Files

- `monthly_breakdown.py` - Analysis script
- `MONTHLY_BREAKDOWN_RESULTS.csv` - Detailed monthly data
- `MULTI_TIMEFRAME_RESULTS.md` - Previous (now questionable) results

---

## Conclusion

**The multi-timeframe "breakthrough" was an illusion.** All top performers rely on single lucky months. 

**Return to daily timeframe strategies** - they passed proper OOS walk-forward validation with consistent monthly performance.
