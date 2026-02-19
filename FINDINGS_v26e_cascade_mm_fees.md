# Fee-Aware Cascade MM Sweep — Findings (v26e)

**Date:** Feb 2026  
**Symbols:** ETHUSDT, SOLUSDT, DOGEUSDT, XRPUSDT  
**Data:** 282 days (2025-05-11 to 2026-02-17), 1-minute OHLC bars  
**Fees:** maker=0.02%, taker=0.055%  
**Script:** `liq_cascade_mm_fees.py`  
**Results:** `results/liq_cascade_mm_fees.txt`

---

## Goal

The original cascade MM strategy (v26d) was tested with 0% fees and showed strong returns. This experiment adds **realistic double-fee modeling**:
- **Entry:** always maker fee (limit order)
- **TP exit:** maker fee (limit order) → round-trip = 0.04%
- **SL/timeout exit:** taker fee (market order) → round-trip = 0.075%

The question: **does the strategy survive real fees?**

---

## Key Finding: Tight TP + Wide SL Is Best

**Wider TP doesn't help.** It means fewer wins → more SL exits (taker fee) → higher average fee. The optimizer converged on: **TP=0.15%, SL=0.50%**.

The trick isn't wider TP. It's **wider SL** — giving losers more room means fewer stop-outs (fewer taker fees) and more trades that eventually revert to TP (maker fee).

---

## Best Config Per Symbol (off=0.20%, TP=0.15%, SL=0.50%)

| Symbol | Fills | WR | Gross | Fee | Net | Total | Sharpe | Max DD |
|--------|-------|----|-------|-----|-----|-------|--------|--------|
| **DOGE** | 802 | 85.3% | +0.065% | 0.045% | **+0.020%** | **+15.9%** | +53 | 5.1% |
| **SOL** | 1,102 | 81.1% | +0.056% | 0.047% | **+0.010%** | **+10.7%** | +26 | 10.1% |
| **ETH** | 1,380 | 80.5% | +0.052% | 0.047% | **+0.005%** | **+7.4%** | +14 | 6.0% |
| **XRP** | 903 | 80.6% | +0.053% | 0.047% | **+0.006%** | **+5.5%** | +16 | 4.8% |

**All 4 symbols net positive over 282 days.** DOGE is the clear winner — fattest gross edge (0.065%), lowest fee drag (85% TP exits = mostly maker fees), best Sharpe.

---

## US-Hours Only (Best Risk-Adjusted)

| Symbol | Fills | WR | Net/trade | Total | Sharpe | Max DD |
|--------|-------|----|-----------|-------|--------|--------|
| **DOGE** | 247 | 87.0% | +0.028% | +6.9% | **+77** | **2.1%** |
| **SOL** | 386 | 83.9% | +0.018% | +6.8% | **+47** | 4.1% |
| ETH | 454 | 81.5% | +0.000% | +0.0% | +0.2 | 6.4% |
| XRP | 294 | 80.6% | -0.006% | -1.9% | -15 | 6.9% |

**DOGE US-hours: Sharpe +77, max DD 2.1%, 87% WR.** Highest quality setup. US hours work best for DOGE and SOL but not ETH/XRP.

---

## Fee Impact Analysis

### 0% maker fee vs 0.02% maker fee (same config: off=0.20%, TP=0.15%, SL=0.50%)

| Symbol | 0% maker total | 0.02% maker total | Fee impact |
|--------|---------------|-------------------|------------|
| DOGE | +45.6% | +15.9% | **-65%** |
| SOL | +50.5% | +10.7% | **-79%** |
| ETH | +57.0% | +7.4% | **-87%** |
| XRP | +38.0% | +5.5% | **-86%** |

**The 0.02% maker fee destroys 65-87% of gross returns.** This is why fee optimization is critical — every basis point of fee reduction matters enormously.

---

## Monthly Breakdown (Best Config, All Hours)

### DOGEUSDT (off=0.20%, TP=0.15%, SL=0.50%)
| Month | Trades | WR | Total |
|-------|--------|----|-------|
| 2025-05 | 389 | 88.9% | **+15.55%** |
| 2025-06 | 249 | 82.7% | **+2.83%** |
| 2025-07 | 60 | 78.3% | -1.77% |
| 2025-08 | 90 | 78.9% | -2.27% |
| 2026-02 | 14 | 100% | **+1.54%** |
| **Total** | **802** | **85.3%** | **+15.89%** |

### SOLUSDT (off=0.20%, TP=0.15%, SL=0.50%)
| Month | Trades | WR | Total |
|-------|--------|----|-------|
| 2025-05 | 356 | 82.6% | **+3.19%** |
| 2025-06 | 502 | 79.1% | **+1.34%** |
| 2025-07 | 83 | 77.1% | **+0.08%** |
| 2025-08 | 113 | 84.1% | **+2.60%** |
| 2026-02 | 48 | 91.7% | **+3.48%** |
| **Total** | **1,102** | **81.1%** | **+10.69%** |

---

## Key Insights

1. **SL width is the #1 fee lever.** SL=0.50% means only 11-12% of trades hit SL (taker fee). SL=0.15% means 37-38% hit SL → average fee jumps from 0.045% to 0.054%.
2. **TP should be tight (0.15%).** Wider TP (0.20-0.25%) reduces TP rate, increases SL/timeout exits, raises average fee.
3. **DOGE has the fattest edge** — 0.065% gross, 85% WR, lowest fee drag.
4. **US hours improve risk-adjusted returns** for DOGE and SOL (higher WR, lower DD).
5. **The strategy is viable but thin** — net edge is only 0.5-2.0 bps per trade after fees.

---

## Implications for Next Steps

- The thin net edge motivates exploring **no-SL configurations** (v26f) to eliminate taker fees entirely
- Also motivates **research-based filters** (v26j) to improve trade quality
- Fee reduction (e.g., VIP tiers, maker rebates) would dramatically improve returns
