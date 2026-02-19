# Integrated Liquidation Strategy — Findings (v26j)

**Date:** Feb 2026  
**Symbols:** DOGEUSDT, SOLUSDT, ETHUSDT, XRPUSDT  
**Data:** 282 days, 1-minute OHLC bars, realistic fees  
**Fees:** maker=0.02%, taker=0.055%

---

## The Problem

The v26h/v26i research showed that:
- **Market-order fade can't overcome fees** (raw edge +1-4 bps, fees 4-7.5 bps)
- **Limit-order entry works** but the baseline strategy has inconsistent months (2-3/5 positive)

We have **months of validated research** (v27-v42) that was never integrated as trading filters. This experiment combines them all.

---

## Research Filters Applied

| Source | Filter | What it does |
|--------|--------|-------------|
| **v33/v42H** | Exclude hours 08,09,13,16 UTC | Removes worst cascade reversion hours |
| **v42g** | Long-only (fade buy-side liqs) | Buy-side cascades revert 2-3 bps better |
| **v26g** | Displacement ≥10 bps | Only trade cascades that actually moved price |
| **v42B** | P97 cascade size (tested) | Larger cascades revert more reliably |
| **v42g** | Clustering 10-30min gap (tested) | Clustered cascades = best reversion |
| **v41** | OOS-validated params | offset=0.15-0.25%, TP=0.15-0.25%, SL=0.50% |

---

## Results: FULL COMBO vs BASELINE

### Monthly Consistency (off=0.15%, TP=0.15%, SL=0.50%)

**BASELINE (no filters, 30min hold):**

| Symbol | May | Jun | Jul | Aug | Feb | Positive |
|--------|-----|-----|-----|-----|-----|----------|
| DOGE | +11.6% | **-1.1%** | **-1.4%** | +1.4% | +0.6% | **3/5** |
| SOL | +8.1% | **-0.2%** | **-1.6%** | **-2.7%** | +4.6% | **2/5** |
| ETH | +9.3% | **-6.6%** | +0.1% | **-0.9%** | +1.1% | **3/5** |
| XRP | -0.1% | **-1.2%** | +0.7% | +5.7% | **-0.3%** | **2/5** |

**FULL COMBO (bad hours + long only + disp≥10, 60min hold):**

| Symbol | May | Jun | Jul | Aug | Feb | Positive |
|--------|-----|-----|-----|-----|-----|----------|
| DOGE | +5.3% | +2.4% | +0.1% | +0.9% | -0.0% | **4/5** |
| SOL | +3.0% | +4.5% | +0.1% | +2.7% | +2.2% | **5/5** ✅ |
| ETH | +9.8% | -2.3% | +1.3% | +2.0% | +0.9% | **4/5** |
| XRP | +1.8% | +2.0% | +1.1% | +4.7% | -0.3% | **4/5** |

### The filters turned losing months into winners:
- **SOL June**: -0.2% → **+4.5%**
- **SOL July**: -1.6% → **+0.1%**
- **SOL August**: -2.7% → **+2.7%**
- **DOGE June**: -1.1% → **+2.4%**
- **DOGE July**: -1.4% → **+0.1%**
- **ETH July**: +0.1% → **+1.3%**
- **ETH August**: -0.9% → **+2.0%**
- **XRP May**: -0.1% → **+1.8%**
- **XRP June**: -1.2% → **+2.0%**

### Positive months: Baseline 10/20 (50%) → Full Combo 17/20 (85%)

---

## Best Configurations (Full Combo, 60min hold)

### With SL=0.50% (off=0.15%, TP=0.15%)

| Symbol | Fills | Total Net | Avg Net/trade | WR | TP% | Pos Months |
|--------|-------|-----------|---------------|-----|-----|------------|
| DOGE | 287 | +8.7% | +0.030% | 88% | 87% | 4/5 |
| SOL | 350 | +12.4% | +0.035% | 89% | 89% | **5/5** |
| ETH | 409 | +11.7% | +0.029% | 88% | 87% | 4/5 |
| XRP | 268 | +9.4% | +0.035% | 89% | 89% | 4/5 |
| **TOTAL** | **1314** | **+42.2%** | **+0.032%** | **88.5%** | — | **17/20** |

### With no SL (off=0.20%, TP=0.15%)

| Symbol | Fills | Total Net | Avg Net/trade | WR | TP% | Pos Months |
|--------|-------|-----------|---------------|-----|-----|------------|
| DOGE | 269 | +11.2% | +0.042% | 95% | 95% | 4/5 |
| SOL | 323 | +15.2% | +0.047% | 92% | 92% | 4/5 |
| ETH | 380 | +3.6% | +0.010% | 91% | 90% | 4/5 |
| XRP | 249 | +13.2% | +0.053% | 92% | 92% | **5/5** |
| **TOTAL** | **1221** | **+43.3%** | **+0.035%** | **92.5%** | — | **17/20** |

---

## Why the Filters Work

### 1. Bad Hours Filter (v33/v42H)
Hours 08, 09, 13, 16 UTC are when cascades revert LEAST. These correspond to:
- **08-09 UTC**: London open — new directional flow overwhelms reversion
- **13 UTC**: Pre-NYSE — positioning ahead of US open
- **16 UTC**: US afternoon — trend continuation phase

Removing these hours eliminates ~25% of trades but removes the worst losers.

### 2. Long-Only Filter (v42g)
Buy-side liquidation cascades (longs getting stopped out → price drops) revert 2-3 bps better than sell-side cascades. This is because:
- Retail is long-biased → more forced selling during drops
- Forced selling creates temporary dislocations that revert
- Sell-side cascades (shorts squeezed) often reflect genuine momentum

### 3. Displacement Filter (v26g)
Only trading cascades that moved price ≥10 bps ensures:
- The cascade is real (not just a single large liquidation)
- There's enough displacement to fill the limit order
- The mean-reversion target is meaningful

### 4. Combined Effect
Each filter alone improves avg return by ~1-2 bps. Combined, they improve avg by ~3-5 bps and dramatically improve monthly consistency. The trade-off is fewer trades (~60-70% reduction), but the remaining trades are much higher quality.

---

## Strategy Parameters (Recommended)

```
CONSERVATIVE (SL=0.50%):
  Entry offset: 0.15%
  Take profit:  0.15% (maker exit)
  Stop loss:    0.50% (taker exit)
  Max hold:     60 minutes
  
AGGRESSIVE (no SL):
  Entry offset: 0.20%
  Take profit:  0.15% (maker exit)
  Stop loss:    none
  Max hold:     60 minutes (taker exit on timeout)

COMMON FILTERS:
  Cascade:      P95, min 2 events within 60s
  Bad hours:    Exclude 08, 09, 13, 16 UTC
  Direction:    Long only (fade buy-side liquidations)
  Displacement: Cascade must have moved price ≥10 bps
  Cooldown:     5 minutes between trades
  Symbols:      DOGE, SOL, ETH, XRP (all 4 simultaneously)
```

---

## Files

- `liq_integrated_strategy.py` — integrated strategy with all research filters
- `results/liq_integrated_strategy.txt` — full results
- `liq_bigmove_strategy.py` — v26h market-order sweep (all negative)
- `liq_bigmove_limit.py` — v26i limit-order sweep (profitable but inconsistent)
