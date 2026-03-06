# Phase 16: Final Combined Strategy

**Date:** 2026-03-06
**Script:** `research_cross_section/phase16_final_combined.py`

---

## Important: Time Alignment Fix

Phases 11b/12/13 had a resample bug: used `.last()` instead of `.first()` when converting 1h data to 8h bars. This caused a 7-hour shift in the forward return window — returns from 07:00–15:00 instead of 00:00–08:00. The funding carry is credited at exactly 00:00/08:00/16:00, so the `.last()` alignment misses this.

Phase 16 uses Phase 10's correct approach:
```python
panel.resample("8h", closed="left", label="left").first()
```

All numbers in this document use the correct alignment.

---

## All Variants Compared (Jan 2025 – Mar 2026, No-Majors, correct alignment)

| Strategy | Sharpe | Sortino | Ann Ret | MaxDD | $1k→ |
|----------|--------|---------|---------|-------|------|
| 1. Baseline (funding+mom24h) | 3.452 | 5.610 | +1385% | -45.4% | $23,424 |
| **4. Funding + mom + f_trend** | **3.470** | **5.795** | **+1311%** | **-32.7%** | **$22,067** |
| 6. Baseline + soft regime | 2.625 | 3.734 | +222% | -21.0% | $3,919 |
| 5. Baseline + binary regime | 2.384 | — | +346% | -34.0% | $5,739 |
| 8. 2×F+trend + binary regime | 1.223 | — | +78% | -37.0% | $1,964 |
| 3. 2×Funding + f_trend | 1.469 | — | +136% | -53.6% | $2,723 |
| 2. Funding only | 1.402 | — | +120% | -43.9% | $2,523 |
| 7. 2×F+trend + soft regime | 0.944 | — | +35% | -36.5% | $1,418 |

---

## Key Results

### Winner: Strategy 4 (Funding + mom_24h + funding_trend)

**Signal construction:**
```python
# At each 8h bar:
z_funding       = cs_zscore(funding)
z_mom24h        = cs_zscore(mom_24h)
z_funding_trend = cs_zscore(funding - funding.shift(24))  # 24h change at 1h resolution

composite = (2*z_funding + z_mom24h + z_funding_trend) / 4
```

**Why 2× weight on funding?** Funding has the strongest IC (+0.203 ICIR) and is the primary signal. The 2:1:1 weighting preserves its dominance while incorporating the two supporting signals.

**Results vs baseline:**
| Metric | Baseline | Strategy 4 | Change |
|--------|---------|-----------|--------|
| Sharpe | 3.452 | 3.470 | +0.5% |
| Sortino | 5.610 | 5.795 | +3.3% |
| MaxDD | -45.4% | -32.7% | **+28% better** |
| $1k→ | $23,424 | $22,067 | -6% (less compounding, less risk) |

### Monthly comparison (Strategy 4 vs Baseline)

| Month | Baseline | Strategy 4 | Diff |
|-------|---------|-----------|------|
| 2025-01 | +16.8% | +26.7% | +9.9% ▲ |
| 2025-02 | +18.7% | +7.8% | -10.9% |
| 2025-03 | +14.9% | +13.7% | -1.2% |
| 2025-04 | +31.3% | +9.2% | -22.1% |
| **2025-05** | **-14.8%** | **+6.2%** | **+21.0% ▲** |
| 2025-06 | -5.7% | -6.4% | -0.7% |
| **2025-07** | **-20.3%** | **-18.0%** | **+2.3%** |
| 2025-08 | +17.7% | +22.6% | +4.9% ▲ |
| 2025-09 | +84.4% | +7.6% | -76.8% |
| **2025-10** | **+133.7%** | **+172.6%** | **+38.9% ▲** |
| 2025-11 | +35.7% | +37.3% | +1.6% |
| **2025-12** | **-13.1%** | **+28.6%** | **+41.7% ▲** |
| 2026-01 | +106.7% | +85.3% | -21.4% |
| 2026-02 | +17.9% | +19.1% | +1.2% |
| 2026-03 | +20.0% | +13.8% | -6.2% |

Strategy 4 avoids the worst months (May -14.8% → +6.2%, Dec -13.1% → +28.6%) at the cost of missing some big months (Sep +84% → +8%). Net effect: lower peak equity but much lower drawdown.

---

## Conservative Option: Soft Regime Filter

For investors who prefer lower MaxDD at the cost of total return:

| Metric | Baseline | Soft Regime |
|--------|---------|-------------|
| Sharpe | 3.452 | 2.625 |
| MaxDD | -45.4% | -21.0% |
| $1k→ | $23,424 | $3,919 |

The soft regime scales positions by `confidence = sqrt(signal_strength_pctile × funding_disp_pctile)` → mapped to [0.25×, 1.0×].

---

## Signals NOT Worth Adding

| Signal | Reason |
|--------|--------|
| Funding only | Sharpe 1.402 — mom_24h is genuinely superadditive |
| OI growth | Negative IC (-0.084 ICIR) |
| Volume anomaly | Negative IC (-0.117 ICIR) |
| BTC-relative | Negative IC (-0.184 ICIR) |
| Dynamic leverage | Counterproductive — cuts monster months |
| Asymmetric regime | Worsens MaxDD (unhedged long) |
| BTC trend gate | Too blunt, drops Sharpe significantly |

---

## Final Recommended Configuration

**Signal:** `(2×z_funding + z_mom24h + z_funding_trend) / 4`
**Universe:** 113 coins (No-Majors exclusion, structural)
**Portfolio:** N=10 long + N=10 short, equal-weight, market-neutral
**Rebal:** Every 8h aligned with Bybit funding settlements (00:00, 08:00, 16:00 UTC)
**Execution:** Limit orders (maker-first, 4 bps/side)
**Regime:** None (full exposure) — or soft threshold for conservative deployment
**Capacity:** $1M–$5M optimal, $10M hard limit

**OOS Performance (Jan 2025 – Mar 2026):**
- Sharpe: **3.47**
- Sortino: **5.80**
- Ann Return: **~1311%** (on notional, unlevered)
- MaxDD: **-32.7%**
- $1k start → **$22,067**
- Win rate: **52.3%** of bars
