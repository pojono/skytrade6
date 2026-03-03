# FINDINGS — XS-2 Production-Grade Dispersion Portfolio

**Date:** 2026-03-03  
**Period:** 2026-01-01 → 2026-02-28  
**Universe:** 52 altcoins (Bybit perps)  
**Strategy:** R3 dispersion portfolio, H=60m, K=5 (long top-5 / short bottom-5 by vol_z)  
**Panel:** 883,584 rows, 655,020 valid  
**Anti-bug:** No lookahead, no survivorship bias, vintage model, strict causal features

---

## VERDICT: NO-GO ❌

The R3 dispersion portfolio does not survive production-grade testing.  
The XS-1 prototype's +5.6bp signal was an artifact of double-counting overlapping vintages.

---

## 1) What Changed vs XS-1

| Property | XS-1 (prototype) | XS-2 (production) |
|----------|---:|---:|
| Entry price | Signal-time close | Next 1m close (t+1m) |
| Exit price | Signal-time + H close | Close at t+H+1m |
| Portfolio model | Single snapshot | Vintage (each signal = independent trade) |
| Overlap handling | None (counted same trade N times) | Explicit (74% overlap measured) |
| Fees | None | 20bp RT per position |
| Slippage | None | Grid: 0/1/2/5 bp/side |
| Gap handling | None | ffill <5m, invalid ≥5m |
| Survivorship | Static universe | Dynamic (requires 3d beta + turnover + valid data) |
| Dispersion pctl | Global | Expanding (causal) |
| Walk-forward | None | Jan→Feb + Feb→Jan |

---

## 2) Baseline Sanity ✓

Unconditional excess returns centered at ~0:  
mean=+0.04bp, med=-3.57bp, std=158.2bp, WR=47.8%, N=655,020

---

## 3) Data Integrity Stats

- **Unified 1m grid:** 84,960 points
- **Forward-filled (<5m gaps):** minimal
- **Invalid (≥5m gaps):** minimal
- **Bug checks:** All passed (no dupes, no lookahead, no future in beta)
- **Entry ≠ signal close:** 93.7% differ (6.3% same when price unchanged between bars)

---

## 4) R3 Regime Stats

- **Activations:** 39,465 / 655,489 eligible (6.0%)
- **Vintages:** 3,227 over 59 days = **57.6/day**
- **Position overlap between consecutive vintages:** **74.0%**

The 74% overlap is critical: consecutive 5m signals mostly select the same coins.  
57.6 vintages/day × 10 positions/vintage × 20bp RT = massive fee drag.

---

## 5) Results by Slippage Level

| Slip/side | Total RT | Mean daily (bp) | Sharpe ann | Hit rate | Max DD (bp) | p_perm |
|---:|---:|---:|---:|---:|---:|---:|
| **0bp** | **20bp** | **-12.67** | **-7.69** | **29%** | **669** | **0.899** |
| 1bp | 22bp | -14.67 | -8.90 | 27% | 779 | 0.903 |
| 2bp | 24bp | -16.67 | -10.12 | 27% | 889 | 1.000 |
| 5bp | 30bp | -22.68 | -13.77 | 21% | 1219 | 0.993 |

**Even at zero slippage (20bp RT fees only), the strategy loses -12.67bp/day.**

---

## 6) Walk-Forward (OOS)

| Period | Slip=0bp mean | Slip=2bp mean | Sharpe |
|--------|---:|---:|---:|
| OOS Feb (train=Jan) | -11.31 bp/day | -15.32 bp/day | -10.06 |
| OOS Jan (train=Feb) | -13.94 bp/day | -17.94 bp/day | -10.07 |

Both OOS periods negative. No month saved.

---

## 7) Root Cause Analysis

### Why XS-1 showed +5.6bp and XS-2 shows -12.67bp:

1. **XS-1 counted the same position ~12× (overlap)**  
   With 74% overlap and 5m rebalancing, a position held for 60m gets included in ~12 consecutive "independent" portfolio snapshots. XS-1 averaged across these as if they were separate trades. The real number of independent trades is ~3,227/12 ≈ 269 per month.

2. **XS-1 had no fees**  
   Each vintage trade touches 10 positions (5 long + 5 short). At 20bp RT, that's 10 × 20bp / 10 = 20bp fee per vintage. With 57 vintages/day, but 74% overlap means ~15 unique position changes/day → still ~300bp/day in fees.

3. **XS-1 used signal-time price, not execution price**  
   Entry at t+1m close and exit at t+H+1m are more realistic. This costs a small amount but is not the main issue.

4. **Gross edge is near zero**  
   The raw long/short spread per vintage, before any fees, is approximately:  
   gross ≈ (-12.67 + 20bp fee) / vintage ≈ +7.3bp/vintage  
   But this is spread across ~57 overlapping vintages/day, so the unique gross edge is ~7.3bp per trade but only ~4-5 truly independent trades/day.

### The fundamental problem:

The **cross-sectional dispersion signal is too persistent** (74% overlap). High vol_z coins stay high vol_z for hours. This means:
- The signal fires on the same coins repeatedly
- Each "new vintage" is essentially the same position
- Fees scale with vintages, but edge scales with independent trades
- Fee/edge ratio is catastrophic

---

## 8) GO/NO-GO Checklist (Slip=2bp)

| Check | Result |
|-------|--------|
| OOS mean > 0 (both halves) | ✗ FAIL (-15.32, -17.94) |
| OOS Sharpe > 1.5 | ✗ FAIL (-10.06, -10.07) |
| Max DD < 800bp | ✗ FAIL (889bp) |
| Full period p < 0.05 | ✗ FAIL (p=1.000) |
| Survives slip=5bp | ✗ FAIL (-22.68bp/day) |

**All checks failed. VERDICT: NO-GO.**

---

## 9) Lessons Learned

1. **Prototype portfolio tests that ignore overlap and fees are dangerous.** The XS-1 +5.6bp was a 4σ mirage caused by counting the same trade 12 times.

2. **Persistent signals + high-frequency rebalancing = fee trap.** A 60m horizon with 5m signal frequency and 74% overlap means fees dominate.

3. **Vintage model is essential** for any high-frequency relative value strategy. Without it, you get inflated N and suppressed fee impact.

4. **Cross-sectional dispersion as a signal is structurally slow-moving.** Dispersion regimes last hours, not minutes. This means the only viable approach would be: signal once at regime onset, hold for regime duration, exit at regime end. But that requires regime change detection, not level detection.

---

## 10) Files

| File | Description |
|------|-------------|
| `flow_research/xs2_production.py` | Production-grade script |
| `output/xs2/xs2_panel.parquet` | Full panel dataset |
| `output/xs2/xs2_results.csv` | Results by slippage level |
| `output/xs2/xs2_vintages_slip2.csv` | Vintage-level detail |
