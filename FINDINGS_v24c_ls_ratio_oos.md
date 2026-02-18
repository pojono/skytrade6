# FINDINGS v24c: LS Ratio Out-of-Sample Validation

**Date:** Feb 2026
**Symbols:** BTCUSDT, SOLUSDT
**In-sample:** Nov 2025 – Jan 2026 (92 days, includes v24's Dec 2025)
**Out-of-sample:** May 1 – Oct 31, 2025 (184 days, 6 months earlier)
**Data:** Binance data warehouse metrics (5-min resolution) + Binance 5m klines

---

## Motivation

v24 found extraordinary results on Dec 2025 data:
- `ls_ratio_top` (Binance top trader long/short ratio) IC = +0.20 at 4h
- Walk-forward combined signal Sharpe 9.3 (BTC), 8.6 (SOL)
- Simple z-threshold backtest profitable after 7 bps fees

v24b attempted OOS validation using Bybit ticker data (May-Aug 2025) but **could not test the LS ratio signal** because Bybit doesn't provide it. The Sharpe dropped from 9+ to 2-3 without LS ratios.

**This study resolves that gap** by downloading Binance metrics (which include LS ratios) from the Binance data warehouse for a completely disjoint 6-month OOS period.

---

## Key Result: The LS Ratio Signal Does NOT Replicate

### IC Comparison at 4h Horizon

| Feature | BTC IS | BTC OOS | SOL IS | SOL OOS |
|---------|--------|---------|--------|---------|
| **ls_top_zscore_24h** | **+0.039** | **+0.003** | **+0.083** | **+0.027** |
| ls_ratio_top | +0.017 | +0.005 | +0.051 | +0.024 |
| ls_global_zscore_24h | +0.019 | -0.006 | +0.079 | +0.007 |
| taker_ratio_4h | +0.048 | +0.005 | +0.073 | +0.010 |
| oi_zscore_24h | +0.010 | +0.013 | -0.026 | +0.024 |

**The flagship `ls_top_zscore_24h` IC drops from +0.039 → +0.003 on BTC (92% degradation) and from +0.083 → +0.027 on SOL (67% degradation).**

The sign is preserved on SOL (positive = momentum), but the magnitude is far too weak to be tradeable. On BTC, the signal effectively vanishes.

### Simple Backtest (z>1 long, z<-1 short, 4h hold, 7 bps fee)

| Signal | BTC IS Avg | BTC OOS Avg | SOL IS Avg | SOL OOS Avg |
|--------|-----------|------------|-----------|------------|
| **ls_top_zscore_24h** | **+2.9 bps** | **-7.6 bps** | **+12.8 bps** | **-1.0 bps** |
| ls_global_zscore_24h | -2.8 bps | -7.9 bps | +7.6 bps | -4.6 bps |
| oi_zscore_24h | -9.6 bps | -5.4 bps | -16.9 bps | -3.8 bps |
| taker_zscore_24h | -7.0 bps | -7.5 bps | -8.5 bps | -9.9 bps |

**Every signal loses money OOS.** The `ls_top_zscore_24h` goes from +2.9 → -7.6 bps on BTC and from +12.8 → -1.0 bps on SOL.

### Walk-Forward Combined Signal

| Feature Set | BTC IS IC | BTC IS Sharpe | BTC OOS IC | BTC OOS Sharpe |
|-------------|----------|--------------|-----------|---------------|
| OHLCV only | -0.047 | -9.11 | +0.040 | -4.59 |
| **OHLCV + OI/F** | **-0.019** | **-5.71** | **-0.018** | **-12.40** |

| Feature Set | SOL IS IC | SOL IS Sharpe | SOL OOS IC | SOL OOS Sharpe |
|-------------|----------|--------------|-----------|---------------|
| OHLCV only | +0.048 | +4.08 | +0.044 | -2.09 |
| **OHLCV + OI/F** | **+0.077** | **+2.31** | **+0.032** | **+0.55** |

**The walk-forward Sharpe collapses from +2.31 → +0.55 on SOL (the only marginally positive result) and is deeply negative on BTC.** The v24 Sharpe 9+ result was clearly period-specific.

### Crowding/Extreme Signals

**Contrarian LS ratio (short when high, long when low):** Fails on both assets in both periods. Negative Sharpe across the board.

**Momentum LS ratio (follow the crowd):**

| Condition | BTC IS | BTC OOS | SOL IS | SOL OOS |
|-----------|--------|---------|--------|---------|
| z < -2.0 → SHORT | **+26.1 bps** | **-10.4 bps** | **+55.3 bps** | **+15.3 bps** |
| z < -1.5 → SHORT | +9.1 bps | -9.4 bps | +34.8 bps | +0.0 bps |
| z > +2.0 → LONG | +1.7 bps | -13.1 bps | -11.3 bps | +10.4 bps |

**The only signal that partially survives OOS is SOL momentum at extreme thresholds (z < -2.0 → SHORT: +15.3 bps OOS).** But this fires only 2,620 times in 184 days, and the BTC version is catastrophically wrong.

---

## Why the Signal Degraded

### 1. LS Ratio Distribution Shifted Dramatically

| Statistic | BTC OOS (May-Oct) | BTC IS (Nov-Jan) | SOL OOS | SOL IS |
|-----------|-------------------|-------------------|---------|--------|
| Mean | 1.24 | 2.48 | 2.70 | 4.02 |
| Std | 0.49 | 0.64 | 0.65 | 0.81 |
| Median | 1.13 | 2.52 | 2.63 | 4.18 |

The LS ratio **doubled** between the OOS and IS periods. In May-Oct 2025, BTC top traders had a mean LS ratio of 1.24 (roughly balanced). By Nov 2025-Jan 2026, it was 2.48 (heavily long-biased). This structural shift means any z-score or threshold calibrated on one period is meaningless in the other.

### 2. OI Change Features Corrupted in OOS Period

The `oi_change_*` features produced NaN ICs in the OOS period, suggesting data quality issues (likely constant OI values in some Binance metrics rows causing division-by-zero in pct_change). This reduced the effective feature set for the walk-forward model.

### 3. Market Regime Was Different

May-Oct 2025 was a different market regime than Nov 2025-Jan 2026. The average 4h return was near zero in the OOS period vs slightly negative in IS. The relationship between positioning and future returns appears to be regime-dependent and non-stationary.

---

## What Partially Survived

### ✅ Sign of LS ratio IC on SOL
The `ls_top_zscore_24h` IC remains positive on SOL in both periods (+0.083 IS, +0.027 OOS). The momentum interpretation (crowd is right) holds. But the magnitude is 3× weaker OOS.

### ✅ SOL extreme momentum at z < -2.0
Shorting SOL when the LS ratio z-score drops below -2.0 earned +55.3 bps IS and +15.3 bps OOS. This is the only signal that survives with meaningful profitability, but it's rare (2,620 trades in 184 days = ~14/day).

### ✅ Taker ratio z < -1.0 → long on SOL (OOS only)
SOL taker ratio contrarian at z < -1.0 earned +1.9 bps OOS (marginal).

---

## What Definitively Failed

### ❌ BTC LS ratio signal
IC drops from +0.039 to +0.003. All backtests negative. Both contrarian and momentum fail. The BTC LS ratio has zero predictive power OOS.

### ❌ Walk-forward combined model on BTC
Sharpe -5.71 IS, -12.40 OOS. Adding OI/funding features makes the model worse, not better.

### ❌ All contrarian LS signals
Buying when LS ratio is low (crowd is wrong) fails catastrophically on both assets in both periods. The crowd is not reliably wrong.

### ❌ Taker ratio signals
All taker ratio signals are negative in both periods on both assets.

### ❌ Global LS ratio
Weaker than top trader LS ratio in every test. Adds no value.

---

## Comparison to v24 and v24b

| Finding | v24 (Dec 2025) | v24b (May-Aug, Bybit) | v24c (May-Oct, Binance) |
|---------|---------------|----------------------|------------------------|
| LS ratio IC@4h BTC | +0.20 | N/A (no data) | **+0.003** |
| LS ratio IC@4h SOL | +0.20 | N/A (no data) | **+0.027** |
| Walk-forward Sharpe BTC | +9.3 | N/A | **-12.40** |
| Walk-forward Sharpe SOL | +8.6 | +2.39 (no LS) | **+0.55** |
| Funding contrarian SOL | +26.7 bps | +7.3 bps | N/A (no premium index) |

**The v24 Sharpe 9+ result was a period-specific artifact.** The signal exists weakly on SOL but is not tradeable after fees.

---

## Conclusions

### The Honest Assessment

1. **The LS ratio signal is real but weak.** It has a positive IC on SOL across both periods (momentum interpretation), but the magnitude (IC ≈ 0.03 OOS) is far too small to generate profits after fees.

2. **The v24 Sharpe 9+ was overfitted to a specific 31-day window.** Even expanding to 92 days (Nov-Jan), the IS Sharpe drops to +2.31 on SOL and -5.71 on BTC. The 31-day Dec 2025 result was a lucky draw.

3. **The LS ratio distribution is non-stationary.** It doubled between May-Oct and Nov-Jan. Any strategy calibrated on one period will fail on the other. This is the fundamental problem — the signal's statistical properties change over time.

4. **BTC has no LS ratio signal.** Zero. The IC is +0.003 OOS. This is noise.

5. **SOL has a marginal signal at extreme thresholds only.** The z < -2.0 momentum signal (+15.3 bps OOS) is the only survivor, but it's too rare and too weak for a standalone strategy.

### Implications for Next Steps

- **Path A (LS ratio directional strategy) is dead.** The signal does not replicate OOS with sufficient strength.
- **Path B (Grid bot) remains the safest bet.** Already proven profitable over 13 months.
- **Path C (Vol arbitrage via options) becomes the highest-priority novel research.** Our vol prediction (R²=0.34) is robust and doesn't depend on non-stationary positioning data.
- The SOL extreme momentum signal (z < -2.0) could be used as a **supplementary filter** for grid bot direction, but not as a standalone strategy.

---

## Files

| File | Description |
|------|-------------|
| `oi_funding_oos_validation.py` | Full OOS validation script |
| `results/oi_funding_oos_validation.txt` | Complete experiment output |
| `FINDINGS_v24c_ls_ratio_oos.md` | This document |

---

**Research Status:** Complete ✅
**Verdict:** LS ratio signal does NOT replicate OOS. v24 Sharpe 9+ was period-specific.
**Next Priority:** Pivot to Path B (grid bot improvements) or Path C (vol arbitrage).
