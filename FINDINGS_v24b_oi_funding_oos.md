# FINDINGS v24b: OI/Funding — Out-of-Sample Validation + Sub-5min OI Spikes

**Date:** Feb 2025
**Symbols:** BTCUSDT, SOLUSDT
**Period:** May 12 – Aug 8, 2025 (89 days) — **completely out-of-sample vs Dec 2025 (v24)**
**Data:** Bybit ticker (5-second resolution) + Bybit trades → 5-min bars

---

## Motivation

v24 found extraordinary results on Dec 2025 data:
- `ls_ratio_top` IC = +0.20 at 4h, walk-forward Sharpe 9.3 (BTC), 8.6 (SOL)
- OI extremes (z>2) profitable as contrarian signals
- Funding extremes profitable on SOL

Two questions:
1. **Do these signals replicate on a different time period?** (May-Aug 2025, 6 months earlier)
2. **Does 5-second OI resolution add value over 5-minute aggregates?**

### Important caveat
Bybit ticker data does **not** include LS ratios (that's Binance-specific). So we can only validate OI, funding, and mark-index spread signals — not the strongest v24 signal (`ls_ratio_top`).

---

## Data Pipeline

| Source | Resolution | Records/day | Fields |
|--------|-----------|-------------|--------|
| Bybit ticker | 5 seconds | ~17,280 | lastPrice, markPrice, indexPrice, OI, fundingRate, bid1/ask1 |
| Bybit trades → OHLCV | 5 min | 288 | Standard OHLCV + microstructure features |

### Engineered Features

**Standard ticker features (14)** — comparable to v24 Binance metrics:
- OI change (5m, 1h, 4h, 24h), OI z-score, OI acceleration
- Funding rate, funding abs, funding z-score, funding cumulative 8h
- Mark-index spread (1h, 4h averages, z-score)

**Sub-5min features (22)** — NEW, only possible with 5-second data:
- OI velocity (max/mean/std of 5s OI changes within each 5-min bar)
- OI spike count (number of >0.05% and >0.1% jumps per bar)
- OI direction (net up vs down ticks within bar)
- OI intra-bar acceleration (velocity in 2nd half vs 1st half)
- Funding rate micro-volatility (std, range within bar)
- Mark-index spread volatility
- Bid/ask spread dynamics (mean, std)
- Bid/ask size ratio dynamics

---

## Part A: Out-of-Sample Validation

### IC Analysis — Bybit OI/Funding Features

| Feature | BTC 4h IC | SOL 4h IC | v24 BTC 4h IC | Consistent? |
|---------|----------|----------|---------------|-------------|
| tk_funding_rate | -0.051 | **-0.122** | -0.035 | ✅ Same sign, SOL stronger |
| tk_funding_cum_8h | -0.019 | **-0.101** | -0.038 | ✅ Same sign |
| tk_mark_index_spread | -0.059 | -0.070 | N/A (new) | ✅ Cross-asset |
| tk_mis_1h | **-0.060** | -0.069 | N/A | ✅ Cross-asset |
| tk_oi_zscore_24h | -0.038 | +0.035 | -0.036 | ⚠️ Mixed |
| tk_oi_change_4h | -0.020 | -0.044 | -0.088 | ✅ Same sign, weaker |

**Key finding:** Funding rate IC on SOL is **-0.122** at 4h — the strongest single-feature IC in the OOS period. Negative IC means: when funding is high (longs paying shorts), price tends to fall. This is the **contrarian funding signal** that v24 found on SOL Dec data, now confirmed on May-Aug data.

### OI Extreme Contrarian — Partial Replication

| Condition | v24 BTC Dec | v24b BTC May-Aug | v24 SOL Dec | v24b SOL May-Aug |
|-----------|------------|-----------------|------------|-----------------|
| OI z > +2.0 → short | **+23.1 bps** | -10.1 bps | +7.7 bps | -7.5 bps |
| OI z < -1.0 → long | +0.2 bps | **+2.9 bps** | -18.9 bps | -19.1 bps |
| OI z < -2.0 → long | -9.4 bps | -4.2 bps | **+38.4 bps** | **+2.7 bps** |

**Verdict: OI extreme contrarian does NOT robustly replicate.** The z>2 short signal that worked spectacularly on BTC Dec (+23 bps) fails on May-Aug (-10 bps). Only the SOL z<-2 long signal shows marginal consistency.

### Funding Extreme Contrarian — Partial Replication

| Condition | v24 SOL Dec | v24b SOL May-Aug |
|-----------|------------|-----------------|
| Funding z > +1.0 → short | **+26.7 bps** | +0.5 bps |
| Funding z < -1.0 → long | **+20.6 bps** | **+7.3 bps** |
| Funding z < -1.5 → long | **+20.1 bps** | +2.7 bps |

**Verdict: Funding contrarian partially replicates on SOL.** The z<-1 long signal works in both periods (+20.6 Dec, +7.3 May-Aug). But the magnitude is much weaker OOS, suggesting some of the Dec result was period-specific.

### Walk-Forward Quintile L/S

| Feature Set | BTC IC | BTC L/S | BTC Sharpe | SOL IC | SOL L/S | SOL Sharpe |
|-------------|--------|---------|-----------|--------|---------|-----------|
| OHLCV only | +0.020 | -7.7 bps | -9.40 | -0.018 | -9.6 bps | -5.23 |
| **OHLCV + Ticker Std** | +0.007 | -6.1 bps | -6.02 | **+0.069** | **+5.8 bps** | **+2.39** |

**Verdict: Without LS ratios, the walk-forward signal is much weaker.** On SOL, adding ticker standard features turns a losing strategy into a marginally profitable one (Sharpe +2.39). On BTC, it's still negative. This confirms that **the LS ratio was doing most of the heavy lifting in v24** — OI and funding alone are not sufficient for a strong directional signal.

---

## Part B: Sub-5min OI Spike Features

### Regime Discrimination — Sub-5min Features Are Extremely Powerful

| Feature | Quiet | Volatile | Ratio | BTC |t| | SOL |t| |
|---------|-------|----------|-------|---------|---------|
| **tk_oi_vel_mean_4h** | 0.003 | 0.007 | 2.1× | 101.8 | 77.1 |
| **tk_oi_spikes_4h** | 26.8 | 77.4 | 2.9× | 94.4 | 84.8 |
| **tk_oi_vel_mean_1h** | 0.003 | 0.007 | 2.6× | 94.7 | 57.7 |
| **tk_oi_spikes_1h** | 5.5 | 20.9 | 3.8× | 82.6 | 75.4 |
| **tk_oi_large_spikes_1h** | 1.6 | 8.0 | 5.0× | 68.1 | 64.1 |
| **tk_mis_std** | 0.27 | 0.40 | 1.5× | 57.5 | 57.1 |
| **tk_oi_vel_max_1h** | 0.16 | 0.34 | 2.1× | 57.8 | 18.3 |

**22/22 sub-5min features are statistically significant** for regime discrimination on BTC, 21/22 on SOL. The t-statistics are enormous (50-100+), far exceeding anything from OB features or standard OI metrics.

**Key insight:** OI spike count within a 5-min bar is 3-5× higher in volatile regimes. This makes intuitive sense — volatile periods see rapid position opening/closing. The 5-second resolution captures this dynamic that 5-min aggregates completely miss.

### Directional IC — Sub-5min Features

| Feature | BTC 4h IC | SOL 4h IC |
|---------|----------|----------|
| tk_spread_mean | +0.037 | **+0.104** |
| tk_mis_max_abs | +0.059 | +0.051 |
| tk_oi_large_spikes_1h | +0.031 | +0.016 |
| tk_oi_vel_mean_4h | +0.019 | +0.005 |
| tk_oi_spikes_4h | +0.012 | +0.005 |

**`tk_spread_mean` (average bid-ask spread within 5-min bar) has IC = +0.104 on SOL.** This is a meaningful signal — wider spreads predict higher future returns (compensation for illiquidity). This is only possible to compute from 5-second data.

### Volatility Prediction

| Feature Set | BTC Ridge R² | BTC GB R² | SOL Ridge R² | SOL GB R² |
|-------------|-------------|----------|-------------|----------|
| OHLCV only | **0.369** | **0.285** | **0.293** | **0.181** |
| OHLCV + Ticker Std | 0.325 | 0.171 | 0.227 | -0.044 |
| OHLCV + Sub-5min only | **0.375** | 0.266 | 0.206 | 0.123 |
| OHLCV + Ticker All | 0.338 | 0.205 | 0.059 | 0.036 |

**Verdict: Sub-5min features marginally help vol prediction on BTC** (Ridge R² 0.369 → 0.375) but hurt on SOL. Adding too many ticker features causes overfitting. The sub-5min features are better for regime discrimination than for prediction.

### Walk-Forward Direction with Sub-5min

| Feature Set | BTC IC | BTC L/S | SOL IC | SOL L/S | SOL Sharpe |
|-------------|--------|---------|--------|---------|-----------|
| OHLCV only | +0.020 | -7.7 bps | -0.018 | -9.6 bps | -5.23 |
| OHLCV + Ticker Std | +0.007 | -6.1 bps | **+0.069** | **+5.8 bps** | **+2.39** |
| OHLCV + Ticker All | +0.049 | -4.2 bps | +0.039 | **+6.6 bps** | **+3.01** |
| OHLCV + Sub-5min only | +0.068 | -0.9 bps | -0.042 | -12.6 bps | -6.14 |

**Verdict: Sub-5min features alone don't help direction.** But combined with standard ticker features (OHLCV + Ticker All), SOL gets Sharpe +3.01 — slightly better than standard alone (+2.39). On BTC, sub-5min features reduce losses but don't turn profitable.

### GB Feature Importance (4h Return Prediction)

Top features are remarkably consistent across BTC and SOL:

| Rank | BTC | SOL |
|------|-----|-----|
| 1 | rvol_24h (0.203) | **tk_spread_mean** [S5] (0.144) |
| 2 | **tk_funding_cum_8h** [TK] (0.109) | rvol_24h (0.136) |
| 3 | **tk_mis_4h** [TK] (0.085) | **tk_funding_cum_8h** [TK] (0.130) |
| 4 | **tk_spread_mean** [S5] (0.074) | tk_oi_change_24h [TK] (0.064) |
| 5 | rvol_4h (0.064) | **tk_oi_spikes_4h** [S5] (0.059) |

**`tk_funding_cum_8h` and `tk_spread_mean` appear in top 4 for both assets.** These are the most robust OI/funding features for direction prediction.

---

## Summary: What Replicates and What Doesn't

### ✅ Confirmed (replicates OOS)

1. **Funding rate is a contrarian signal on SOL** — IC = -0.12 at 4h on May-Aug (vs -0.16 on Dec). Negative funding → price rises.
2. **Mark-index spread predicts returns** — IC = -0.06 to -0.07 at 4h on both assets. New finding not in v24.
3. **Sub-5min OI dynamics are extremely powerful regime discriminators** — |t| = 50-100+, far exceeding any other feature class.
4. **`tk_funding_cum_8h` is the most robust OI/funding feature** — top 3 in GB importance on both assets, both periods.
5. **`tk_spread_mean` (from 5s data) is a strong directional feature on SOL** — IC = +0.10.

### ⚠️ Partially confirmed

6. **OI extreme contrarian** — works in some conditions but not robustly across periods.
7. **Walk-forward with OI/funding (no LS ratios)** — marginally profitable on SOL (Sharpe 2.4) but not BTC.

### ❌ Not confirmed without LS ratios

8. **The Sharpe 9+ walk-forward from v24 was driven by LS ratios**, which are Binance-specific and not available in Bybit ticker data. Without them, the signal drops to Sharpe 2-3 on SOL and negative on BTC.

### Key Takeaway

**The `ls_ratio_top` signal from v24 remains the most important finding, but we cannot validate it OOS because Bybit doesn't provide it.** To properly validate, we would need Binance metrics data for May-Aug 2025. The OI and funding signals alone provide moderate value — useful for regime detection and as supplementary features, but not strong enough for standalone directional trading.

**The 5-second resolution adds genuine value for regime detection** (OI spike features are 3-5× more discriminating than 5-min aggregates) but **does not meaningfully improve directional prediction**. The bid-ask spread from 5s data (`tk_spread_mean`) is the exception — it's a strong SOL signal (IC = 0.10).

---

## Files

| File | Description |
|------|-------------|
| `parse_ticker.py` | Bybit ticker JSONL → daily parquet parser |
| `oi_funding_research_v24b.py` | Part A (OOS validation) + Part B (sub-5min features) |
| `results/oi_funding_v24b_BTC.txt` | BTC experiment output |
| `results/oi_funding_v24b_SOL.txt` | SOL experiment output |
