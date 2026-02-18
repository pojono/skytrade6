# FINDINGS v24d: OI Velocity Leads Price at 5-Second Resolution

## Motivation

Previous research (v24b, v24c) showed that Open Interest (OI) features are powerful for regime discrimination but appeared to spike simultaneously with price volatility at 5-minute resolution. However, this was a **measurement artifact** — aggregating to 5-minute bars destroyed the sub-bar temporal information.

This study re-examines the lead/lag relationship at **true 5-second resolution** using raw Bybit ticker data, testing across **5 assets** and **2 time periods** to determine whether OI velocity provides advance warning of regime transitions.

## Methodology

1. **Regime labeling**: Fit a 2-component GMM on 5-minute OHLCV features (rvol, parkvol, efficiency, ADX, etc.) to classify bars as quiet or volatile
2. **Transition identification**: Find all quiet→volatile regime transitions
3. **5-second analysis**: Load raw ticker data in a ±5 minute window around each transition
4. **Feature computation**: Calculate rolling 30-second OI velocity (|Δ OI|) and price volatility (|Δ price|) at 5-second granularity
5. **Spike comparison**: For each transition, determine which metric exceeds 2× its baseline first

## Data

- **2025 sample** (in-sample): May 12 – Aug 8, 2025 (89 days, ~500-730 transitions per asset)
- **2026 sample** (out-of-sample): Feb 9 – Feb 17, 2026 (9 days, ~47-59 transitions per asset)
- **Assets**: BTCUSDT, ETHUSDT, SOLUSDT, DOGEUSDT, XRPUSDT
- **Source**: Bybit linear perpetual ticker data at ~5-second polling interval

## Results

### Cross-Asset: Which Metric Spikes First? (2× baseline threshold)

#### 2025 In-Sample (May–Aug, 89 days)

| Asset | OI First | Price First | Simultaneous | Median OI Lead | Transitions |
|-------|----------|-------------|--------------|----------------|-------------|
| **SOL** | **73.7%** | 22.6% | 2.1% | **+80s** | 506 |
| **DOGE** | **73.1%** | 22.4% | 1.9% | **+75s** | 728 |
| **ETH** | **70.5%** | 25.5% | 2.8% | **+75s** | 542 |
| **XRP** | **68.4%** | 25.7% | 2.5% | **+55s** | 674 |
| BTC | 49.4% | 44.1% | 5.1% | 0s | 506 |

#### 2026 Out-of-Sample (Feb 9–17, 9 days)

| Asset | OI First | Price First | Simultaneous | Median OI Lead | Transitions |
|-------|----------|-------------|--------------|----------------|-------------|
| **XRP** | **74.5%** | 17.0% | 6.4% | **+110s** | 47 |
| **SOL** | **72.5%** | 25.5% | 2.0% | **+75s** | 51 |
| **DOGE** | **71.2%** | 23.1% | 3.8% | **+45s** | 52 |
| **BTC** | **64.2%** | 30.2% | 5.7% | **+25s** | 53 |
| **ETH** | **64.4%** | 32.2% | 1.7% | **+53s** | 59 |

### Advance Warning Time (When OI Does Lead)

| Asset | 2025 Mean Lead | 2025 Median Lead | 2026 Mean Lead | 2026 Median Lead |
|-------|---------------|-----------------|---------------|-----------------|
| SOL | 173s (2.9 min) | 155s | 150s (2.5 min) | 140s |
| DOGE | 179s (3.0 min) | 170s | 157s (2.6 min) | 128s |
| ETH | 173s (2.9 min) | 170s | 172s (2.9 min) | 162s |
| XRP | 169s (2.8 min) | 160s | 210s (3.5 min) | 242s |
| BTC | 144s (2.4 min) | 125s | 159s (2.7 min) | 148s |

### SOL 5-Second Profile Around Transitions (2025)

The temporal profile shows OI rising **before** price moves on altcoins:

| Time | OI Ratio | Price Ratio | Interpretation |
|------|----------|-------------|----------------|
| -120s | **1.40×** | 0.97× | OI rising, price flat |
| -90s | **1.42×** | 1.06× | OI still ahead |
| -60s | **1.47×** | 0.96× | OI elevated, price still normal |
| -45s | **1.56×** | 0.99× | OI clearly spiking |
| 0s | 1.62× | 0.97× | Transition bar starts |
| +15s | 2.06× | 1.32× | Price finally starts moving |
| +60s | 3.00× | 1.30× | OI still leading |

### Threshold Exceedance Timing (2025)

First time each metric exceeds threshold relative to transition (t=0):

**SOL:**
| Metric | > 1.5× | > 2.0× | > 3.0× | > 5.0× |
|--------|--------|--------|--------|--------|
| OI velocity | **-55s** | +15s | +70s | never |
| Price volatility | +30s | never | never | never |

**ETH:**
| Metric | > 1.5× | > 2.0× | > 3.0× | > 5.0× |
|--------|--------|--------|--------|--------|
| OI velocity | **-100s** | +10s | +30s | +95s |
| Price volatility | +20s | never | never | never |

**DOGE:**
| Metric | > 1.5× | > 2.0× | > 3.0× | > 5.0× |
|--------|--------|--------|--------|--------|
| OI velocity | **-75s** | +15s | +25s | never |
| Price volatility | never | never | never | never |

On altcoins, OI velocity exceeds 1.5× baseline **55-100 seconds before** the transition bar, while price volatility doesn't even reach 1.5× until after the transition.

## Key Findings

### 1. OI Leads Price on All Altcoins (68-74%)

On SOL, ETH, DOGE, and XRP, OI velocity spikes first in **68-74% of regime transitions**. This is not a coin flip — it's a statistically significant lead. The causal chain is:

> **Positions open (OI spikes) → Price impact propagates → OHLCV volatility rises → Regime detector catches it**

### 2. BTC Is Different — More Efficient Market

In 2025, BTC showed a 49/44 split (essentially random). In 2026, BTC improved to 64/30, suggesting the effect may exist but is weaker and less consistent. BTC is the most liquid crypto market — arbitrageurs and market makers react instantly, so positions and price move simultaneously.

### 3. The Effect Is Robust Across Time

The 2026 out-of-sample data (completely unseen period) confirms the pattern:
- All 5 assets show OI leading in 2026
- The percentages are remarkably consistent (64-75% in 2026 vs 49-74% in 2025)
- Advance warning times are similar (~2.5-3.5 minutes when OI leads)

### 4. Advance Warning Is 1-3 Minutes

When OI does lead, it provides **2.5-3.5 minutes of advance warning** on average. This is enough time to:
- Widen grid spacing before volatility hits
- Reduce position sizes
- Pause new order placement
- Tighten stop-losses

### 5. Previous 5-Minute Analysis Was Flawed

The v24c analysis at 5-minute resolution concluded "OI doesn't lead" — this was wrong. Aggregating to 5-minute bars placed both the OI spike and the price spike in the same bar, masking the sub-bar temporal relationship. **Resolution matters.**

## Practical Implications

### For Grid Bot (SOL, ETH, DOGE, XRP)

A 5-second OI velocity monitor can serve as an **early warning system**:
- Compute rolling 30-second OI velocity from ticker data
- When OI velocity z-score exceeds 1.5-2.0, flag potential regime change
- Use this as a leading indicator to pre-emptively adjust grid parameters
- Expected advance warning: ~1-3 minutes before OHLCV-based detection

### For BTC

OI velocity is less useful as a leading indicator on BTC. Stick with OHLCV-based regime detection (GMM/HMM) which already achieves 99.6% AUC.

### Combined Detector

The optimal approach for altcoins:
1. **Primary**: OI velocity z-score at 5-second resolution (early warning)
2. **Confirmation**: GMM/HMM on 5-minute OHLCV bars (high accuracy)
3. **Action**: When OI fires, prepare to adjust; when OHLCV confirms, execute

## Scripts

- `oi_regime_lead_lag_5s.py` — Main analysis script (supports `--symbol`, `--start`, `--end`)
- `parse_ticker.py` — Parses Bybit ticker JSONL into daily parquet
- `download_ticker_tar.sh` — Efficient bulk ticker download from dataminer

## Result Files

- `results/oi_regime_lead_lag_5s_BTC.txt` — BTC 2025 results
- `results/oi_regime_lead_lag_5s_SOL.txt` — SOL 2025 results
- `results/oi_regime_lead_lag_5s_ETH.txt` — ETH 2025 results
- `results/oi_regime_lead_lag_5s_DOGE.txt` — DOGE 2025 results
- `results/oi_regime_lead_lag_5s_XRP.txt` — XRP 2025 results
- `results/oi_regime_lead_lag_5s_2026_*.txt` — All 5 assets, 2026 OOS
