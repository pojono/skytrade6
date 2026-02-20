# Broad Validation: Vol Dip-Buying on 20 Coins

## Results Summary — Sorted by Monthly Sharpe

| # | Symbol | Months | Pos% | Beat Rand% | Total | Ann. | mSharpe | MaxDD | Trades | WR | Long% | B&H |
|---|--------|--------|------|-----------|-------|------|---------|-------|--------|-----|-------|-----|
| 1 | **SOL** | 23/32 | 72% | 81% | +63.7% | +23.9% | **+1.84** | 5.7% | 115 | 64% | 99% | +746% |
| 2 | **POL** | 7/12 | 58% | 75% | +27.9% | +27.9% | **+1.83** | 1.1% | 46 | 57% | 95% | -72% |
| 3 | **BNB** | 22/32 | 69% | 56% | +40.5% | +15.2% | **+1.50** | 4.2% | 127 | 60% | 94% | +150% |
| 4 | **ADA** | 22/32 | 69% | 66% | +90.0% | +33.7% | **+1.48** | 16.1% | 122 | 59% | 98% | +14% |
| 5 | **XRP** | 21/32 | 66% | 72% | +104.1% | +39.0% | **+1.42** | 19.1% | 133 | 60% | 95% | +320% |
| 6 | **UNI** | 19/32 | 59% | 62% | +59.6% | +22.4% | **+1.32** | 14.6% | 126 | 56% | 96% | -33% |
| 7 | **SUI** | 15/28 | 54% | 61% | +52.2% | +22.4% | **+1.13** | 17.0% | 91 | 55% | 99% | -26% |
| 8 | **ARB** | 19/30 | 63% | 67% | +37.5% | +15.0% | +0.78 | 16.8% | 123 | 59% | 99% | -93% |
| 9 | **LINK** | 23/31 | 74% | 74% | +36.7% | +14.2% | +0.71 | 22.4% | 113 | 60% | 95% | +58% |
| 10 | **LTC** | 16/31 | 52% | 55% | +27.8% | +10.8% | +0.70 | 21.7% | 128 | 54% | 96% | -22% |
| 11 | ETH | 18/32 | 56% | 62% | +19.6% | +7.3% | +0.59 | 10.8% | 119 | 51% | 95% | +65% |
| 12 | NEAR | 19/31 | 61% | 65% | +30.9% | +12.0% | +0.58 | 24.7% | 112 | 57% | 97% | -18% |
| 13 | OP | 16/31 | 52% | 55% | +34.1% | +13.2% | +0.52 | 29.4% | 125 | 53% | 95% | -86% |
| 14 | ATOM | 19/31 | 61% | 68% | +22.1% | +8.6% | +0.46 | 19.6% | 125 | 57% | 97% | -75% |
| 15 | BTC | 17/32 | 53% | 59% | +10.2% | +3.8% | +0.45 | 12.4% | 143 | 52% | 86% | +312% |
| 16 | DOGE | 14/32 | 44% | 56% | +11.4% | +4.3% | +0.26 | 19.8% | 140 | 51% | 96% | +42% |
| 17 | FIL | 19/31 | 61% | 58% | +5.3% | +2.1% | +0.13 | 12.2% | 132 | 52% | 97% | -69% |
| 18 | AVAX | 20/32 | 62% | 62% | +1.3% | +0.5% | +0.03 | 32.7% | 107 | 57% | 91% | -14% |
| 19 | DOT | 18/31 | 58% | 58% | -21.8% | -8.5% | -0.52 | 39.6% | 114 | 52% | 99% | -69% |
| 20 | APT | 14/32 | 44% | 50% | -38.8% | -14.5% | -0.53 | 48.9% | 129 | 44% | 96% | -75% |

## Aggregate Statistics

- **18/20 symbols positive** (90%)
- **13/20 with mSharpe > 0.5** (65%)
- **19/20 beat random >50% of months** (95%)
- Only 2 losers: DOT (-21.8%) and APT (-38.8%)

## Key Observations

### 1. The Signal Is Real and Broad
18 out of 20 symbols are net positive over 2.5+ years of walk-forward testing with no lookahead bias. This is not curve-fitting to a few assets — the signal generalizes across the crypto market.

### 2. Not Correlated with Buy-and-Hold
Several symbols with **negative** B&H still have **strong positive** strategy returns:
- **ADA**: B&H +14%, Strategy +90% — strategy massively outperforms
- **UNI**: B&H -33%, Strategy +60% — profitable in a bear asset
- **ARB**: B&H -93%, Strategy +38% — profitable while asset lost 93%
- **OP**: B&H -86%, Strategy +34%
- **LTC**: B&H -22%, Strategy +28%
- **ATOM**: B&H -75%, Strategy +22%

This is strong evidence the signal is **not** just leveraged buy-and-hold.

### 3. Tier Classification

**Tier A (mSharpe > 1.0, trade):**
SOL, POL, BNB, ADA, XRP, UNI, SUI — 7 symbols

**Tier B (mSharpe 0.5–1.0, monitor):**
ARB, LINK, LTC, ETH, NEAR, OP — 6 symbols

**Tier C (mSharpe < 0.5, skip):**
ATOM, BTC, DOGE, FIL, AVAX, DOT, APT — 7 symbols

### 4. Long Bias Confirmed but Not Fatal
Average long trade percentage: 91–99% across all symbols. The strategy almost never goes short. However, the fact that it works on assets with negative B&H (UNI -33%, ARB -93%, OP -86%) proves the long trades are **selective dip-buying**, not passive long exposure.

### 5. The Two Losers
- **DOT**: -21.8%, mSharpe -0.52. B&H was -69%, so strategy lost less than holding. Dips in DOT don't recover.
- **APT**: -38.8%, mSharpe -0.53. B&H was -75%. Same pattern — sustained decline, dips are traps.

Both are assets in structural decline. The strategy correctly identifies vol spikes + dips, but the dips don't bounce in dying assets.

## Protocol

- **Walk-forward**: 6-month warmup, then monthly out-of-sample
- **No lookahead**: All rolling windows backward-looking (pandas `.rolling()`)
- **Fixed parameters**: threshold=2.0, hold=4h, fees=4bps RT — never optimized per symbol or period
- **Random baseline**: 5-seed random direction per month
- **Data**: Bybit linear perpetual 1h klines, Jan 2023 – Feb 2026
