# Tick-Level Microstructure Findings (Feb 24, 2026)

## Overview
We extracted raw trade-level (tick) data for `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, and `DOGEUSDT` for a single day (February 24, 2026) to analyze the microstructure properties across the four markets (Binance Spot, Binance Futures, Bybit Spot, Bybit Futures).

## Summary Metrics

| Symbol | Market | Trades/Sec | Avg Trade Size | Taker Buy % | Large Trades (>$100k) |
|--------|--------|------------|----------------|-------------|-----------------------|
| **BTC** | Binance Fut | 66.6 | **$2,465** | 49.0% | **6,723** |
| | Bybit Fut | 34.6 | $2,280 | 49.3% | 5,138 |
| | Binance Spot| **87.4** | $194 | 48.3% | 689 |
| | Bybit Spot | 17.4 | $694 | 48.7% | 180 |
| **ETH** | Binance Fut | 100.1| $1,156 | 48.4% | 3,315 |
| | Bybit Fut | 25.1 | **$1,262** | 50.3% | 1,061 |
| | Binance Spot| 76.4 | $121 | 47.0% | 107 |
| | Bybit Spot | 7.9 | $386 | 48.8% | 7 |
| **SOL** | Binance Fut | 24.7 | $924 | 49.1% | 242 |
| | Bybit Fut | 8.2 | **$1,445** | 50.7% | 113 |
| | Binance Spot| 14.2 | $225 | 49.9% | 50 |
| | Bybit Spot | 2.3 | $294 | 48.4% | 6 |
| **DOGE** | Binance Fut | 13.7 | **$398** | 49.8% | 24 |
| | Bybit Fut | 4.1 | $385 | 48.6% | 2 |
| | Binance Spot| 9.7 | $75 | **54.1%** | 5 |
| | Bybit Spot | 0.4 | $183 | 45.6% | 0 |

---

## Key Microstructure Insights

### 1. The Spot Market is Retail Driven (High Frequency, Tiny Size)
* **Binance Spot** has the highest *number* of trades per second (87 trades/sec for BTC), but the *average trade size* is extraordinarily small ($194 for BTC, $121 for ETH). 
* This points to Binance Spot being entirely dominated by retail algorithms, grid bots, and tiny fractional purchases.

### 2. Institutional Whales Live on Futures
* **Futures trade sizes are ~10x larger** than spot trade sizes. For BTC, the average futures trade is ~$2,400 while the spot average is ~$194. 
* **Large Trade Density:** On Binance Futures, there were **6,723 trades over $100,000** for BTC in a single day. On Binance Spot, there were only **689** (a 10x differential). Bybit Futures is also a massive whale hub, showing 5,138 trades >$100k.

### 3. Bybit Futures vs. Binance Futures (Quality vs. Quantity)
* While Binance Futures processes more trades per second, **Bybit Futures often has a higher average trade size** (e.g., $1,262 vs $1,156 for ETH, and $1,445 vs $924 for SOL). 
* This indicates that Bybit's perpetual liquidity is incredibly solid and is actively utilized by large institutional traders, not just retail punters. 

### 4. Bybit Spot is Illiquid 
* The data confirms the low volume findings from our previous analysis. DOGE only traded at **0.4 trades per second** on Bybit Spot compared to 9.7 trades/sec on Binance Spot. 
* Furthermore, Bybit Spot saw almost zero "whale" trades (only 7 large ETH trades and 6 large SOL trades in an entire 24-hour period). 

### 5. Taker Imbalance (Predictive Signal?)
* Notice the DOGE taker breakdown: **Binance Spot Taker Buy % was extremely high at 54.1%**, while Bybit Spot was at 45.6%. 
* These massive retail taker imbalances on Binance Spot often act as a short-term momentum or exhaustion signal, especially for retail-heavy meme coins.
