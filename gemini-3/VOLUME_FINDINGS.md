# Volume Analysis Findings (Jan 2025 - Mar 2026)

## Overview
We aggregated the total quote volume (in USDT) across the 140 common symbols for the 4 primary markets:
1. Binance Futures
2. Binance Spot
3. Bybit Futures
4. Bybit Spot

## Aggregate Market Share
The total volume traded across these 140 symbols in this period is staggering (~$36 Trillion). The distribution reveals absolute dominance by the Futures markets, and absolute dominance by Binance overall.

| Market | Total Volume (USDT) | Market Share (Avg per Symbol) |
|--------|---------------------|-------------------------------|
| **Binance Futures** | $22.14 Trillion | 66.4% |
| **Bybit Futures** | $9.59 Trillion | 22.6% |
| **Binance Spot** | $3.19 Trillion | 7.6% |
| **Bybit Spot** | $1.17 Trillion | 3.4% |

### Key Insights
1. **Futures is King:** Over **89% of all trading volume** occurs in the perpetual futures markets. Spot markets account for only ~11% of the total volume.
2. **Binance Dominance:** Binance controls roughly **74%** of the total volume across these symbols (66.4% Futures + 7.6% Spot).
3. **Bybit's Strong Point is Futures:** Bybit captures a very respectable ~23% of the futures market share. However, its spot market is a relative ghost town, capturing only 3.4% of the volume.

## Top 10 Symbols by Volume
The top 10 most traded symbols follow similar distributions, but some nuances exist:

| Rank | Symbol | Total Volume | Binance Fut % | Bybit Fut % | Binance Spot % | Bybit Spot % |
|------|--------|--------------|---------------|-------------|----------------|--------------|
| 1 | BTCUSDT | $11.95 T | 58.6% | 29.9% | 7.3% | 4.1% |
| 2 | ETHUSDT | $9.72 T | 66.1% | 24.1% | 7.3% | 2.4% |
| 3 | SOLUSDT | $2.92 T | 58.5% | 28.7% | 9.6% | 3.0% |
| 4 | XRPUSDT | $1.60 T | 53.3% | 29.7% | 12.4% | 4.4% |
| 5 | DOGEUSDT | $938 B | 62.1% | 23.6% | 12.0% | 2.1% |
| 6 | SUIUSDT | $530 B | 56.8% | 26.6% | 13.8% | 2.6% |
| 7 | BNBUSDT | $460 B | 65.7% | 10.6% | 21.8% | 1.8% |
| 8 | TRUMPUSDT| $425 B | 59.3% | 21.4% | 15.7% | 3.4% |
| 9 | ADAUSDT | $403 B | 57.6% | 26.4% | 12.9% | 2.9% |
| 10 | FARTCOINUSDT| $304 B | 54.7% | 45.2% | 0.0% | 0.0% |

### Nuanced Insights from Top 10
* **Bybit is highly competitive in BTC and XRP:** Bybit captures almost 30% of the futures market for BTC and XRP, significantly higher than its average 22.6% share.
* **BNB is Heavily Binance Skewed:** Unsurprisingly, Binance Coin (BNB) sees virtually zero traction on Bybit futures (only 10.6%), and trades heavily on Binance Spot (21.8%).
* **Meme Coins and Spot Volumes:** Assets like XRP, DOGE, SUI, and TRUMP see a higher percentage of their volume on Binance Spot (12-15%) compared to the majors like BTC and ETH (7%).
* **Meme Coin Futures Dominance:** `FARTCOINUSDT` has literally 0% volume on spot in our dataset, with a massive 45.2% of its futures volume happening on Bybit, showing Bybit's strength in listing high-volatility meme coin perpetuals early.

## Strategic Implications
1. **Price Discovery happens on Binance Futures:** With 66.4% of all volume, Binance Futures is the undisputed leader in price discovery.
2. **Bybit Spot is a Laggard:** With only 3.4% market share, Bybit Spot is highly susceptible to latency arbitrage and microstructure inefficiencies. If price moves sharply on Binance Futures, Bybit Spot will likely be the last to react, making it a prime target for lead-lag arbitrage strategies (assuming fees can be overcome).
3. **Statistical Arbitrage Focus:** When trading the Binance vs. Bybit spread, you should treat Binance as the "true" price and expect Bybit to revert to Binance, not the other way around. 
