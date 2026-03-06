# Price Correlation Analysis Findings (Futures vs Spot, Binance vs Bybit)

## Overview
We analyzed 1-minute close prices across 140 common symbols on Binance and Bybit.
For each symbol, we looked at 4 distinct data sources:
1. Binance Futures
2. Binance Spot
3. Bybit Futures
4. Bybit Spot

The data evaluated was from January 2025 onwards. The correlation of 1-minute returns was computed across all 4 data streams.

## Results Summary
The 4 data streams are **highly correlated, but NOT perfectly correlated**. The correlation usually ranges from `0.85` to `0.95`. Large-cap tokens like BTC and ETH show near-perfect correlation (>0.99), while smaller caps can vary significantly.

### Pairwise Median Correlations

| Pair | Median Correlation | Note |
|------|--------------------|------|
| **Binance Futures ↔ Bybit Futures** | 0.9750 | Highest correlation. Futures markets track each other very closely across exchanges. |
| **Binance Futures ↔ Binance Spot** | 0.9358 | Strong internal correlation between spot and futures on Binance. |
| **Bybit Futures ↔ Bybit Spot** | 0.9178 | Slightly lower internal correlation on Bybit. |
| **Binance Spot ↔ Bybit Spot** | 0.8828 | Lowest direct correlation. Spot markets can deviate more across exchanges due to liquidity differences. |

### Top & Bottom Correlated Symbols

Large cap tokens are consistently the highest correlated. Small cap and newer tokens show the weakest correlation.

* **Highest Correlated (Consistent across pairs):**
  * `ETHUSDT` (~0.996)
  * `BTCUSDT` (~0.995)
  * `XRPUSDT` (~0.993)

* **Lowest Correlated (Inconsistent pricing):**
  * `DASHUSDT` (~ -0.01 to 0.00) - Shows virtually zero correlation between Bybit and Binance on spot, indicating a possible data anomaly, severe illiquidity, or differing definitions of the ticker.
  * `ZECUSDT` (~ -0.005) - Similar to DASH.
  * `DYDXUSDT` (~ 0.60) - Low correlation between futures markets.
  * `SIGNUSDT` (~ 0.51) - Poor cross-exchange spot correlation.

## Conclusion
* **Are all four perfectly correlated?** No. 1-minute return correlation averages around 0.92-0.97 for most pairs, and lower for spot-to-spot comparisons. 
* **Arbitrage/Divergence Potential:** The gap from perfect correlation (especially the 0.88 median correlation between Binance Spot and Bybit Spot, or the 0.93 median between Futures and Spot on the same exchange) suggests significant microstructure deviations at the 1-minute level, creating potential for statistical arbitrage and mean-reversion strategies.
* **Warning:** Assets like `DASH` and `ZEC` exhibit zero correlation between Bybit Spot and other markets, which usually indicates delistings, frozen markets, or drastically different liquidity profiles.

## Statistical Arbitrage Strategy Exploration

We built a mean-reversion statistical arbitrage model using the rolling 60-minute Z-score of the log price spread between **Binance Futures** and **Bybit Futures**. The strategy enters a long/short spread position when the Z-score exceeds `±2.5` and exits when it reverts to `0.0`.

### Gross Profitability (No Fees)
The raw divergence (gross edge) shows an extremely high win rate (>99%) and positive average returns across the board:

| Symbol | Gross Win Rate | Avg Hold | Gross Edge (Per Trade) |
|--------|----------------|----------|-------------------------|
| BTCUSDT | 99.5% | 9.5 mins | 0.88 bps |
| ETHUSDT | 99.8% | 8.1 mins | 1.12 bps |
| SOLUSDT | 99.9% | 7.4 mins | 1.68 bps |
| XRPUSDT | 99.9% | 7.2 mins | 1.44 bps |
| DOGEUSDT| 100.0%| 5.4 mins | 1.76 bps |
| DYDXUSDT| 100.0%| 2.9 mins | 20.46 bps |
| WLDUSDT | 100.0%| 4.6 mins | 3.26 bps |

### Feasibility After Fees
A cross-exchange pairs trade requires taking **4 market actions** per round-trip (Open Leg 1, Open Leg 2, Close Leg 1, Close Leg 2). 

* **Average VIP 0 Taker Fees:** ~5.5 bps per transaction.
* **Total Fee Burden:** `4 * 5.5 bps = 22 bps` per trade.

Comparing the Gross Edge to the Fee Burden:
* **Large Caps (BTC, ETH, SOL):** Generate ~1-2 bps of gross edge per trade, which is completely destroyed by the 22 bps fee burden. 
* **Small Caps (DYDX):** Generates ~20.4 bps of gross edge per trade. This is *almost* profitable, but still falls short of the 22 bps needed to break even.

### Conclusion on Feasibility
1. **As a pure taker strategy:** It is **NOT** profitable. The divergences close too fast (average hold times of 3-9 mins) and the spread magnitude is too small to overcome 22 bps of taker fees.
2. **Pathways to Profitability:**
   * **Maker/Taker Asymmetry:** If we can post limit orders (Maker) on at least 2 or 3 of the legs, the fee burden drops to nearly 0 bps (Bybit Maker is ~0.02% or 2 bps, Binance Maker is ~0.02% or 2 bps). If we can execute all 4 legs as maker, the fee is ~8 bps, making DYDX highly profitable and WLD close to profitable.
   * **High VIP Tiers:** At high VIP tiers, taker fees can drop to ~2-3 bps (total burden 8-12 bps), making highly volatile tokens like DYDX or WLD feasible.
   * **Spot vs Futures Basis (Funding Arbitrage):** Utilizing the same principles but holding for the funding rate payment instead of pure price reversion (which we know from memory data has been successfully backtested for longer hold times).
