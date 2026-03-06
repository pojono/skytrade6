# Master Backtest Plan: Hidden Patterns Strategies

We will systematically build and backtest the three S-Tier strategies derived from our microstructure research. 

## Strategy 1: "The Cascade Sniper" (High-Frequency Reversion)
**Objective:** Trade localized flash crashes and spoofed breakouts.
* **Signals:** 
  * *Long:* Open Interest drop > 2% (Flush) + Mark/Index Basis Z-score < -3.5 (Flash Crash).
  * *Short:* Price at 60m High + CVD Bearish Divergence (Spoofing).
* **Horizon:** 15 to 60 minutes.
* **Assets:** High-volatility altcoins (SUI, WLD, DYDX, ENA).
* **Requirements:** 1-minute Klines, Mark Price, Index Price, Open Interest, Tick Data (for CVD).

## Strategy 2: "The Whale Shadow" (Medium-Frequency Flow)
**Objective:** Fade extreme retail sentiment.
* **Signals:** 
  * *Short:* Retail Account L/S Z-score > 1.5 AND Whale Margin L/S Z-score < -1.5 (Bearish Divergence).
  * *Long:* Retail Account L/S Z-score < -1.5 AND Whale Margin L/S Z-score > 1.5 (Bullish Divergence).
* **Filter:** Premium Index Z-score confirms overextension (> 2.5 or < -2.5).
* **Horizon:** 4 hours.
* **Assets:** Mid/Large Caps (SOL, ETH, XRP, DYDX).
* **Requirements:** 1-minute Klines, Binance Top Trader L/S Ratios, Premium Index.

## Strategy 3: "The Powder Keg Swing" (Low-Frequency Swing)
**Objective:** Exploit maximum system leverage cascades.
* **Signals:**
  * *Short (Powder Keg):* Open Interest (USD) 7-day Z-score > 2.0 AND Funding Rate 7-day Z-score > 2.0.
  * *Long (Despair Pit):* Open Interest 7-day Z-score > 2.0 AND Funding Rate 7-day Z-score < -2.0.
* **Horizon:** 24 hours.
* **Assets:** All symbols.
* **Requirements:** Hourly/1m Klines, Open Interest USD metrics, Funding Rates.

---

## Technical Architecture
1. `backtest_core.py`: A generic, vectorized backtesting engine that takes a DataFrame with a `signal` column (1 for long, -1 for short, 0 for neutral), applies taker fees (e.g., 5 bps), simulates forward holding periods, and computes PnL, Win Rate, Max Drawdown, and Sharpe Ratio.
2. `strat_1_cascade_sniper.py`: Signal generator and runner for Strategy 1.
3. `strat_2_whale_shadow.py`: Signal generator and runner for Strategy 2.
4. `strat_3_powder_keg.py`: Signal generator and runner for Strategy 3.
