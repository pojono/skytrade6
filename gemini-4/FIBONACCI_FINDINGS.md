# Quantitative Evaluation of Fibonacci Retracement Theory in Crypto Markets

## Methodology

To empirically approve or reject the validity of Fibonacci/Golden Ratio retracement levels, I designed a quantitative study over 20 major crypto assets across multiple market structures.

1. **Assets Analysed:** `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `DOGEUSDT`, `XRPUSDT`, `ADAUSDT`, `AVAXUSDT`, `LINKUSDT`, `DOTUSDT`, `MATICUSDT`, `LTCUSDT`, `UNIUSDT`, `ATOMUSDT`, `BCHUSDT`, `NEARUSDT`, `OPUSDT`, `ARBUSDT`, `APTUSDT`, `SUIUSDT`, and `INJUSDT`.
2. **Timeframes Analysed:** 5-minute, 15-minute, 30-minute, 1-hour, 4-hour, and 1-day candles to capture both micro-scalping and macro-trend fractals.
3. **Swing Detection:** Utilized `scipy.signal.find_peaks` to identify local highs and lows across 3 different window sizes (5, 10, 20 periods) representing different scales of price swings. Overlapping or consecutive swings of the same type were filtered out.
4. **Retracement Calculation:** For every alternating 3-point swing ($A \rightarrow B \rightarrow C$), the retracement ratio was calculated as $\frac{|B - C|}{|B - A|}$.
5. **Data Volume:** This massive expansion yielded a highly robust dataset of **1,973,737 valid market retracements** (bounded between 5% and 150% retracement of the original move).

## Findings

A Kernel Density Estimation (KDE) was fitted to the empirical distribution of these ~2 million retracements to find naturally occurring "attractor" levels where the market tends to reverse.

The analysis found two distinct, stable peaks in the retracement distribution:
1. **0.5856** (clustering near the **0.618** Golden Ratio)
2. **0.9992** (clustering near the **1.000** Full Retracement)

### Distance to Fibonacci Levels
We evaluated the distance between the empirically discovered peaks and the standard Fibonacci sequence `[0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272]`:
- The peak at `0.5856` has a distance of **0.0324** to the nearest Fibonacci level (0.618).
- The peak at `0.9992` has a distance of **0.0008** to the nearest level (1.0).

The average distance from the true market peaks to their nearest Fibonacci counterpart is **0.0166**.

### Statistical Significance (Monte Carlo Simulation)
Is an average error of 0.0166 statistically significant, or could any random set of peaks land this close to Fibonacci levels by pure chance?

We ran a 10,000-iteration Monte Carlo simulation, throwing random peaks across the `[0.05, 1.5]` domain and calculating their average distance to the nearest Fibonacci levels.
- **Expected random distance:** `0.0635`
- **Observed empirical distance:** `0.0166`
- **P-value:** `0.0489`

## Strategy Backtest

We attempted to build an automated, systematic trading strategy to blindly capitalize on this statistical anomaly. 

**Rules:**
1. Dynamically identify a confirmed swing (at least 1% move, pulling back >2.0 ATR).
2. Filter for macro-trend alignment (Price > 50 SMA for Longs).
3. Place a limit order at exactly the `0.618` Golden Ratio retracement level.
4. Stop Loss placed 1 ATR behind the origin of the swing (below the 1.0 level).
5. Take Profit placed at a strict 2:1 Reward-to-Risk ratio.

**Results (BTC, ETH, SOL - 1h timeframe):**
- Total Trades: 199
- Win Rate: 34.7%
- Trade-Level Sharpe: -0.04
- **Net Result: Unprofitable**

### Why does a valid statistical anomaly lose money?
This exposes a classic quantitative trading trap: **Structural Alpha $\neq$ Execution Alpha**.

1. **Adverse Selection:** A limit order placed at 0.618 catches falling knives. While the market *frequently* reverses at 0.618, the times your limit order is actually filled are disproportionately the times the market is crashing straight through to the 1.0 level (the second major peak we discovered).
2. **Missing Confluence:** The statistical proof only states that 0.618 is a frequent "attractor" or "terminus" for pullbacks. It does *not* mean every 0.618 level will hold. 
3. **Execution Edge required:** To make this profitable, you cannot use blind limit orders. You must use the 0.618 level as a **Zone of Interest** and require a secondary microstructure trigger (e.g., a tick-level orderbook absorption, a spike in delta, or a volatility contraction) *at* the level before firing a market order to enter.
