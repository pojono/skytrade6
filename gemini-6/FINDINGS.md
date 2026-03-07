# Gemini-6 Research Findings: Microstructure CVD Divergence & Dual-Market Execution

## 1. Initial Hypothesis: Tick-Level Smart Money Divergence
We hypothesized that we could extract edge from the raw `_trades.csv.gz` tick data by splitting the Cumulative Volume Delta (CVD) into two tiers:
*   **Whale Flow (Smart Money):** Trades > 98th percentile of daily volume.
*   **Retail Flow (Dumb Money):** Trades < 20th percentile of daily volume.

The trigger was a **Divergence**: Price making a new local high/low, but Whale CVD and Retail CVD moving aggressively in opposite directions.

### Initial Validation (SUIUSDT & SOLUSDT)
Using a 1-month sample (July 2025), we found that fading Retail FOMO/Panic was highly profitable in small samples, generating **+0.68% to +1.03%** forward returns over 1-4 hours.

## 2. Removing Lookahead Bias & Massive Scale Testing
We refactored the engine to eliminate lookahead bias by using strictly backward-looking Rolling Z-Scores (3-day window) for the CVDs, and dynamically sizing the Whale/Retail dollar thresholds per day.

We then scaled this to a massive Out-of-Sample test: **30 volatile altcoins over 6 months (July - Dec 2025)**.

### The Reality Check: Edge Decay
At scale (171 events), the raw structural edge decayed entirely:
*   **Avg 1h Edge:** -0.06%
*   **Avg 2h Edge:** -0.09%
*   **Avg 4h Edge:** -0.27%
*   **Win Rate (4h):** 48.0%

**Conclusion:** "Structural Alpha != Execution Alpha." When Retail panics at a 4h low and Whales absorb it passively via limit orders, the market maker does NOT immediately reverse the price. They continue to bleed the price lower to hunt for more retail liquidations because there is no aggressive market buying yet.

## 3. The Solution: Dual-Market Lead-Lag (Spot Trigger)
To fix the timing issue, we hypothesized that the Binance Spot market leads the Binance Futures market during true structural reversals. We updated the architecture:

1.  **Structural Setup (Futures):** Futures Retail is heavily selling (Z < -1.5) AND Futures Whales are passively absorbing (Z > 1.5).
2.  **Execution Trigger (Spot):** A sudden 1-minute spike in **Spot Whale Buying** (>3x the rolling 1h average).

### Dual-Market Results (5 Volatile Coins, 2 Months)
The Spot execution trigger successfully tightened the edge and made it tradable:
*   **Avg 1h Edge:** +0.29%
*   **Avg 2h Edge:** +0.37% (Clears the 20 bps taker fee hurdle)
*   **Win Rate:** 57.7%

## 4. The Microstructure Filter: L2 Ask Absorption
To further optimize execution for <100ms infrastructure, we analyzed the raw `_bookDepth` and `ob200` data at the exact millisecond of our Dual-Market triggers.

We discovered that we can filter out losing trades by watching the Market Maker's behavior on the Ask wall (top 5% of the book):
*   **Winning Trades (Clean Squeeze):** The Ask volume remains stable or drops (*Ask Absorption*). The market maker pulls liquidity to let the spot buying drive the price up.
*   **Losing Trades (Failed Signal):** The Ask volume jumps significantly (>2%) the moment the Spot buying arrives (*Wall Replenishment*). The market maker is actively suppressing the price to bleed it lower.

## 5. Architectural Proposal for Production
Based on these findings, a live execution system should operate as a 3-step sequence:

1.  **State Machine (Futures):** Maintain rolling 4h CVDs for Whales/Retail locally. Wait for the "Primed" state (Retail Panic + Whale Absorption).
2.  **Trigger (Spot):** Listen to the `aggTrade` websocket. If a massive Spot Whale buy hits, initiate the execution sequence.
3.  **Execution Filter (Orderbook):** In <100ms, snapshot the `depth` websocket. 
    *   If Ask wall is replenishing -> **ABORT**.
    *   If Ask wall is thinning/absorbing -> **EXECUTE MARKET BUY** on Futures.
