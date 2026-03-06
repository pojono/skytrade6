# Microstructure Edge: Dual-Market Lead-Lag & Orderbook Absorption

## The Thesis
In highly fragmented crypto markets, price discovery happens across different participants and venues simultaneously.
Our massive-scale backtest (6 months, 30 coins) proved that "Structural Alpha" (e.g., Futures Retail selling while Futures Whales buy) decays rapidly if executed as a naked limit/market order. The market maker will happily bleed the price lower to hunt liquidations before allowing the reversal.

To fix this, we need **Execution Alpha**. We require a trigger that tells us exactly *when* the passive accumulation ends and the aggressive reversal begins.

## 1. The Trigger: Spot Whale Flow (Lead-Lag)
We tested a dual-market architecture where we monitor the **Binance Futures** market for the structural setup, but we use the **Binance Spot** market as the execution trigger.

**The Setup:**
1. **Futures Retail** is net selling heavily over 4 hours (Z-score < -1.5).
2. **Futures Whales** are passively absorbing this flow (Z-score > 1.5).
3. **Trigger:** A sudden 1-minute spike in **Spot Whale Buying** (>3x the 1h rolling average).

### Results (2-Month Sample, 5 Volatile Coins)
When adding the Spot execution trigger, the edge immediately tightened:
- **Avg 1h Edge:** +0.29%
- **Avg 2h Edge:** +0.37%
- **Win Rate:** 57.7%
This +37 bps edge is achievable and cleanly clears the 20 bps taker fee hurdle, validating that Spot heavily leads Futures during structural reversals.

## 2. The Filter: L2 Orderbook Ask Absorption
Even with the Spot trigger, the market maker might still be suppressing the price by flashing massive Ask walls.

We wrote a tool to ingest the raw ms-precision `ob200` data and `bookDepth` data exactly at the moment of our triggers to see what the market makers were doing with their liquidity.

**Findings:**
1. When the Forward Return is positive (a clean squeeze), we observed **Orderbook Absorption**. The total volume of Asks in the top 5% of the book remained stable or dropped slightly *despite* heavy Spot buying. This indicates the market maker is pulling liquidity to let the price fly.
2. When the Forward Return is negative (a failed signal), we observed **Wall Replenishment**. The total volume of Asks jumped significantly (+2.33% in our sample event) the exact moment the spot buying arrived. The market maker was absorbing the buying pressure to continue bleeding the asset lower.

## Architectural Proposal for Production
Since our infrastructure supports <100ms decision latency, we can perfectly trade this:

1. **State Machine:** Maintain rolling 4h CVDs for Futures Whales/Retail locally.
2. **Trigger:** Listen to the `aggTrade` websocket for Spot. If a cluster of >$10k trades hits within a few seconds while the State Machine is 'Primed', prepare to fire.
3. **Execution Filter:** At the exact moment of the Spot spike, snapshot the top 5% of the `depth` websocket. If the Ask wall is heavily stacked/replenishing compared to 5 seconds ago, **ABORT**. If the Ask wall is thinning or stable, **EXECUTE MARKET BUY** on Futures.
