# Deployment Architecture: Volatility-Adjusted Powder Keg

## The "Golden Cluster" Universe
Based on rigorous 1-minute high-fidelity tick simulation, we discovered that extreme leverage setups ("Powder Kegs") are heavily manipulated by market makers on low-cap altcoins via stop-hunting wicks. 

To execute this strategy profitably, we **must strictly limit the trading universe** to the "Golden Cluster"—high-liquidity assets with massive, thick orderbooks that cannot be easily spoofed or wick-hunted.

### The Golden Cluster
1. **BTCUSDT** (Bitcoin)
2. **SOLUSDT** (Solana)
3. **LINKUSDT** (Chainlink)
4. **AVAXUSDT** (Avalanche)
5. **NEARUSDT** (NEAR Protocol)
6. **WLDUSDT** (Worldcoin - high volatility, but highly mathematical squeezes)

*(Note: We explicitly exclude SUI, ENA, and ETH based on the high-fidelity backtest proving they either chop out or don't cascade cleanly enough within the 24h window).*

---

## Execution Logic (The "Wide Net" Squeeze Catcher)

To survive the intra-candle HFT noise leading up to a liquidation cascade, the deployment script must enforce the following strict execution rules:

1. **No Trailing Stops:** Do not submit stop-loss orders to the exchange. Market makers will hunt them.
2. **Inverse-ATR Sizing:** The portfolio risk manager must calculate the 24-hour Average True Range (ATR) immediately before entry. The position size is dynamically scaled so that a 1x ATR move against the position equals exactly 1% of total portfolio equity.
3. **Hard Take Profit:** A limit order placed exactly 10% away from the execution entry price.
4. **Hard Time Stop:** A cron job or websocket monitor that forcefully closes the position exactly 24 hours after entry, regardless of PnL. If the cascade hasn't occurred in 24 hours, the setup is invalid.

## System Architecture

### 1. Data Ingestion Layer (Websockets)
* **Binance Futures API:** Subscribes to 1-minute Klines (for ATR calculation) and Open Interest streams.
* **Bybit Futures API:** Subscribes to Funding Rate streams.

### 2. State Management Layer (Redis / Memory)
* Maintains a rolling 168-hour (7-day) buffer of Hourly Close, Open Interest, and Funding Rates for the Golden Cluster.
* Calculates Z-Scores continuously.

### 3. Execution Layer
* When `OI_Z > 2.0` AND `FR_Z > 2.0`, it triggers a **SHORT**.
* When `OI_Z > 2.0` AND `FR_Z < -2.0`, it triggers a **LONG**.
* Calculates size: `Trade Size = (Portfolio Equity * 0.01) / ATR_Percent`.
* Executes Market Order.
* Immediately places a Limit Take Profit Order (+10%).
* Registers the trade in the Time-Stop Database (to be closed in 24h).

