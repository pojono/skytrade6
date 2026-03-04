# Cross-Exchange Pattern Research — Experiment 1

## Thesis

When the same asset trades on multiple exchanges (Bybit vs Binance), temporary dislocations in price, premium, volume, and open interest create **predictable short-term price movements**. These dislocations arise because:

1. **Price leads/lags** — One exchange often moves first (informed flow), the other follows. If Bybit spikes while Binance lags, the lagging exchange will catch up.
2. **Premium divergence** — The gap between futures price and index price differs across exchanges. Convergence is mechanical.
3. **Volume imbalance** — A sudden volume spike on one exchange (but not the other) signals informed order flow. The direction of that flow predicts the next move.
4. **OI divergence** — When OI surges on one exchange but not the other, it signals new positioning that creates directional pressure.
5. **Funding rate divergence** — Different FR across exchanges creates arbitrage pressure that moves prices.

## Why This Should Work

- With 116 symbols, we get massive statistical power (116 × 240 days × 1440 min/day ≈ 40M datapoints)
- Cross-exchange features are **structural** — not curve-fitted pattern recognition
- Fee hurdle: 20bps round-trip (taker) or 8bps (maker). We target moves > 50bps.
- Altcoins are less efficient than BTC/ETH → more edge

## Data Available

- **116 common symbols** on Bybit + Binance
- **Date range:** 2025-07-01 to 2026-03-03 (~8 months)
- **Data types per exchange:**
  - 1m OHLCV klines
  - 1m mark price klines
  - 1m premium index klines
  - Funding rate history
  - Open interest (5min, Bybit) / metrics (5min, Binance)
  - Long/short ratio (5min, Bybit) / metrics (5min, Binance)

## Fee Assumptions

- **Taker fee:** 0.10% (10 bps) per leg → 20 bps round-trip
- **Maker fee:** 0.04% (4 bps) per leg → 8 bps round-trip
- Strategy must target **net profit after fees**, focusing on large moves > 50bps

## Pipeline

1. `load_data.py` — Parallel loader for all 116 symbols, both exchanges
2. `features.py` — Cross-exchange feature engineering (5m aggregation)
3. `discover.py` — Signal discovery: which features predict >50bps moves?
4. `backtest.py` — Walk-forward backtest with realistic fees
5. `optimize.py` — Parameter sweep + robustness checks

## Key Signals to Investigate

### Tier 1: Price-based cross-exchange signals
- **Price divergence** (Bybit close − Binance close) / midpoint — mean-reverting?
- **Return lead/lag** — does 5m return on exchange A predict next 5m on exchange B?
- **Premium spread** — (Bybit premium − Binance premium) divergence & convergence

### Tier 2: Volume/flow-based signals
- **Volume ratio spike** — sudden change in (Bybit volume / Binance volume)
- **Taker buy ratio divergence** — Binance has taker_buy_volume; compare to total
- **Volume-weighted price divergence** — VWAP divergence across exchanges

### Tier 3: Positioning signals
- **OI divergence acceleration** — OI growing faster on one exchange
- **LS ratio divergence** — crowd positioning differs → predict squeeze
- **Funding rate spread** — FR(Bybit) − FR(Binance) extreme values

### Tier 4: Composite/interaction signals
- **Volume spike + price divergence** — flow on one exchange + lag on the other
- **OI surge + premium spike** — new leveraged positions pushing premium out of line
