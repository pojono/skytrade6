# Microstructure Window Analysis

- Input trades: /home/ubuntu/Projects/skytrade6/codex-exp-1/out/candidate_trades_v3_replayopt.csv
- Date window: 2026-02-01 through 2026-03-02
- Trades analyzed: 3090
- Symbols: CRVUSDT, GALAUSDT, SEIUSDT
- Winners under frozen 25% model: 1496
- Losers under frozen 25% model: 1594
- Mean modeled net PnL: 4.2159 bps
- Median modeled net PnL: -1.6044 bps

## Feature Means (Winners vs Losers)

| Feature | Winners | Losers | Delta |
|---|---:|---:|---:|
| bybit_book_spread_bps | 3.4219 | 3.5260 | -0.1040 |
| bybit_top5_imbalance | -0.0419 | -0.0285 | -0.0133 |
| bybit_trade_imbalance_5s | 0.0470 | 0.0795 | -0.0325 |
| bybit_trade_count_5s | 4.5314 | 4.1920 | 0.3394 |
| binance_trade_imbalance_5s | -0.6381 | -0.6596 | 0.0215 |
| binance_trade_count_5s | 3.5782 | 3.4216 | 0.1566 |
| binance_depth_imbalance_1pct | 0.0813 | 0.0740 | 0.0072 |
| bybit_book_lag_ms | 0.0000 | 0.0000 | 0.0000 |
| binance_depth_lag_ms | 23367.6471 | 22531.9950 | 835.6521 |

## Coverage

- Missing bybit orderbook enrichments: 0
- Missing bybit trade enrichments: 0
- Missing binance depth enrichments: 0
- Missing binance trade enrichments: 0

## Notes

- This is a first-pass microstructure overlay on the selected downloaded window only.
- Bybit enrichments use true L2 order book updates from the ob200 feed.
- Binance enrichments use trade flow plus aggregated `bookDepth` percentage buckets, not true top-of-book quotes.
- Results here are descriptive. They help identify where the 1-minute edge may depend on intraminute conditions.
