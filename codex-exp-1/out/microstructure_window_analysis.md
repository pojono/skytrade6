# Microstructure Window Analysis

- Input trades: /home/ubuntu/Projects/skytrade6/codex-exp-1/out/candidate_trades_v3_replayopt.csv
- Date window: 2026-02-24 through 2026-03-02
- Trades analyzed: 561
- Symbols: CRVUSDT, GALAUSDT, SEIUSDT
- Winners under frozen 25% model: 295
- Losers under frozen 25% model: 266
- Mean modeled net PnL: 6.2514 bps
- Median modeled net PnL: 1.5621 bps

## Feature Means (Winners vs Losers)

| Feature | Winners | Losers | Delta |
|---|---:|---:|---:|
| bybit_book_spread_bps | 3.5219 | 3.7013 | -0.1794 |
| bybit_top5_imbalance | -0.0411 | -0.0157 | -0.0253 |
| bybit_trade_imbalance_5s | -0.0362 | 0.0842 | -0.1204 |
| bybit_trade_count_5s | 4.6814 | 3.5564 | 1.1250 |
| binance_trade_imbalance_5s | -0.6452 | -0.6293 | -0.0160 |
| binance_trade_count_5s | 3.8746 | 2.6203 | 1.2543 |
| binance_depth_imbalance_1pct | 0.0546 | 0.0439 | 0.0107 |
| bybit_book_lag_ms | 0.0000 | 0.0000 | 0.0000 |
| binance_depth_lag_ms | 23769.4915 | 22334.5865 | 1434.9051 |

## Coverage

- Missing bybit orderbook enrichments: 0
- Missing bybit trade enrichments: 0
- Missing binance depth enrichments: 0
- Missing binance trade enrichments: 0

## Notes

- This is a first-pass microstructure overlay on the downloaded 7-day window only.
- Bybit enrichments use true L2 order book updates from the ob200 feed.
- Binance enrichments use trade flow plus aggregated `bookDepth` percentage buckets, not true top-of-book quotes.
- Results here are descriptive. They help identify where the 1-minute edge may depend on intraminute conditions.