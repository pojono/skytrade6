# Microstructure Filter Report

- Train days: 2026-02-24, 2026-02-25, 2026-02-26, 2026-02-27, 2026-02-28
- Test days: 2026-03-01, 2026-03-02

## Baseline (No Extra Microstructure Gate)

- Train: 362 trades, 6.9477 bps, 53.04% win rate
- Test: 199 trades, 4.9847 bps, 51.76% win rate

## Selected Filter (Train Only)

```json
{
  "min_bybit_trade_count_5s": 0.0,
  "min_binance_trade_count_5s": 0.0,
  "max_bybit_book_spread_bps": 4.5,
  "max_bybit_trade_imbalance_5s": 0.2
}
```

- Train: 224 trades, 8.9789 bps, 58.93% win rate
- Test: 151 trades, 4.6207 bps, 51.66% win rate

## Hypothesis-Driven Gate (Activity + Tight Bybit Book)

This is not chosen on the holdout. It is a hand-picked follow-through from the descriptive microstructure findings:

```json
{
  "min_bybit_trade_count_5s": 4.0,
  "min_binance_trade_count_5s": 0.0,
  "max_bybit_book_spread_bps": 4.5,
  "max_bybit_trade_imbalance_5s": 1.0
}
```

- Train: 92 trades, 9.4215 bps, 58.70% win rate
- Test: 66 trades, 7.8444 bps, 59.09% win rate

- Kept trades in full 7-day window: 375 / 561
