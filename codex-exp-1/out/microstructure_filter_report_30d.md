# Microstructure Filter Report

- Train days: 2026-02-01, 2026-02-02, 2026-02-03, 2026-02-04, 2026-02-05, 2026-02-06, 2026-02-07, 2026-02-08, 2026-02-09, 2026-02-10, 2026-02-11, 2026-02-12, 2026-02-13, 2026-02-14, 2026-02-15, 2026-02-16, 2026-02-17, 2026-02-18, 2026-02-19, 2026-02-20, 2026-02-21, 2026-02-22, 2026-02-23
- Test days: 2026-02-24, 2026-02-25, 2026-02-26, 2026-02-27, 2026-02-28, 2026-03-01, 2026-03-02

## Baseline (No Extra Microstructure Gate)

- Train: 2529 trades, 3.7644 bps, 47.49% win rate
- Test: 561 trades, 6.2514 bps, 52.58% win rate

## Selected Filter (Train Only)

```json
{
  "min_bybit_trade_count_5s": 2.0,
  "min_binance_trade_count_5s": 0.0,
  "max_bybit_book_spread_bps": 4.5,
  "max_bybit_trade_imbalance_5s": 1.0
}
```

- Train: 901 trades, 4.7455 bps, 51.28% win rate
- Test: 199 trades, 8.1528 bps, 59.30% win rate

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

- Train: 694 trades, 4.5536 bps, 50.29% win rate
- Test: 158 trades, 8.7627 bps, 58.86% win rate

- Kept trades in full analysis window: 1100 / 3090
