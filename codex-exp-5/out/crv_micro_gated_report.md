# CRV Micro-Gated Rule

## Scope

- Symbol: CRVUSDT
- Audited days: 2026-02-01, 2026-02-02, 2026-02-03, 2026-02-04, 2026-02-05, 2026-02-06, 2026-02-07, 2026-02-08, 2026-02-09, 2026-02-10, 2026-02-11, 2026-02-12, 2026-02-13, 2026-02-14, 2026-02-15, 2026-02-16, 2026-02-17, 2026-02-18, 2026-02-19, 2026-02-20, 2026-02-21, 2026-02-22, 2026-02-23, 2026-02-24, 2026-02-25, 2026-02-26, 2026-02-27, 2026-02-28, 2026-03-01, 2026-03-02
- Taker fee round trip: 20.00 bps

## Gates

- Min signal score: 18.00
- Max Bybit trigger book spread: 8.00 bps
- Require flow fading: False

## Comparison

- Baseline recent triggers: 90
- Baseline avg net after taker fee: 9.8842 bps
- Baseline win rate: 62.22%
- Filtered triggers: 14
- Filtered avg net after taker fee: 21.5115 bps
- Filtered win rate: 85.71%

## Monthly

| Month | Trades | Avg Net Taker bps |
|---|---:|---:|
| 2026-02 | 14 | 21.5115 |

## Interpretation

- This is a recent microstructure-covered sample, not the full historical backtest window.
- The point of this rule is to test whether adding execution-aware gates improves the surviving CRV edge materially.
