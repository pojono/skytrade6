# Entry Cost Model Case Study

This study targets immediate entry cost instead of 4-hour PnL.

## Scope

- Covered symbols: ARBUSDT, ASTERUSDT, BTCUSDT, ENAUSDT, ETHUSDT, FARTCOINUSDT, LINKUSDT, SOLUSDT, TIAUSDT, WIFUSDT, WLDUSDT
- Coverage window: 2025-11-01 through 2026-01-31
- Entry cost proxy uses Binance `agg_trades_futures` and `book_depth` only.
- Because usable 5-second labels are only present in the January 2026 coverage, this study uses an internal time split within the extracted sample when no earlier training rows exist.
- For each signal, signal price = last trade before timestamp.
- Entry cost targets for a long buy:
  - `entry_vwap5_bps`: next-5s trade VWAP vs signal price
  - `entry_worst5_bps`: worst trade price in next 5s vs signal price

## Broad Research Set

- Broad signal definition: `ls_z >= 1.0`, `taker_z >= 0.0`, risk-on regime, positive 4h momentum.
- Train rows: 10
- Test rows: 13
- Unfiltered test avg (4h net after 20 bps): -74.34 bps
- Avg next-5s VWAP cost in test: 1.07 bps

## Best Entry-Cost Reject Rule

- `entry_vwap5_bps <= 0.37`
- `entry_worst5_bps <= 1.77`
- `signed_share_10s >= -0.3000`
- `vol_10s <= 0.000456`
- Broad train avg after filter: -36.85 bps on 3 rows
- Broad test avg after filter: -158.93 bps on 5 rows
- Broad test improvement vs unfiltered: -84.59 bps

## Apply Same Rule To Strict Strategy Subset

- Strict covered test rows before filter: 3
- Strict covered test avg before filter: -116.54 bps
- Strict covered avg next-5s VWAP cost: 0.50 bps
- Strict covered test rows after filter: 2
- Strict covered test avg after filter: -275.21 bps
- Strict covered avg next-5s VWAP cost after filter: 0.07 bps

## Interpretation

- If filtering on expected entry cost improves 4-hour holdout PnL, execution cost is a practical gating variable.
- If it does not, then bad 4-hour outcomes are not being driven primarily by immediate entry price impact in this sample.