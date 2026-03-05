# Execution-Aware Filter Case Study

This is a microstructure case study on the locally covered Binance symbols with true execution data.

## Scope

- Execution data source: local `parquet/{symbol}/binance/book_depth` and `agg_trades_futures`.
- Coverage window: `2025-11-01` through `2026-01-31`.
- Covered symbols in this run: ARBUSDT, ASTERUSDT, BTCUSDT, ENAUSDT, ETHUSDT, FARTCOINUSDT, LINKUSDT, SOLUSDT, TIAUSDT, WIFUSDT, WLDUSDT.
- This is not a full-universe filter. It is a covered-subset test only.

## Broad Research Set

- Broad signal definition for learning: `ls_z >= 1.0`, `taker_z >= 0.0`, risk-on regime, positive 4h momentum.
- Train rows with execution features: 19
- Test rows with execution features: 77
- Unfiltered train avg: -63.62 bps
- Unfiltered test avg: -15.38 bps

## Best Execution Reject Rule

- `signed_share_1m >= -0.1173`
- `depth_imbalance_1 >= -0.1000`
- `depth_total_1 >= 48,427,867`
- `rv_1m <= 0.001299`
- Broad train avg after filter: 24.60 bps on 6 rows
- Broad test avg after filter: -42.41 bps on 7 rows
- Broad train improvement vs unfiltered: 88.22 bps
- Broad test improvement vs unfiltered: -27.03 bps
- Broad test hit rate after filter: 14.3%

## Apply Same Rule To Strict Strategy Subset

- Strict subset = the actual current strategy conditions on the covered symbols (`ls_z >= 2.0`, `taker_z >= 0.5`).
- Strict covered test rows before filter: 17
- Strict covered test avg before filter: 20.48 bps
- Strict covered test rows after filter: 0
- Strict covered test avg after filter: no rows kept

## Interpretation

- The pre-2026 microstructure-trained rules did not improve the broader January covered holdout; they made it worse.
- On the actual strict covered holdout, the learned rules were so conservative that they rejected everything.
- That means execution context is likely relevant, but these simple transferable reject rules are not good enough yet.
- Sample sizes are still limited, so this is directional evidence only.