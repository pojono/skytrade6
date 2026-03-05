# Post-Entry Path Case Study

This study tests whether 5-30 minute follow-through after entry is the real gating variable.

## Scope

- Covered symbols: ARBUSDT, ASTERUSDT, BTCUSDT, ENAUSDT, ETHUSDT, FARTCOINUSDT, LINKUSDT, SOLUSDT, TIAUSDT, WIFUSDT, WLDUSDT
- Uses Binance `agg_trades_futures` on the widened January-covered symbol set.
- Internal time split within the extracted sample (train/test) because the usable post-entry paths are concentrated in January 2026.

## Broad Research Set

- Train rows: 52
- Test rows: 44
- Unfiltered broad test avg: -39.84 bps
- Average broad test 30s return: 0.79 bps
- Average broad test 5m return: -3.37 bps

## Best Post-Entry Rule

- `ret_30s_bps >= 0.00`
- `ret_5m_bps >= 1.09`
- `path_low_5m_bps >= -12.45`
- `buy_share_5m >= 0.456`
- Broad train avg after filter: -19.37 bps on 19 rows
- Broad test avg after filter: -17.15 bps on 13 rows
- Broad test improvement vs unfiltered: 22.69 bps

## Apply Same Rule To Strict Strategy Subset

- Strict covered test rows before filter: 11
- Strict covered test avg before filter: -18.98 bps
- Strict covered test rows after filter: 3
- Strict covered test avg after filter: -23.65 bps

## Correlation Clue

- Corr(`ret_30s_bps`, 4h net): -0.076
- Corr(`ret_5m_bps`, 4h net): -0.076
- Corr(`path_low_5m_bps`, 4h net): -0.206

## Interpretation

- If this improves holdout materially, the real problem is post-entry fade, not entry friction.
- If it does not, then even short-term follow-through is not a stable separator on this sample.