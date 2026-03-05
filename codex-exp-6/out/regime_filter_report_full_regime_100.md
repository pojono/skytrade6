# Regime Filter Report

- Date range: `2026-01-02` to `2026-03-02`
- Universe size: `99`
- Horizon: `240` minutes
- Base expected net (all signals): `3.27` bps
- Minimum samples per rule: `1200`

## Top Rules

- rv_60 <= q50 (8.5661): expected `2.27` bps, worst split `-16.88` bps, trade share `50.00%`, samples `9454`
- signal_align_240 > 0 AND rv_60 <= q50: expected `2.27` bps, worst split `-16.88` bps, trade share `50.00%`, samples `9454`
- rv_60 <= q65 (10.4541): expected `2.72` bps, worst split `-16.94` bps, trade share `65.00%`, samples `12289`
- all_signals: expected `3.27` bps, worst split `-18.86` bps, trade share `100.00%`, samples `18907`
- signal_align_240 > 0: expected `3.27` bps, worst split `-18.86` bps, trade share `100.00%`, samples `18907`
- signal_align_h > 0: expected `3.27` bps, worst split `-18.86` bps, trade share `100.00%`, samples `18907`
- rv_240 <= q50 (9.4333): expected `2.57` bps, worst split `-20.66` bps, trade share `50.00%`, samples `9454`
- signal_align_60 > 0 AND rv_60 <= q50: expected `3.36` bps, worst split `-20.73` bps, trade share `36.71%`, samples `6940`
- signal_align_60 > 0: expected `3.99` bps, worst split `-21.19` bps, trade share `77.14%`, samples `14585`
- signal_align_240 > 0 AND rv_60 <= q50 AND disp_240 <= q50: expected `3.48` bps, worst split `-21.64` bps, trade share `28.56%`, samples `5400`

## Practical Decision

Treat a rule as `trade` only if it keeps meaningful sample size and improves worst-split net expectancy versus `all_signals`.
Everything else is `no-trade` by default.
