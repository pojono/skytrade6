# Rolling Universe Selector

- Input: `/home/ubuntu/Projects/skytrade6/codex-8/out/universe_screen_micro_events.csv`
- Train window: `5` days
- Max active symbols: `8`
- Entry rank: `8`
- Keep rank: `12`
- Min train events: `20`
- Min mean edge: `2.00` bps
- Min positive-day share: `0.60`

## Summary

- `baseline_all_symbols`: folds=4, mean_events=538.2, mean_pair_net_15m=+2.79 bps, mean_pair_net_30m=+3.61 bps, mean_pair_win_15m=47.8%, positive_folds=4/4
- `rolling_active_universe`: folds=4, mean_events=73.5, mean_pair_net_15m=+25.14 bps, mean_pair_net_30m=+27.22 bps, mean_pair_win_15m=77.9%, positive_folds=4/4

## Daily Active Sets

- `2026-03-01`: active=7 [DYDXUSDT,CHZUSDT,CRVUSDT,OPNUSDT,HUMAUSDT,ATHUSDT,GIGGLEUSDT], active15=+32.89 bps vs baseline15=+3.29 bps
- `2026-03-02`: active=8 [DYDXUSDT,CHZUSDT,CRVUSDT,OPNUSDT,HUMAUSDT,ATHUSDT,GIGGLEUSDT,GALAUSDT], active15=+24.18 bps vs baseline15=+3.02 bps
- `2026-03-03`: active=8 [DYDXUSDT,CHZUSDT,CRVUSDT,OPNUSDT,HUMAUSDT,ATHUSDT,GIGGLEUSDT,GALAUSDT], active15=+27.57 bps vs baseline15=+3.36 bps
- `2026-03-04`: active=4 [DYDXUSDT,CRVUSDT,HUMAUSDT,GALAUSDT], active15=+15.91 bps vs baseline15=+1.50 bps
