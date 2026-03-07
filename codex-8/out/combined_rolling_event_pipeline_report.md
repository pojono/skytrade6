# Combined Rolling Pipeline

- Events input: `/home/ubuntu/Projects/skytrade6/codex-8/out/event_microstructure_active_set.csv`
- Active set input: `/home/ubuntu/Projects/skytrade6/codex-8/out/rolling_universe_micro_active_symbols.csv`
- Selection quantile: `0.60`
- Folds: `4`

## Summary

- `active_baseline`: folds=4, mean_events=73.5, mean_pair_net_15m=+25.14 bps, mean_pair_net_max_15m=+61.58 bps, mean_pair_win_15m=77.9%, positive_folds=4/4
- `active_state_top_quantile`: folds=4, mean_events=28.2, mean_pair_net_15m=+18.75 bps, mean_pair_net_max_15m=+45.72 bps, mean_pair_win_15m=80.1%, positive_folds=4/4
- `active_model_top_quantile`: folds=4, mean_events=33.8, mean_pair_net_15m=+41.42 bps, mean_pair_net_max_15m=+93.95 bps, mean_pair_win_15m=82.0%, positive_folds=4/4

## Top Mean Feature Importances

- `gap_bps`: `0.4141`
- `crowding_gap`: `0.1182`
- `bb_spread_bps_ob`: `0.0645`
- `rel_ret_5m_bps`: `0.0391`
- `rel_ret_15m_bps`: `0.0362`
- `bb_signed_notional_60s`: `0.0261`
- `bn_depth_imbalance_1`: `0.0237`
- `gap_z_60`: `0.0198`
- `ob_gap_bps`: `0.0181`
- `gap_z_240`: `0.0177`
- `oi_gap_30m_z_240`: `0.0137`
- `bb_top5_imbalance`: `0.0134`
- `bn_taker_imbalance`: `0.0133`
- `sym_OPNUSDT`: `0.0131`
- `sec_gap_bps`: `0.0128`
