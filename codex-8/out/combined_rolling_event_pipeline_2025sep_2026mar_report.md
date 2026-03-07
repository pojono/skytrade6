# Combined Rolling Pipeline

- Events input: `codex-8/out/event_microstructure_active_union_2025sep_2026mar.csv`
- Active set input: `codex-8/out/rolling_universe_micro_2025sep_2026mar_active_symbols.csv`
- Selection quantile: `0.60`
- Folds: `155`

## Summary

- `active_baseline`: folds=155, mean_events=91.8, mean_pair_net_15m=+10.48 bps, mean_pair_net_max_15m=+31.25 bps, mean_pair_win_15m=71.4%, positive_folds=155/155
- `active_state_top_quantile`: folds=155, mean_events=35.5, mean_pair_net_15m=+10.54 bps, mean_pair_net_max_15m=+28.93 bps, mean_pair_win_15m=74.3%, positive_folds=155/155
- `active_model_top_quantile`: folds=155, mean_events=41.6, mean_pair_net_15m=+17.87 bps, mean_pair_net_max_15m=+45.75 bps, mean_pair_win_15m=77.7%, positive_folds=155/155

## Top Mean Feature Importances

- `gap_bps`: `0.3197`
- `rel_ret_15m_bps`: `0.1165`
- `rel_ret_5m_bps`: `0.0624`
- `gap_z_60`: `0.0568`
- `ob_gap_bps`: `0.0280`
- `gap_z_240`: `0.0262`
- `bb_spread_bps_ob`: `0.0222`
- `crowding_gap`: `0.0169`
- `bn_depth_imbalance_5`: `0.0167`
- `bb_realized_vol_15m_bps`: `0.0145`
- `bb_top20_imbalance_chg_5s`: `0.0142`
- `ob_gap_change_15s`: `0.0139`
- `bn_depth_imbalance_1`: `0.0136`
- `sec_gap_change_5s`: `0.0133`
- `sec_gap_bps`: `0.0132`
