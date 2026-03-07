# Event Microstructure Walk-Forward

- Input: `/home/ubuntu/Projects/skytrade6/codex-8/out/event_microstructure_dataset_wf.csv`
- Train window: `5` days
- Selection quantile: `0.60`
- Distinct days: `9`
- Walk-forward folds: `4`

## Summary

- `baseline`: folds=4, mean_events=55.8, mean_pair_net_15m=+9.21 bps, mean_pair_net_max_15m=+30.98 bps, mean_pair_win_15m=58.2%, positive_folds=4/4
- `state_top_quantile`: folds=4, mean_events=21.5, mean_pair_net_15m=+7.67 bps, mean_pair_net_max_15m=+30.68 bps, mean_pair_win_15m=53.0%, positive_folds=4/4
- `model_top_quantile`: folds=4, mean_events=20.2, mean_pair_net_15m=+16.26 bps, mean_pair_net_max_15m=+44.58 bps, mean_pair_win_15m=77.6%, positive_folds=4/4

## Top Mean Feature Importances

- `bb_spread_bps_ob`: `0.1553`
- `rel_ret_15m_bps`: `0.0762`
- `gap_bps`: `0.0602`
- `bb_top20_imbalance`: `0.0469`
- `crowding_gap`: `0.0448`
- `gap_z_240`: `0.0390`
- `bb_top5_imbalance`: `0.0306`
- `bb_mid_gap_vs_trades_bps`: `0.0282`
- `ob_gap_bps`: `0.0280`
- `rel_ret_5m_bps`: `0.0252`
- `bn_depth_imbalance_1`: `0.0239`
- `bb_top5_pull_pressure_5s`: `0.0223`
- `gap_z_60`: `0.0218`
- `bb_top5_pull_pressure`: `0.0201`
- `tod_cos`: `0.0196`
