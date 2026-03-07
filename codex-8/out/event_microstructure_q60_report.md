# Event Microstructure Report

- Input: `/home/ubuntu/Projects/skytrade6/codex-8/out/event_microstructure_dataset.csv`
- Train/Test cutoff day: `2026-03-03`
- Selection quantile: `0.60`

## Dataset

- Events: `321`
- Symbols: `3`
- Train events: `204`
- Test events: `117`
- Baseline test pair net 15m: `+8.20 bps`

## Test Summary

- `baseline`: events=117, pair_net_15m=+8.20 bps, pair_net_max_15m=+26.79 bps, pair_win_15m=51.3%, pair_win_max_15m=74.4%, gap_close_15m=+16.20 bps
- `state_top_quantile`: events=36, pair_net_15m=+5.90 bps, pair_net_max_15m=+29.85 bps, pair_win_15m=47.2%, pair_win_max_15m=77.8%, gap_close_15m=+13.90 bps
- `model_top_quantile`: events=20, pair_net_15m=+21.15 bps, pair_net_max_15m=+48.40 bps, pair_win_15m=85.0%, pair_win_max_15m=100.0%, gap_close_15m=+29.15 bps

## Top Feature Importances

- `bb_spread_bps_ob`: `0.1016`
- `sec_gap_change_15s`: `0.0756`
- `gap_bps`: `0.0602`
- `bb_top20_imbalance`: `0.0568`
- `rel_ret_5m_bps`: `0.0549`
- `bb_mid_gap_vs_trades_bps`: `0.0440`
- `bn_depth_imbalance_5`: `0.0425`
- `gap_z_240`: `0.0391`
- `rel_ret_15m_bps`: `0.0330`
- `bb_best_sz_imbalance`: `0.0315`
- `ob_gap_bps`: `0.0315`
- `crowding_gap`: `0.0275`
- `bn_taker_imbalance`: `0.0273`
- `ob_gap_change_15s`: `0.0210`
- `tod_cos`: `0.0203`
