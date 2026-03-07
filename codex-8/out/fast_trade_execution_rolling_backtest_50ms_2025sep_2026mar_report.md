# Fast Trade Execution Rolling Backtest

- Events input: `codex-8/out/event_microstructure_active_union_2025sep_2026mar.csv`
- Active set input: `codex-8/out/rolling_universe_micro_2025sep_2026mar_active_symbols.csv`
- Selection quantile: `0.60`
- Hold minutes: `15`
- Entry delay ms: `50`
- Base fee assumption: `8.00` bps
- Bybit spread cross multiplier: `1.00`
- Max concurrent positions: `4`
- Max daily trades: `12`
- Folds: `155`

## Summary

- `active_fast_exec_baseline`: folds=155, mean_events=12.0, mean_fast_net_15m=-11.30 bps, mean_fast_gap_close_15m=+0.27 bps, mean_fast_win_15m=21.0%, mean_entry_decay=19.19 bps, mean_entry_fill_lag=10519.4 ms, positive_folds=10/155
- `active_fast_exec_state_top_quantile`: folds=155, mean_events=12.0, mean_fast_net_15m=-10.98 bps, mean_fast_gap_close_15m=+0.66 bps, mean_fast_win_15m=19.1%, mean_entry_decay=18.03 bps, mean_entry_fill_lag=12109.1 ms, positive_folds=4/155
- `active_fast_exec_model_top_quantile`: folds=155, mean_events=11.8, mean_fast_net_15m=-7.04 bps, mean_fast_gap_close_15m=+3.42 bps, mean_fast_win_15m=29.3%, mean_entry_decay=15.96 bps, mean_entry_fill_lag=8454.1 ms, positive_folds=20/155

## Top Mean Feature Importances

- `ob_gap_bps`: `0.2070`
- `bb_spread_bps_ob`: `0.1283`
- `rel_ret_15m_bps`: `0.0543`
- `sec_gap_bps`: `0.0412`
- `rel_ret_5m_bps`: `0.0372`
- `gap_bps`: `0.0299`
- `sec_gap_change_5s`: `0.0293`
- `ob_gap_change_5s`: `0.0288`
- `crowding_gap`: `0.0223`
- `bn_depth_imbalance_1`: `0.0199`
- `ob_gap_change_15s`: `0.0196`
- `bb_realized_vol_15m_bps`: `0.0185`
- `bb_top20_imbalance`: `0.0180`
- `sec_gap_change_15s`: `0.0170`
- `bb_mid_gap_vs_trades_bps`: `0.0169`
