# Execution Aware Rolling Backtest

- Events input: `codex-8/out/event_microstructure_active_union_2025sep_2026mar.csv`
- Active set input: `codex-8/out/rolling_universe_micro_2025sep_2026mar_active_symbols.csv`
- Panel input: `codex-8/out/dislocation_panel_active_union_2025sep_2026mar.csv.gz`
- Selection quantile: `0.60`
- Hold minutes: `15`
- Entry delay minutes: `1`
- Base fee assumption: `8.00` bps
- Spread cross multiplier: `0.50`
- Stale book multiplier: `0.25`
- Adverse imbalance multiplier: `0.50`
- Max concurrent positions: `4`
- Max daily trades: `12`
- Folds: `155`

## Summary

- `active_exec_baseline`: folds=155, mean_events=12.0, mean_exec_net_15m=-10.80 bps, mean_delay_net_15m=-8.10 bps, mean_exec_win_15m=22.9%, mean_extra_cost=2.70 bps, mean_entry_decay=18.59 bps, positive_folds=7/155
- `active_exec_state_top_quantile`: folds=155, mean_events=12.0, mean_exec_net_15m=-10.35 bps, mean_delay_net_15m=-7.35 bps, mean_exec_win_15m=21.3%, mean_extra_cost=2.99 bps, mean_entry_decay=17.58 bps, positive_folds=7/155
- `active_exec_model_top_quantile`: folds=155, mean_events=11.9, mean_exec_net_15m=-8.62 bps, mean_delay_net_15m=-6.95 bps, mean_exec_win_15m=26.4%, mean_extra_cost=1.67 bps, mean_entry_decay=17.90 bps, positive_folds=13/155

## Top Mean Feature Importances

- `sec_gap_bps`: `0.1001`
- `bb_mid_gap_vs_trades_bps`: `0.0933`
- `bb_spread_bps_ob`: `0.0746`
- `rel_ret_5m_bps`: `0.0502`
- `gap_bps`: `0.0455`
- `ob_gap_bps`: `0.0414`
- `rel_ret_15m_bps`: `0.0399`
- `ob_gap_change_15s`: `0.0372`
- `ob_gap_change_5s`: `0.0368`
- `bb_top20_imbalance`: `0.0284`
- `sec_gap_change_5s`: `0.0268`
- `crowding_gap`: `0.0247`
- `bn_depth_imbalance_1`: `0.0221`
- `sec_gap_change_15s`: `0.0212`
- `gap_z_240`: `0.0196`
