# Codex 8

State-conditioned cross-exchange dislocation research.

This track starts from two constraints already established elsewhere in the repo:

- naive cross-exchange lead/lag is not robust enough after fees
- plain dislocation reversion can work, but only in narrow, regime-dependent slices

`codex-8` focuses on a more defensible question:

Can we rank long-only Bybit dislocation setups using Binance + Bybit state variables such as
spread extremity, premium divergence, positioning divergence, and open-interest pressure?

## Files

- `CURRENT_STATE.md`: concise status memo for the current research state, what worked, what failed, and the next step.
- `build_dislocation_panel.py`: joins Bybit and Binance minute bars plus 5-minute positioning fields into one minute panel.
- `analyze_state_conditioned_reversion.py`: engineers dislocation features, fits a chronological train/test logistic model, and writes a report.
- `build_event_microstructure_dataset.py`: extracts dislocation events and enriches them with trailing second-level trade/depth features; it can also restrict work to an exact `symbol,date` set via `--required-symbol-days`.
- `analyze_event_microstructure.py`: ranks event quality for pair-style gap closure and writes the first event-driven report.
- `walkforward_event_microstructure.py`: runs daily rolling walk-forward validation on the event dataset.
- `screen_dislocation_universe.py`: screens the recent shared-symbol universe and writes a fresh evidence-based shortlist.
- `rolling_universe_selector.py`: builds a no-lookahead rolling active symbol set from prior-day event economics.
- `combined_rolling_event_pipeline.py`: combines the rolling active universe with the event microstructure model fold by fold.
- `execution_aware_rolling_backtest.py`: applies delayed entry, explicit microstructure haircuts, and overlap limits on top of the rolling active universe.
- `fast_trade_execution_rolling_backtest.py`: applies a fast trade-print execution proxy with sub-second signal timing.

## Default universe

The initial default run uses a liquid shared universe:

- `BTCUSDT`
- `ETHUSDT`
- `SOLUSDT`
- `XRPUSDT`
- `DOGEUSDT`
- `BNBUSDT`

## Default run

```bash
python3 codex-8/build_dislocation_panel.py
python3 codex-8/analyze_state_conditioned_reversion.py
```

Long-window selector run:

```bash
python3 codex-8/screen_dislocation_universe.py \
  --start-date 2025-09-01 \
  --end-date 2026-03-04 \
  --min-days 185 \
  --require-micro \
  --train-days 30 \
  --min-train-events 40 \
  --min-test-events 20 \
  --output-prefix universe_screen_micro_2025sep_2026mar

python3 codex-8/rolling_universe_selector.py \
  --input codex-8/out/universe_screen_micro_2025sep_2026mar_events.csv \
  --train-days 30 \
  --max-active-symbols 8 \
  --entry-rank 8 \
  --keep-rank 12 \
  --min-train-events 40 \
  --min-mean-edge-bps 2.0 \
  --min-positive-day-share 0.6 \
  --output-prefix rolling_universe_micro_2025sep_2026mar
```

Defaults:

- date range: `2025-07-01` to `2026-03-04`
- hold horizon: `15` minutes
- entry universe: rows where Bybit is at least `6 bps` cheaper than Binance
- fee assumption: `8 bps` round trip

## Outputs

- `out/dislocation_panel.csv.gz`
- `out/dislocation_panel_summary.json`
- `out/state_reversion_summary.csv`
- `out/state_reversion_score_buckets.csv`
- `out/state_reversion_feature_weights.csv`
- `out/state_reversion_monthly_test.csv`
- `out/state_reversion_report.md`
- `out/event_microstructure_dataset.csv`
- `out/event_microstructure_report.md`
- `out/event_microstructure_walkforward_folds.csv`
- `out/event_microstructure_walkforward_summary.csv`
- `out/event_microstructure_walkforward_report.md`
- `out/universe_screen_micro_top_candidates.csv`
- `out/rolling_universe_micro_active_symbols.csv`
- `out/rolling_universe_micro_daily_performance.csv`
- `out/rolling_universe_micro_report.md`
- `out/universe_screen_micro_2025sep_2026mar_events.csv`
- `out/universe_screen_micro_2025sep_2026mar_report.md`
- `out/rolling_universe_micro_2025sep_2026mar_active_symbols.csv`
- `out/rolling_universe_micro_2025sep_2026mar_daily_performance.csv`
- `out/rolling_universe_micro_2025sep_2026mar_monthly_summary.csv`
- `out/rolling_universe_micro_2025sep_2026mar_report.md`
- `out/rolling_universe_micro_2025sep_2026mar_required_symbol_days.csv`
- `out/dislocation_panel_active_union_2025sep_2026mar.csv.gz`
- `out/event_microstructure_active_union_2025sep_2026mar.csv`
- `out/combined_rolling_event_pipeline_2025sep_2026mar_folds.csv`
- `out/combined_rolling_event_pipeline_2025sep_2026mar_summary.csv`
- `out/combined_rolling_event_pipeline_2025sep_2026mar_feature_importance.csv`
- `out/combined_rolling_event_pipeline_2025sep_2026mar_report.md`
- `out/execution_aware_rolling_backtest_2025sep_2026mar_folds.csv`
- `out/execution_aware_rolling_backtest_2025sep_2026mar_summary.csv`
- `out/execution_aware_rolling_backtest_2025sep_2026mar_monthly.csv`
- `out/execution_aware_rolling_backtest_2025sep_2026mar_feature_importance.csv`
- `out/execution_aware_rolling_backtest_2025sep_2026mar_report.md`
- `out/fast_trade_execution_rolling_backtest_50ms_2025sep_2026mar_folds.csv`
- `out/fast_trade_execution_rolling_backtest_50ms_2025sep_2026mar_summary.csv`
- `out/fast_trade_execution_rolling_backtest_50ms_2025sep_2026mar_monthly.csv`
- `out/fast_trade_execution_rolling_backtest_50ms_2025sep_2026mar_feature_importance.csv`
- `out/fast_trade_execution_rolling_backtest_50ms_2025sep_2026mar_report.md`
- `out/combined_rolling_event_pipeline_folds.csv`
- `out/combined_rolling_event_pipeline_summary.csv`
- `out/combined_rolling_event_pipeline_report.md`

## First objective

The first objective is not production deployment. It is to answer a narrower research question:

Does ML ranking improve the quality of candidate dislocation trades over:

- the raw candidate universe
- a simple hand-built state score

If yes, the next step is walk-forward validation and explicit execution modeling.

## Current State

See `CURRENT_STATE.md` for the current research status.

Short version:

- rolling no-lookahead symbol selection works on the long window
- event microstructure ranking improves the active sleeve
- conservative execution-aware backtests are negative
- the datalake supports serious Bybit execution modeling, but not symmetric execution-grade Binance fills

So the correct next step is a Bybit-only orderbook execution simulator with Binance used as signal input.
