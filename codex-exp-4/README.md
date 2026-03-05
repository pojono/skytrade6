# codex-exp-4

This experiment is isolated to `codex-exp-4` and reads source data from `../datalake`.

## Goal

Search for repeatable pre-move signals across **Binance + Bybit** perpetual futures data that can support a strategy after a conservative **8 bps round-trip maker fee** assumption (`0.04%` per side).

## What is implemented

`analyze_premove_signals.py`:

- loads overlapping 1-minute Binance and Bybit data for selected symbols
- merges minute bars with 5-minute OI / long-short metrics
- builds pre-move features (momentum, basis, volume z-scores, taker imbalance, OI changes, long-short changes, realized vol)
- labels "big moves" by future return over a configurable horizon
- tests threshold rules on a strict chronological train/test split
- ranks rules by out-of-sample performance and writes CSV outputs to `out/`

`microstructure_edge_scan.py`:

- builds per-second features from Binance trades, Bybit trades, and Binance `bookDepth`
- measures signed aggressive flow across both exchanges
- computes compact depth imbalance features from Binance depth snapshots
- tests short-horizon threshold rules plus simple depth+flow conjunctions
- caches per-second feature tables in `out/cache/` for faster reruns

`bybit_orderbook_edge_scan.py`:

- streams Bybit `orderbook.jsonl` and reconstructs a live top-of-book state
- emits one summary row per second with spread, top-5/top-20 imbalance, and pull-pressure features
- joins those orderbook features with cross-exchange trade flow
- tests short-horizon rules and simple orderbook+flow conjunctions

`event_regime_edge_scan.py`:

- uses cached joined microstructure tables
- converts raw signals into shock-style events with a cooldown
- evaluates those events under explicit regime filters (`wide_spread`, `high_abs_flow`, `extreme_pull`, `high_abs_gap`)
- keeps train/test chronology fixed and reuses the train-derived event direction on the test set

`walkforward_event_regime.py`:

- builds on the cached joined microstructure files
- runs rolling walk-forward validation across multiple days
- pools prior days as training, selects the best event/regime rule on that window, and tests it on the next day
- writes fold-level and summary CSVs for robustness assessment

`cross_exchange_leadlag_model.py`:

- tests a different hypothesis class: direct price lead/lag between Binance and Bybit
- builds short-horizon return, relative-move, gap-change, and simple flow/spread-normalized features
- fits a rolling walk-forward ridge model on prior days
- keeps only the largest-magnitude predictions as trade candidates and evaluates them after fees

`rare_event_precision_filter.py`:

- extracts only rare shock states instead of scoring every second
- labels whether those shock events are followed by a large move
- trains a walk-forward classifier optimized for **precision** on dangerous events
- measures lift over the baseline danger rate rather than average return

## Default run

```bash
python3 codex-exp-4/analyze_premove_signals.py
```

Default settings:

- symbols: `BTCUSDT ETHUSDT SOLUSDT DOGEUSDT`
- overlap window: last `45` common trading days
- move horizon: `15` minutes
- big-move threshold: `35` bps
- fee assumption: `8` bps round trip

## Current finding

On the current default run, the existing bar + 5-minute positioning data does **not** show a robust single-threshold edge that survives the 8 bps fee assumption out of sample.

The best out-of-sample single-factor rule found by the current sweep was still negative after fees.

This means the next credible step is:

1. move to richer microstructure inputs (`trades`, possibly top-of-book / depth)
2. test cross-exchange lead/lag features directly
3. require agreement from multiple pre-move signals instead of single-factor thresholds

## Microstructure update

I downloaded:

- Binance `trades` for `BTCUSDT` and `SOLUSDT` on `2026-03-02` and `2026-03-03`
- Binance `bookDepth` for the same symbols and dates
- Bybit `trades` and `orderbook` for the same symbols and dates

Current microstructure finding:

- extreme trade-flow bursts and depth imbalances do show directional information
- the best out-of-sample candidates come from **depth imbalance + cross-exchange flow confirmation**
- Bybit orderbook spread and pull-pressure also show directional information
- event-style scans reduce noise, but still do not produce a fee-surviving train+test candidate on the current two-day sample
- rolling walk-forward validation over `2026-02-24` through `2026-03-03` is also negative on every test fold for both `BTCUSDT` and `SOLUSDT`
- the direct cross-exchange lead/lag price model is also negative on every walk-forward fold
- the best current use of the signal is as a **rare-event danger detector**, not a direct alpha model
- but the current rules are still **not robustly positive on both train and test**, so the edge is not yet production-grade

The useful part is that the search space now looks materially better than the 1-minute bar-only study, and it points toward queue-pressure / liquidity-withdrawal features as the likely next source of edge.

## Walk-forward update

Using an 8-day sample (`2026-02-24` to `2026-03-03`) with rolling 3-day train windows:

- `BTCUSDT`: mean test result was `-5.36 bps` per trade, with `0/5` positive test folds
- `SOLUSDT`: mean test result was `-8.54 bps` per trade, with `0/5` positive test folds

This is a much stronger negative result than the earlier single split. It means the currently discovered event/regime rules are not just overfit to one day; they still fail once evaluated in a rolling forward process.

## Lead/Lag update

Using the same 8-day window, a direct cross-exchange lead/lag model based on Binance-vs-Bybit relative price moves also failed:

- `BTCUSDT` at a 60-second horizon: mean test result `-7.84 bps`, `0/5` positive test folds
- `SOLUSDT` at a 60-second horizon: mean test result `-7.75 bps`, `0/5` positive test folds

This is important because it rules out the simpler thesis that one exchange is cleanly leading the other in a way that is easy to monetize after fees with a short-horizon directional model.

## Rare-Event update

On a broader 10-day pooled sample (`2026-02-22` to `2026-03-03`) across:

- `BTCUSDT`
- `SOLUSDT`
- `DOGEUSDT`
- `1000PEPEUSDT`
- `GUNUSDT`

the rare-event precision model extracted `31,570` shock events with a baseline danger rate of `25.0%`.

Walk-forward results:

- mean test precision: `34.2%`
- mean test lift: `1.23x`

So the wider sample confirms the effect, but at a lower strength than the earlier smaller sample. The signal remains useful for **selecting elevated-risk shock events**, but it is not strong enough to reclassify this as a standalone tradable alpha.

## Outputs

- `out/best_rule_summary.csv`
- `out/rule_leaderboard.csv`
- `out/feature_conditionals.csv`
- `out/microstructure_rule_leaderboard.csv`
- `out/microstructure_top_candidates.csv`
- `out/bybit_orderbook_rule_leaderboard.csv`
- `out/bybit_orderbook_top_candidates.csv`
- `out/event_regime_rule_leaderboard.csv`
- `out/event_regime_top_candidates.csv`
- `out/walkforward_event_regime_folds.csv`
- `out/walkforward_event_regime_summary.csv`
- `out/cross_exchange_leadlag_folds.csv`
- `out/cross_exchange_leadlag_summary.csv`
- `out/rare_event_dataset.csv`
- `out/rare_event_precision_folds.csv`
- `out/rare_event_precision_summary.csv`
- `out/cache/*.csv`
