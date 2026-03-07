# Codex 8 Current State

Date: `2026-03-07`

## Objective

`codex-8` started as a no-lookahead research track for cross-exchange dislocation events.

The working question became:

- can Binance + Bybit state and microstructure features identify tradable Bybit dislocation events
- with a rolling active symbol set chosen only from prior data

## What Is Proven

### 1. Rolling symbol selection is real

Long-window no-lookahead selector run:

- window: `2025-09-01` to `2026-03-04`
- folds: `155` daily folds
- eligible symbols: `46`

Result:

- all-symbol baseline: `+2.25 bps`
- rolling active universe: `+10.48 bps`
- positive in `155/155` folds

Main files:

- `out/universe_screen_micro_2025sep_2026mar_report.md`
- `out/rolling_universe_micro_2025sep_2026mar_report.md`
- `out/rolling_universe_micro_2025sep_2026mar_monthly_summary.csv`

### 2. Event microstructure ranking improves the active sleeve

Combined no-lookahead rolling pipeline:

- event dataset: `out/event_microstructure_active_union_2025sep_2026mar.csv`
- folds: `155`

Result:

- active baseline: `+10.48 bps`
- active state sleeve: `+10.54 bps`
- active model sleeve: `+17.87 bps`
- positive in `155/155` folds

Main files:

- `out/combined_rolling_event_pipeline_2025sep_2026mar_report.md`
- `out/combined_rolling_event_pipeline_2025sep_2026mar_summary.csv`

Interpretation:

- there is real predictive structure in the public data
- rolling selection + event ranking is better than raw broad-universe trading

## What Failed

### 3. Conservative taker-style execution breaks the edge

Execution-aware backtest assumptions:

- `1` minute entry delay
- `15` minute hold
- `8` bps base round-trip fee
- extra spread/staleness/imbalance haircuts
- `4` max concurrent positions
- `12` max daily trades

Result:

- execution baseline: `-10.80 bps`
- execution state sleeve: `-10.35 bps`
- execution model sleeve: `-8.62 bps`

Main files:

- `out/execution_aware_rolling_backtest_2025sep_2026mar_report.md`
- `out/execution_aware_rolling_backtest_2025sep_2026mar_summary.csv`

Main reason:

- mean extra execution cost for model sleeve: only `1.67 bps`
- mean entry decay: `17.90 bps`

Interpretation:

- the first execution-aware result is negative mainly because `1m` delay is far too slow for this edge

### 4. Fast trade-based proxy still does not clear costs

Fast trade-fill proxy assumptions:

- signal clock: `50ms`
- entry/exit filled at next observed Bybit and Binance trade prints
- `8` bps base round-trip fee
- Bybit spread crossing penalty
- same portfolio caps

Result:

- fast baseline: `-11.30 bps`
- fast state sleeve: `-10.98 bps`
- fast model sleeve: `-7.04 bps`

Main files:

- `out/fast_trade_execution_rolling_backtest_50ms_2025sep_2026mar_report.md`
- `out/fast_trade_execution_rolling_backtest_50ms_2025sep_2026mar_summary.csv`

Interpretation:

- the model still helps versus baseline
- but next-trade execution is not a valid final fill model
- observed fill lag remained multi-second because the proxy waits for market prints

## Data Reality

Execution-grade data is asymmetric.

- Bybit has high-frequency orderbook data via `_orderbook.jsonl.gz`
- Binance in this datalake only has sparse `bookDepth` snapshots, not execution-grade quote data

So the correct production-style strategy design is:

- use Binance + Bybit as signal inputs
- execute on Bybit
- model Bybit entry and exit from the Bybit orderbook at `+50ms` or `+100ms`
- do not force a symmetric executable pair-trade model from data that does not support it

## Current Conclusion

Current evidence says:

- research alpha: yes
- rolling no-lookahead selection: yes
- event ranking improvement: yes
- deployable strategy from current taker-style backtests: no

That is not a dead end. It means the next step is narrower and better defined.

## Next Step

Build a Bybit-only execution simulator:

- signal from cross-exchange features at time `t`
- Bybit fill from orderbook at `t + 50ms` or `t + 100ms`
- Bybit exit from orderbook with explicit size and crossing assumptions
- portfolio/risk constraints on overlapping positions

This is the correct next test for the actual infra and the actual datalake.
