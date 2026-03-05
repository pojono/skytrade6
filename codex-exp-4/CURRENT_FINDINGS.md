# Current Findings (codex-exp-4)

## Objective

Build a profitable strategy using Binance + Bybit data that survives maker-fee assumptions (modeled as `8 bps` round-trip in most tests).

## Data Used

Primary inputs from `../datalake`:

- Binance: `trades`, `bookDepth`, plus baseline 1m/metrics where needed
- Bybit: `trades`, `orderbook` (`ob200`), plus baseline 1m/positioning where needed

Core symbols explored:

- `BTCUSDT`
- `SOLUSDT`
- `DOGEUSDT`
- `1000PEPEUSDT`
- `1000BONKUSDT`
- `GUNUSDT`
- `INITUSDT`

## What Was Implemented

Key scripts in this folder:

- `analyze_premove_signals.py`: bar/positioning pre-move signals
- `microstructure_edge_scan.py`: per-second trades + Binance depth scans
- `bybit_orderbook_edge_scan.py`: per-second Bybit orderbook pressure scans
- `event_regime_edge_scan.py`: event + regime filtered rules
- `walkforward_event_regime.py`: rolling walk-forward rule validation
- `cross_exchange_leadlag_model.py`: direct venue lead/lag model
- `quote_safety_filter.py`: market-making danger suppression filter
- `rare_event_precision_filter.py`: precision-first dangerous-event classifier
- `rare_event_directional_strategy.py`: rare-event directional passive-entry simulator

## Main Results

### 1) Broad directional alpha attempts were not robust

- Single-factor and conjunction rule scans: generally negative after fees out-of-sample.
- Walk-forward event/regime rules (`2026-02-24` to `2026-03-03`):
  - `BTCUSDT`: mean test about `-5.36 bps`, `0/5` positive folds.
  - `SOLUSDT`: mean test about `-8.54 bps`, `0/5` positive folds.
- Cross-exchange lead/lag model (same window): negative across all folds.

Conclusion: simple directional strategies are not robust enough in current form.

### 2) Quote-safety filter signal exists but is weak

- On majors and thin symbols, filter usually blocked around `30%` of time.
- Reduction in unsafe windows was generally small (often near zero on thin names).

Conclusion: useful as a weak risk signal, not a standalone edge.

### 3) Rare-event precision path is the strongest signal shape

- On broader pooled sample (`BTCUSDT`, `SOLUSDT`, `DOGEUSDT`, `1000PEPEUSDT`, `GUNUSDT`, 10 days):
  - events: `31,570`
  - base danger rate: `25.0%`
  - mean test precision: `34.2%`
  - mean test lift: `1.23x`

Interpretation: shock-event danger detection is real and stable enough to improve selection quality.

### 4) First profitable candidate found under optimistic execution assumptions

From `rare_event_directional_strategy.py`:

- Config (`best_v1`):
  - symbols: `BTCUSDT`, `SOLUSDT`, `GUNUSDT`
  - `entry-z=3.5`
  - `move-bps=15`
- Result:
  - folds: `3`
  - mean test expected net: `+7.17 bps` per signal
  - positive folds: `3/3`
  - mean precision: `~45%`
  - mean trades/day: `~13`

Saved artifacts:

- `out/rare_event_directional_dataset_best_v1.csv`
- `out/rare_event_directional_folds_best_v1.csv`
- `out/rare_event_directional_summary_best_v1.csv`

## Critical Caveat

The profitable candidate is likely fragile:

- small fold count (`3`)
- low trade count
- highly selective symbol subset
- sensitive to execution assumptions

When execution realism was tightened (larger effective queue, partial taker capture, fill caps, slippage/adverse drift), the strategy often dropped to zero usable folds or negative expectancy.

## Current State

Most accurate summary:

- There is a real rare-event microstructure signal.
- It can be turned into positive expectancy under optimistic assumptions.
- It is not yet robust enough under stricter execution assumptions to claim production-ready profitability.

## Recommended Next Steps (Execution-Focused)

1. Freeze candidate parameters and validate on strict holdout dates/symbols (no retuning).
2. Optimize directly under realistic execution assumptions, not optimistic proxy first.
3. Enforce minimum fold/trade counts in model selection to reduce overfit risk.
4. Move from pooled model to symbol-specific models for top contributors.
5. Compare two deployment paths:
   - rare-event directional passive-entry strategy
   - rare-event signal as one-sided quoting/risk throttle layer

## Output Files to Check First

- `out/rare_event_directional_summary_best_v1.csv`
- `out/rare_event_directional_folds_best_v1.csv`
- `out/rare_event_precision_summary.csv`
- `out/walkforward_event_regime_summary.csv`
- `out/cross_exchange_leadlag_summary.csv`
