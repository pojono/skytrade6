# Master Findings: Codex Exp 3

This document consolidates the current state of research in `codex-exp-3`.

## Objective

Find a strategy from local `datalake` data that remains positive after realistic costs, with explicit focus on:

- not trusting prior repo headline PnL
- using raw data rebuilds
- surviving maker fee `0.04%` per side (`8 bps` round trip)
- validating across Binance and Bybit
- adding execution realism from trade-level data

## Data Scope

- Core rebuilt sample: `rebuilt_binance_samples_4h.csv`
  - Rows: `146,268`
  - Time range: `2025-07-15 00:05 UTC` to `2026-03-03 16:05 UTC`
- Train/test split:
  - Train: through `2025-12-31`
  - Test: from `2026-01-01`
- Selected strategy trades (main path): `63` symbols, `114` timestamps total, `27` test timestamps
- Trade-level execution study (full test basket):
  - `42` signal rows
  - `26` symbols
  - 2026 test-period signal windows only

## Research Arc

### 1) Rejected Idea

Fresh cross-exchange positioning-confirmation model (both venues must confirm) did not survive `8 bps` all-maker in train+test.

Conclusion:
- attractive hypothesis, but too weak after fee-aware validation

### 2) Surviving Signal Family

The robust signal is rare-event long-only continuation:

- extreme Binance top-trader long bias (`ls_z`)
- positive Binance taker-flow confirmation (`taker_z`)
- positive 4h momentum
- risk-on breadth filter
- 4-hour hold

This is:
- not spread arbitrage
- not mean reversion
- a selective continuation effect

### 3) Independent Rebuild

Removed dependence on earlier experiment artifacts by rebuilding Binance sample directly from raw `datalake/binance` in `codex-exp-3`.

### 4) Cross-Venue Repricing

Repriced selected entries on Bybit and averaged Binance/Bybit returns.

Conclusion:
- effect persists cross-venue
- less likely to be a Binance-only backtest artifact

### 5) Partial Funding Adjustment

Added Bybit-leg funding adjustment over each 4h hold.

Observed impact:
- very small on average (`~0.223 bps` in test)
- does not materially change the edge

### 6) Trade-Level Execution Validation

Downloaded and analyzed trade prints around actual signal times.

Top-line full-basket execution findings:
- Binance maker-fill proxy in first 60s: `100.0%`
- Bybit maker-fill proxy in first 60s: `90.5%`
- Binance 60s VWAP drift: `-1.39 bps`
- Bybit 60s VWAP drift: `-1.59 bps`

Interpretation:
- entry microstructure does not invalidate the strategy globally
- execution risk is concentrated in specific symbols

High-risk execution symbols identified:
- `LINKUSDT`, `ENAUSDT`, `AVAXUSDT`, `XLMUSDT`, `XRPUSDT` (plus low-fill edge cases like `PAXGUSDT`, `BARDUSDT`)

### 7) Execution-Aware Selection

Tested:

- hard blacklist
- soft execution penalty in ranking

Outcome:
- soft penalty is better default than hard blacklist
- preserves opportunity while downweighting bad execution names

## Current Main Strategy (Default)

Signal rule (every 4h at `HH:05 UTC`):

- `ls_z >= 2.0`
- `taker_z >= 0.5`
- `mom_4h > 0`
- `oi_med_3d >= 20,000,000`
- `breadth_mom >= 0.60`
- `median_ls_z >= 0.0`
- rank by execution-adjusted score
- take top `3`
- hold `4h`

Cost basis:
- `8 bps` maker round trip
- plus symbol-level execution drag estimate in stricter comparisons
- with partial Bybit funding adjustment available

## Winner/Loser Reverse-Engineering Attempts

Compared several approaches on same candidate set (`candidate_quality_dataset.csv`):

- `baseline_soft`
- `strict_threshold`
- `blended_rank`
- `empirical_winrate` (walk-forward prior win-rate style)

Findings:
- naive winner/loser classifier-style approach did **not** beat baseline
- blended heuristic rank underperformed baseline
- strict threshold produced strongest bps but very low frequency (thin sample risk)

## Strategy Mode Comparison

Three practical modes were compared:

1. `default_soft`
2. `high_conviction`
3. `hybrid_regime` (high-conviction only when breadth is very strong, else default-soft)

Recent test results:

- `high_conviction`: very high bps, but sparse and less stable month-to-month
- `default_soft`: broadest coverage, lower bps, stable default
- `hybrid_regime`: best compromise so far (higher bps than default with less sparsity risk than pure high-conviction)

## What Is Achieved

- A fee-surviving strategy family was identified and rebuilt from raw data
- Cross-venue validation passed
- Partial funding realism passed
- Trade-level entry realism passed at portfolio level
- Execution-aware ranking improved practical robustness
- A stronger regime mode (`hybrid_regime`) emerged as a promising tactical variant

## What Is Not Fully Solved Yet

- Full Binance funding adjustment in the same hold-window framework
- Full queue/partial-fill modeling (beyond simple first-minute proxies)
- Exit-side execution realism at same depth as entry-side checks
- Longer out-of-sample horizon beyond current 2026 window

## Current Practical Recommendation

If one mode is needed now:
- use `default_soft` as primary baseline

If you can support mode switching:
- add `hybrid_regime` as an optional high-quality mode

Avoid relying on:
- pure high-conviction alone (too sparse so far)
- hard symbol blacklists as the default mechanism

## Key Output Files

Core:

- `FINDINGS_codex_exp_3.md`
- `revalidate_exp2_on_bybit.py`
- `revalidated_exp2_portfolio.csv`
- `revalidated_exp2_portfolio_execution_soft.csv`
- `revalidated_exp2_portfolio_execution_filtered.csv`

Execution research:

- `TRADE_ENTRY_FINDINGS_full.md`
- `trade_entry_feasibility_full.csv`
- `trade_entry_feasibility_summary_full.csv`
- `EXECUTION_FILTER_FINDINGS.md`

Approach/mode comparisons:

- `SELECTION_APPROACH_FINDINGS.md`
- `selection_approach_comparison.csv`
- `HIGH_CONVICTION_FINDINGS.md`
- `high_conviction_portfolio.csv`
- `STRATEGY_MODE_FINDINGS.md`
- `strategy_mode_comparison.csv`
