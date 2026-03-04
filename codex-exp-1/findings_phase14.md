# Findings Phase 14: Forward-Only Daily Paper Runner

This phase implements the correct next validation step for the current best candidate:

- stop re-scoring the same historical window
- freeze the strategy spec
- append only new days into a rolling paper log

## Implemented Runner

New script:

- `rolling_paper_replayopt_25.py`

This runner is dedicated to the frozen best current candidate:

- input: `out/candidate_trades_v3_replayopt.csv`
- allocation: `25%`
- one open position
- one open per symbol
- selector mode `spread`
- daily cap `3`
- daily loss stop `1%`
- monthly loss stop `3%`
- size-aware slippage enabled

Output log:

- `out/rolling_paper_log_replayopt_25.csv`

Each row includes:

- day
- month
- equity at start of day
- realized day PnL
- realized day return
- month-to-date realized PnL
- equity at end of day

## Forward-Only Behavior

The runner reads the existing log and:

- skips days already written
- starts new simulation days from the last logged ending equity
- preserves month-to-date realized PnL needed for the monthly risk stop

That makes it safe to rerun as new data arrives.

## Validation

First run on the current dataset:

- appended days: `194`
- ending equity: `$107,638.99`

Second run immediately after:

- appended days: `0`
- ending equity: `$107,638.99`

This confirms the runner is idempotent on unchanged input.

## Current Log State

Latest rows in the rolling log:

- `2026-02-26`: equity end `$106,938.45`
- `2026-02-27`: equity end `$107,127.96`
- `2026-02-28`: equity end `$107,384.55`
- `2026-03-01`: equity end `$107,468.72`
- `2026-03-02`: equity end `$107,638.99`

So the log is ready for true forward appends once new trade rows appear in the frozen filtered input.

## Why This Matters

This is the transition point from research to validation:

- research asks whether a historical edge exists
- forward-only paper logging asks whether the edge persists without reusing the same sample

This is the correct way to reduce incremental overfitting risk from here.
