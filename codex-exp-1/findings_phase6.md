# Phase 6 Findings

## Goal

Add portfolio-level risk controls and a rolling forward paper-trading log.

This phase adds:

1. Daily loss stops
2. Monthly loss stops
3. An append-style rolling paper log by day

## New Tools

Added:

- `codex-exp-1/rolling_paper_candidate.py`

Extended:

- `codex-exp-1/paper_trade_candidate.py`

New controls in the paper simulator:

- `--daily-loss-stop-pct`
- `--monthly-loss-stop-pct`

## Live-Style Portfolio Configuration

Frozen basket:

1. `CRVUSDT`
2. `GALAUSDT`
3. `SEIUSDT`

Execution rules:

- max open positions: `1`
- max open per symbol: `1`
- max symbol allocation: `10%`
- daily cap: `3` per symbol
- selector: `spread`
- moderate dynamic slippage:
  - fixed extra: `1 bps`
  - spread coefficient: `0.10`
  - velocity coefficient: `0.05`

Risk stops:

- daily loss stop: `1%`
- monthly loss stop: `3%`

## Paper Result With Risk Stops

Report:

- `codex-exp-1/out/paper_report_v3_live.md`

Result:

- filled trades: `1,421`
- final capital: `$105,358.62`
- total PnL: `$5,358.62`
- average net edge: `3.6742 bps`
- win rate: `58.69%`

## Key Observation

These stop thresholds did **not** bind in the tested history.

The result is effectively unchanged from the prior conservative live-style run.

That implies:

- the current strategy does not hit large enough daily or monthly drawdowns in this sample to trigger the chosen kill switches
- the stops are reasonable as safety rails, but they are not yet the main driver of the result

## Rolling Daily Log

Output:

- `codex-exp-1/out/rolling_paper_log_v3.csv`

This produces a day-by-day record with:

- start equity
- day PnL
- day return
- month-to-date realized PnL
- end equity

The current run appended `209` daily rows and finished at:

- ending equity: `$105,358.62`

## Practical Conclusion

The current strategy now has:

1. fixed symbols
2. fixed filters
3. no-peek validation
4. explicit selector logic
5. explicit exposure caps
6. explicit risk-stop hooks
7. a rolling daily paper-trading log

That is enough structure to treat it as a serious paper-trading candidate.
