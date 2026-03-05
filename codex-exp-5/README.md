# codex-exp-5

This experiment is isolated to `codex-exp-5` and reads source data from `../datalake`.

## Goal

Build a strategy that remains profitable after realistic fees:

- maker: `0.04%` per side (`8` bps round trip)
- taker: `0.10%` per side (`20` bps round trip)

## Current Strategy

`extreme_spread_crv.py` backtests a strict, short-horizon **cross-exchange mean reversion** rule on `CRVUSDT`:

- compare synchronized 1-minute closes on Binance and Bybit
- enter only when the spread is extremely stretched (`>= 32` bps)
- require confirming positioning and carry agreement:
  - long/short differential
  - open-interest differential
  - basis / premium differential
- cap execution to the first `3` qualifying trades per day
- hold for exactly `1` minute

Default command:

```bash
python3 codex-exp-5/extreme_spread_crv.py
```

## Current Result

On the current local datalake snapshot:

- train months (`2025-11` to `2026-01`) average net after taker fees: `+2.0451` bps/trade
- test months (`2026-02` to `2026-03`) average net after taker fees: `+9.8842` bps/trade
- test win rate after taker fees: `62.22%`

Generated outputs:

- `out/trade_log.csv`
- `out/report.md`

## Scanner

`scan_extreme_spread.py` applies the same strict rule to a symbol list and ranks survivors.

Example:

```bash
python3 codex-exp-5/scan_extreme_spread.py --symbols CRVUSDT,GALAUSDT,CAKEUSDT,ETHFIUSDT,AEROUSDT --workers 8
```

The scanner now prints live per-symbol progress, elapsed time, and rolling ETA while it runs.

Current verified scan result on that tested shortlist:

- only `CRVUSDT` passed the minimum trade and split filters while staying positive after the `20` bps taker round trip

Scanner outputs:

- `out/scan_leaderboard.csv`
- `out/scan_report.md`

## Microstructure Follow-Up

`crv_microstructure_audit.py` audits recent `CRVUSDT` triggers on the days where local Binance trades, Bybit trades, and Bybit order book data all exist.

`crv_micro_gated_rule.py` then applies a tighter execution-aware filter to that recent sample.

Default command:

```bash
python3 codex-exp-5/crv_micro_gated_rule.py
```

Current recent-sample result:

- baseline audited sample: `90` triggers, `+9.8842` bps average after taker fees
- filtered sample (`score >= 18`, Bybit book spread `<= 8` bps): `14` triggers, `+21.5115` bps average after taker fees

## Multi-Symbol Micro Scan

`multi_symbol_micro_scan.py` scans all symbols with local two-exchange microstructure coverage and searches for execution-aware gates that produce a positive basket with at least two positive symbols.

Default command:

```bash
python3 codex-exp-5/multi_symbol_micro_scan.py --workers 6
```

Current best recent-sample basket:

- gate: `score >= 20`, Bybit trigger book spread `<= 10` bps
- symbols: `CRVUSDT`, `GALAUSDT`
- combined result: `18` trades, `+13.0217` bps average after taker fees, `77.78%` win rate

## Broader Trade-Flow Scan

`multi_symbol_trade_flow_scan.py` expands beyond the order-book-limited subset by using only overlapping Binance + Bybit trade flow on symbols with enough shared coverage.

Default command:

```bash
python3 codex-exp-5/multi_symbol_trade_flow_scan.py --workers 8
```

Current broadest useful result:

- `16` symbols scanned (with at least `5` shared trade days)
- no basket with `3+` individually positive symbols was found
- best broader positive portfolio:
  - `CRVUSDT`, `GALAUSDT`, `GUNUSDT`
  - gate: `score >= 18`
  - `82` trades
  - `+6.9674` bps average after taker fees

## Anti-Overfit Validator

`anti_overfit_validate.py` enforces a hard anti-overfit bar:

- temporal train / validation / holdout split checks
- minimum per-symbol trades per split
- minimum number of positive symbols
- positive portfolio average on all splits
- concentration cap on holdout top-symbol contribution

Default run:

```bash
python3 codex-exp-5/anti_overfit_validate.py
```

Current status:

- no configuration passes the strict anti-overfit bar on current local data
- even a relaxed diagnostic run still produced no passing configuration

## Walk-Forward Robust Strategy

`robust_walkforward_strategy.py` builds a daily walk-forward strategy:

- gate selection each day uses only prior data
- symbol-breadth and concentration constraints are enforced in gate selection
- selected gate is applied to next day only

Default run:

```bash
python3 codex-exp-5/robust_walkforward_strategy.py
```

Current result on local data:

- days traded: `20`
- total trades: `28`
- avg net after taker fees: `+5.7083` bps
- win rate: `57.14%`
- symbol mix: `CRVUSDT`, `GALAUSDT`, `GUNUSDT`, `INITUSDT`

## Data Note

The checked-in `datalake` currently contains `binance/` and `bybit/` only. `datalake/okx/` is not present in this workspace, so this strategy uses the two exchanges with available local coverage.
