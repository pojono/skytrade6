# codex-exp-5 Findings

## Objective

Find a strategy that remains profitable after:

- maker fees: `0.04%` per side (`8` bps round trip)
- taker fees: `0.10%` per side (`20` bps round trip)

All work in this folder uses local data from `../datalake`.

## What Was Tested

The current research tested a strict **cross-exchange 1-minute mean-reversion** idea between Binance and Bybit:

- compare synchronized 1-minute closes
- detect extreme spread dislocations
- fade the spread (short rich venue, long cheap venue)
- require positioning and carry confirmation:
  - long/short differential
  - open-interest differential
  - basis / premium differential
- hold for exactly one minute
- cap trades per symbol per day

This logic is implemented in:

- `extreme_spread_crv.py`
- `scan_extreme_spread.py`

## Best Validated Result

The strongest validated result is `CRVUSDT`.

Using the current strict rule:

- minimum spread: `32` bps
- minimum score: `14`
- daily cap: `3`
- full taker round trip: `20` bps

Result from `out/report.md`:

- train months (`2025-11` to `2026-01`): `+2.0451` bps/trade after taker fees
- test months (`2026-02` to `2026-03`): `+9.8842` bps/trade after taker fees
- test win rate: `62.22%`

This is the clearest evidence of a currently usable edge in this workspace.

## Full-Universe Scan

A full scan was run across `116` shared Binance+Bybit symbols using the same strict rule.

Summary from `out/scan_report.md`:

- `34` symbols passed minimum overlap / trade-count filters
- only `2` symbols were positive after full `20` bps taker costs

Profitable symbols after taker fees:

1. `CRVUSDT`
   - train: `104` trades, `+2.0451` bps
   - test: `90` trades, `+9.8842` bps
2. `KAVAUSDT`
   - train: `27` trades, `+0.0903` bps
   - test: `18` trades, `+2.4300` bps

Interpretation:

- `CRVUSDT` is the only strong result.
- `KAVAUSDT` is positive but lower confidence because the sample is thin.
- the edge is not broad across the universe under current assumptions.

## Important Near-Miss

`SIRENUSDT` was the closest non-profitable symbol:

- train: `53` trades, `+17.5368` bps
- test: `88` trades, `-0.4487` bps

This matters because it is close to break-even after taker fees and may become positive with better filtering or execution.

## Current Interpretation

The current evidence suggests:

- the edge is **real but highly concentrated**
- it is not a general “all-symbol” mean-reversion effect
- it likely appears in specific microstructure / liquidity regimes
- broad scanning alone will not produce a robust portfolio without tighter conditioning

The working hypothesis is that some symbols temporarily exhibit slower cross-exchange inventory balancing, creating short-lived overreactions that revert within a minute.

## Constraints

- current validated work uses `binance/` and `bybit/` only
- this workspace does not currently contain `datalake/okx/`
- current backtests are signal-validity tests, not full execution-quality simulations
- actual fill quality may differ from 1-minute close proxies

## Recommended Next Step

The highest-value next step is:

1. decompose `CRVUSDT` and `KAVAUSDT` by regime
2. identify when the edge is strongest
3. test microstructure-level execution around the trigger minute

This is more likely to improve edge quality than broadening the same scan across more symbols without changing the model.

## Microstructure Pivot Result

The strongest immediate follow-up was a **CRVUSDT microstructure audit** using local Binance trades, Bybit trades, and Bybit order book data on the audited recent days.

Result from `out/crv_micro_report.md`:

- `90` recent CRV triggers were audited
- avg net after full `20` bps taker fees stayed at `+9.8842` bps
- positive trigger share stayed at `62.22%`

Most useful observations:

- higher-score trades (`score >= 20`) improved to `+13.1726` bps
- tighter Bybit top-of-book conditions (`<= 12` bps) improved to `+11.1532` bps
- larger displayed top depth did **not** improve results
- trades with the strongest obvious pre-entry flow alignment were **not** the best subset

Interpretation:

- the promising continuation is not “more symbols”
- the promising continuation is a **stricter CRV regime filter**
- specifically: keep the existing signal, then add a microstructure gate that prefers tighter books and avoids chasing the most obvious ongoing burst

## Current Best Pivot

A stricter recent-sample CRV rule was tested in `crv_micro_gated_rule.py` with:

- `score >= 18`
- Bybit trigger book spread `<= 8` bps

Result from `out/crv_micro_gated_report.md`:

- baseline recent sample: `90` triggers, `+9.8842` bps average after taker fees
- filtered recent sample: `14` triggers, `+21.5115` bps average after taker fees
- filtered recent win rate: `85.71%`

Interpretation:

- this does not yet prove a full-history improvement
- but it is the strongest current direction in this folder
- the research should now focus on turning this from a recent-sample microstructure filter into a broader validated CRV execution rule

## First Multi-Symbol Version

To avoid relying on one coin, a multi-symbol microstructure scan was run across all symbols with local two-exchange microstructure coverage:

- `1000BONKUSDT`
- `1000PEPEUSDT`
- `BTCUSDT`
- `CRVUSDT`
- `DOGEUSDT`
- `GALAUSDT`
- `GUNUSDT`
- `INITUSDT`
- `SEIUSDT`
- `SOLUSDT`

Result from `out/multi_symbol_micro_scan.md`:

- the best currently found multi-symbol gate was:
  - `score >= 20`
  - Bybit trigger book spread `<= 10` bps
  - no extra flow filter
- this produced a **2-symbol** recent-sample basket:
  - `CRVUSDT`
  - `GALAUSDT`
- combined result:
  - `18` trades
  - `+13.0217` bps average after full `20` bps taker fees
  - `77.78%` win rate

Interpretation:

- this is the first result in this folder that is not purely single-coin
- it is still a small recent-sample basket, not a broad portfolio
- but it is the most credible path so far toward a non-single-asset strategy

## Broader Trade-Flow Expansion

To expand beyond the `10` symbol order-book-limited universe, a second scan was run on the broader set of symbols that already have overlapping Binance + Bybit `trades.csv` coverage and enough overlap to be worth testing (`16` symbols with at least `5` shared days).

Result from `out/multi_symbol_trade_flow_scan.md`:

- with a requirement of at least `3` individually positive symbols, **no** qualifying portfolio was found
- with a relaxed requirement of at least `2` positive symbols, the best basket was still dominated by:
  - `CRVUSDT`
  - `GALAUSDT`
- best 2-positive broad trade-flow basket:
  - gate: `score >= 20`, positive combined flow
  - `22` trades
  - `+12.3003` bps average after full `20` bps taker fees
- there is also a broader but weaker `3`-symbol portfolio:
  - `CRVUSDT`, `GALAUSDT`, `GUNUSDT`
  - gate: `score >= 18`, no extra flow filter
  - `82` trades
  - `+6.9674` bps average after taker fees
  - only `2` of those `3` symbols are individually positive

Interpretation:

- the research now clearly shows the edge can be tested beyond `10` symbols
- but the edge still does **not** generalize into a robust `3+` positive-symbol basket on current local data
- the strongest current broadening step is a small basket centered on `CRVUSDT` and `GALAUSDT`, with `GUNUSDT` as a weaker add-on if the goal is more trade count rather than cleaner per-trade edge

## Anti-Overfit Validation Outcome

A dedicated anti-overfit validator was added (`anti_overfit_validate.py`) with:

- temporal day-sequence train/validation/holdout splits
- minimum per-symbol split trade counts
- required number of positive symbols
- required positive portfolio performance on all splits
- concentration cap on holdout top-symbol contribution

Current result:

- strict bar (`>=5` positive symbols): no passing configuration
- relaxed diagnostic bar (`>=2` positive symbols, lower holdout trade and concentration requirements): still no passing configuration

Interpretation:

- current local data does not support claiming a robust, diversified strategy
- current edges are still too concentrated and/or too unstable across temporal splits
- this is exactly the failure mode the anti-overfit framework is designed to expose

## Best Current Deployable Candidate (Still Limited)

A practical walk-forward strategy was built in `robust_walkforward_strategy.py` to reduce overfit risk:

- day-by-day gate re-selection using only prior data
- breadth and concentration constraints in selection
- next-day-only execution

Result from `out/robust_walkforward_report.md`:

- traded days: `20`
- total trades: `28`
- avg net after taker fees: `+5.7083` bps
- win rate: `57.14%`

Caveat:

- this is currently the best implementable candidate in this folder
- it is safer than a static fit, but still limited by sparse symbol coverage and concentration risk
