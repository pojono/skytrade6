# Current Findings (March 4, 2026)

## Scope and Data
- Work isolated under `codex-exp-2`.
- Strategy research is based on raw datalake-derived Binance data and rebuilt backtests; prior repo PnL claims were not trusted.
- Fees modeled explicitly at round-trip taker cost (`20 bps`) unless stated otherwise.
- Later stages include causal entry logic and raw orderbook-based execution simulation.

## High-Level Summary
- The original broad 4h continuation idea does **not** look production-safe.
- Static symbol whitelists and tabular per-trade ML classifiers overfit and did not improve holdout robustness.
- Execution-aware work changed the direction: the most credible variant is now
  - delayed entry (`+30s`),
  - short follow-through confirmation,
  - shorter hold (`60m`),
  - with orderbook-based fill simulation.
- On the current execution-covered sample, this variant is positive after full fees, but still small-sample.

## What Failed (Important Negative Findings)
- Broad symmetric long/short variants failed after realistic fees.
- Symbol concentration is real; many symbols are weak/negative while a small subset drives returns.
- Train-only static symbol whitelists did not transfer to holdout.
- Per-trade classifiers (v1 and richer v2 feature sets) looked strong in CV but failed on holdout.
- Simple execution-threshold filters (spread/depth/imbalance snapshots) were unstable and often harmful.
- Immediate 5s entry-cost filtering did not explain 4h losers and did not rescue performance.
- Rolling earned-symbol eligibility filters did not rescue the base variant.

## Key Directional Shift
- Delayed confirmation plus shorter holding period works better than instant-entry 4h holds.
- The signal appears closer to short-horizon continuation than long-horizon continuation.
- The first 30s to 5m post-signal path carries useful information; 4h outcomes are less stable.

## Latest Execution-Aware Result (Causal, Raw Orderbook)
Using `orderbook_walkforward_30s_60m.py` with:
- Entry decision at `signal + 30s` (causal),
- Exit at `entry + 60m`,
- Fills from actual `book_depth` snapshots,
- reference from last `agg_trade` before execution timestamp,
- full `20 bps` round-trip fees.

Latest run (after adding ETC/ZEC/BCH/ATOM/KITE/ICP; `HBAR` still processing):
- Execution feature universe: `23` symbols.
- Test rows: `41`.
- Best train-selected gate:
  - `ret_30s_bps >= 4.2823`
  - `buy_share_30s >= 0.5`

Performance:
- `$10k` notional:
  - Unfiltered test: `-30.64 bps`
  - Filtered test: `+33.59 bps`
  - Improvement: `+64.23 bps`
- `$50k` notional:
  - Filtered test: `+33.59 bps`
- `$100k` notional:
  - Filtered test: `+33.57 bps`

Kept test trades:
- `8` trades across `8` symbols:
  - `APTUSDT`, `BCHUSDT`, `BNBUSDT`, `ETHUSDT`, `SOLUSDT`, `SUIUSDT`, `UNIUSDT`, `XMRUSDT`.
- Largest positive addition in latest pass: `BCHUSDT` (kept test trade around `+100.5 bps`).

## Interpretation
- This is the strongest result so far under explicit fees plus orderbook-based execution simulation.
- It is now meaningfully positive in the current covered sample.
- It is still **not production-ready** because:
  - kept-trade count is small (`8`),
  - result may be sensitive to event clustering,
  - we still need broader coverage and portfolio-level constraints.

## Coverage Status
- Triggered broad universe (current research mask): `68` symbols.
- Currently covered with local parquet `agg_trades_futures + book_depth`: `41` symbols.
- Included in latest orderbook feature sample: `23` symbols.
- `HBARUSDT` conversion is still running from an earlier broad-source pull; it is not yet included in the latest rerun.

## Current Best Candidate Strategy (Research Candidate, Not Deployment)
- Signal family: long-side continuation under risk-on conditions.
- Entry logic: wait 30s after signal, then require strong short-horizon continuation.
- Gate:
  - `ret_30s_bps >= 4.2823`
  - `buy_share_30s >= 0.5`
- Exit: 60 minutes after entry.
- Execution model: orderbook-based fill approximation + full taker fees.

## Risks Still Open
- Small effective sample after gating.
- Potential selection/coverage bias while universe expansion is incomplete.
- No portfolio overlap/capital contention model in final decision layer yet.
- No missed-fill/latency stress test beyond current approximation.

## Next Steps (Highest Value)
1. Finish `HBARUSDT` parquet conversion and rerun both delayed + orderbook studies.
2. Continue narrow Binance-only expansion for next missing names (`FFUSDT`, `FILUSDT`, `RIVERUSDT`, `DASHUSDT`, etc.).
3. Add portfolio-level simulation:
   - overlapping signals,
   - capital constraints,
   - position limits,
   - execution queueing.
4. Stress test the gate with adverse fill assumptions and stricter slippage buffers.

## Related Files
- `FINDINGS_codex_exp_2.md`
- `FINDINGS_symbol_classification.md`
- `FINDINGS_symbol_whitelist.md`
- `FINDINGS_per_trade_classifier.md`
- `FINDINGS_per_trade_classifier_v2.md`
- `FINDINGS_execution_case_study.md`
- `FINDINGS_entry_cost_case_study.md`
- `FINDINGS_post_entry_path_case_study.md`
- `FINDINGS_delayed_confirmation_case_study.md`
- `FINDINGS_orderbook_walkforward_30s_60m.md`
