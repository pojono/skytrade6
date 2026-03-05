# Current Findings (Expansion Update) — 2026-03-04

## Scope of this update
- Expanded Binance raw orderbook/trade coverage and converted new symbols to parquet.
- Re-ran:
  - `codex-exp-2/delayed_confirmation_strategy_case_study.py`
  - `codex-exp-2/orderbook_walkforward_30s_60m.py`

Newly added in this pass (full Jul 1, 2025 -> Feb 28, 2026 depth+aggTrades):
- `AXSUSDT`
- `AIXBTUSDT`
- `BIOUSDT`
- `KAITOUSDT`

Also finalized previously pending conversions:
- `RIVERUSDT`, `SAHARAUSDT`, `HUSDT`, `STRKUSDT`, `LDOUSDT`

## Data coverage status
- Symbols present in `samples_4h.csv`: **116**
- Symbols with Binance `book_depth` parquet available: **61**
- Coverage improved this pass: **56 -> 61**

## Most reliable result (execution-aware, fee-aware)
From `orderbook_walkforward_30s_60m_results.csv`:

- Base test (no gate): about **-30.64 bps**
- Gated test (best row, $10k notional): **+28.06 bps**
- Improvement vs base: **+58.70 bps**
- Fee model already included in simulation (`maker 0.04%`, `taker 0.10%` assumptions in strategy scripts)

Best gate (selected on train, evaluated on test):
- `ret_30s_bps >= 1.4718`
- `buy_share_30s >= 0.7006`

Kept test trades:
- **10 trades**
- Symbols: `APTUSDT,BCHUSDT,ETHUSDT,ONDOUSDT,SOLUSDT,SUIUSDT,UNIUSDT,XMRUSDT,ZECUSDT`

## Classifier path status (delayed confirmation)
From `delayed_confirmation_grid.csv`:
- Best out-of-sample row still has **negative test avg bps** (about **-39.38 bps**).
- Conclusion: delayed-confirmation classifier is still **not production-ready**.

## Production readiness assessment
Current answer: **not ready for production yet**.

Why:
- Positive execution-aware alpha exists, but concentrated in very few trades (**10**) and limited symbols.
- Coverage is still partial (**61/116** symbols).
- Robustness is not yet demonstrated across wider symbol universe and longer out-of-sample windows.
- Classifier branch remains negative OOS.

What is promising:
- Orderbook-gated branch remains positive after fees in current walk-forward.
- Slippage impact in this run is low enough not to erase edge at tested notionals.

## Bybit vs Binance execution comparison (new)
Cross-venue report:
- `codex-exp-2/FINDINGS_orderbook_cross_venue_compare_30s_60m.md`
- `codex-exp-2/orderbook_cross_venue_compare_30s_60m_results.csv`

Method:
- Same signal timestamps and same Binance agg-trade reference prices.
- Only execution depth source changed: Binance `book_depth` vs Bybit `orderbook/bybit_futures`.
- Same 20 bps round-trip fee model.
- Same gate as Binance walk-forward (`ret_30s_bps >= 1.471836`, `buy_share_30s >= 0.700624`).

Current overlap achieved (latest run):
- 145 feature rows across 13 symbols:
  - `1000PEPEUSDT,BNBUSDT,BTCUSDT,ETHUSDT,LINKUSDT,ONDOUSDT,PENGUUSDT,SOLUSDT,SUIUSDT,TAOUSDT,TRUMPUSDT,WIFUSDT,XRPUSDT`
  - (test subset symbols depend on notional feasibility)

Key result (test overlap subset, latest):
- At $10k notional:
  - Binance avg: `-26.58 bps`
  - Bybit avg: `-29.22 bps`
  - Bybit minus Binance: `-2.65 bps`
- At $50k notional:
  - Binance avg: `-21.68 bps`
  - Bybit avg: `-24.75 bps`
  - Bybit minus Binance: `-3.08 bps`
- At $100k notional:
  - Binance avg: `-30.58 bps`
  - Bybit avg: `-33.53 bps`
  - Bybit minus Binance: `-2.95 bps`

Interpretation:
- On current overlap subset, Bybit depth simulation is slightly worse than Binance (about 1 to 5 bps depending on slice).
- This does **not** invalidate Binance-only positive gated result; it shows venue choice materially affects net edge.
- We need wider Bybit overlap before making a final venue-level ranking.

## Next actions (priority)
1. Continue filling `bookDepth + aggTrades` for remaining high-frequency missing symbols, convert to parquet.
2. Re-run walk-forward after each +5 to +10 symbols and track metric stability:
   - `test_kept_avg_bps`
   - `test_kept_rows`
   - symbol concentration
3. Add stricter stability checks before any production decision:
   - minimum kept trades threshold
   - max per-symbol weight
   - rolling window degradation checks
