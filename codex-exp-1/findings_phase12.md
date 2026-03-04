# Findings Phase 12: Downside Analysis for the Frozen 25% Filtered Variant

This phase freezes the current best moderate-size deployment candidate:

- replay-optimized pre-trade filter
- `25%` per-trade allocation
- size-aware slippage enabled

Source run:

- `out/paper_report_v3_replayopt_sized25.md`

Downside analysis tool:

- `analyze_downside.py`

Primary outputs:

- `out/downside_report_v3_replayopt_sized25.md`
- `out/downside_weekly_v3_replayopt_sized25.csv`
- `out/downside_monthly_v3_replayopt_sized25.csv`

## Strategy Variant Under Review

Performance summary from the replay:

- `897` fills
- final capital: `$107,638.99`
- total PnL: `$7,638.99`
- win rate: `53.29%`
- average net edge: `3.2849 bps`

## Realized Drawdown

The downside pass reconstructs realized equity from the fill stream in exit order.

Results:

- Max drawdown: `$270.08`
- Max drawdown percent: `0.25%`
- Peak before drawdown: `$106,567.18`
- Trough at max drawdown: `$106,297.10`
- Drawdown window:
  - start: `2026-02-21 02:08:00 UTC`
  - end: `2026-02-23 00:38:00 UTC`

Interpretation:

- In this historical sample, the realized equity curve is unusually smooth.
- That is encouraging, but it should be treated cautiously because:
  - the strategy is single-slot
  - fills are modeled, not exchange-executed
  - unrealized intrabar excursions are not captured

So this is a realized-PnL drawdown, not a worst-path market-to-market drawdown.

## Weekly Stability

- Weeks observed: `31`
- Positive weeks: `26`
- Negative weeks: `5`
- Best week: `2026-W09` with `$1,092.87`
- Worst week: `2025-W34` with `-$102.94`

Worst weeks:

- `2025-W34`: `-$102.94`
- `2025-W43`: `-$78.71`
- `2025-W32`: `-$56.13`
- `2025-W33`: `-$51.00`
- `2025-W42`: `-$13.60`

This is consistent with a strategy that has many small realized outcomes rather than occasional large losses in the current sample.

## Monthly Stability

- Months observed: `8`
- Positive months: `7`
- Negative months: `1`

Monthly PnL:

- `2025-08`: `-$181.75`
- `2025-09`: `$393.21`
- `2025-10`: `$160.67`
- `2025-11`: `$1,027.00`
- `2025-12`: `$1,006.99`
- `2026-01`: `$1,273.94`
- `2026-02`: `$3,704.56`
- `2026-03`: `$254.43` (partial month)

Interpretation:

- The variant is not monotonic, but the loss months are limited in this sample.
- Performance is concentrated in `2026-02`, so the strategy is still somewhat regime-sensitive.

## Practical Conclusion

For the current frozen `25%` filtered variant:

- realized downside is mild in-sample
- weekly and monthly stability are better than the raw win rate might imply
- the strategy still appears operationally viable under the current model

But the correct risk caveat is:

- the smoothness may be partly an artifact of modeled one-minute exits and a single-slot book
- this is not yet a substitute for live paper trading against real venue conditions

So the current evidence supports this as the best moderate-size research candidate, not as a production-ready guarantee.
