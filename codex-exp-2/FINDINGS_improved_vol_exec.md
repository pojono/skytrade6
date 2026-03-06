# Improvement Attempt: Vol-Target + Execution Gate

## Scope
- Data source: raw `datalake/binance` only
- Window: `2025-01-01` to `2026-03-04` available; strategy warmup starts trading later
- Candidate rows (top-5 pre-gate): `264`

## Gate (chosen on train only)
- Max staleness: `40s`
- Depth quantile: `0.2` -> min depth_1pct `1,597,752` USD
- Flow quantile: `0.3` -> min flow_30m `1,160,931` USD

## Full-Period Equity from $1,000
- Baseline (equal-weight top-3), mixed 14 bps: `$1,522.27`
- Improved (vol-target + exec gate), mixed 14 bps: `$1,158.74`
- Baseline all-taker 20 bps: `$1,387.36`
- Improved all-taker 20 bps: `$1,107.15`

## Test-Only (2026+)
- Baseline test timestamps: `29`
- Improved test timestamps: `12`
- Baseline test avg net bps (mixed): `17.64`
- Improved test avg net bps (mixed): `-25.49`

## Variant Sweep (what actually helps)
- `baseline_top5_equal` is best in this sweep:
  - Final equity mixed: `$1,539.44`
  - Final equity all-taker: `$1,403.03`
  - Test avg net bps (mixed): `17.26`
- `baseline_top3_equal` (current baseline):
  - Final equity mixed: `$1,522.27`
  - Final equity all-taker: `$1,387.36`
  - Test avg net bps (mixed): `17.64`
- `vol_target_top3/top5` both underperform baseline.
- All tested depth+flow gating variants underperform strongly due reduced opportunity set and fragile test sample counts.

## Recommendation
- Do **not** enable the tested execution gate in production yet.
- Keep execution checks as monitoring features, not hard filters, until we have denser book/trade coverage and more test months.
- If you want a deployable change now, switch portfolio construction from top-3 to **top-5 equal-weight** and keep existing signal thresholds.

## Files
- `improved_vol_exec_trades.csv`
- `improved_vol_exec_monthly.csv`
- `improved_vol_exec_variant_sweep.csv`
