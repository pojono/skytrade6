# Monthly Breakdown and Equity Curve (Best Config)

## Setup
- Initial equity: `$1,000.00`
- Mixed fee assumption (production): `14.0 bps` round-trip
- All-taker reference: `20.0 bps` round-trip
- Config: `ls_z>=2.0`, `taker_z>=0.5`, `oi_med_3d>=20000000`, `top_n=5`, `breadth>=0.6`, `median_ls>=0.0`

## Trade Counts
- Total decision timestamps: `155`
- Train (<= 2025-12-31): `126`
- Test (>= 2026-01-01): `29`

## Equity Results
- Final equity (mixed 14 bps): `$1,539.44`
- Final equity (all-taker 20 bps): `$1,403.03`

## Test-Only Snapshot
- Test compounded return (mixed): `4.75%`
- Test compounded return (all-taker): `2.95%`
- Test avg net bps/trade (mixed): `17.26`
- Test avg net bps/trade (all-taker): `11.26`

## Files
- `best_config_trades.csv`
- `best_config_monthly_breakdown.csv`
- `best_config_equity_curve.csv`