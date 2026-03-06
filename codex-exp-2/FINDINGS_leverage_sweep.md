# Leverage Sweep (Aggressive Variants)

## Setup
- Base strategy: current `best_config_trades.csv` (top-5 equal, mixed-fee net returns at 1x).
- Leverage grid: `[1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]`
- Liquidation rule: if per-trade leveraged return <= -100%, equity is set to zero from that point onward.
- Stress scenario amplifies negative trade returns by `1.8x` before leverage.

## Best Non-Liquidating Variants
- `close_to_close` 8x: final `$6314.63`, return `531.5%`, maxDD `-73.6%`
- `close_to_close` 10x: final `$5814.26`, return `481.4%`, maxDD `-83.2%`
- `close_to_close` 6x: final `$5527.00`, return `452.7%`, maxDD `-60.6%`
- `close_to_close` 5x: final `$4776.04`, return `377.6%`, maxDD `-52.8%`
- `close_to_close` 12x: final `$4254.42`, return `325.4%`, maxDD `-90.1%`

## Files
- `leverage_sweep_summary.csv`
- `leverage_sweep_monthly.csv`
- `leverage_sweep_equity.csv`