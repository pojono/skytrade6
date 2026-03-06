# Money Flow Report

## Coverage
- Futures-common universe: 3 symbols.
- Full spot+futures universe: 3 symbols.
- Spot analysis window: 13 daily observations.

## Core Pattern
- Best cross-sectional 3-day signal: `n/a` with mean daily Spearman IC nan across 0 days.
- Best symbol state: `n/a` with n/a average 3-day forward return over 0 observations.
- Worst symbol state: `n/a` with n/a average 3-day forward return over 0 observations.

## Interpretation
- `up|spot_high|oi_down` is the cleanest accumulation signature: price is already rising, spot share is above its own baseline, and open interest is not expanding with it. That usually behaves like cash-led demand or short-cover plus spot follow-through.
- `up|spot_low|oi_up` is the opposite regime: futures dominate the tape while leverage expands into strength. That behaves more like a crowded chase and has materially weaker follow-through.
- When this pattern broadens across many coins on the same day, the equal-weight universe also separates: high breadth in `up|spot_high|oi_down` days leads to 1.14% next-3-day universe return, while high breadth in `up|spot_low|oi_up` days leads to -4.90%.

## Files
- `symbol_daily_flows.csv`: per-symbol daily feature set.
- `money_flow_feature_ic.csv`: feature-level predictive ranking.
- `money_flow_state_summary.csv`: forward returns by money-flow state.
- `money_flow_breadth_summary.csv`: market-wide breadth regime results.
