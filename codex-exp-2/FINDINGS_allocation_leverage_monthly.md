# Monthly Breakdown: Allocation % and Leverage Sweep

- Grid allocations: `[0.1, 0.25, 0.5, 0.75, 1.0]`
- Grid leverage: `[1, 2, 3, 5, 8, 10]`
- Return source: `best_config_trades.csv` (`ret_net_mixed`, i.e. after mixed fees).
- Portfolio model: only `allocation` part of capital is deployed each signal; undeployed capital stays in cash.
- If leveraged sleeve return <= -100%, allocated sleeve is fully lost for that step (sleeve wipe).

## Top Configurations by Final Equity
- alloc=1.00, lev=8x: final=$6314.63, return=531.5%, maxDD=-73.6%, test_comp=18.8%, sleeve_wipes=0
- alloc=0.75, lev=10x: final=$6230.68, return=523.1%, maxDD=-70.6%, test_comp=19.1%, sleeve_wipes=0
- alloc=1.00, lev=10x: final=$5814.26, return=481.4%, maxDD=-83.2%, test_comp=15.5%, sleeve_wipes=0
- alloc=0.75, lev=8x: final=$5527.00, return=452.7%, maxDD=-60.6%, test_comp=18.7%, sleeve_wipes=0
- alloc=1.00, lev=5x: final=$4776.04, return=377.6%, maxDD=-52.8%, test_comp=17.5%, sleeve_wipes=0
- alloc=0.50, lev=10x: final=$4776.04, return=377.6%, maxDD=-52.8%, test_comp=17.5%, sleeve_wipes=0
- alloc=0.50, lev=8x: final=$3912.69, return=291.3%, maxDD=-44.1%, test_comp=15.4%, sleeve_wipes=0
- alloc=0.75, lev=5x: final=$3691.11, return=269.1%, maxDD=-41.7%, test_comp=14.7%, sleeve_wipes=0

## Files
- `allocation_leverage_summary.csv`
- `allocation_leverage_monthly.csv`
- `allocation_leverage_equity.csv`