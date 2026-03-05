# Orderbook Walk-Forward: 30s Entry, 60m Exit

This is a causal, execution-aware walk-forward on the covered raw signal stream.
- Entry decision time: signal + 30s
- Exit time: entry + 60m
- Fees: 20 bps round-trip
- Entry and exit fills are simulated from actual `book_depth` snapshots using cumulative depth buckets (0% to 5%) and a piecewise-linear fill model.
- Reference price uses the last observed `agg_trade` before each execution timestamp.

## Size Sweep

### $10,000

- Train rows with full execution data: 42
- Test rows with full execution data: 41
- Mean test entry impact: 0.00 bps
- Mean test exit impact: 0.00 bps
- Unfiltered test avg after execution + fees: -30.64 bps
- Train-chosen gate: `ret_30s_bps >= 1.47`, `buy_share_30s >= 0.701`
- Filtered test avg: 28.06 bps on 10 rows
- Improvement vs unfiltered: 58.70 bps

### $50,000

- Train rows with full execution data: 42
- Test rows with full execution data: 41
- Mean test entry impact: 0.01 bps
- Mean test exit impact: 0.02 bps
- Unfiltered test avg after execution + fees: -30.67 bps
- Train-chosen gate: `ret_30s_bps >= 1.47`, `buy_share_30s >= 0.701`
- Filtered test avg: 28.06 bps on 10 rows
- Improvement vs unfiltered: 58.73 bps

### $100,000

- Train rows with full execution data: 42
- Test rows with full execution data: 41
- Mean test entry impact: 0.24 bps
- Mean test exit impact: 0.35 bps
- Unfiltered test avg after execution + fees: -31.23 bps
- Train-chosen gate: `ret_30s_bps >= 1.47`, `buy_share_30s >= 0.701`
- Filtered test avg: 28.04 bps on 10 rows
- Improvement vs unfiltered: 59.27 bps

## Bottom Line

- A causal 30-second gate combined with orderbook-based fills stays positive on the covered test sample at $10,000 notional.