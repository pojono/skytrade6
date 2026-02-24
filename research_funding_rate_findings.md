# Funding Rate Edge Research — Binance vs Bybit

**Date:** 2026-02-23
**Data:** 2 days (2026-02-22 to 2026-02-23), 487 matched symbols, 1.2M snapshots at 1-min resolution
**Script:** `research_funding_rate_edge.py`

## Key Findings

### 1. Extreme Funding Rates
- Negative extremes dominate: POWERUSDT hit -2.0% (BN) / -2.5% (BB) funding caps
- 11 coins had |FR| > 50bp during the window
- Negative extremes are 10x larger than positive extremes

### 2. Cross-Exchange FR Spread
- Mean spread (BN - BB): -0.20 bp (no directional bias)
- Mean |spread|: 2.05 bp
- p99: 20.28 bp, p99.9: 63.59 bp
- Fat tails — occasional large divergences

### 3. Funding Interval Mismatch (Main Finding)
- 94 coins have **1h funding on Bybit** but **8h on Binance**
- Same rate compounds 8x more on Bybit
- During extreme negative FR: annualized spread can be 100-1800% (extrapolated)
- Top candidates: LAUSDT, AZTECUSDT, AXSUSDT, ENSOUSDT

### 4. Execution Costs
- Cross-exchange price spread: 4.3% average (high!)
- Bybit bid-ask: 4.25 bps median (tight for extreme FR coins: 1.8 bps)
- Price spread widens to ~1% during extreme FR events

### 5. Settlement Timing
- No significant edge around settlement times (±10min)
- FR spread only marginally wider near settlement

## Verdict
The funding interval mismatch (1h vs 8h) is a real structural inefficiency, but:
- 4.3% cross-exchange price spread is the main barrier
- Extreme FR conditions are transient (need more data to measure persistence)
- Small-cap coins limit capacity
- Need weeks/months of data to properly evaluate

## Next Steps
- Accumulate more days of symbol=ALL data
- Measure FR persistence (how long do extreme FR windows last?)
- Track cross-exchange price spread dynamics during FR normalization
- Evaluate if a signal-based approach (enter when FR diverges, exit when it converges) is viable
- Consider spot vs perp arb as alternative to cross-exchange perp arb
