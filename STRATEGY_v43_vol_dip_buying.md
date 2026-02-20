# Volatility Dip-Buying Strategy — v43 Final Specification

## Overview

**Core idea:** When realized volatility spikes AND price has dipped in the last 4 hours, go long. This is a "buy the dip in high volatility" strategy that exploits the tendency of crypto to V-recover after sharp drops.

**Signal:** Combined z-score of:
1. `rvol_z` — 24h realized volatility z-scored vs 168h (1-week) lookback
2. `mr_4h` — 4h price mean-reversion z-score (negative = price dropped)

`combined = (rvol_z + mr_4h) / 2`

Trade when `|combined| > 2.0`. Direction: long if combined > 0, short if < 0.

## Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Signal threshold | 2.0 | Higher = fewer but better trades |
| Hold period | 4 hours | Sweet spot: 1h too short, 8h+ diminishing returns |
| Entry | Limit order at close of signal bar | Maker fee 0.02% |
| Exit | Limit order 4h later | Maker fee 0.02% |
| RT fees | 4 bps (0.04%) | Maker + maker |
| Cooldown | 4 bars (4h) after exit | Avoid overtrading |
| Warmup | 168 bars (7 days) | For z-score calculation |

## Recommended Symbols

| Symbol | 3yr Total | Avg bps/trade | WR | Walk-Forward | Recommended |
|--------|-----------|---------------|-----|-------------|-------------|
| **SOL** | +72.1% | +51.9 | 63.3% | 75% pos months | ✅ Primary |
| **XRP** | +108.3% | +68.5 | 59.5% | 68% pos months | ✅ Primary |
| **DOGE** | +29.6% | +17.6 | 56.5% | 62% pos months | ✅ Secondary |
| BTC | +12.9% | +7.6 | 53.8% | 59% pos months | ⚠️ Marginal |
| ETH | +30.9% | +21.5 | 52.8% | 52% pos months | ⚠️ Marginal |

## Regime Filter (Optional)

For SOL and DOGE, adding a DD(30d)>15% filter improves consistency:
- Skip long trades when price is >15% below its 30-day high
- SOL: 60% → 78% positive walk-forward months
- DOGE: 62% → 79% positive walk-forward months

**Do NOT use regime filters on XRP** — they hurt performance.

## Risk Characteristics

| Metric | SOL | XRP | DOGE |
|--------|-----|-----|------|
| Trades/year | ~45 | ~50 | ~55 |
| Capital deployed | ~2% of time | ~2% | ~2% |
| Max drawdown | 13.4% | 24.1% | 21.8% |
| Sharpe (annualized) | ~9.8 | ~6.9 | ~2.5 |
| Long/Short ratio | 97/3 | 93/7 | 96/4 |

## Known Weaknesses

1. **Heavily long-biased** — 90-97% of trades are long. Short side has too few trades to validate statistically.
2. **Regime-dependent** — Works when dips are V-shaped recoveries. Fails in sustained grinding bear markets (SOL 2025Q4: -2.2%, 2026Q1: -7.2%).
3. **Low frequency** — ~4-5 trades/month per symbol. Slow to compound.
4. **Not delta-neutral** — Directional risk during 4h hold period.
5. **Tested on 3 years only** — Jan 2023 to Feb 2026 was net bullish for crypto.

## Implementation Notes

### Signal Calculation (Python)
```python
import pandas as pd, numpy as np

def compute_signal(close_prices_1h):
    """Input: pandas Series of 1h close prices, indexed by timestamp."""
    c = close_prices_1h.values.astype(np.float64)
    ret = pd.Series(np.diff(c, prepend=c[0]) / np.maximum(c, 1e-8) * 10000,
                    index=close_prices_1h.index)
    
    # Realized volatility z-score
    rvol = ret.rolling(24, min_periods=8).std()
    rvol_z = ((rvol - rvol.rolling(168, min_periods=48).mean()) /
              rvol.rolling(168, min_periods=48).std().clip(lower=1e-8))
    
    # 4h mean-reversion z-score
    r4 = ret.rolling(4).sum()
    mr_4h = -((r4 - ret.rolling(48, min_periods=12).mean() * 4) /
              (ret.rolling(48, min_periods=12).std().clip(lower=1e-8) * 2))
    
    combined = (rvol_z + mr_4h) / 2
    return combined
```

### Entry Logic
```
Every hour:
  1. Compute combined signal from latest 168+ hours of 1h OHLCV
  2. If |combined| > 2.0 AND no position AND cooldown expired:
     a. If combined > 0: place LIMIT BUY at current price
     b. If combined < 0: place LIMIT SELL at current price
  3. Set exit: LIMIT order at entry_price ± 0 (market close in 4h)
     — In practice, place a limit order at the expected 4h close price
     — Or simply close at market after 4h (adds ~3.5 bps taker fee)
```

### Position Sizing
- Max 1 position per symbol at a time
- Suggested: 5-10% of portfolio per trade
- With 3 symbols (SOL, XRP, DOGE): max 15-30% deployed at any time
- Actual deployment: ~2% of time per symbol → ~6% total

### Kill Switches
- Stop trading if 3 consecutive losses
- Stop trading if monthly drawdown > 5%
- Stop trading if signal hasn't fired in 30 days (market regime change)

## Validation Summary

- **19 strategy variants tested** in v43 research (v43a through v43s)
- **16 failed completely** (fee wall, no signal, simulation artifacts, execution cost)
- **This strategy is the only one that survived** 3-year walk-forward on multiple symbols
- Signal beats random baseline on all 5 symbols (z-score +0.83 to +4.95)
- Fee-robust: profitable even at 20 bps RT on XRP/SOL

## Honest Assessment

This is a **conditional buy-the-dip** strategy. It works because crypto tends to V-recover after sharp volatility spikes. The edge is real (beats random consistently) but not large — average +18 to +69 bps per trade depending on symbol.

The main risk is a regime change where dips stop recovering (sustained bear market). The regime filter helps on SOL/DOGE but reduces trade count significantly.

**Recommendation:** Paper trade for 1-2 months before going live. Start with SOL (best Sharpe, best walk-forward). Add XRP if SOL confirms. Use small position sizes (5% per trade).
