# Momentum Continuation Strategy — AUDIT VERDICT: REJECTED

## Strategy
Long on 2 consecutive up bars (close > open), short on 2 consecutive down bars. Hold 2 bars. 1h OHLC, Bybit data, 2025-07 to 2026-02.

## CRITICAL BUG: Lookahead Bias in Entry Timing

The backtest enters at `open[i]` where `i` is the bar whose signal depends on `close[i]`.

```
signal[i] = 1  if close[i] > open[i]  AND  close[i-1] > open[i-1]
entry_price = open[i]
```

**The signal uses bar i's close to decide to enter at bar i's open.** This is textbook lookahead bias — we're using future information (will this bar close up?) to execute a trade at the bar's open price, which is chronologically *before* the close is known.

### Impact: Complete Strategy Invalidation

When fixed to enter at `open[i+1]` (next bar open, which is the first actionable price after the signal is known), results flip from massively profitable to massively losing:

| Symbol   | Original WR | Original Avg PnL | Fixed WR | Fixed Avg PnL | Fixed Total |
|----------|-------------|-------------------|----------|---------------|-------------|
| SOLUSDT  | 58.5%       | +0.33%            | 40.6%    | -0.21%        | -302%       |
| ETHUSDT  | 55.6%       | +0.29%            | 36.7%    | -0.17%        | -250%       |
| BTCUSDT  | 47.4%       | +0.08%            | 32.1%    | -0.19%        | -279%       |
| DOGEUSDT | 59.1%       | +0.45%            | 41.4%    | -0.14%        | -196%       |
| XRPUSDT  | 58.4%       | +0.32%            | 40.7%    | -0.14%        | -208%       |

**Every single symbol goes from profitable to heavily losing.** The fixed results are indistinguishable from random trading (random baseline: -0.20% avg PnL).

### Why the Bug Creates Fake Alpha

When bar i closes up (signal = long), entering at `open[i]` guarantees the first portion of the trade (bar i itself) is profitable by construction: `close[i] > open[i]` is literally the signal condition. The strategy is effectively "buying at the open and selling at the close of a bar that we already know closed up," which is not possible in real-time trading.

## Secondary Issue: Broad Test Only Covered 10 Symbols

The FINDINGS.md and IMPLEMENTATION_GUIDE.md claim "235 common symbols" were tested, but `broad_test_results_classified.csv` contains only 10 rows (BTC, ETH, SOL, XRP, DOGE, HYPE, RIVER, ADA, NEAR). The "broad robustness" claim is overstated.

## Conclusion

**STRATEGY REJECTED.** The entire reported edge (58% WR, 0.3% avg PnL, Sharpe 10+) is a lookahead artifact. With correct entry timing, the strategy loses ~0.17% per trade across all symbols — worse than random. There is zero tradeable edge.

## MA Crossover Strategy Test

Tested MA Crossover (fast 10, slow 50) on 20 symbols, all show negative returns and Sharpe ratios.

Example results:

- BNBUSDT: 1920 trades, 35.62% WR, -0.0019 avg P&L, -3.63% total, Sharpe -1.66

- AVAXUSDT: 1920 trades, 40.52% WR, -0.0016 avg P&L, -3.11% total, Sharpe -0.96

- HBARUSDT: 1920 trades, 40.89% WR, -0.0020 avg P&L, -3.77% total, Sharpe -1.10

All symbols tested show similar losing results. No edge found.

Audit date: 2026-03-04
