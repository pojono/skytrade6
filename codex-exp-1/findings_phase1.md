# Phase 1 Findings

## Tested Configuration

- Universe: up to 111 common symbols with at least 90 overlap days
- History used per symbol: most recent 45 overlap days
- Out-of-sample: last 15 days
- Entry rule: one-bar spread-reversion, trigger at `|spread| >= 10 bps`
- Cost assumption: `6 bps` round trip

Command used:

```bash
python3 codex-exp-1/cross_exchange_edge_scan.py \
  --min-overlap-days 90 \
  --min-signal-bps 10 \
  --fee-bps-roundtrip 6 \
  --test-days 15 \
  --recent-days 45 \
  --max-symbols 120 \
  --workers 4 \
  --executor thread
```

## Portfolio Conclusion

The unfiltered strategy is not good enough as a broad portfolio.

Full-universe result:

- Symbols analyzed: `111`
- Test signals: `571,088`
- Test average net PnL: `-3.0803 bps`

This means the naive rule is still structurally negative after realistic fees, even though many individual symbols look good in isolation.

## Why The Raw Scan Looked Better At First

Small pilot baskets can overstate the edge because:

- symbol selection matters
- thresholds reduce trade count unevenly
- some symbols show extremely low out-of-sample sample sizes
- a few high-quality symbols can mask a broadly negative universe

That is why broad expansion and minimum-signal filtering are mandatory.

## Surviving Symbols

Using strict survival criteria:

- positive test average net PnL
- positive total average net PnL
- at least 500 test signals
- at least 2,000 total signals

The current survivors are:

1. `CRVUSDT`
2. `GALAUSDT`
3. `KAVAUSDT`
4. `ARBUSDT`
5. `SEIUSDT`
6. `NEARUSDT`
7. `FILUSDT`
8. `XPLUSDT`

These come from `codex-exp-1/out/survivor_symbols.txt`.

## Research Implication

The edge, if real, is not "trade all symbols when spread exceeds 10 bps."

The edge is more likely one of:

1. Symbol-specific
2. Regime-dependent
3. Strong only when spread dislocation aligns with positioning imbalance
4. Sensitive to spread threshold and trade density

## Next Necessary Step

Do not widen the same raw rule further.

Instead:

1. Build a symbol-focused second-pass analysis for the survivor list
2. Add filters from Binance `metrics.csv` and Bybit funding / OI / long-short ratio
3. Require the filtered version to improve both:
   - average net PnL
   - proportion of profitable symbols
4. Re-test on the same out-of-sample window before extending history

## Practical Direction

The best next hypothesis is:

"Spread mean reversion works only when the expensive venue also shows stretched positioning."

That is the next test worth building.
