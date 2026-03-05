# CRV Microstructure Audit

## Scope

- Symbol: CRVUSDT
- Purpose: test whether the surviving CRV edge is concentrated in recent trade-flow / liquidity conditions that are observable with local microstructure data.
- Audited days: 2026-02-01, 2026-02-02, 2026-02-03, 2026-02-04, 2026-02-05, 2026-02-06, 2026-02-07, 2026-02-08, 2026-02-09, 2026-02-10, 2026-02-11, 2026-02-12, 2026-02-13, 2026-02-14, 2026-02-15, 2026-02-16, 2026-02-17, 2026-02-18, 2026-02-19, 2026-02-20, 2026-02-21, 2026-02-22, 2026-02-23, 2026-02-24, 2026-02-25, 2026-02-26, 2026-02-27, 2026-02-28, 2026-03-01, 2026-03-02
- Trigger count audited: 90

## Aggregate

- Avg net after 20 bps taker fee: 9.8842 bps
- Positive trigger share: 62.22%
- Avg combined signed flow before entry: 3120.32 USD
- Avg Bybit top-of-book spread: 78.1993 bps
- Avg Bybit top depth: 24011.47 USD

## Buckets

- Flow aligned with signal: 51 trades, avg net 6.3989 bps
- Flow not aligned: 39 trades, avg net 14.4418 bps
- High score (>=20): 29 trades, avg net 13.1726 bps
- Higher top depth (>=25k USD): 34 trades, avg net 4.6644 bps
- Tight Bybit book (<=12 bps): 32 trades, avg net 11.1532 bps

## Interpretation

- This is a recent-days execution/regime audit, not a replacement for the full historical signal test.
- Higher-score signals remain the cleanest positive subset and should stay central to the filter.
- Tighter Bybit top-of-book conditions improve average outcome and look like a useful execution gate.
- Larger displayed top depth does not help here; the better trades are not simply the deepest-book moments.
- The best trades often arrive after the most obvious aggressive flow burst has already started fading, not while it is still strongly aligned.
