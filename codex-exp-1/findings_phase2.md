# Phase 2 Findings

## Tested Filter

The strongest dense filter found on the survivor basket was:

- spread entry threshold: `10 bps`
- round-trip cost: `6 bps`
- long/short divergence threshold: `>= 0.15`
- cross-venue OI-change divergence threshold: `>= 5 bps`
- carry divergence threshold: `>= 2 bps`

Carry proxy:

- Binance: `mark / index - 1`
- Bybit: `premium_index_kline_1m` close

## Important Generalization Result

This filter did **not** generalize as a broad all-symbol strategy.

On the full eligible universe sweep, all 48 tested threshold combinations were still negative.

Best broad configuration:

- `ls >= 0.15`
- `oi >= 5 bps`
- `carry >= 0`

Broad full-universe result:

- train avg net: `-3.4743 bps`
- test avg net: `-3.2549 bps`

So there is no evidence yet for a universal cross-exchange portfolio edge from this rule.

## Candidate Robust Basket

Even though the broad universe stays negative, a smaller filtered basket does survive.

Using the stricter filter:

- `ls >= 0.15`
- `oi >= 5 bps`
- `carry >= 2 bps`

And requiring:

- filtered train signals `>= 500`
- filtered test signals `>= 500`
- positive filtered train avg net
- positive filtered test avg net

The credible symbols are:

1. `CRVUSDT`
2. `GALAUSDT`
3. `SEIUSDT`
4. `FILUSDT`
5. `XPLUSDT`

Weighted basket performance across these 5 names:

- filtered train avg net: `6.8182 bps` on `10,083` signals
- filtered test avg net: `6.7217 bps` on `7,320` signals

## Interpretation

The current evidence supports a **clustered edge**, not a universal one.

That means:

- symbol selection is part of the strategy
- the filter is doing useful work
- the edge appears concentrated in a narrow set of names with persistent microstructure differences

## Practical Next Step

The next validation step is stricter:

1. Freeze this 5-symbol basket
2. Extend the validation window beyond the recent 45 days
3. Test threshold stability across:
   - spread threshold
   - cost assumptions
   - recent-window length
4. Reject the basket if the weighted test edge collapses materially

Until that longer validation passes, this is a promising candidate, not a production-ready conclusion.
