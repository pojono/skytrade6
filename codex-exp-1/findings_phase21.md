# Findings Phase 21: What Kills The Edge In Strict Fill Replay

This phase breaks down the strict fill replay slippage into four components:

- Bybit entry
- Bybit exit
- Binance entry
- Binance exit

Source file:

- `out/strict_fill_replay_train_gate_30d.csv`

This is the strict replay of the `193` accepted fills from the current best microstructure-gated 30-day sleeve.

## Average Slippage By Component

Across all `193` fillable trades:

- Bybit entry slippage: `5.1678 bps`
- Bybit exit slippage: `5.0532 bps`
- Binance entry slippage: `2.2938 bps`
- Binance exit slippage: `1.9651 bps`

Total average strict execution slippage:

- `14.4799 bps`

The core point is simple:

- Bybit accounts for about `10.22 bps` of the average `14.48 bps`
- Binance accounts for about `4.26 bps`

So the majority of the strict execution drag is on the Bybit side.

## Median And Tail

Median component slippage:

- Bybit entry: `5.1479 bps`
- Bybit exit: `4.9602 bps`
- Binance entry: `2.1845 bps`
- Binance exit: `1.9511 bps`

90th percentile:

- Bybit entry: `6.3803 bps`
- Bybit exit: `6.1731 bps`
- Binance entry: `3.1894 bps`
- Binance exit: `2.6292 bps`

This reinforces the same conclusion:

- normal-case Bybit slippage is already large
- the tail makes it worse

## By Symbol

### CRVUSDT

- trades: `88`
- strict average net: `+5.6425 bps`

Average component slippage:

- Bybit entry: `5.0062 bps`
- Bybit exit: `4.9522 bps`
- Binance entry: `1.8051 bps`
- Binance exit: `1.7454 bps`

Interpretation:

- `CRVUSDT` still survives strict replay on average

### GALAUSDT

- trades: `87`
- strict average net: `-7.2887 bps`

Average component slippage:

- Bybit entry: `5.4344 bps`
- Bybit exit: `5.3097 bps`
- Binance entry: `2.8495 bps`
- Binance exit: `2.1917 bps`

Interpretation:

- `GALAUSDT` breaks badly under strict replay
- both venues are worse than on `CRVUSDT`, but Bybit is still the largest contributor

### SEIUSDT

- trades: `18`
- strict average net: `-10.0124 bps`

Average component slippage:

- Bybit entry: `4.6689 bps`
- Bybit exit: `4.3071 bps`
- Binance entry: `1.9971 bps`
- Binance exit: `1.9440 bps`

Interpretation:

- `SEIUSDT` also breaks
- even with slightly lower slippage than `GALAUSDT`, the edge is too small to survive

## Worst Cases

The biggest strict execution losses cluster in:

- `GALAUSDT`
- some `SEIUSDT`
- occasional stressed `CRVUSDT`

Examples:

- `GALAUSDT` on `2026-02-06` reached `31.63 bps` strict execution slippage on one trade
- several `GALAUSDT` trades exceeded `19 bps`
- some `SEIUSDT` trades exceeded `24 bps`

Those tails are large enough to destroy a short-horizon spread-capture trade.

## The Real Diagnosis

The strict replay failure is not “everything is slightly worse.”

The real diagnosis is:

- the strategy is paying too much crossing the book on Bybit
- and the remaining gross edge in `GALAUSDT` and `SEIUSDT` is not large enough to absorb that cost

So the current failure mode is mainly:

- Bybit entry/exit crossing cost
- amplified by weaker symbols

## Most Likely Path Forward

If this strategy is to be salvaged, the most plausible paths are:

1. Reduce Bybit crossing cost.
   - better order placement
   - passive or semi-passive logic
   - tighter symbol/time selection

2. Concentrate on the only symbol still surviving strict replay on average.
   - `CRVUSDT`

3. Reconsider whether `GALAUSDT` and `SEIUSDT` should be traded live at all.

The evidence right now points to execution mechanics, especially on Bybit, as the main bottleneck.
