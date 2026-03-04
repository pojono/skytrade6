# Findings Phase 19: Account-Level Replay With The Microstructure Gate

This phase answers the practical question that matters:

- does the microstructure gate improve account-level replay, not just per-trade quality?

Script:

- `replay_microstructure_gate.py`

Output:

- `out/microstructure_replay_comparison_30d.md`

The replay uses the same frozen `25%` live-style assumptions on the same overlapping 30-day high-resolution window.

## Compared Variants

1. Base 30-day window
   - no extra microstructure gate
2. Train-selected microstructure gate
   - selected from the constrained train-only search
3. Hypothesis gate
   - simple hand-constrained activity + tight-book gate

## Results

### Base 30-Day Window

- `213` fills
- `$3,818.55` total PnL
- `7.0423 bps` average net edge
- `58.22%` win rate

### Train-Selected Gate

Gate:

```json
{
  "min_bybit_trade_count_5s": 2.0,
  "min_binance_trade_count_5s": 0.0,
  "max_bybit_book_spread_bps": 4.5,
  "max_bybit_trade_imbalance_5s": 1.0
}
```

Result:

- `193` fills
- `$3,917.35` total PnL
- `7.9684 bps` average net edge
- `61.66%` win rate

Effect vs base:

- fewer fills (`-20`)
- higher total PnL (`+$98.80`)
- higher edge per fill
- materially higher win rate

This is a real account-level improvement, even if the dollar lift is modest over this 30-day slice.

### Hypothesis Gate

Gate:

```json
{
  "min_bybit_trade_count_5s": 4.0,
  "min_binance_trade_count_5s": 0.0,
  "max_bybit_book_spread_bps": 4.5,
  "max_bybit_trade_imbalance_5s": 1.0
}
```

Result:

- `190` fills
- `$3,772.39` total PnL
- `7.8004 bps` average net edge
- `58.95%` win rate

Effect vs base:

- fewer fills
- slightly lower total PnL
- higher edge per fill

So this gate improves trade quality, but not total dollars in the 30-day replay.

## Practical Conclusion

The best current optional microstructure add-on is the train-selected gate:

- it improves account-level dollars
- it improves win rate
- it improves average edge

The improvement is not huge, so this should be treated as:

- a quality/robustness enhancer
- not a radically new strategy

But it is still important because it is the first high-resolution filter that:

- survives a longer holdout
- and also survives conversion into account-level replay

## Current Best Reading

The strategy is strongest when the existing 1-minute signal is combined with:

- modest recent Bybit activity
- a tighter local Bybit book

That is exactly the kind of condition a real execution system should care about.
