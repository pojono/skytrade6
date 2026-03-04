# Findings Phase 15: Historical Stress Matrix For The Frozen 25% Filtered Sleeve

This phase stress-tests the current best candidate without changing the strategy logic.

Only execution assumptions were worsened.

Runner:

- `stress_test_replayopt_25.py`

Outputs:

- `out/stress_matrix_replayopt_25.csv`
- `out/stress_matrix_replayopt_25.md`

## Frozen Strategy

The strategy remains fixed:

- replay-optimized filtered `25%` sleeve
- same input trade stream: `out/candidate_trades_v3_replayopt.csv`
- same selector, caps, stops, and symbol universe

Only these execution parameters were stressed:

- round-trip fee
- fixed extra slippage
- spread slippage coefficient
- velocity slippage coefficient
- size slippage coefficient

## Stress Results

### Base

- PnL: `$7,638.99`
- Avg net: `3.2849 bps`
- Win rate: `53.29%`
- Max realized drawdown: `0.25%`

### Higher Fees

- fee increased from `6` to `8 bps`
- PnL: `$2,918.30`
- Avg net: `1.2849 bps`
- Win rate: `47.94%`
- Max realized drawdown: `0.79%`

Still positive, but much weaker.

### Higher Fixed Slippage

- extra slippage increased from `1` to `2 bps`
- PnL: `$5,252.21`
- Avg net: `2.2849 bps`
- Win rate: `50.39%`
- Max realized drawdown: `0.35%`

Still clearly positive.

### Higher Variable Slippage

- spread coeff increased from `0.10` to `0.15`
- velocity coeff increased from `0.05` to `0.08`
- PnL: `$6,140.96`
- Avg net: `2.6599 bps`
- Win rate: `51.84%`
- Max realized drawdown: `0.29%`

Still strong.

### Higher Size Slippage

- size slippage increased from `1.5` to `2.0 bps`
- PnL: `$5,843.90`
- Avg net: `2.5349 bps`
- Win rate: `51.51%`
- Max realized drawdown: `0.31%`

Still strong.

### Harsh Combo

- fee `8 bps`
- extra slippage `2 bps`
- spread coeff `0.15`
- velocity coeff `0.08`
- size slippage `2.0 bps`

Result:

- PnL: `-$2,419.67`
- Avg net: `-1.0901 bps`
- Win rate: `42.14%`
- Max realized drawdown: `3.88%`

This breaks the strategy.

### Very Harsh Combo

- fee `10 bps`
- extra slippage `3 bps`
- spread coeff `0.20`
- velocity coeff `0.10`
- size slippage `2.5 bps`

Result:

- PnL: `-$11,406.67`
- Avg net: `-5.4044 bps`
- Win rate: `30.92%`
- Max realized drawdown: `11.68%`

This decisively breaks the strategy.

## Practical Interpretation

The frozen `25%` filtered sleeve survives moderate execution deterioration:

- `5` of `7` tested scenarios remain positive

But it does not survive truly harsh execution assumptions.

That means:

- the edge is real enough to handle moderate cost slippage
- the edge is not large enough to tolerate severe fill degradation

This is a healthy, realistic result. It means the strategy is not fake, but it is execution-sensitive.

## Production Implication

The current strategy is viable only if actual live execution quality is closer to:

- the base case
- or the moderate stress cases

If real trading behaves more like the harsh combo, the edge is not enough.

So the correct production gate from here is:

- validate live execution quality first
- do not assume the historical edge survives poor fills
