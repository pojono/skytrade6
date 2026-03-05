# Per-Trade Classifier V2

This version adds causal symbol-history features on top of the base signal.

## Added Feature Families

- prior signal outcome for the same symbol
- rolling 3-signal average and hit rate for the symbol
- number of prior signals seen for the symbol
- hours since last signal in the same symbol
- current cross-sectional rank within the timestamp
- interaction and magnitude terms (`abs_mom`, `ls*taker`, centered breadth)

## Best CV Configuration

- Model: `logistic`
- Threshold: `0.70`
- Max positions: `1`
- CV avg: 135.94 bps
- CV hit rate: 87.5%

## Holdout Comparison

- Base holdout avg: 26.95 bps across 27 decisions
- Base holdout hit rate: 55.6%
- V2 chosen-model holdout avg: -55.20 bps across 12 decisions
- V2 chosen-model holdout hit rate: 41.7%

## Holdout Diagnostic Across Top CV Candidates

- Best test result among top CV candidates: `logistic` threshold `0.65` max-pos `2`
- Best test avg among those candidates: -43.27 bps
- Best test decision count among those candidates: 12

## Interpretation

- If this still fails to beat the base strategy, richer static/tabular features are not enough yet.
- That would imply the missing edge is more about execution context or non-stationary symbol behavior than simple classification.