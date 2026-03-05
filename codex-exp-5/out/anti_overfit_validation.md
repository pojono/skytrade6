# Anti-Overfit Validation

A config is accepted only if it passes all of:
- minimum positive symbols: 2
- minimum trades per symbol per split: 3
- minimum holdout trades total: 15
- positive portfolio avg in train, validation, and holdout
- holdout top-symbol contribution <= 0.70

## Result

No configuration passed the anti-overfit bar on current local data.
