# Microstructure Replay Comparison (30d Window)

- All three variants use the same frozen 25% live-style replay assumptions.
- Only the trade eligibility differs via the microstructure gate.

| Variant | Filled Trades | Total PnL | Avg Net | Win Rate | Final Capital |
|---|---:|---:|---:|---:|---:|
| Base 30d window | 213 | 3818.55 | 7.0423 bps | 58.22% | 103818.55 |
| Train-selected gate | 193 | 3917.35 | 7.9684 bps | 61.66% | 103917.35 |
| Hypothesis gate | 190 | 3772.39 | 7.8004 bps | 58.95% | 103772.39 |
