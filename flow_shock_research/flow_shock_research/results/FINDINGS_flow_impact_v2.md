# Flow Impact Detector v2 - Findings

## Methodology Change

**v1 (Z-Score):** Measured volume spikes
```
flow_z = (volume - mean) / std
```
**Problem:** Assumes normal distribution, measures activity not impact

**v2 (Flow Impact):** Measures ability to move market
```
Flow Impact = Aggressive Volume / Top Book Depth
```
**Improvement:** Normalizes by available liquidity, regime-independent

## Results Comparison

| Metric | Z-Score (v1) | Flow Impact (v2) |
|--------|--------------|------------------|
| Events/day | 2.3 (z>30) | 1.1 (impact>0.6) |
| Total events | 210 (92d) | 105 (92d) |
| Detection basis | Volume spike | Market impact |
| Regime handling | Poor (mean/std) | Good (median/MAD) |
| Directional | No | Yes (buy/sell) |

## Flow Impact Interpretation

| Flow Impact | Meaning |
|-------------|----------|
| 0.05 | Noise |
| 0.2 | Activity |
| 0.5 | Stress |
| 1.0 | Market punched through |
| >1.5 | Forced flow |

**Detected events:** 0.75 - 118.62

## Key Improvements

1. **Market Impact Normalization:** $500k volume on thin book ≠ thick book
2. **Signed Flow:** Tracks buy vs sell aggression separately
3. **Robust Statistics:** Median/MAD handles regime changes
4. **Burst Detection:** Requires sustained same-direction flow

## Next Steps

1. Analyze price behavior around Flow Impact events
2. Compare profitability: z-score vs flow impact triggers
3. Optimize impact threshold (0.6 vs 0.8 vs 1.0)
4. Add order flow toxicity metrics
