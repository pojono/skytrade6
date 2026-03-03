# Flow Shock Detector - Quick Sample Analysis

**Sample Period:** 7 days from 2025-05-11 to 2025-08-10
**Sampled Dates:** 2025-05-11, 2025-05-24, 2025-06-06, 2025-06-19, 2025-07-02, 2025-07-15, 2025-07-28
**Total Trades:** 2,967,013
**Detection Threshold:** flow_z > 3.0σ

## Results Summary

| Window | Events | Avg/Day | Median/Day | Est. Total (92d) | Target Range % |
|--------|--------|---------|------------|------------------|----------------|
| 5s | 92643 | 18528.6 | 21261 | 1217594 | 0% |
| 10s | 91625 | 18325.0 | 21104 | 1204214 | 0% |
| 15s | 87055 | 17411.0 | 19951 | 1144151 | 0% |
| 30s | 77732 | 15546.4 | 18136 | 1021621 | 0% |

## Interpretation

**Target:** 1-5 high-quality events per day

- Shorter windows (5-10s) detect more frequent spikes
- Longer windows (30s) filter for sustained volume shocks
- Adjust z-threshold or window size to hit target range

**Next Steps:**
1. Analyze price behavior around detected events
2. Test different z-thresholds (2.5, 3.5, 4.0)
3. Add directional filters (buy vs sell aggression)
4. Validate on full dataset
