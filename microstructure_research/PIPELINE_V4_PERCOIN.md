# Pipeline V4 — Per-Coin Selection with Hard Gates

## Changes from V3

### Problem with V3
- Tier 2 passed 94% of features (980/1040) — barely filtering
- No minimum effect size — features with |r|=0.001 could pass
- Cross-coin clustering forced both coins into same feature set
- Tier 5 ranked but never eliminated — 182 features all reached final output

### V4 Fixes

| Component | V3 | V4 |
|-----------|----|----|
| **Tier 2 filter** | sign≥70%, SNR≥0.5, streak≤3, regime | + **\|full_r\| ≥ 0.03** |
| **Tier 4 scope** | Cross-coin (union of both) | **Per-coin** (separate clustering) |
| **Tier 5 output** | Ranking only, no elimination | **Hard OOS filter**: mean_oos > 0.02, pct_positive ≥ 60% |
| **Final output** | 1 universal feature list | **Per-coin, per-target survivor lists** |

## Results

### Tier 2: Effect Size Filter Impact

| Target | SOL v3 | SOL v4 | DOGE v3 | DOGE v4 |
|--------|--------|--------|---------|---------|
| tgt_ret_1 | 419 | **219** | 395 | **205** |
| tgt_ret_5 | 371 | **268** | 408 | **244** |
| tgt_profitable_long_5 | 279 | **53** | 285 | **38** |
| tgt_profitable_short_5 | 285 | **54** | 295 | **55** |
| tgt_sharpe_10 | 297 | **194** | 404 | **282** |

The min |r| ≥ 0.03 filter cuts binary targets aggressively (AUC-based metrics have smaller effect sizes) while continuous targets see moderate reduction.

### Tier 4: Per-Coin Clustering

| | V3 (cross-coin) | V4 SOL | V4 DOGE |
|--|-----------------|--------|---------|
| Input features | 980 | 574 | 583 |
| Cluster reps | 182 | 218 | 213 |

Per-coin clustering produces slightly more representatives because coin-specific features no longer get merged with cross-coin counterparts.

### Tier 5: Hard OOS Filter

| | SOL | DOGE |
|--|-----|------|
| Features evaluated | 218 | 213 |
| Unique features surviving | **217** | **211** |
| Targets with survivors | **66** | **73** |
| Total (feature, target) pairs | **4,216** | **4,108** |

### Top Features by Target Breadth (most versatile)

**SOL:**
| Feature | #targets | avg OOS |
|---------|----------|---------|
| `is_swing_low` | 51 | +0.078 |
| `is_swing_high` | 43 | +0.061 |
| `atr_percentile_14` | 37 | +0.100 |
| `arrival_time_kurtosis_z` | 35 | +0.092 |
| `volume_autocorr` | 35 | +0.097 |
| `max_down_sweep_z` | 34 | +0.124 |
| `value_area_overlap_pct_z` | 33 | +0.135 |

**DOGE:**
| Feature | #targets | avg OOS |
|---------|----------|---------|
| `is_swing_low` | 50 | +0.084 |
| `is_swing_high` | 44 | +0.062 |
| `support_touches_20` | 38 | +0.074 |
| `max_monotonic_run` | 37 | +0.093 |
| `eigen_ratio` | 36 | +0.119 |
| `donchian_width_bps_10` | 33 | +0.077 |
| `hilbert_std_amplitude` | 32 | +0.076 |

### Key Profitable Target Survivors

**tgt_profitable_long_5:**
- SOL: 55/218 features pass hard OOS filter
- DOGE: 49/213 features pass hard OOS filter

**tgt_profitable_short_5:**
- SOL: 59/218 pass
- DOGE: 48/213 pass

## Pipeline Flow Summary (V4)

```
Tier 1:  1,040 features → 1,040  (scoring only)
Tier 2:  1,040 → ~575 per coin   (|r|≥0.03 + stability gates)
Tier 4:  ~575  → ~215 per coin   (per-coin redundancy clustering)
Tier 5:  ~215  → per-target lists (hard OOS: mean>0.02, pct≥60%)
```

## Key Takeaways

1. **Per-coin models are the right approach** — SOL and DOGE have different top features beyond the universal ones (`is_swing_low/high`)
2. **min |r| ≥ 0.03 is effective** — cuts Tier 2 from 94% pass rate to ~55%, removing noise
3. **Hard OOS filter works** — for `tgt_profitable_long_5`, only ~50/215 features survive per coin
4. **Output is now actionable** — `tier5_{SYMBOL}_{tf}_survivors.csv` gives per-coin, per-target feature lists ready for multi-feature modeling
5. **Universal features confirmed** — `is_swing_low`, `is_swing_high`, `atr_percentile_14` survive on both coins independently
6. **Coin-specific features discovered** — SOL favors `volume_autocorr`, `value_area_overlap_pct_z`; DOGE favors `support_touches_20`, `eigen_ratio`, `inside_bar`
