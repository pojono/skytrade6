# Exit ML v3: Feature Experiments

**Date:** 2026-02-28  
**Data:** 81,511 ticks (100ms), 139 settlements, 30 symbols  
**Method:** 70/30 temporal split, LogReg + HGBC, permutation importance  
**Target:** `near_bottom_10` (within 10 bps of eventual minimum)

---

## Experiment Results

| Experiment | #Features | LR Test AUC | HGBC Test AUC | LR Δ | HGBC Δ |
|-----------|-----------|-------------|---------------|------|--------|
| **Baseline v2** | 46 | 0.7649 | 0.7885 | — | — |
| + OB depth (L5-L50) | 62 | 0.7693 | 0.7784 | +0.0045 | **-0.0100** |
| + CVD | 52 | 0.7601 | 0.7921 | -0.0048 | **+0.0036** |
| + Sequence | 55 | 0.7701 | 0.7910 | **+0.0052** | +0.0026 |
| + FR regime | 52 | 0.7649 | 0.7885 | +0.0000 | +0.0000 |
| ALL combined | 83 | 0.7659 | 0.7716 | +0.0011 | **-0.0169** |

---

## Key Findings

### 1. Sequence features are the clear winner
- **LogReg +0.0052** (0.7649 → 0.7701) — largest LR improvement
- **HGBC +0.0026** (0.7885 → 0.7910) — improves both models
- Low overfit gap (LR: -0.010, HGBC: +0.204 — same as baseline)
- Key new features: bounce count, consecutive new lows, price range, inter-trade time, reversals

### 2. CVD helps HGBC but hurts LogReg
- HGBC +0.0036 — the nonlinear model can use CVD velocity/acceleration
- LogReg -0.0048 — CVD adds noise for the linear model
- Suggests CVD signal is nonlinear (e.g., "CVD flattening after steep drop" = bottom)

### 3. OB depth HURTS performance
- LogReg +0.0045 (tiny gain) but HGBC **-0.0100** (significant drop)
- 16 extra features add noise without meaningful signal
- The L1 imbalance already captures the important orderbook information
- Deeper levels are too noisy at 100ms resolution during settlement chaos

### 4. FR regime interactions add NOTHING
- Both LR and HGBC: exactly 0.0000 delta
- The model already has `fr_bps` and `fr_abs_bps` — interactions are redundant
- FR regime is implicitly captured by other features (drop depth correlates with FR)

### 5. ALL combined is WORSE than baseline for HGBC
- HGBC -0.0169 — the 37 extra features cause overfitting
- LR +0.0011 — marginal improvement shows some signal exists but is drowned
- LogReg overfit gap flips positive (+0.005) for the first time — a warning sign
- **More features ≠ better performance with 139 settlements**

---

## Feature Importance (Top 5, consistent across all experiments)

1. **distance_from_low_bps** — how far above running minimum (always #1 or #2)
2. **pct_of_window_elapsed** — time progression through 60s window
3. **running_min_bps** — depth of drop so far
4. **drop_rate_bps_per_s** — rate of descent (slowing = exhaustion)
5. **time_since_new_low_ms** — no new lows = bottom forming

The top 5 are the SAME across all experiments. New features don't displace them.

---

## Recommendations

### Immediate: Add sequence features only
- bounce_count, consecutive_new_lows, price_range_2s/5s, price_std_2s/5s
- avg_inter_trade_ms, max_inter_trade_ms, reversals_2s
- Expected gain: +0.005 AUC (both models), zero overfit risk

### Do NOT add:
- OB depth (hurts HGBC, minimal LR gain)
- FR regime interactions (zero improvement)
- ALL features combined (overfits with current data size)

### Future (when data grows to 500+ settlements):
- Revisit CVD — the nonlinear signal may become learnable with more data
- Revisit OB depth at higher resolution (50ms ticks)
- Ensemble stacking (needs more data to avoid overfitting the meta-model)

---

## Technical Notes

- Dataset: `exit_ml_experiments_full.parquet` (81,511 ticks × 92 cols, 21 MB)
- OB.50 features computed via vectorized numpy from maintained bid/ask dicts
  (dict state replayed at each tick, `np.fromiter` + vectorized ops, no sorting)
- Full feature extraction: 116s for 141 JSONL files (single pass)
- Evaluation per experiment: ~25-30s (train/test only, no LOSO)
