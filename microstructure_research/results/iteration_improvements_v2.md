# Strategy Iteration Improvements v2 — Post-Audit Results

**Date:** 2025-01-XX
**Asset:** SOLUSDT 4h, 12 WFO periods (~2 years)
**Execution:** Open-to-open (no lookahead bias)

## Context

After fixing the exit price lookahead bias (commit e67729d), both strategies saw significant performance drops:
- Directional: +145% → +58% (Sharpe 4.58 → 2.44)
- Straddle: +267% → +49% (Sharpe 4.32 → 1.16)

Root cause: old code held trades ~1 extra bar, capturing momentum continuation outside the model's prediction horizon.

## Improvement Approach

Tested systematically via iteration runners:
1. **Longer hold periods** (4-5 bars) — explicitly capture momentum continuation
2. **Higher confidence threshold floor** (0.56) — filter low-quality signals
3. **Convex sizing** (confidence²) — concentrate capital on highest-conviction trades
4. **Combined configs** — best of above

## Directional Strategy Results

| Config | Net% | Sharpe | PF | MaxDD | Pos | Trades | AvgTrd |
|--------|------|--------|-----|-------|-----|--------|--------|
| **v3_early_exit** (baseline) | **+102.4%** | **4.05** | **1.80** | -19.7% | 11/12 | 361 | +0.637% |
| v8_high_thresh | +104.2% | 3.86 | 1.76 | -19.7% | 11/12 | 365 | +0.612% |
| v6_hold4 | +93.3% | 2.97 | 1.55 | -20.4% | 8/12 | 333 | +0.540% |
| v7_hold5 | +83.9% | 2.41 | 1.47 | -12.5% | 11/12 | 310 | +0.549% |
| v11_hold5_combo | +57.4% | 2.58 | 1.51 | **-8.1%** | 10/12 | 307 | +0.594% |
| v9_convex_size | +48.9% | 4.05 | 1.80 | -12.9% | 11/12 | 361 | +0.637% |
| v10_combined_v2 | +48.4% | 2.96 | 1.56 | -12.3% | 11/12 | 331 | +0.554% |

**Best config: v3_early_exit** — 15 base + 4 regime models, LightGBM meta, hold 3, early exit, linear sizing.
- Already 2x better than the main WFO script's v4 config (+58%) because it uses 19 models vs 8.
- Convex sizing halves return but maintains same Sharpe — useful for risk reduction.

## Straddle Strategy Results

| Config | Net% | Sharpe | PF | MaxDD | Pos | Trades | AvgTrd |
|--------|------|--------|-----|-------|-----|--------|--------|
| **v9_high_thresh** | **+122.4%** | **3.81** | 1.74 | -18.6% | 10/12 | 367 | +0.613% |
| v5_simple_size (baseline) | +117.0% | 3.55 | 1.67 | -18.6% | 10/12 | 370 | +0.577% |
| v7_hold4 | +106.9% | 3.36 | 1.68 | -25.0% | 9/12 | 332 | +0.639% |
| v8_hold5 | +102.0% | 2.79 | 1.56 | -25.6% | 8/12 | 317 | +0.611% |
| v11_hold4_combo | +70.2% | 3.76 | **1.79** | -16.9% | 9/12 | 332 | +0.733% |
| v10_convex | +65.1% | 3.55 | 1.67 | **-9.4%** | 10/12 | 370 | +0.577% |

**Best config: v9_high_thresh** — 6 vol + 13 dir models, LightGBM meta, hold 3, early exit, no vol gate, confidence sizing, threshold floor 0.56.
- Massive improvement from +49% (old v2_no_gate) to +122% — driven by expanded model set + simplified sizing.
- v11_hold4_combo has best PF (1.79) and highest avg trade (+0.733%).

## Key Findings

1. **Model count matters most**: Going from 8→19 base models (directional) or 14→19 (straddle) is the single biggest improvement.
2. **Simplified sizing helps straddle**: Removing vol_gate and using pure confidence sizing outperforms vol×dir product sizing.
3. **Higher threshold floor (0.56)** gives marginal improvement on both strategies.
4. **Convex sizing** halves absolute return but maintains Sharpe — good for risk-averse deployment.
5. **Longer hold periods** (4-5 bars) don't help much — the 3-bar hold with early exit is already optimal.
6. **Both strategies now significantly outperform B&H** (-17.7%) by 100%+ with realistic execution.

## Next Steps

- Update main WFO scripts with best iteration configs
- Consider running both strategies in parallel (uncorrelated signals)
- Test on additional assets/timeframes
