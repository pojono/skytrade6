# Phase 13: Improved Regime Filter

**Date:** 2026-03-06
**Script:** `research_cross_section/phase13_regime_v2.py`

---

## Variants Tested

| Variant | Sharpe | Ann Ret | MaxDD | Active% |
|---------|--------|---------|-------|---------|
| Baseline (no filter) | 2.177 | +405% | -55.1% | 100% |
| Phase-5 regime (symm, 30th pctile) | 1.825 | +188% | -50.3% | 23% |
| A: Asymmetric (short gate only) | 1.948 | +335% | -68.6% | 23% |
| B: BTC trend gate (asym short) | 1.418 | +157% | -72.7% | 16% |
| C: Soft threshold (scale 0.25–1.0x) | 1.976 | +139% | -25.3% | 100% |
| A+B: Asym + BTC trend | 1.418 | +157% | -72.7% | 16% |
| A+C: Asym + soft | 1.942 | +285% | -59.3% | 100% |

*Best by Sharpe: Baseline (no filter) at 2.177*

---

## Key Findings

### 1. Asymmetric filter makes MaxDD worse
Filtering only the short leg while leaving longs always on creates an unhedged long position during flat regimes. This increases MaxDD from -55% to -69% — the opposite of what we wanted. The short leg provides critical hedging. If you suppress it, you're just running a long-only crypto portfolio.

### 2. BTC trend gate is highly destructive
Adding a BTC uptrend veto on shorts further reduces active time to 16% and drops Sharpe to 1.418. The idea was sound (don't short alts when BTC is surging), but the implementation removes too many bars where the strategy actually profits from going short.

### 3. Soft threshold is the best regime variant
The soft threshold (confidence-scaled position size) achieves:
- MaxDD: -25.3% (vs -55.1% baseline) — **55% MaxDD reduction**
- Sharpe: 1.976 (vs 2.177 baseline) — moderate Sharpe cost
- Always active (no flat periods)

This is the most conservative deployment option: you're always in the market but sized small when regime conditions are unclear.

### 4. All filters reduce Sharpe on full-period data
The baseline wins on Sharpe in every comparison. Regime filtering always costs alpha because you inevitably miss some good bars when the filter is conservative.

---

## Recommendation

| Use Case | Recommended Config |
|----------|--------------------|
| Maximum alpha | No regime filter (full exposure) |
| Conservative / lower volatility | Soft threshold (scale 0.25–1.0x by confidence) |
| Binary on/off | Phase 5 walk-forward filter (active ~28% of bars) |

**Do NOT use asymmetric filtering (short-only) — it worsens MaxDD.**
**Do NOT use BTC trend gate — too blunt, destroys too much signal.**
