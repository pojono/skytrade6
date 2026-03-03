# FINDINGS — Stage A: Tail-Asymmetry Analysis on REG_OI_FUND

**Date:** 2026-03-03  
**Period:** 2026-01-01 → 2026-02-28  
**Coins:** 8 altcoins  
**Baseline:** Volatility-matched (ATR quintile + hour-of-day), 10x oversampling  
**Stats:** Binomial test, bootstrap CI, BH FDR correction

---

## 1) Result: SCENARIO 1 — No Tail Uplift

**PASS=0, WEAK=3, FAIL=5**

**Zero significant tail uplift hits** at (q_fdr < 0.10, regime_prob ≥ 5%, uplift ≥ 2x).

REG_OI_FUND does NOT amplify the probability of extreme moves when controlling for volatility.

---

## 2) Per-Symbol Verdict

| Symbol | Verdict | Best uplift | H | k (ATR) | N regime |
|--------|---------|---:|---:|---:|---:|
| 1000BONKUSDT | WEAK | 1.93x | 30m | 4 | 14 |
| AIXBTUSDT | WEAK | 1.74x | 15m | 4 | 30 |
| 1000RATSUSDT | WEAK | 1.71x | 60m | 4 | 20 |
| ARCUSDT | FAIL | 1.24x | 30m | 2 | 61 |
| 1000TURBOUSDT | FAIL | 1.20x | 15m | 3 | 14 |
| ARBUSDT | FAIL | 1.17x | 15m | 4 | 52 |
| ATOMUSDT | FAIL | 1.09x | 30m | 3 | 68 |
| APTUSDT | FAIL | 1.01x | 60m | 2 | 54 |

The 3 "WEAK" coins (BONK, AIXBT, RATS) show ~1.7-1.9x uplift at 4 ATR threshold, but:
- Small N (14-30 regime signals)
- None pass FDR correction (all q > 0.88)
- Likely noise in small samples

---

## 3) Cross-Symbol Average Uplift

| Horizon | 1 ATR | 2 ATR | 3 ATR | 4 ATR |
|---------|---:|---:|---:|---:|
| 15m | 1.06x | 1.09x | 1.12x | 1.14x |
| 30m | 0.98x | 1.05x | 1.06x | 1.06x |
| 60m | 1.03x | 1.07x | 1.12x | 1.16x |

Average uplift is **1.0–1.2x** everywhere. Even at 4 ATR (rare tail), only 1.16x.
This is indistinguishable from noise after FDR correction.

---

## 4) Why No Tail Uplift Despite Range Expansion?

The range expansion (1.5-3.9x from regime research) was measured **in raw basis points**.

When we normalize by ATR (i.e., control for the volatility that already exists at signal time), the tails are NOT elevated. This means:

> REG_OI_FUND identifies periods that already HAVE high ATR.
> The range expansion is a selection effect, not a causal amplification.

In other words: OI_FUND fires when the market is already volatile, and the forward range simply reflects that existing volatility. The regime doesn't create NEW tail risk beyond what ATR already predicts.

---

## 5) Directional Tail Asymmetry (§8)

| Symbol | Bias (60m) | Pattern |
|--------|-----------|---------|
| 1000TURBOUSDT | UP | P(ret>k) > P(ret<-k) at all k |
| ARBUSDT | UP | P(ret>k) > P(ret<-k) at all k |
| AIXBTUSDT | DOWN | P(ret<-k) > P(ret>k) at all k |
| APTUSDT | DOWN | Mild down bias at k≥2 |
| ATOMUSDT | DOWN | Mild down bias at k=2-3 |
| 1000BONKUSDT | NEUTRAL | Symmetric |
| 1000RATSUSDT | NEUTRAL | Symmetric |
| ARCUSDT | NEUTRAL | Symmetric |

Directional biases exist but are **coin-specific** (consistent with earlier findings) and the uplifts are small (0.5-1.5x), not actionable.

---

## 6) Conditional Expectation (§9)

When tails DO occur (|ret| > k ATR), the conditional mean depth is similar between regime and baseline:

- At k=3 ATR, 60m: regime CE ≈ 7-8 ATR, baseline CE ≈ 7-9 ATR
- No evidence that regime tails are "deeper" than baseline tails

---

## 7) What This Means for Strategy

### Dead ends (confirmed):
1. **Breakout** — killed by whipsaw (proved earlier)
2. **Fade-first-break** — killed by continuation before retrace (proved earlier)
3. **Tail capture** — killed by this analysis: no tail uplift after vol normalization

### The fundamental problem:
REG_OI_FUND is a **high-ATR selector**, not a **tail amplifier**. It correctly identifies volatile periods, but ATR already captures this. There's no residual edge in the tails beyond what a simple "ATR is high" filter would give.

### Remaining viable use:
- **Risk management signal**: when OI_FUND fires, expect 1.5-3.9x raw range → widen stops / reduce size
- **Regime label for other models**: combine with other factors that DO have edge

---

## 8) Files

| File | Description |
|------|-------------|
| `flow_research/tail_asymmetry.py` | Full analysis script |
| `flow_research/output/regime/tail_uplift.csv` | Per (symbol, horizon, threshold) results |
| `flow_research/output/regime/tail_summary.csv` | Per-symbol verdict |
