# Phases 17–21: Curve Flattening Research

**Date:** 2026-03-06
**Goal:** Reduce the lumpiness of the equity curve — fewer monster months, fewer bad months.

---

## Background

The Phase 16 strategy (Sharpe 3.47, MaxDD -32.7%) generates ~70% of its profit in 3–4 months
(Oct 2025 +173%, Jan 2026 +85%, Sep 2025 +8%). This creates an irregular equity curve.

Four approaches were tested to flatten it.

---

## Phase 17: Adaptive N (Variable Portfolio Width)

**Idea:** Use more positions (N=20) when signal dispersion is low, fewer (N=5) when high.

| Variant | Sharpe | MaxDD | $1k→ |
|---------|--------|-------|------|
| Fixed N=5 | **3.675** | -58.9% | $191,186 |
| Fixed N=10 (baseline) | 3.470 | -32.7% | $22,067 |
| Fixed N=15 | 3.176 | -27.4% | $4,511 |
| Fixed N=20 | 2.502 | -23.6% | $2,417 |
| Adaptive N (5/10/20) | 3.057 | -34.7% | $18,369 |
| Adaptive N (5/15) | 3.017 | -34.2% | $13,626 |

**Key findings:**
- N=5 gives the highest Sharpe (3.675) but also worst MaxDD (-58.9%). Oct 2025 was +506%.
- Adaptive N (switching between 5 and 20) **hurts** — diluting with N=20 in weak regimes reduces returns without reducing MaxDD
- Smaller N = higher alpha but more volatile. There is a clear N vs stability trade-off.
- **Verdict:** Adaptive N does not flatten the curve — it just changes the risk level. Use N=10 or N=15 based on risk preference, not dynamically.

---

## Phase 18: Rolling Sharpe Sizing

**Idea:** Scale position size based on recent 30-bar (10-day) Sharpe.

- **A. Edge-scaling:** High Sharpe → up to 1.5×; negative Sharpe → 0.5×
- **B. Harvest mode:** Cap monthly gains at 30%, then scale down to 25%
- **C. Inverse scaling:** Reduce to 0.5× when Sharpe > 5 (hot streak) OR Sharpe < 0 (bad regime)
- **D. Harvest + Inverse:** Both combined

| Variant | Sharpe | Sortino | MaxDD | $1k→ |
|---------|--------|---------|-------|------|
| Baseline (1×) | 3.470 | 5.795 | -32.7% | $22,067 |
| A: Edge-scaling | 3.005 | 4.439 | -36.4% | $15,150 |
| B: Harvest mode | 3.296 | 5.467 | -32.7% | $11,397 |
| **C: Inverse scaling** | **3.444** | **5.777** | **-20.0%** | $9,105 |
| D: Harvest + Inverse | 3.412 | 6.139 | -21.5% | $7,694 |

**Key findings:**
- **Inverse scaling (C)** reduces MaxDD from -32.7% to -20.0% with minimal Sharpe cost (3.44 vs 3.47)
- The magic is in scaling DOWN when the strategy is hot (Sharpe > 5) — the subsequent reversals are dampened
- D has the highest Sortino (6.139) — best downside-adjusted performance
- **Verdict: Inverse scaling is effective. Harvest + Inverse gives best risk-stability.**

---

## Phase 19: Mean-Reversion Counter-Cycle Layer

**Idea:** Add a layer using -mom_8h (short recent winners, long recent losers) to offset momentum crashes.

| Variant | Sharpe | MaxDD | $1k→ |
|---------|--------|-------|------|
| Main strategy | 3.470 | -32.7% | $22,067 |
| MR only | **-1.506** | -93.1% | $91 |
| 90% Main + 10% MR | 3.314 | -33.9% | $14,100 |
| 80% Main + 20% MR | 3.037 | -38.0% | $8,792 |

**Key findings:**
- The MR layer (negative 8h momentum) is **catastrophically bad** as a standalone: Sharpe -1.506, $1k → $91
- Correlation with main strategy: -0.18 (slightly negative, theoretically good)
- But the MR layer is so bad that every blend with it hurts
- Phase 1 showed mom_8h has ICIR -0.172 — it mean-reverts, but apparently not consistently enough to trade profitably after fees
- **Verdict: Reject entirely. MR layer destroys value.**

---

## Phase 20: Funding Saturation Gate

**Idea:** Reduce exposure when universe average funding is at extremes (overheated or too low).

| Variant | Sharpe | MaxDD | $1k→ |
|---------|--------|-------|------|
| Baseline | 3.470 | -32.7% | $22,067 |
| A: High-funding gate (>95%) | 3.382 | -32.5% | $18,907 |
| B: Low-funding gate (<10%) | 3.472 | -31.7% | $18,877 |
| C: Sweet-zone gate | 3.385 | -31.7% | $16,174 |
| E: Continuous scale | 3.262 | -25.2% | $8,961 |

**Critical finding:** High-funding bars (>95th percentile) earn **46 bps/bar** on average vs 27 bps otherwise. The strategy makes MORE money when funding is extreme, not less. Gating on high funding would skip the best bars.

Low-funding bars still earn 24.5 bps — almost as good as average. The gate improves MaxDD marginally (-31.7% vs -32.7%).

**Verdict:** Marginal improvement from low-funding gate. High-funding gate is counterproductive. Not worth the complexity.

---

## Phase 21: Final Stable Strategy (Combined Best)

Combines Phase 16 signal with inverse scaling (Phase 18C).

| Strategy | Sharpe | Sortino | MaxDD | Positive months | $1k→ |
|----------|--------|---------|-------|----------------|------|
| Baseline N=10 (Phase 16) | 3.470 | 5.795 | -32.7% | 11/15 (73%) | $22,067 |
| **N=10 + Inverse scale** | **3.546** | **5.997** | **-21.1%** | **13/15 (87%)** | $10,404 |
| N=10 + Harvest+Inverse | 3.379 | 5.997 | -19.9% | — | $7,116 |
| N=10 + Low-funding gate | 3.472 | 5.826 | -31.7% | — | $18,877 |
| N=15 + Inverse scale | 3.644 | 6.184 | -15.5% | — | $3,479* |

*N=15 has data gaps in first 5 months of 2025 (insufficient valid composite values).

### RECOMMENDED: N=10 + Inverse Scaling

**Inverse scaling rule:**
```python
rolling_sharpe = rets.rolling(30).apply(sharpe_fn)

if rolling_sharpe > 5:
    scale = 0.5    # hot streak — harvest
elif rolling_sharpe > 1:
    scale = 1.0    # normal — full exposure
elif rolling_sharpe > 0:
    scale = 0.75   # weak — slight reduce
else:
    scale = 0.5    # negative — protect capital
```

**Performance vs baseline:**
| Metric | Baseline (Phase 16) | **Final Stable** | Change |
|--------|--------------------|--------------------|--------|
| Sharpe | 3.470 | **3.546** | **+2.2%** |
| Sortino | 5.795 | **5.997** | +3.5% |
| MaxDD | -32.7% | **-21.1%** | **-35%** |
| Positive months | 11/15 (73%) | **13/15 (87%)** | +14pp |
| Worst month | -18.0% | **-6.7%** | **63% better** |
| Best month | +172.6% | +107.9% | still excellent |
| Monthly vol | ~40%+ | **28.6%** | much smoother |

**Why this works:**
- In monster months (Oct +173%), the rolling Sharpe shoots above 5 → scale drops to 0.5×
- This automatically harvests profits and reduces exposure heading into the reversal
- In bad months (Jul -18%), rolling Sharpe goes negative → scale drops to 0.5×
- Position sizes shrink exactly when conditions are worst

**The result:** Both Sharpe AND MaxDD improve simultaneously — a genuine free lunch from timing.

---

## Overall Conclusions

| Idea | Works? | Reason |
|------|--------|--------|
| Adaptive N | No | Switching to N=20 in weak regimes dilutes returns |
| Vol-targeting (Phase 12) | No | Cuts allocation in best months |
| Harvest mode | Partial | Reduces best months without improving Sharpe |
| **Inverse scaling** | **YES** | Improves Sharpe AND MaxDD |
| MR counter-cycle layer | No | Catastrophically bad standalone |
| Funding saturation gate | Marginal | Tiny MaxDD improvement, high-funding gate is harmful |

**Single best improvement: Inverse scaling based on rolling 30-bar Sharpe.**
