# Research Findings v15 — ML-Powered Grid Bot Improvements

**Date:** 2026-02-16
**Method:** Incremental A/B testing of 7 ML enhancements vs Fix 1.00% (24h) baseline
**Period:** 2025-01-01 → 2026-01-31 (387 sim days after warmup)
**Fee:** 2 bps maker per fill (Bybit VIP0)
**Quick validation:** SOLUSDT (best grid bot asset), then BTC + ETH for top candidates

---

## Baseline

**Fix 1.00% (24h):** Fixed 1% grid spacing, 24h rebalance interval.

| Symbol | PnL | Sharpe | MaxDD |
|--------|-----|--------|-------|
| BTC | +$789 | 0.83 | -$837 |
| ETH | +$901 | 0.45 | -$2,002 |
| SOL | +$2,151 | 0.85 | -$2,424 |

---

## Step-by-Step Results (SOLUSDT quick validation)

| Step | Strategy | PnL | Sharpe | MaxDD | vs Base | Verdict |
|------|----------|-----|--------|-------|---------|---------|
| S0 | Fix 1.00% (24h) | +$2,151 | 0.85 | -$2,424 | — | Baseline |
| S1 | Direct range/2 f1.0% | +$1,318 | 0.53 | -$2,395 | -$833 | ❌ Worse |
| S1b | Direct range/2 f0.5% | +$963 | 0.35 | -$2,397 | -$1,187 | ❌ Worse |
| S2 | P90 range/2 f1.0% | +$1,091 | 0.44 | -$1,730 | -$1,059 | ❌ Worse (but best MaxDD) |
| S2b | P90 range/2 f0.5% | +$842 | 0.32 | -$1,782 | -$1,309 | ❌ Worse |
| S3 | Breakout widen 2× | +$2,024 | 0.81 | -$2,424 | -$127 | ⚠️ Close |
| S3b | Breakout pause | +$1,732 | 0.69 | -$2,424 | -$418 | ❌ Worse |
| S4 | Consolidation tighten | +$1,313 | 0.51 | -$3,035 | -$838 | ❌ Worse |
| S4b | Consol + Brk combo | +$1,061 | 0.42 | -$3,035 | -$1,089 | ❌ Worse |
| S5 | Adaptive rebalance | +$695 | 0.30 | -$2,757 | -$1,456 | ❌ Worse on SOL |
| S6 | 4h-range rebalance | +$618 | 0.41 | -$2,934 | -$1,532 | ❌ Worse |
| S7 | Asymmetry adjust | +$2,451 | 0.90 | -$2,624 | +$300 | ✅ Better on SOL |

**SOL quick validation:** Only S7 beat baseline. S3 was close. S5 hurt badly.

---

## Cross-Symbol Validation (BTC, ETH, SOL)

Ran all strategies on BTC and ETH to check robustness:

### S5: Adaptive Rebalance (vol-based: 8h/24h/48h)

| Symbol | PnL | Sharpe | MaxDD | vs Base |
|--------|-----|--------|-------|---------|
| **BTC** | **+$1,355** | **1.84** | -$715 | **+$567** ✅ |
| **ETH** | **+$1,136** | **0.60** | -$1,773 | **+$234** ✅ |
| SOL | +$695 | 0.30 | -$2,757 | -$1,456 ❌ |

**Best on BTC (Sharpe 1.84!) and ETH, but worst on SOL.** The adaptive rebalance helps on lower-vol assets (BTC, ETH) where the 24h fixed interval is too frequent, but hurts on high-vol SOL where frequent rebalancing is needed to limit inventory.

### S7: Asymmetry Adjustment (tighten when symmetric, widen when trending)

| Symbol | PnL | Sharpe | MaxDD | vs Base |
|--------|-----|--------|-------|---------|
| BTC | +$612 | 0.58 | -$914 | -$176 ❌ |
| ETH | -$24 | 0.02 | -$2,612 | -$925 ❌ |
| **SOL** | **+$2,451** | **0.90** | -$2,624 | **+$300** ✅ |

**Only works on SOL.** Not robust across assets.

### S3: Breakout Widen (2× spacing when breakout prob > 0.3)

| Symbol | PnL | Sharpe | MaxDD | vs Base |
|--------|-----|--------|-------|---------|
| BTC | +$857 | 0.91 | -$839 | +$69 ✅ |
| ETH | +$882 | 0.44 | -$2,002 | -$19 ≈ |
| SOL | +$2,024 | 0.81 | -$2,424 | -$127 ≈ |

**Consistently close to baseline, slightly positive on BTC.** Doesn't hurt, but doesn't help much either. The breakout model (AUC=0.65) isn't strong enough to make a meaningful difference.

### S1: Direct Range Prediction

| Symbol | PnL | Sharpe | MaxDD | vs Base |
|--------|-----|--------|-------|---------|
| BTC | +$827 | 0.87 | -$834 | +$38 ≈ |
| ETH | +$732 | 0.37 | -$2,001 | -$169 ❌ |
| SOL | +$1,318 | 0.53 | -$2,395 | -$833 ❌ |

**Marginal on BTC, worse on ETH/SOL.** Confirms v11 finding: direct range ≈ vol×k, no real improvement.

### S2: P90 Range (Ridge × 1.7)

| Symbol | PnL | Sharpe | MaxDD | vs Base |
|--------|-----|--------|-------|---------|
| **BTC** | **+$926** | **1.01** | -$828 | **+$137** ✅ |
| ETH | +$608 | 0.29 | -$2,104 | -$293 ❌ |
| SOL | +$1,091 | 0.44 | -$1,730 | -$1,059 ❌ |

**Best Sharpe on BTC (1.01) with lowest MaxDD.** The wider P90 grid reduces drawdowns but also reduces fills and total profit on higher-vol assets.

---

## Key Insights

### 1. No single ML improvement beats baseline on all 3 symbols

The fixed 1.00% grid is surprisingly hard to beat. Each ML enhancement helps on some assets but hurts on others:
- **S5 (adaptive rebalance):** Best for BTC/ETH, worst for SOL
- **S7 (asymmetry):** Best for SOL, worst for ETH
- **S2 (P90 range):** Best risk-adjusted for BTC, worse for SOL/ETH

### 2. Asset-specific optimal strategies

| Asset | Best Strategy | PnL | Sharpe |
|-------|-------------|-----|--------|
| **BTC** | S5: Adaptive rebalance | +$1,355 | **1.84** |
| **ETH** | S5: Adaptive rebalance | +$1,136 | 0.60 |
| **SOL** | S7: Asymmetry adjust | +$2,451 | 0.90 |

### 3. Why adaptive spacing hurts (S1, S1b, S2b)

All adaptive spacing strategies performed worse than fixed 1.00%. The reason: when the model predicts low vol, it tightens the grid below 1.00%, which increases fills but each fill captures less spread. The extra fills don't compensate for the reduced per-fill profit. **The 1.00% floor is already near-optimal.**

### 4. Why consolidation tightening hurts (S4)

Tightening during consolidation (low range_compression) increases fills but also increases MaxDD (-$3,035 vs -$2,424). Consolidation often precedes breakouts — tightening right before a breakout is the worst possible timing.

### 5. Why adaptive rebalance helps BTC but hurts SOL (S5)

- **BTC (low vol):** The 24h rebalance is too frequent — it closes profitable inventory too early. Extending to 48h during calm periods lets profits accumulate.
- **SOL (high vol):** The 24h rebalance is already too infrequent — SOL's large moves build dangerous inventory. Shortening to 8h during high vol helps, but the 48h calm periods let inventory grow too much.

### 6. Breakout detection is too weak to matter (S3)

The breakout model (AUC=0.65) correctly identifies ~55% of breakouts, but the false positive rate means it widens the grid unnecessarily ~30% of the time. The net effect is nearly zero.

---

## Combo Strategy Results (BTC, ETH, SOL)

Tested 4 combinations of the best individual strategies:

| Strategy | BTC PnL | BTC Sharpe | ETH PnL | ETH Sharpe | SOL PnL | SOL Sharpe | Beats base |
|----------|---------|-----------|---------|-----------|---------|-----------|------------|
| **S0: Baseline** | +$789 | 0.83 | +$901 | 0.45 | +$2,151 | 0.85 | — |
| **S5: AdaptRebal** | **+$1,355** | **1.84** | **+$1,136** | **0.60** | +$695 | 0.30 | 2/3 |
| S7: Asymmetry | +$612 | 0.58 | -$24 | 0.02 | **+$2,451** | **0.90** | 1/3 |
| C1: S5+S7 | +$730 | 0.89 | +$860 | 0.43 | -$274 | -0.07 | 0/3 |
| **C2: S5+S3** | **+$1,250** | **1.73** | **+$1,031** | **0.55** | +$903 | 0.39 | 2/3 |
| C3: S5+S3+S7 | +$609 | 0.78 | +$793 | 0.40 | -$224 | -0.05 | 0/3 |
| C4: S5+S2 | +$825 | 1.25 | +$477 | 0.29 | +$539 | 0.25 | 0/3 |

### Combo Takeaways

1. **S5 alone is still the best for BTC/ETH.** No combo improved on it.

2. **C2 (AdaptRebal + Breakout Widen) is the most robust combo** — beats baseline on BTC (+$462, Sharpe 1.73) and ETH (+$129, Sharpe 0.55), and doesn't crash SOL as badly as S5 alone (+$903 vs +$695). It's the best "universal" strategy.

3. **Adding S7 (asymmetry) to any combo destroys SOL.** S7 is SOL-specific — when combined with adaptive rebalance, the two fight each other and produce losses.

4. **More ML ≠ better.** The triple combo (C3) is worse than any individual component. The improvements are not additive — they interfere with each other.

---

## Conclusion

**The fixed 1.00% (24h) grid is a strong baseline that's hard to beat with ML.**

### Best universal strategy: C2 (Adaptive Rebalance + Breakout Widen)
- Beats baseline on BTC (+$462) and ETH (+$129)
- Positive on SOL (+$903) but below baseline (-$1,247)
- Uses vol prediction for rebalance timing + breakout detection for grid widening

### Best per-asset strategies:

| Asset | Strategy | PnL | Sharpe | Improvement |
|-------|----------|-----|--------|-------------|
| **BTC** | S5: Adaptive rebalance | +$1,355 | 1.84 | +72% PnL, +122% Sharpe |
| **ETH** | S5: Adaptive rebalance | +$1,136 | 0.60 | +26% PnL, +33% Sharpe |
| **SOL** | S7: Asymmetry adjust | +$2,451 | 0.90 | +14% PnL, +6% Sharpe |

### What ML adds to grid bots

The ML predictions (vol, range, breakout, asymmetry) provide:
- **Meaningful value for rebalance timing** (S5) on lower-vol assets (BTC, ETH)
- **Marginal value for grid spacing** — the fixed 1.00% is near-optimal
- **No value when combined naively** — improvements interfere with each other

**Final recommendation:** Per-asset configuration with S5 for BTC/ETH and S0/S7 for SOL.

---

## Files

| File | Description |
|------|-------------|
| `grid_bot_v15.py` | All 7 ML improvements + 4 combos + baseline |
| `results/grid_v15_combo_BTC.txt` | BTCUSDT full results (17 strategies) |
| `results/grid_v15_combo_ETH.txt` | ETHUSDT full results (17 strategies) |
| `results/grid_v15_combo_SOL.txt` | SOLUSDT full results (17 strategies) |
| `results/grid_v15_SOL.txt` | SOLUSDT quick validation (13 strategies) |
| `results/grid_v15_BTC.txt` | BTCUSDT initial cross-validation |
| `results/grid_v15_ETH.txt` | ETHUSDT initial cross-validation |
| `PLAN_v15_ml_grid_improvements.md` | Original plan with 7 steps |
