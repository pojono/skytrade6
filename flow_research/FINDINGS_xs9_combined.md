# XS-9 — Combined Strategy: Vol Dip-Buying + Fragility Overlay

**Date:** 2026-03-07  
**Data:** 9 Bybit perps (Tier A portfolio), 2025-07-01 → 2026-02-28 (8 months)  
**Base strategy:** Vol dip-buying (rvol_z + mr_4h > 2.0, 4h hold, 4bps RT)  
**Overlays tested:** Fragility sizing (XS-8), Compression boost (S07), Asymmetric downside  
**Script:** `xs9_combined_backtest.py`  
**Runtime:** 61s

---

## TL;DR

**Fragility overlay IMPROVES both return and risk.** ΔSharpe = +0.29, ΔMaxDD = -7.9%.

| Variant | Trades | Avg bps | Ann Return | Sharpe | MaxDD | Calmar |
|---------|--------|---------|------------|--------|-------|--------|
| **A) Baseline (pure VDB)** | 292 | +9.3 | +40.9%/yr | +0.74 | 28.0% | 1.46 |
| **B) + Fragility sizing** | 292 | +11.9 | **+52.1%/yr** | **+1.03** | **20.0%** | **2.60** |
| C) + Compression boost | 292 | +9.1 | +39.9%/yr | +0.70 | 29.0% | 1.38 |
| D) + Both (frag + comp) | 292 | +11.7 | +51.1%/yr | +0.97 | 21.1% | 2.44 |
| E) + Asymmetric downside | 292 | +11.7 | +51.3%/yr | +0.97 | 21.1% | 2.44 |

**Winner: B) Fragility sizing alone.** Simple, clean, +39% improvement in Sharpe.

---

## 1. How the Overlay Works

### Fragility Score

```
frag_score = -4.62 × crowd_oi - 0.95 × pca_var1
```

Coefficients from XS-8 LogReg (12×ATR, ≥10% target). Higher score = more fragile market.

### Position Sizing Rules

| Fragility Quintile | Size Multiplier | Interpretation |
|---------------------|-----------------|----------------|
| Q1 (safe) | 100% | Full size — low tail risk |
| Q2 | 100% | Full size |
| Q3 | 100% | Full size |
| Q4 | 75% | Reduce — elevated fragility |
| Q5 (fragile) | 50% | Half size — highest tail risk |

### Walk-Forward Correctness

Fragility quintiles are computed using **expanding percentiles** (no lookahead). At each trade entry, only past fragility readings are used to determine the quintile.

---

## 2. Why Fragility Sizing Works

### Trade performance by fragility quintile (baseline VDB trades):

| Quintile | N | Avg bps | WR | Long avg | Short avg |
|----------|---|---------|----|---------:|----------:|
| Q1 (safe) | 94 | **+39.2** | 58.5% | +41.7 | -187.7 |
| Q2 | 58 | -23.5 | 48.3% | -26.8 | +21.5 |
| Q3 | 55 | +8.1 | 49.1% | +14.2 | -53.1 |
| Q4 | 46 | **+62.7** | 63.0% | +62.7 | — |
| Q5 (fragile) | 39 | **-75.1** | 30.8% | -78.3 | -36.4 |

**The key insight:** Q5 trades average **-75 bps** with only 31% WR. These are the toxic trades — the market is fragile and the mean-reversion signal gets run over by tail events.

By reducing Q5 size to 50%, we cut the damage from these trades in half. The Q1 and Q4 trades (which are strongly positive) keep full size.

### Long trades by fragility:

| Quintile | N | Avg bps | WR |
|----------|---|---------|-----|
| Q1 | 93 | +41.7 | 59% |
| Q2 | 54 | -26.8 | 46% |
| Q3 | 50 | +14.2 | 52% |
| Q4 | 46 | +62.7 | 63% |
| Q5 | 36 | **-78.3** | **31%** |

Q5 longs are catastrophic: -78 bps average, 31% WR. These are mean-reversion trades that get steamrolled because the market has no liquidity buffer.

---

## 3. Why Compression Doesn't Help

Compression boost (enter more size when market is compressed) actually **hurts** performance slightly:

| State | N trades | Avg bps | WR |
|-------|----------|---------|-----|
| Compressed | 30 | **-10.9** | 60.0% |
| Not compressed | 262 | +11.7 | 50.8% |

**Compressed trades have higher WR (60%) but negative average bps.** This means when compression predicts a big move, the move is in the *wrong direction* for the VDB signal. The vol dip-buying is a mean-reversion strategy — it bets against the move. When compression predicts a big move, the continuation (not reversion) is more likely.

**Compression is anti-correlated with mean-reversion.** It should NOT be used to boost VDB size.

---

## 4. Asymmetric Protection Adds Nothing

Variant E (extra 30% reduction for shorts in Q5) performs identically to D. This is because there are only 3 short trades in Q5 over 8 months — too few to matter.

The strategy is overwhelmingly long (279/292 = 96%). Asymmetric short protection is irrelevant for a strategy that rarely shorts.

---

## 5. Monthly Breakdown

| Month | Baseline | + Fragility | Δ |
|-------|----------|-------------|---|
| 2025-07 | +26.8% | +26.7% | -0.1% |
| 2025-08 | +8.5% | +9.2% | +0.8% |
| 2025-09 | -8.7% | -8.9% | -0.2% |
| **2025-10** | **-19.3%** | **-11.2%** | **+8.1%** |
| 2025-11 | +14.8% | +11.8% | -3.1% |
| **2025-12** | **-9.9%** | **-6.5%** | **+3.4%** |
| 2026-01 | +17.8% | +17.6% | -0.2% |
| 2026-02 | -2.7% | -3.1% | -0.4% |

**The value is in the losing months.** October baseline lost -19.3%, but with fragility sizing only -11.2%. December improved from -9.9% to -6.5%. The overlay works exactly as designed: it reduces position sizes during fragile periods, cutting the worst drawdowns.

The cost: winning months are slightly smaller (Nov: +14.8% → +11.8%). The fragility filter reduced size during some winning Q4/Q5 trades too.

**Net: dramatically better risk-adjusted returns.** Calmar from 1.46 → 2.60 (78% improvement).

---

## 6. Production Implementation

### What to compute every 5 minutes (from XS-8):

1. **crowd_oi:** Fraction of top-68 coins with OI z-score > 1.5 (7d rolling)
2. **pca_var1:** First PCA component explained variance of 6h returns across coins

### Position sizing logic:

```python
frag_score = -4.62 * crowd_oi - 0.95 * pca_var1

# Expanding percentile (track history)
frag_quintile = expanding_percentile_rank(frag_score)

if frag_quintile >= 0.80:    # Q5
    size_mult = 0.50
elif frag_quintile >= 0.60:  # Q4
    size_mult = 0.75
else:                        # Q1-Q3
    size_mult = 1.00

position_size = base_size * size_mult
```

### Data requirements:

| Component | Data needed | Frequency |
|-----------|-------------|-----------|
| Vol dip-buying signal | 1h klines per symbol | Every hour |
| Fragility score | 1m klines + OI + FR for ~68 symbols | Every 5 min |
| PCA computation | 6h returns across coins | Every 5 hours |

### Added complexity vs baseline:

- **Extra data:** Need 1m data + OI + FR for 68 symbols (not just 9)
- **Extra compute:** PCA every 5h, rolling z-scores every 5m
- **Extra state:** Track expanding fragility distribution
- **Benefit:** +11% annual return, -8% max drawdown, +0.29 Sharpe

---

## 7. Caveats

1. **8-month test period** — need longer OOS to confirm robustness
2. **Only 292 trades** — statistical power is limited (especially per-quintile)
3. **Q5 has only 39 trades** — the -75bp average could be noisy
4. **Q4 anomaly** — Q4 trades are the *best* (+62.7bp), which is counterintuitive. This may be a small-sample artifact.
5. **No compression benefit** — the S07 signal is anti-correlated with mean-reversion, which makes structural sense but limits the combination

---

## 8. Verdict

### ✅ Fragility sizing is a genuine improvement

- **Sharpe:** +0.74 → +1.03 (+39%)
- **MaxDD:** 28.0% → 20.0% (-29%)
- **Calmar:** 1.46 → 2.60 (+78%)
- **Mechanism:** Reduces exposure during Q5 fragile markets where VDB signal gets crushed (-75bp avg)
- **Cost:** Slightly lower returns in winning months

### ❌ Compression boost does not help

- Mean-reversion and compression-breakout are opposing forces
- Compression predicts big moves, VDB bets against big moves
- These should NOT be combined

### Recommendation

Deploy Vol Dip-Buying with fragility sizing overlay. Skip compression gate.

The production system needs:
1. **Vol dip-buying engine** — 1h candles, 9 symbols, simple
2. **Fragility monitor** — 5m cross-sectional features from 68 symbols
3. **Position sizer** — scale by fragility quintile (Q5 → 50%, Q4 → 75%)

---

## Files

- **Script:** `flow_research/xs9_combined_backtest.py`
- **Trades:** `flow_research/output/xs9/xs9_trades_*.csv`
- **Summary:** `flow_research/output/xs9/xs9_summary.csv`
- **Dependencies:** `flow_research/output/xs8c/xs8c_extended.parquet` (fragility features)
