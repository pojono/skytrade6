# Phase 15: Capacity Model

**Date:** 2026-03-06
**Script:** `research_cross_section/phase15_capacity.py`

---

## Market Impact Model

```
impact_bps = 10 × sqrt(order_size_usd / daily_volume_usd)
```

- `order_size_usd` = AUM / 20 positions × turnover_fraction per rebal
- `daily_volume_usd` = sum of last 3 bars' 8h turnover (= 24h volume proxy)
- `IMPACT_COEFF = 10 bps` per sqrt(fraction of daily volume)

This is a standard square-root market impact model, conservative for limit orders.

---

## Results

| AUM | Net Sharpe | Ann Ret | MaxDD | vs Baseline |
|-----|-----------|---------|-------|------------|
| No impact (baseline) | 2.177 | +406% | -55.1% | — |
| $100k | 2.121 | +379% | -55.2% | -3% |
| $250k | 2.089 | +364% | -55.2% | -4% |
| $500k | 2.052 | +348% | -55.3% | -6% |
| $1M | 2.001 | +326% | -55.3% | -8% |
| $2M | 1.927 | +297% | -55.5% | -11% |
| $5M | 1.782 | +244% | -56.0% | -18% |
| $10M | 1.619 | +194% | -56.6% | -26% ← capacity limit |
| $25M | 1.294 | +114% | -59.4% | -41% |
| $50M | 0.929 | +50% | -65.8% | -57% |
| $100M | 0.412 | -9% | -73.2% | -81% |

**Capacity estimate (Sharpe -25% from baseline): ~$10M AUM**

---

## Key Findings

1. **Sweet spot: $1M–$5M AUM** — Sharpe 1.78–2.00, Ann Ret 244–326%
2. **Degradation accelerates above $10M** — liquidity in meme/small-cap coins limits scale
3. **$100M is breakeven** — fees + impact consume all alpha
4. **Small-cap concentration** — JELLYJELLY, COAI, PIPPIN, MYX have thin daily volumes ($1–10M/day), so position sizes exceeding ~$50k each trigger meaningful impact

---

## Practical Deployment Sizing

| Fund Stage | AUM | Expected Net Sharpe |
|-----------|-----|---------------------|
| Seed / family office | $500k | ~2.05 |
| Small fund | $2M | ~1.93 |
| Target optimal | $5M | ~1.78 |
| Hard limit | $10M | ~1.62 |

**Recommendation:** Deploy at $1–5M AUM initially. Monitor slippage per coin. Exclude coins where position size exceeds 2% of their daily volume.
