# Grid Bot Research Summary — v15 + v16 + v17

**Date:** 2026-02-16
**Period:** 2025-01-01 → 2026-01-31 (387 sim days after warmup)
**Setup:** $10K capital, 5 grid levels, 2bps maker fees (Bybit VIP0)

---

## Bug Post-Mortem

v16 and v17 originally used a **broken simulator** that lacked:
1. Level deactivation (same level filled every bar → 47× inflated fills)
2. Position sizing (raw price units instead of USD)
3. Paired buy/sell activation

This made the baseline appear as a massive loss (-$290K on SOL) when it's actually
profitable (+$2,151). All v16/v17 findings were re-run with the correct v15 simulator
on 2026-02-16. The results below are all from the **corrected** simulator.

---

## Baseline: Fix 1.00% (24h)

| Symbol | PnL | PnL/day | Sharpe | MaxDD | Fills/day |
|--------|-----|---------|--------|-------|-----------|
| **SOL** | **+$2,151** | +$5.55 | 0.85 | -$2,424 | 9.3 |
| **BTC** | **+$789** | +$2.04 | 0.83 | -$837 | 4.4 |
| **ETH** | **+$901** | +$2.33 | 0.45 | -$2,002 | 8.0 |

The fixed 1% grid with 24h rebalance is **profitable on all 3 assets** and
surprisingly hard to beat.

---

## Winners by Asset

### SOLUSDT — Best: S7 Asymmetry Adjust (v15)

| Strategy | PnL | PnL/day | Sharpe | MaxDD | vs Base |
|----------|-----|---------|--------|-------|---------|
| S0: Baseline | +$2,151 | +$5.55 | 0.85 | -$2,424 | — |
| **S7: Asymmetry adj** | **+$2,451** | **+$6.33** | **0.90** | -$2,624 | **+$300** |
| N3: Efficiency adapt (v16) | +$2,299 | +$5.93 | 0.89 | -$2,136 | +$148 |
| N4: VPIN adapt (v16) | +$2,179 | +$5.63 | 0.88 | -$2,587 | +$28 |

S7 adjusts grid asymmetry based on predicted upside fraction — tightens when
symmetric (more fills), widens when trending (less adverse selection).
Only works on SOL (fails on BTC/ETH).

### BTCUSDT — Best: S5 Adaptive Rebalance (v15)

| Strategy | PnL | PnL/day | Sharpe | MaxDD | vs Base |
|----------|-----|---------|--------|-------|---------|
| S0: Baseline | +$789 | +$2.04 | 0.83 | -$837 | — |
| **S5: Adaptive rebal** | **+$1,355** | **+$3.50** | **1.84** | **-$715** | **+$567** |
| C1: Toxic+AdaptRebal (v16) | +$1,281 | +$3.31 | 1.75 | -$715 | +$492 |
| N6: Multifractal (v16) | +$849 | +$2.19 | 0.90 | -$800 | +$60 |

S5 uses vol prediction to adapt rebalance timing: 8h during high vol, 48h during
calm periods. On BTC, the 24h default is too frequent — extending to 48h in calm
periods lets profitable inventory accumulate longer.

### ETHUSDT — Best: S5 Adaptive Rebalance (v15) / C3 VPIN+Toxic+AdRebal (v16)

| Strategy | PnL | PnL/day | Sharpe | MaxDD | vs Base |
|----------|-----|---------|--------|-------|---------|
| S0: Baseline | +$901 | +$2.33 | 0.45 | -$2,002 | — |
| **S5: Adaptive rebal** | **+$1,136** | **+$2.93** | **0.60** | -$1,773 | **+$234** |
| **C3: VPIN+Toxic+AdRebal** | **+$1,228** | **+$3.17** | **0.65** | -$1,786 | **+$326** |
| C1: Toxic+AdaptRebal (v16) | +$1,131 | +$2.92 | 0.60 | -$1,757 | +$230 |

C3 combines VPIN-based spacing adjustment + toxic flow pause + adaptive rebalance.
Slightly better than S5 alone on ETH, but the improvement is marginal.

---

## Best Universal Strategy: S5 Adaptive Rebalance (v15)

If you must pick ONE strategy for all assets:

| Symbol | S5 PnL | S5 Sharpe | vs Base PnL | vs Base Sharpe |
|--------|--------|-----------|-------------|----------------|
| BTC | +$1,355 | **1.84** | +$567 (+72%) | +122% |
| ETH | +$1,136 | 0.60 | +$234 (+26%) | +33% |
| SOL | +$695 | 0.30 | -$1,456 (-68%) | -65% |

S5 is the best risk-adjusted strategy on BTC (Sharpe 1.84) and ETH, but
**hurts SOL** where the 24h rebalance is already near-optimal.

---

## What DOESN'T Work

### Regime filtering / pausing (v17) — HARMFUL

Every form of pausing (vol, efficiency, ADX, parkvol) with force-close-on-pause
**destroys profitability**. The grid's natural mean-reversion cycle is interrupted,
and inventory is liquidated at the worst possible time (during vol spikes).

| Strategy | SOL PnL | vs Base |
|----------|---------|---------|
| R2: Vol pause (1.5x) | -$9,469 | -$11,620 |
| R7: Vol+Eff pause | -$11,218 | -$13,368 |
| V1: 30min rebal + vol pause | -$241 | -$2,392 |

### Informed rebalance (v16 N5/N5b) — HARMFUL

Triggers too many rebalances, generating excess fees and crystallizing losses
the grid would naturally recover from.

| Symbol | N5b PnL | vs Base |
|--------|---------|---------|
| SOL | -$1,260 | -$3,411 |
| BTC | +$132 | -$657 |
| ETH | -$1,187 | -$2,088 |

### Adaptive spacing (v15 S1/S1b/S2b) — HARMFUL

Tightening below 1% increases fills but each fill captures less spread.
The extra fills don't compensate. **1.00% is near-optimal.**

### Combining multiple ML signals (v15 C1/C3) — HARMFUL

Improvements are not additive — they interfere with each other.
Triple combos (S5+S3+S7) are worse than any individual component.

---

## Final Recommendations

### Per-asset optimal configuration

| Asset | Strategy | Expected PnL/year | Sharpe |
|-------|----------|-------------------|--------|
| **BTC** | S5: Adaptive rebalance | ~+$1,355 | 1.84 |
| **ETH** | S5: Adaptive rebalance | ~+$1,136 | 0.60 |
| **SOL** | S7: Asymmetry adjust | ~+$2,451 | 0.90 |

### Universal fallback

**Fix 1.00% (24h)** — profitable on all 3 assets with no ML required.

### Key principles learned

1. **The fixed grid is a strong baseline** — hard to beat with any signal
2. **Rebalance timing > grid spacing** — when to cut inventory matters more than grid width
3. **Less is more** — simpler strategies outperform complex ML combos
4. **Don't pause a profitable grid** — regime filtering destroys the mean-reversion cycle
5. **Asset-specific tuning matters** — no single enhancement works everywhere

---

## Files

| File | Description |
|------|-------------|
| `grid_bot_v15.py` | ML improvements (7 strategies + 4 combos) — **correct simulator** |
| `grid_bot_v16.py` | Novel microstructure signals (10 strategies + 5 combos) — **fixed** |
| `grid_bot_v17.py` | Regime filtering (35+ strategies) — **fixed** |
| `FINDINGS_v15_ml_grid.md` | v15 findings (original, correct) |
| `FINDINGS_v16_novel_grid.md` | v16 findings (corrected 2026-02-16) |
| `FINDINGS_v17_regime_grid.md` | v17 findings (corrected 2026-02-16) |
| `results/grid_v15_*.txt` | v15 raw results |
| `results/grid_v16_*_fixed.txt` | v16 corrected results |
| `results/grid_v17_SOL_fixed.txt` | v17 corrected results |
