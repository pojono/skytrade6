# FINDINGS v17 — Regime-Filtered Grid Bot

> **⚠️ CORRECTED 2026-02-16:** Original v17 results used a broken simulator (same bugs as v16).
> All results below are from the **fixed simulator** ported from v15. The original narrative
> ("99.7% loss reduction") was comparing broken numbers. With the correct simulator, the
> baseline is **already profitable** (+$2,151 on SOL), and **all regime filtering strategies
> make things worse**. See "Bug Post-Mortem" in FINDINGS_v16.

## Motivation

v16 (with broken sim) suggested N5b informed rebalance reduced losses by 90%.
v17 explored regime filtering — pausing the grid during high-vol/trending periods
and force-closing inventory on pause transitions.

## Method

- **Vol pause:** Pause when Ridge-predicted vol > threshold × median
- **Efficiency pause:** Pause when backward-looking efficiency_4h > threshold (trending)
- **ADX pause:** Pause when ADX_4h > 0.30 (strong trend)
- **Parkvol pause:** Pause when parkvol_1h > 1.5× median (no ML, pure backward)
- **Informed rebalance:** Rebalance within 1h when composite informed flow z > 1.5
- **Short rebalance:** 30min, 1h, 2h base rebalance intervals
- **Force-close on pause:** When grid transitions to paused, close all inventory at market
- **Capital:** $10K, 5 levels, 2bps maker fees

## Results — SOLUSDT (387 days, $95→$294) — CORRECTED

### Every v17 strategy is worse than baseline

| Strategy | PnL | PnL/day | Fills | Sharpe | MaxDD | vs Base |
|----------|-----|---------|-------|--------|-------|---------|
| **S0: Fix 1% (24h)** | **+$2,151** | **+$5.55** | 3,597 | **0.85** | -$2,424 | — |
| Ref: N5b InfRebal | -$1,260 | -$3.25 | 8,134 | -0.53 | -$2,105 | -$3,411 |
| R1: Vol pause (2x) | -$8,710 | -$22.49 | 4,463 | -3.37 | -$8,922 | -$10,860 |
| R2: Vol pause (1.5x) | -$9,469 | -$24.45 | 4,390 | -5.18 | -$9,589 | -$11,620 |
| R3: Eff pause (>0.35) | -$3,415 | -$8.82 | 4,609 | -1.56 | -$4,695 | -$5,565 |
| R4: Eff pause (>0.30) | -$1,008 | -$2.60 | 5,003 | -0.30 | -$2,063 | -$3,158 |
| R6: Parkvol pause (1.5x) | -$11,340 | -$29.28 | 4,260 | -6.20 | -$11,523 | -$13,490 |
| C1: VolPause+InfRebal | -$2,668 | -$6.89 | 4,798 | -4.79 | -$2,695 | -$4,819 |
| U2: 1%+1hR+VolP | -$421 | -$1.09 | 4,052 | -1.43 | -$427 | -$2,572 |
| V1: 1%+30mR+VolP | -$241 | -$0.62 | 2,064 | -1.38 | -$256 | -$2,392 |
| V5: 1%+30mR+WideVP | -$71 | -$0.18 | 770 | -0.78 | -$111 | -$2,221 |

**Not a single v17 strategy achieves positive PnL.** The baseline (+$2,151) is the best.

### Why regime filtering destroys profitability

The force-close-on-pause mechanism is the culprit:
- When the grid pauses, it **closes all inventory at market price**
- In a bull market, inventory is typically long (buy fills dominate)
- Closing long inventory during a dip (which triggers the vol pause) **crystallizes losses**
- When the grid resumes, it rebuilds from scratch — missing the recovery
- The grid's natural mean-reversion (buy dips, sell rallies) is **interrupted** by pausing

**V5 (30min rebal + wide vol pause)** loses only -$71 over 387 days, but it only has
770 fills (2/day) — the grid barely runs. It's essentially "don't trade" which is
slightly better than "trade and lose from force-closing."

### The paradox

- **Without force-close:** Pausing just freezes inventory, which accumulates drift risk
- **With force-close:** Pausing crystallizes losses at the worst time (during vol spikes)
- **Either way:** Pausing hurts more than it helps when the baseline is already profitable

## Key Findings (CORRECTED)

### 1. The baseline grid bot is PROFITABLE — no enhancement needed

With the correct simulator, the Fix 1.00% (24h) grid produces:

| Symbol | PnL | PnL/day | Sharpe | MaxDD |
|--------|-----|---------|--------|-------|
| SOL | +$2,151 | +$5.55 | 0.85 | -$2,424 |
| BTC | +$789 | +$2.04 | 0.83 | -$837 |
| ETH | +$901 | +$2.33 | 0.45 | -$2,002 |

This is a **genuine, realistic result**: $10K capital, 2bps fees, 5 levels, 1% spacing.
The grid earns ~$5,876 in grid profits on SOL, pays ~$718 in fees, for net +$2,151.

### 2. Regime filtering is harmful in a bull market

Every form of pausing (vol, efficiency, ADX, parkvol) makes things worse.
The grid bot's natural behavior — buying dips and selling rallies — is already
the correct response to short-term mean-reversion within a trend.

Pausing interrupts this natural cycle and forces inventory liquidation at bad times.

### 3. Informed rebalance (N5b) is harmful

N5b triggers 21 fills/day vs baseline's 9.3. The excess rebalancing generates
fees and crystallizes small losses that the grid would naturally recover from.

### 4. Short rebalance intervals are harmful

Even without pausing, shorter rebalance intervals (1h, 30min) hurt because
they close inventory before the grid can complete round-trips.

### 5. The correct v15 findings hold

v15's conclusions are validated:
- **S5 (adaptive rebalance)** is best for BTC/ETH (Sharpe 1.84 on BTC)
- **S7 (asymmetry adjust)** is best for SOL (+$2,451, Sharpe 0.90)
- **Fixed 1.00% (24h)** is a strong universal baseline

## Conclusion

**The entire v17 research direction was misguided** — it was built on the false premise
(from the broken v16 simulator) that the baseline grid bot was losing money.

In reality, the baseline is profitable, and the correct research direction is:
1. **Use v15's per-asset strategies** (S5 for BTC/ETH, S7 for SOL)
2. **Do NOT add regime filtering or informed rebalance** — they destroy value
3. **Focus on other strategy types** (not grid bots) for further alpha

## Files

| File | Description |
|------|-------------|
| `grid_bot_v17.py` | Regime-filtered grid bot (fixed simulator) |
| `results/grid_v17_SOL_fixed.txt` | SOL results with correct simulator |
| `results/grid_v17_SOL.txt` | SOL results with broken simulator (historical) |
| `results/grid_v17_BTC.txt` | BTC results with broken simulator (historical) |
| `results/grid_v17_ETH.txt` | ETH results with broken simulator (historical) |
