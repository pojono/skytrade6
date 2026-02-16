# FINDINGS v16 — Novel Microstructure Signals for Grid Bot

> **⚠️ CORRECTED 2026-02-16:** Original v16 results used a broken simulator (no level
> deactivation, no position sizing, no paired buy/sell). All results below are from the
> **fixed simulator** ported from v15. The original findings (N5b "90% loss reduction")
> were artifacts of the broken sim. See "Bug Post-Mortem" section at bottom.

## Motivation

v6 tested novel academic signals (VPIN, toxic flow, herding, efficiency, entropy, multifractal)
as **directional trading strategies** → mostly failed (Sharpe < 0.10 after fees).

**Key insight:** These signals contain information about market *microstructure state*, not direction.
We repurpose them as **grid bot regime controllers**:

| Signal | Grid Bot Use | Rationale |
|--------|-------------|-----------|
| Toxic flow | **Pause** grid | Informed traders will pick off your limits |
| Herding runs | **Widen** grid | Momentum building, avoid getting run over |
| Efficiency | **Tighten** when low | Choppy = grid paradise; trending = grid death |
| VPIN | **Widen** when high | Informed trading = adverse selection risk |
| Composite informed | **Smart rebalance** | Cut inventory when informed flow spikes |
| Multifractal ratio | **Preemptive widen** | Clustered vol = spike incoming |

## Data & Method

- **Tick-level features** computed from raw trade data, aggregated to 5m bars
- Novel features: VPIN, toxic_flow, avg_run_length, run_imbalance, aggressive_ratio, size_entropy, multifractal_ratio
- Rolling z-scores (24h lookback) for threshold-based strategies
- Baseline: Fix 1.00% spacing, 24h rebalance ($10K capital, 5 levels, 2bps fees)
- Reference: S5 adaptive rebalance from v15

## Results — Individual Novel Strategies (CORRECTED)

### SOLUSDT (387 days, $95–$294)

| Strategy | PnL | PnL/day | Fills/d | Sharpe | MaxDD | vs Base |
|----------|-----|---------|---------|--------|-------|---------|
| S0: Fix 1.00% (24h) | +$2,151 | +$5.55 | 9.3 | 0.85 | -$2,424 | — |
| S5ref: AdaptRebal | +$695 | +$1.79 | 11.4 | 0.30 | -$2,757 | -$1,456 |
| N1: Toxic pause (z>2) | +$1,933 | +$4.99 | 9.3 | 0.77 | -$2,476 | -$218 |
| N1b: Toxic pause (z>1.5) | +$1,672 | +$4.32 | 9.2 | 0.67 | -$2,519 | -$479 |
| N2: Herding widen | +$2,050 | +$5.29 | 9.1 | 0.82 | -$2,411 | -$101 |
| **N3: Efficiency adapt** | **+$2,299** | **+$5.93** | 10.7 | **0.89** | -$2,136 | **+$148** ✅ |
| N4: VPIN adapt spacing | +$2,179 | +$5.63 | 9.5 | 0.88 | -$2,587 | +$28 |
| N5: Informed rebalance | +$359 | +$0.93 | 16.5 | 0.27 | -$2,264 | -$1,791 |
| N5b: Informed rebal soft | -$1,260 | -$3.25 | 21.0 | -0.53 | -$2,105 | -$3,411 |
| N6: Multifractal adapt | +$2,110 | +$5.45 | 9.3 | 0.84 | -$2,410 | -$40 |

### BTCUSDT (387 days, $74K–$126K)

| Strategy | PnL | PnL/day | Fills/d | Sharpe | MaxDD | vs Base |
|----------|-----|---------|---------|--------|-------|---------|
| S0: Fix 1.00% (24h) | +$789 | +$2.04 | 4.4 | 0.83 | -$837 | — |
| **S5ref: AdaptRebal** | **+$1,355** | **+$3.50** | 5.2 | **1.84** | -$715 | **+$567** ✅ |
| N1: Toxic pause (z>2) | +$706 | +$1.82 | 4.3 | 0.75 | -$878 | -$83 |
| N1b: Toxic pause (z>1.5) | +$625 | +$1.61 | 4.3 | 0.67 | -$922 | -$164 |
| N2: Herding widen | +$754 | +$1.95 | 4.2 | 0.81 | -$883 | -$35 |
| N3: Efficiency adapt | +$801 | +$2.07 | 5.4 | 0.79 | -$867 | +$13 |
| N4: VPIN adapt spacing | +$892 | +$2.30 | 4.6 | 0.88 | -$836 | +$103 |
| N5: Informed rebalance | -$288 | -$0.74 | 5.5 | -0.14 | -$1,064 | -$1,077 |
| N5b: Informed rebal soft | +$132 | +$0.34 | 5.7 | 0.40 | -$613 | -$657 |
| **N6: Multifractal adapt** | **+$849** | **+$2.19** | 4.4 | **0.90** | -$800 | **+$60** ✅ |

### ETHUSDT (387 days, $1,397–$4,951)

| Strategy | PnL | PnL/day | Fills/d | Sharpe | MaxDD | vs Base |
|----------|-----|---------|---------|--------|-------|---------|
| S0: Fix 1.00% (24h) | +$901 | +$2.33 | 8.0 | 0.45 | -$2,002 | — |
| **S5ref: AdaptRebal** | **+$1,136** | **+$2.93** | 9.3 | **0.60** | -$1,773 | **+$234** ✅ |
| N1: Toxic pause (z>2) | +$832 | +$2.15 | 7.9 | 0.42 | -$2,079 | -$69 |
| N1b: Toxic pause (z>1.5) | +$359 | +$0.93 | 7.8 | 0.19 | -$2,109 | -$542 |
| N2: Herding widen | +$781 | +$2.02 | 7.7 | 0.41 | -$2,016 | -$120 |
| N3: Efficiency adapt | +$380 | +$0.98 | 9.1 | 0.21 | -$2,013 | -$521 |
| N4: VPIN adapt spacing | +$877 | +$2.26 | 8.1 | 0.44 | -$1,980 | -$24 |
| N5: Informed rebalance | -$3,436 | -$8.87 | 12.5 | -1.68 | -$4,096 | -$4,338 |
| N5b: Informed rebal soft | -$1,187 | -$3.06 | 15.7 | -0.81 | -$1,518 | -$2,088 |
| **N6: Multifractal adapt** | **+$957** | **+$2.47** | 8.0 | **0.47** | -$1,984 | **+$55** ✅ |

## Results — Combo Strategies (CORRECTED)

### SOLUSDT

| Strategy | PnL | PnL/day | Sharpe | vs Base |
|----------|-----|---------|--------|---------|
| C1: Toxic+AdaptRebal | +$583 | +$1.50 | 0.26 | -$1,568 |
| C2: Eff+AdaptRebal | +$768 | +$1.98 | 0.32 | -$1,382 |
| C3: VPIN+Toxic+AdRebal | +$562 | +$1.45 | 0.26 | -$1,589 |
| C4: Herd+InfRebal | +$343 | +$0.88 | 0.27 | -$1,808 |
| C5: Eff+Toxic+InfRebal | -$195 | -$0.50 | 0.01 | -$2,346 |

### BTCUSDT

| Strategy | PnL | PnL/day | Sharpe | vs Base |
|----------|-----|---------|--------|---------|
| **C1: Toxic+AdaptRebal** | **+$1,281** | **+$3.31** | **1.75** | **+$492** ✅ |
| C2: Eff+AdaptRebal | +$1,187 | +$3.07 | 1.50 | +$399 ✅ |
| C3: VPIN+Toxic+AdRebal | +$1,049 | +$2.71 | 1.42 | +$260 ✅ |
| C4: Herd+InfRebal | -$33 | -$0.09 | 0.10 | -$822 |
| C5: Eff+Toxic+InfRebal | -$358 | -$0.93 | -0.20 | -$1,147 |

### ETHUSDT

| Strategy | PnL | PnL/day | Sharpe | vs Base |
|----------|-----|---------|--------|---------|
| C1: Toxic+AdaptRebal | +$1,131 | +$2.92 | 0.60 | +$230 ✅ |
| C2: Eff+AdaptRebal | +$684 | +$1.77 | 0.37 | -$217 |
| **C3: VPIN+Toxic+AdRebal** | **+$1,228** | **+$3.17** | **0.65** | **+$326** ✅ |
| C4: Herd+InfRebal | -$3,536 | -$9.13 | -1.81 | -$4,437 |
| C5: Eff+Toxic+InfRebal | -$3,508 | -$9.06 | -1.72 | -$4,410 |

## Key Findings (CORRECTED)

### 1. N5b (Informed Rebalance) is HARMFUL, not helpful

The original finding that N5b reduced losses by 90% was **entirely an artifact of the broken simulator**.
With the correct simulator:

| Symbol | Baseline PnL | N5b PnL | vs Base | Verdict |
|--------|-------------|---------|---------|---------|
| SOL | +$2,151 | -$1,260 | -$3,411 | ❌ Destroys profitability |
| BTC | +$789 | +$132 | -$657 | ❌ Much worse |
| ETH | +$901 | -$1,187 | -$2,088 | ❌ Destroys profitability |

**Why it fails:** Informed-flow-triggered rebalancing generates too many rebalances (21 fills/day on SOL
vs 9.3 baseline). Each rebalance closes inventory at market, crystallizing small losses. The cumulative
cost of excessive rebalancing far exceeds any benefit from cutting inventory before trends.

### 2. No novel strategy consistently beats baseline

The fixed 1.00% (24h) grid remains extremely hard to beat:

| Symbol | Best Novel Strategy | vs Base | Magnitude |
|--------|-------------------|---------|-----------|
| SOL | N3: Efficiency adapt | +$148 | +7% PnL |
| BTC | N6: Multifractal adapt | +$60 | +8% PnL |
| ETH | N6: Multifractal adapt | +$55 | +6% PnL |

Improvements are **marginal** (6-8%) and not consistent across assets.

### 3. S5 (Adaptive Rebalance from v15) remains the best enhancement for BTC/ETH

| Symbol | S5 PnL | vs Base | Sharpe |
|--------|--------|---------|--------|
| BTC | +$1,355 | +$567 (+72%) | 1.84 |
| ETH | +$1,136 | +$234 (+26%) | 0.60 |
| SOL | +$695 | -$1,456 (-68%) | 0.30 |

Confirms v15 finding: adaptive rebalance helps lower-vol assets, hurts high-vol SOL.

### 4. Combos with AdaptRebal work on BTC, not SOL

C1 (Toxic+AdaptRebal) on BTC: +$1,281 (Sharpe 1.75) — best combo.
But all combos hurt SOL. The adaptive rebalance component dominates combo performance.

### 5. Informed rebalance (N5, N5b) is consistently harmful

Both N5 and N5b are **worse than baseline on all 3 symbols**. The informed flow signal
triggers too many rebalances, and each rebalance has a cost (closing inventory at market).

## Bug Post-Mortem

The original v16 simulator had 3 critical bugs vs v15's correct implementation:

1. **No level deactivation** — same grid level filled every bar price stayed below it,
   causing 47× inflated fill counts (437/day vs 9.3/day on SOL)
2. **No position sizing** — stored raw prices instead of (qty, cost) tuples;
   PnL was in "price units" not USD
3. **No paired buy/sell activation** — all levels checked independently every bar

These bugs made the baseline appear as a massive loss ($-290K on SOL) when it's actually
profitable (+$2,151). The "90% loss reduction" from N5b was comparing broken numbers.

## Recommendations (CORRECTED)

1. **Discard N5/N5b (informed rebalance)** — harmful on all assets
2. **Keep S5 (adaptive rebalance)** for BTC/ETH — confirmed by both v15 and v16
3. **N3 (efficiency adapt) and N6 (multifractal)** show marginal promise but need more testing
4. **The fixed 1.00% (24h) baseline is genuinely strong** — hard to beat with any signal
