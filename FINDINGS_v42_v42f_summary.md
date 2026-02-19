# FINDINGS v42–v42f: New Signal Research Summary

## Overview

Systematic exploration of 15+ new signal ideas across v42–v42f.
**Two major validated edges found**, plus one massive new discovery (cross-symbol contagion).

---

## Experiment Scorecard

| EXP | Idea | Result | Status |
|-----|------|--------|--------|
| A | Spot-Futures Basis Mean-Reversion | Real but <1 bps — below fees | ❌ DEAD |
| B | Cascade Size Filtering (P95→P97→P99) | Monotonic improvement, P97 sweet spot | ✅ WINNER |
| C | OI Divergence | No predictive power | ❌ DEAD |
| D | Funding Rate Pre-Settlement | No edge | ❌ DEAD |
| E | Intraday Seasonality | Consistent hour-of-day patterns | ✅ WINNER |
| F | Combined Cascade Size + Hour Filter | P97+good hours = 100% WR but few trades; P95 no-bad practical | ✅ CONFIRMED |
| G | Spot-Futures Volume Imbalance | No actionable signal | ❌ DEAD |
| H | Cascade Hour-of-Day Interaction | 08,09,13,16 UTC are bad hours; confirms seasonality | ✅ CONFIRMED |
| I | Vol Compression Straddle | 99% WR on 30d → 0/7 windows on 60d. Pure overfitting. | ❌ DEAD |
| J | VPIN Toxicity Filter | HIGH toxicity cascades revert MORE (+2.3 bps). Counter-intuitive. | ⚠️ MINOR |
| K | Trade Imbalance Momentum | Directional signal exists but <2 bps — below fees | ❌ DEAD |
| L | Cross-Symbol Cascade Contagion | **ETH→SOL 91% WR, +5.3 bps, +19%/30d** | ✅ **MAJOR WINNER** |
| M | Post-Cascade Vol Expansion | All configs negative | ❌ DEAD |
| N | Whale Trade Detection (P95/P99/P99.9) | Zero predictive power at any threshold | ❌ DEAD |
| O | Large Trade Mean-Reversion | Zero signal | ❌ DEAD |

**Hit rate: 4/15 ideas validated (27%)** — typical for systematic research.

---

## Major Finding: Cross-Symbol Cascade Contagion (v42f)

### The Idea
When ETH has a liquidation cascade, other symbols (SOL, DOGE) often cascade within seconds to minutes. By detecting the ETH cascade first, we can enter cascade MM positions on SOL/DOGE **before or during** their own cascades.

### 60-Day OOS Results (ALL pairs profitable)

**Config: offset=0.15%, TP=0.15%, SL=0.50%**

| Trigger → Target | Trades | Win Rate | Avg Net | Total Return | Sharpe |
|------------------|--------|----------|---------|--------------|--------|
| ETH→ETH (same) | 616 | 92.0% | +5.8 bps | +35.84% | 196 |
| ETH→SOL | 600 | 90.7% | +4.7 bps | +28.31% | 145 |
| ETH→DOGE | 612 | 89.9% | +4.2 bps | +25.64% | 124 |
| SOL→ETH | 356 | 89.9% | +4.4 bps | +15.55% | 132 |
| SOL→SOL (same) | 363 | 92.8% | +6.1 bps | +22.11% | 208 |
| SOL→DOGE | 358 | 91.3% | +5.0 bps | +18.03% | 158 |
| DOGE→ETH | 202 | 93.1% | +6.3 bps | +12.63% | 217 |
| DOGE→SOL | 203 | 95.6% | +8.0 bps | +16.17% | 340 |
| DOGE→DOGE (same) | 211 | 91.5% | +5.3 bps | +11.28% | 172 |

### Walk-Forward OOS (train=41d, test=18d)

**16/18 pairs OOS positive** at off=0.15. Only 2 failures: DOG-triggered pairs at off=0.20 (small sample, n=21-26).

Key OOS results (off=0.15):
- **ETH→ETH**: train +0.71%/d → test +0.38%/d ✅
- **ETH→SOL**: train +0.53%/d → test +0.36%/d ✅
- **ETH→DOG**: train +0.49%/d → test +0.31%/d ✅
- **SOL→SOL**: train +0.41%/d → test +0.29%/d ✅
- **SOL→DOG**: train +0.29%/d → test +0.34%/d ✅ (test > train!)

### Rolling 15-Day Window Stability

**28/28 windows positive (100%)** across all 4 tested pairs:
- ETH→SOL: 7/7 ✅
- ETH→DOGE: 7/7 ✅
- SOL→SOL: 7/7 ✅
- ETH→ETH: 7/7 ✅

### Combined Triggers (same + cross symbol)

| Target | Same Only | Cross Only (ETH) | Combined | Combined Trades |
|--------|-----------|-------------------|----------|-----------------|
| SOL | +22.11% | +28.31% | **+37.55%** | 765 |
| DOGE | +11.28% | +25.64% | **+32.35%** | 688 |

**Combined triggers increase total return by 70%+ vs same-symbol only.**

---

## Confirmed Enhancements to Cascade MM

### 1. Cascade Size Filtering (EXP B)
- P97 threshold outperforms P95 by ~2 bps/trade
- Monotonic: larger cascades → better reversion
- Trade-off: fewer trades (P97 = ~60% of P95 count)

### 2. Intraday Seasonality (EXP E + H)
- **Bad hours** (avoid): 08, 09, 13, 16 UTC
- **Good hours**: 00-07, 11, 14-15, 17-23 UTC
- Excluding bad hours preserves ~75% of trades, removes most losers

### 3. VPIN Toxicity (EXP J — minor)
- Counter-intuitive: HIGH toxicity cascades revert MORE (+2.3 bps vs LOW)
- Interpretation: toxic flow = forced liquidations = more mechanical reversion
- Not strong enough to be a primary filter, but confirms cascade MM thesis

---

## Dead Ends (important lessons)

1. **Vol Compression Straddle** — looked amazing on 30d (99% WR) but 0/7 on 60d. Classic overfitting to a specific regime. Always validate OOS.
2. **Whale trades** — zero signal at P95/P99/P99.9. Individual large trades are noise.
3. **Trade imbalance** — directional signal exists but magnitude (~1-2 bps) is below fee wall.
4. **OI divergence, funding rate** — no predictive power for short-term returns.
5. **Spot-futures volume imbalance** — no actionable signal despite theoretical appeal.
6. **Post-cascade vol expansion** — entering after cascade settles loses the edge entirely.

---

## Recommended Strategy Configuration

### Production Cascade MM with All Enhancements

```
Symbols:     ETH, SOL, DOGE (run all 3 simultaneously)
Triggers:    Cross-symbol contagion (ETH cascades trigger SOL+DOGE entries too)
Cascade:     P95 threshold, min 2 events within 60s
Hours:       Exclude 08, 09, 13, 16 UTC
Entry:       Limit order at ±0.15% from cascade end price (fade direction)
TP:          0.15% (maker fee exit)
SL:          0.50% (taker fee exit)
Max hold:    30 minutes
Cooldown:    5 minutes between trades
Fees:        Maker 0.02%, Taker 0.055%

Expected (60d backtest):
  - Combined ETH+SOL→SOL: +37.55%, 765 trades, 91% WR
  - Combined ETH+DOG→DOG: +32.35%, 688 trades, 91% WR
  - ETH→ETH (same): +35.84%, 616 trades, 92% WR
```

### Key Risk: All edges are from the same 60-day period (May-Jul 2025)
Need to validate on additional months as data becomes available.

---

## Scripts & Results

| File | Description |
|------|-------------|
| `research_v42_new_signals.py` | EXP A-E: initial 5 experiments (7d) |
| `research_v42b_cascade_size_season.py` | EXP B+E expanded to 30d, walk-forward |
| `research_v42c_next_ideas.py` | EXP F-J: combined filters, vol imbalance, VPIN |
| `research_v42d_vol_straddle_oos.py` | Vol straddle 60d OOS (killed it) |
| `research_v42e_more_ideas.py` | EXP K-O: imbalance, contagion, whales |
| `research_v42f_contagion_oos.py` | Contagion 60d OOS validation (RAM-safe) |
| `results/v42f_contagion_oos.txt` | Full contagion OOS results |
