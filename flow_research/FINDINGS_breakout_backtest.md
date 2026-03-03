# FINDINGS — Dual-Stop Breakout on REG_OI_FUND

**Date:** 2026-03-03  
**Period:** 2026-01-01 → 2026-02-28 (59 days)  
**Coins:** 8 altcoins  
**Grid:** 3×3×3×2 = 54 configs/symbol, 3 slippage levels  
**Fees:** 20bp RT (taker)  
**Cooldown:** 60 min

---

## 1) Go/No-Go Summary

| Symbol | Verdict | E/sig (bp) | PF | P5 (bp) | Weeks +/total | Slip=5bp |
|--------|---------|---:|---:|---:|---|---|
| **1000BONKUSDT** | **GO** | +11.1 | 2.25 | -48 | 4/6 | OK (+4.9) |
| **1000RATSUSDT** | **GO** | +13.0 | 1.71 | -61 | 3/5 | FAIL (-7.4) |
| ARCUSDT | GO* | +5.4 | 1.17 | -72 | 6/9 | FAIL (-2.8) |
| AIXBTUSDT | NO-GO | +2.0 | 1.11 | -77 | 4/7 | FAIL |
| 1000TURBOUSDT | NO-GO | -2.9 | 0.72 | -33 | 1/4 | FAIL |
| APTUSDT | NO-GO | -5.9 | 0.62 | -46 | 2/8 | FAIL |
| ARBUSDT | NO-GO | -4.1 | 0.73 | -46 | 4/8 | FAIL |
| ATOMUSDT | NO-GO | -13.5 | 0.10 | -34 | 0/9 | FAIL |

**Only 1000BONKUSDT is robustly positive** — survives slip=5bp with +4.9bp/signal.

1000RATSUSDT is positive at slip=0 and slip=2, but dies at slip=5bp. Marginal.

ARCUSDT has 6/9 positive weeks and PF>1, but slim edge that vanishes at slip=2bp.

5 of 8 coins are **clear NO-GO** — negative expectancy even at slip=0.

---

## 2) Best Configs (slip=0)

| Symbol | k_entry | k_sl | k_tp | TO | N_trades | Trig% | TP% | SL% | WR | Mean net | PF |
|--------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000BONKUSDT | 0.3 | 0.8 | 2.0 | 30 | 10 | 71% | 60% | 40% | 70% | +15.6 | 2.25 |
| 1000RATSUSDT | 0.5 | 1.0 | 2.0 | 60 | 20 | 100% | 50% | 50% | 60% | +13.0 | 1.71 |
| ARCUSDT | 0.5 | 0.5 | 2.0 | 30 | 58 | 95% | 36% | 64% | 36% | +5.6 | 1.17 |
| AIXBTUSDT | 0.3 | 1.0 | 2.0 | 30 | 25 | 83% | 52% | 44% | 52% | +2.3 | 1.11 |

**Pattern in winning configs:**
- **k_tp=2.0** dominates — need wide TP to catch the range expansion tail
- **k_sl ≥ 0.8** — tight SL (0.5) gets chopped out in high-vol regimes
- **k_entry 0.3–0.5** — lower entry distance = higher trigger rate
- **TO 30 vs 60 matters little** — most of the action happens in first 30 min

---

## 3) Slippage Sensitivity

| Symbol | slip=0 | slip=2 | slip=5 |
|--------|---:|---:|---:|
| 1000BONKUSDT | **+11.1** | **+7.0** | **+4.9** |
| 1000RATSUSDT | **+13.0** | **+11.0** | -7.4 |
| ARCUSDT | **+5.4** | +0.0 | -2.8 |
| AIXBTUSDT | **+2.0** | -1.8 | -5.8 |
| All others | negative | worse | much worse |

The edge is **thin** — 2-5bp slippage kills most coins. Only BONK has enough margin.

---

## 4) Walk-Forward Results

| Symbol | Train | Best params | Test mean (bp) | Test WR | Test PF |
|--------|-------|-------------|---:|---:|---:|
| 1000RATSUSDT | JAN | k_e=0.7,sl=0.5,tp=2.0,TO=30 | **+16.4** | 71% | 2.26 |
| 1000RATSUSDT | FEB | k_e=0.5,sl=1.0,tp=2.0,TO=60 | +1.3 | 46% | 1.05 |
| 1000BONKUSDT | JAN | k_e=0.3,sl=0.8,tp=2.0,TO=30 | -11.0 | 50% | 0.24 |
| 1000BONKUSDT | FEB | k_e=0.3,sl=0.5,tp=1.5,TO=30 | -0.1 | 50% | 1.00 |
| All others | either | any | **negative** | <50% | <1.0 |

**Walk-forward is disappointing.** The best in-sample configs don't transfer well OOS. Only 1000RATSUSDT JAN→FEB shows a positive OOS result.

---

## 5) Why It Fails (Diagnosis)

The regime research confirmed **1.5–3.9x range expansion** during REG_OI_FUND. So why doesn't the breakout work?

### 5.1 Range expansion ≠ directional breakout continuation

The expanded range often manifests as **whipsaw** — price breaks one side, then reverses through the other. The dual-stop catches the initial breakout, but:
- **64% SL rate** on ARCUSDT (the most active coin) means price routinely reverses after triggering entry
- High SL% despite wide SL levels (0.5–1.0 ATR) = the reversal is violent

### 5.2 Fee drag is fatal for non-directional strategies

20bp RT fees consume ~50-100% of the edge. Even BONK's +15.6bp mean is slim:
- Average TP hit = +40–90bp gross, but only 60% of the time
- Average SL hit = -30–50bp gross, 40% of the time
- Fees eat the profit asymmetry

### 5.3 Signal sparsity limits sample size

After 60m cooldown: 10-68 trades per symbol over 59 days. Too few for stable statistics.
BONK's "GO" status rests on N=10 trades — not reliable.

### 5.4 ATR-based levels are too blunt

ATR_15m captures recent vol but doesn't account for the asymmetric nature of OI_FUND regimes.
The range expansion is real, but its direction is determined by liquidation cascades that ATR can't predict.

---

## 6) What Would Work Instead

Given these findings, the range expansion during REG_OI_FUND is **real but not exploitable via breakout**. Better approaches:

1. **Options/straddles** (if available) — directly buy volatility during the regime
2. **Wider stops on existing positions** — use REG_OI_FUND as a risk-management signal, not a trade entry
3. **Directional model per coin** — since some coins consistently drift one way during OI_FUND (BONK down, RATS up), a per-coin directional strategy could work better than non-directional breakout
4. **Shorter horizon** — 60m may be too long; the expansion might happen in first 5-15m with a subsequent mean-reversion that kills the breakout

---

## 7) Files

| File | Description |
|------|-------------|
| `flow_research/backtest_breakout.py` | Full backtest implementation |
| `flow_research/output/regime/breakout_trades.parquet` | All trade records |
| `flow_research/output/regime/breakout_report.csv` | Summary by (symbol, params, slippage) |
| `flow_research/output/regime/breakout_weekly.csv` | Weekly stability breakdown |

---

## 8) Bottom Line

**The dual-stop breakout on REG_OI_FUND is NOT a viable strategy.** The range expansion is real (confirmed by regime research), but it manifests as whipsaw rather than directional continuation. Fees and slippage consume the thin edge. Walk-forward confirms no parameter stability.

The main value of REG_OI_FUND remains as a **volatility/risk signal**, not a trade entry trigger.
