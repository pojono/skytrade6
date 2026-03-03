# XS-7 v2 — Convex Extraction on S07 (Exit Engineering)

**Date:** 2026-03-03  
**Data:** 52→64 Bybit perps, 2026-01-01 → 2026-02-28  
**Signal:** S07 compress_oi (rv_6h ≤ P20_expanding AND oi_z ≥ 1.5), 5m grid, 24h cooldown from placement  
**Grid:** 4 phases — Base (36), Trailing (30), Partial (20), Combo (6) = 92 configs  
**Fees:** taker=10bp, maker=2bp. Slippage: {0,5,10}bp  
**Walk-forward:** Train Jan / Test Feb AND reverse, ±24h purge  
**Script:** `xs7v2_bracket_backtest.py`  
**Runtime:** 15.4min

---

## TL;DR

**NO-GO ❌** — Exit engineering (trailing, partial, closer TP) does NOT solve the fundamental problem.

All 92 configs are positive in both WF splits (avg net >0), but the strategy remains **structurally tail-dependent**: median is always negative (-24bp to -60bp), top-5 trades carry 290-820% of total PnL, and weekly stability is poor (1-3/4 weeks positive). The v2 improvements are marginal:

- **Closer TP (b=2.0 vs v1's b=3):** TP hit rate improves from 3% → 6-10%, but still 90% of trades exit via TIME
- **Trailing stops:** Add 2-9% TRAIL exits, improve median by ~8bp at best (from -32bp to -24bp), but can hurt net mean
- **Partial exits:** Nearly zero impact on median or concentration (partial fills too rare)
- **Combo (trail + partial):** Best avg net is +66bp but only 40 trades in Feb (statistically meaningless)

**The 389bp MFE on TIME exits is a mirage** — it's the peak of random walk, not a capturable signal. Trailing stops at 1.0×ATR capture some of it but at the cost of cutting winners short.

---

## 1. v2 vs v1 Comparison

| Metric | v1 (a=1.0,b=3,c=1.5) | v2 base (a=1.2,b=2.0,c=2.0) | v2 base (a=0.8,b=2.0,c=1.5) |
|--------|----------------------|------------------------------|------------------------------|
| N trades (Feb) | 68 | 40 | 86 |
| Net mean (bp) | +23 | **+68** | +15 |
| Net median (bp) | -27 | **-60** | -32 |
| PF | 1.13 | **1.41** | 1.09 |
| TP exits | 3% | **10%** | 6% |
| TIME exits | 96% | 90% | 91% |
| Max DD (bp) | -2202 | -1621 | -3623 |
| Mean MFE (bp) | 389 | 415 | 375 |
| Weeks positive | 3/4 | 3/4 | 2/4 |

**Observations:**
- Closer TP (b=2.0) doubles TP hit rate (3% → 6-10%) — genuine improvement
- a=1.2 has fewer trades (40 vs 86) but higher mean — driven by fewer losers
- Median is WORSE for a=1.2 (-60bp) because the few TP hits are enormous while all TIME exits lose similarly
- a=0.8 has more trades but lower mean — diluted by more noise triggers

---

## 2. Trailing Stop Results

Best trailing configs (a=0.8, b=2.0, c=1.5, test_fwd, slip=5bp):

| trail_start | trail_gap | Net mean | Median | TP% | TRAIL% | TIME% |
|-------------|-----------|----------|--------|-----|--------|-------|
| OFF | OFF | +14.8 | -31.9 | 6% | 0% | 91% |
| 1.0 | 0.8 | +14.2 | **-23.6** | 5% | **9%** | 83% |
| 1.0 | 1.0 | **+19.5** | **-23.6** | 6% | **7%** | 84% |
| 1.0 | 1.2 | +18.3 | -27.2 | 6% | 5% | 86% |
| 1.5 | 0.8 | +4.4 | -31.9 | 5% | 2% | 90% |
| 1.5 | 1.0 | +12.5 | -31.9 | 6% | 1% | 90% |

**Analysis:**
- trail_start=1.0 (activate after 1×ATR move) does convert 7-9% of TIME exits to TRAIL exits
- Median improves modestly: -32bp → -24bp (8bp better)
- But net mean is flat or slightly worse — the trailing stop cuts some winners that would have reached TP
- trail_start=1.5 is too high to trigger — barely any TRAIL exits (0-2%)
- **Tight gap (0.8) is worse** than wider gap (1.0-1.2) — gets shaken out of trending moves

---

## 3. Partial Exit Results

Partial exits at tp1=1.5-2.0×ATR on the a=1.2 base config:

| p1 | tp1 | Net mean | Partial fill rate | vs base |
|----|-----|----------|-------------------|---------|
| 0.3 | 1.5 | +58.6 | 12.5% | -9bp |
| 0.5 | 1.5 | +52.9 | 12.5% | -15bp |
| 0.3 | 2.0 | +67.3 | 10.0% | -0.4bp |
| 0.5 | 2.0 | +67.3 | 10.0% | -0.4bp |

**Analysis:**
- Partial fills occur on 10-12% of trades — slightly above TP rate
- tp1=2.0 partial is barely different from no-partial (same TP level basically)
- tp1=1.5 reduces mean by 9-15bp because it forces early profit-taking on winners
- **Partial exits don't help median at all** — the 88% of trades that don't reach tp1 are unaffected
- p1=0.3 vs 0.5 makes almost no difference

---

## 4. Combo Results (Trail + Partial)

| trail | partial | Net mean (avg both splits) | vs base avg |
|-------|---------|---------------------------|-------------|
| 1.0/1.2 | 0.5/1.5 | **+66.2** | +18bp |
| 1.5/0.8 | 0.5/1.5 | +65.1 | +17bp |
| 1.0/1.0 | 0.5/1.5 | +65.0 | +17bp |

All combo configs are on a=1.2, b=2.0, c=2.0 (only base config that overlaps in top-3 of both trailing and partial). The improvement is driven by the trailing component, not partial.

**But with only 40 trades in Feb (test_fwd) and 53 in Jan (test_rev), the difference between +48bp (base) and +66bp (combo) is well within noise.**

---

## 5. Tail Dependency (the core problem)

For the best base config (a=1.2, b=2.0, c=2.0, slip=5bp, test_fwd):

| Metric | Value |
|--------|-------|
| Total net PnL | +2,710bp |
| Remove top-1 trade | +277bp (-90%) |
| Remove top-2 trades | -1,229bp (negative!) |
| Remove top-3 trades | -2,575bp |
| Remove top-5 trades | -5,113bp |
| Top-5 concentration | 289% of total |

**Without the single best trade, the strategy makes 10% of its total PnL. Without the top 2, it's negative.** This is unchanged from v1 and no amount of exit engineering fixes it — the edge IS the tail.

---

## 6. Cost Sensitivity

Best base config (a=1.2, b=2.0, c=2.0, test_fwd):

| Slippage | Net mean | PF |
|----------|----------|-----|
| 0bp | +108.9 | 1.68 |
| 5bp | +99.1 | 1.60 |
| 10bp | +89.3 | 1.52 |

Strategy survives high friction (still +89bp at 10bp slip) because the winners are so large that fees are irrelevant. The losers are small TIME-exit losses where fees are a large fraction.

---

## 7. Bug Guards

- **Entry == signal close:** 0.0% ✅ (anti-lookahead working)
- **Double trigger rate:** 0.0% ✅ (bracket width sufficient)
- **Cooldown violations:** 0 ✅
- **NaN in critical fields:** None ✅

---

## 8. Why Exit Engineering Failed

The **hypothesis** was: "TIME exits have mean MFE of 389bp, so trailing/partial should capture some of that MFE and improve the median."

**Reality:** The 389bp MFE is the peak of a random walk that then reverts. For most TIME exits:
1. Price triggers the bracket (moves 1×ATR in one direction)
2. Price continues 2-4×ATR further (creating the MFE)  
3. Price reverts back toward entry over the remaining 8-20h
4. Trade exits at TIME with small loss

A trailing stop at 1.0×ATR does catch some of the #2→#3 transition, converting ~7% of TIME exits to TRAIL exits. But:
- The gap (0.8-1.2×ATR) is wide enough that most of the MFE is lost before the trail triggers
- The trail also cuts genuine trend-followers short, reducing TP hits from 10% to 5-8%
- Net effect: +5bp to the mean at best, -8bp to median at best

**Partial exits** fail because tp1 at 1.5-2.0×ATR is reached by only 10-12% of trades — almost exactly the same trades that would have hit the full TP. There's no "middle ground" population to exploit.

---

## 9. GO/NO-GO Assessment

### Mini-criteria check (92 configs, slip=5bp):

| Criterion | Pass rate | Blocking? |
|-----------|-----------|-----------|
| OOS mean > 0 both splits | 82/92 (89%) | No |
| N trades ≥ 50 combined | 92/92 (100%) | No |
| Double trigger ≤ 1% | 92/92 (100%) | No |
| PnL top-5 trades ≤ 80% | **0/92 (0%)** | **YES** |
| Net median ≥ -10bp | **0/92 (0%)** | **YES** |

**No config passes mini-criteria. Both blockers are structural.**

### GO criteria: 0/92 pass

The strategy cannot meet GO criteria because:
1. It's fundamentally tail-dependent (top-5 trades always >200% of total PnL)
2. Median is always negative (-24bp to -60bp)
3. These are not fixable with exit engineering — they're properties of the signal

### Verdict: **NO-GO ❌**

S07 (compress_oi) remains a "rare jackpot" signal. Exit engineering cannot convert it into a stable convex profile. The edge exists but is:
- Too concentrated in tail events (2-4 trades per month across 50+ coins)
- Too thin per trade (most trades are small losers)
- Too dependent on specific altcoin blow-ups

---

## 10. What Would Make This Deployable

1. **More data:** 6+ months to see if tail frequency (~2 TP/month) is stable
2. **Regime filter:** S07 might work better in ranging markets vs trending — untested
3. **Symbol selection:** Pre-filter coins by historical S07 uplift (but sample too small)
4. **Accept the profile:** Deploy with very small size as a "lottery ticket" allocation, expecting negative months 40% of the time but large positive months 60% of the time

---

## Files

- **Script:** `flow_research/xs7v2_bracket_backtest.py`
- **Trade log:** `flow_research/output/xs7v2/xs7v2_trades.csv`
- **Config report:** `flow_research/output/xs7v2/xs7v2_report.csv`
- **Equity curves:** `flow_research/output/xs7v2/xs7v2_equity.csv`
- **Auto-findings:** `flow_research/output/xs7v2/FINDINGS_xs7v2.md`
