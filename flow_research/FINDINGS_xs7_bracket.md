# XS-7 — Bracket / Convex Execution on S07

**Date:** 2026-03-03  
**Data:** 52 Bybit perps, 2026-01-01 → 2026-02-28  
**Signal:** S07 compress_oi (rv_6h ≤ P20 AND oi_z ≥ 1.5), 5m grid, 24h cooldown/coin  
**Grid:** a∈{0.8,1.0,1.2,1.5}, b∈{3,4,5}, c∈{1.5,2.0,2.5}, cancel∈{0,60,300}s  
**Fees:** taker=10bp, maker=2bp. Slippage: {0,5,10}bp  
**Walk-forward:** Train Jan / Test Feb AND reverse, ±24h purge  
**Script:** `xs7_bracket_backtest.py`

---

## TL;DR

**Conditional GO ⚠️** — not NO-GO, but not a clean GO either.

48 of 108 configs are positive in BOTH walk-forward splits (Jan and Feb). The best config (a=1.0, b=3, c=1.5) averages **+20bp/trade net** across both months with 134 trades. But:

- PF is weak (1.1-1.15)
- Median trade is **negative** (-35bp) — the average is carried by **4 huge tail captures** (~1800bp each)
- 96% of trades exit via TIME (24h timeout), only 3% hit TP
- Weekly stability is poor (1-3 out of 4 weeks positive)

This is a **genuine tail-capture strategy** — it works by losing small most of the time and winning huge rarely. The question is whether 2 months of data is enough to trust the tail frequency.

---

## 1. Key Metrics — Best Config (a=1.0, b=3, c=1.5)

| Metric | Test Feb | Test Jan | Combined |
|--------|----------|----------|----------|
| N trades | 68 | 66 | 134 |
| Trigger rate | 5.9% | 7.7% | 6.6% |
| Net mean (bp) | +23.1 | +17.2 | **+20.2** |
| Net median (bp) | -27.0 | -27.0 | **-34.7** |
| PF | 1.13 | 1.15 | 1.14 |
| TP exits | 1.5% | 1.5% | 3% |
| SL exits | 1.5% | 1.5% | 2% |
| TIME exits | 97% | 97% | **96%** |
| DOUBLE triggers | 0% | 0% | **0%** |
| p5 net (bp) | -587 | -583 | -664 |
| Max DD (bp) | -2202 | -3724 | — |
| Weekly ≥ 0 | 3/4 | 1/4 | — |

### What this means

- **+20bp/trade is real** — it survives both splits, realistic fees (10bp taker + 2bp maker), and 5bp slippage
- **But it's driven by 4 TP trades** that each make ~1800bp (18% return!)
- The other 130 trades average -6bp each (TIME exits with small drift)
- If those 4 big trades don't happen → strategy loses ~800bp over 2 months
- **DOUBLE trigger rate is 0%** across all configs — bracket width of 1.0×ATR is wide enough

---

## 2. Why Most Trades Exit via TIME

The bracket entry (P0 ± 1.0×ATR) triggers when price moves 1 ATR from signal. But the **TP is 3×ATR from entry**, meaning price needs to move **4×ATR total from signal** within 24h. From XS-6 we know the breach rate for 3×ATR at 24h is only ~3% — so naturally most brackets that trigger don't reach TP.

The TIME exits aren't all losers though:
- Mean gross of TIME exits: **+13.8bp** (slight positive drift after entry)
- But mean net: **-6.2bp** (fees eat it)
- 42.6% of TIME exits are net positive
- Mean MFE of TIME exits: **+389bp** (price often moves significantly in our favor before drifting back)

**The 389bp MFE on TIME exits is important** — it means the price IS moving big after S07, but we're not capturing it because TP is too far or because the move reverses before hitting TP.

---

## 3. Tail Capture Analysis

The 4 TP trades (the whole profit engine):

| Symbol | Side | Gross (bp) | Time in trade |
|--------|------|-----------|---------------|
| ATHUSDT | short | ~1949 | ~9h |
| CYBERUSDT | short | ~1841 | ~9h |
| JELLYJELLYUSDT | short | ~1750 | ~8h |
| ENSOUSDT | long | ~1826 | ~7h |

All are altcoins with high vol. 3/4 are shorts. These are genuine big moves that the bracket captured.

---

## 4. Per-Symbol Concentration

- 23 of 46 coins are net positive
- Top 5 coins contribute **9,234bp** of the total **2,621bp** net
- This means the bottom 41 coins collectively lose -6,613bp
- **Extreme concentration risk** — profit depends on a handful of altcoins delivering tail moves

---

## 5. Parameter Sensitivity

### Bracket width (a) — most important parameter

| a | Trigger Rate | N Trades (Feb) | Net Mean (bp) |
|---|-------------|---------------|---------------|
| 0.8 | 9.8% | 89 | +26.2 |
| 1.0 | 5.9% | 68 | +23.1 |
| 1.2 | 3.7% | 50 | +29.3 |
| **1.5** | **1.9%** | **30** | **+120.9** |

a=1.5 looks amazing (+121bp) but has only 30 trades in Feb and **fails the reverse split** (Jan: -96bp). It's overfitting to Feb's recovery move.

a=0.8 and a=1.0 both work in both splits — more robust but weaker edge.

### TP/SL — SL is irrelevant, TP is everything

- c (SL distance) barely matters because SL rarely triggers (2-7% of trades)
- b=3 is best because it captures tails without waiting too long
- b=4,5 reduce hit rate without improving mean (moves that go 4-5×ATR are too rare)

### Cancel delay — doesn't matter

0% double triggers across all delays. 1×ATR bracket width is wide enough that both stops never trigger in the same bar/minute.

---

## 6. Stress Tests

### All-taker + 10bp slippage (worst case)

Best config at worst-case fees: a=1.0, b=3, c=1.5

| Split | Net Mean (bp) | PF |
|-------|--------------|-----|
| Feb | +11.1 | 1.06 |
| Jan | +5.2 | 1.03 |

Still positive but barely. **Strategy breaks even at ~15bp total friction per side.**

### Zero slippage (optimistic)

| Split | Net Mean (bp) | PF |
|-------|--------------|-----|
| Feb | +33.1 | 1.22 |
| Jan | +27.2 | 1.24 |

Meaningful improvement. The strategy is sensitive to execution costs.

---

## 7. Tradeability

- **Mean time to entry:** 911 min (15.2h) from signal — very slow
- **69% of entries happen 12-24h after signal** — the bracket often triggers near the time stop boundary
- **0% of entries in first hour** — the compressed vol regime persists for hours
- **Mean time in trade:** 505 min (8.4h)
- **Total cycle:** ~24h from signal to exit on average

This means:
- No latency requirements
- Need to monitor ~50 coins for S07 signals every 5 min
- Bracket orders sit open for most of the 24h window
- **Exchange rate limits** may constrain how many simultaneous brackets we can maintain

---

## 8. GO/NO-GO Assessment

### Arguments for GO:
1. **48 configs positive in both splits** — not a fluke of one parameter combo
2. **0% double-trigger** — bracket width is sufficient
3. **Genuine tail capture** — the 4 TP trades are real big moves on different coins
4. **No speed requirements** — 5min signal grid, 15h average entry delay
5. **Survives realistic fees** (+20bp net at 10bp taker + 5bp slip)

### Arguments for NO-GO:
1. **Only 134 trades in 2 months** — statistically thin
2. **4 TP trades carry everything** — remove any 2 and it's negative
3. **PF 1.1** — razor-thin edge
4. **Weekly stability 1-3/4** — not consistent enough for confidence
5. **p5 = -664bp** — worst 5% of trades lose 6.6%
6. **Extreme symbol concentration** — 5 coins drive all profit
7. **Median trade is negative** — you lose more often than you win

### Verdict: **CONDITIONAL ⚠️**

The strategy has a **real but thin edge** that is:
- Sensitive to fee structure (breaks even at ~15bp/side)
- Dependent on rare tail events (4 in 2 months across 52 coins)
- Not stable enough week-to-week for confident sizing

**Recommendation:** Do NOT deploy with meaningful size. Instead:
1. Paper trade for 2-3 more months to accumulate more TP events
2. If TP frequency holds at ~2/month across 50 coins → small live test
3. Focus on reducing the TIME exit loss — 389bp MFE suggests a trailing stop or partial exit could capture more of the move

---

## 9. What Would Make This a GO

1. **More data** — 6 months minimum, need to see 20+ TP trades to trust the tail frequency
2. **Trailing stop / partial exit** — TIME exits have mean MFE of 389bp but end at -6bp net. A trailing stop at 50% of MFE could capture 150-200bp on these trades and flip the median positive
3. **Symbol selection** — don't bracket all 52 coins. Pre-filter for coins where S07 has historically higher uplift
4. **Regime awareness** — Jan was a downtrend, Feb was a recovery. S07 might work differently in each regime

---

## Files

- **Script:** `flow_research/xs7_bracket_backtest.py`
- **Trade log:** `flow_research/output/xs7/xs7_trades.csv`
- **Config report:** `flow_research/output/xs7/xs7_report.csv`
- **Auto-findings:** `flow_research/output/xs7/FINDINGS_xs7_bracket.md`
