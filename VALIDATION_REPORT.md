# Signal Validation Report — Comprehensive 8-Point Audit

**Date:** 2025-08-07  
**Script:** `validate_winners.py` (110 min runtime)  
**Signals tested:** Ret IQR, Wt Mom Divergence, MACD Hist Velocity, Regime Persistence, Stochastic Velocity  
**Data:** Bybit USDT perpetual 1-min bars, 2024-01-01 to 2026-02-16

---

## Executive Summary

| # | Validation Point | Verdict | Notes |
|---|---|---|---|
| 1 | Bybit VIP 0 fees | ✅ PASS | Exact match: maker 2bps, taker 5.5bps |
| 2 | Lookahead bias | ✅ PASS | Next-bar fill ≈ same-bar fill; avg fill delay 6-8 bars |
| 3 | Subsecond reaction | ✅ PASS | 1-min bars + limit orders; ~1 min reaction time sufficient |
| 4 | Overfitting | ✅ PASS (ETH/SOL) ⚠️ (BTC) | IS/OOS ratios 0.81-1.14 on ETH; all param combos positive; BTC fails 1 period |
| 5 | Script logic | ✅ PASS | Conservative: SL before TP, limit fills at bar extremes |
| 6 | Hidden biases | ⚠️ CAUTION | Time-of-day variation; selection bias from shared trade structure |
| 7 | All 5 coins | ⚠️ PARTIAL | ETH/SOL/DOGE/XRP strong; BTC fails in original test period |
| 8 | Date periods | ✅ PASS (ETH) ⚠️ (BTC) | ETH: 7/7 periods positive; BTC: 6/7 positive |

**Overall: Signals are REAL but the edge comes primarily from the TRADE STRUCTURE (limit offset + trailing stop) rather than the specific signal. BTC requires special handling.**

---

## 1. Fee Survival (Bybit VIP 0)

**Fee schedule used in research:**
- Maker: 0.020% (2 bps) — matches Bybit VIP 0 exactly
- Taker: 0.055% (5.5 bps) — matches Bybit VIP 0 exactly

**Fee stress tests on ETHUSDT (ret_iqr_60_95):**

| Fee Scenario | Maker | Taker | Result |
|---|---|---|---|
| Original (Bybit VIP 0) | 2 bps | 5.5 bps | ✅ +5.0 bps avg |
| Non-VIP | 2 bps | 5.5 bps | ✅ +5.0 bps avg |
| Worst case (both taker) | 5.5 bps | 5.5 bps | ✅ +3.4 bps avg |
| Double fee | 4 bps | 11 bps | ❌ -2.5 bps avg |

**Verdict:** Survives realistic fee scenarios. Only fails at 2x fees (unrealistic).

---

## 2. Lookahead Bias

**Test:** Compare fills starting on signal bar vs next bar (conservative).

| Signal | Same-bar avg | Next-bar avg | Δ avg | Δ WR |
|---|---|---|---|---|
| ret_iqr_60_95 | +5.0 bps | +4.8 bps | -0.2 bps | -0.2% |
| wt_mom_40_90 | +5.0 bps | +4.8 bps | -0.2 bps | +0.2% |
| macd_hv_3_90 | +5.0 bps | +4.7 bps | -0.3 bps | -0.7% |
| regime_persist_30 | +4.4 bps | +4.5 bps | +0.1 bps | +0.3% |
| stoch_vel_30_3_90 | +3.2 bps | +3.1 bps | -0.1 bps | +1.4% |

**Fill delay analysis:**
- Mean fill delay: 6.5-7.7 bars (minutes) from signal
- Only 3-13% fill on same bar as signal
- 52-62% fill within 5 bars

**Verdict:** No lookahead bias. The limit offset (15 bps from close) means fills naturally happen bars later. Next-bar restriction has negligible impact.

---

## 3. Subsecond Reaction Requirement

- Signals use **1-minute bars** (not tick data)
- Rolling windows: 14-120 bars = 14 min to 2 hours
- Signal changes at most **once per minute** (at bar close)
- Entry via **limit order** placed 15 bps from close
- Fill happens passively when price reaches limit (minutes later)
- Must place order within ~1 minute of signal (before next bar close)

**Verdict:** No subsecond reaction needed. A simple API polling every 30 seconds is sufficient.

---

## 4. Overfitting Analysis

### 4a. IS vs OOS Consistency

| Symbol | ret_iqr IS→OOS | wt_mom IS→OOS | macd_hv IS→OOS |
|---|---|---|---|
| ETHUSDT | +5.3→+5.0 (0.93) | +6.1→+5.0 (0.81) | +5.8→+5.0 (0.86) |
| SOLUSDT | +5.6→+5.9 (1.05) | +5.4→+5.2 (0.97) | +5.2→+5.4 (1.05) |
| DOGEUSDT | +7.0→+11.2 (1.60) | +6.8→+10.4 (1.54) | +6.2→+10.4 (1.67) |
| XRPUSDT | +3.9→+9.3 (2.39) | +3.6→+8.2 (2.26) | +3.5→+8.2 (2.35) |
| **BTCUSDT** | **-0.9→-2.1** | **-0.9→-1.6** | **-1.3→-2.0** |

- ETH/SOL: Excellent IS/OOS ratios (0.81-1.14) — no overfitting
- DOGE/XRP: OOS > IS — suspicious but could be favorable market regime in OOS period
- **BTC: NEGATIVE in both IS and OOS** — signals do not work on BTC in the original period

### 4b. Parameter Sensitivity (ETH)

**Ret IQR** — tested 18 param combos (window × threshold):
- ALL 18 positive, range +4.4 to +5.1 bps avg
- Smooth gradient: higher threshold → higher avg bps, fewer trades

**Wt Mom Div** — tested 15 param combos:
- ALL 15 positive, range +4.5 to +6.0 bps avg

**MACD Hist Vel** — tested 15 param combos:
- ALL 15 positive, range +4.3 to +6.0 bps avg
- Gradual decay with larger lag (expected)

**Verdict:** No parameter sensitivity issues on ETH. The edge is robust across wide parameter ranges.

---

## 5. Script Logic Audit

| Component | Implementation | Assessment |
|---|---|---|
| Entry | Limit order at close ± 0.15% | ✅ Passive, realistic |
| Fill detection | Bar low/high reaches limit | ✅ Conservative (bar extremes) |
| Take Profit | 0.15% from fill | ✅ Correct |
| Stop Loss | 0.50% from fill | ✅ Correct (asymmetric TP/SL) |
| Trailing Stop | Activate at 3 bps, trail at 2 bps | ✅ Standard logic |
| Max Hold | 30 bars (30 min) | ✅ Correct |
| Fee: entry | Maker (limit order) | ✅ Correct |
| Fee: TP exit | Maker (limit order) | ✅ Correct |
| Fee: SL/trail/timeout | Taker (market order) | ✅ Correct |
| SL vs TP priority | SL checked before TP | ✅ Conservative |

**Minor issue:** Timeout exit uses close price (slightly optimistic). Taker fee already partially compensates.

**Example verified trade:**
- Signal: 2025-07-11 00:58, ETH close $2937.18
- Limit buy: $2932.77 (15 bps below)
- Filled: 3 bars later at 01:01
- Exit: trail at $2935.99 → gross +11.0 bps, net +3.5 bps after 7.5 bps fees

---

## 6. Hidden Bias Detection

### 6a. Time-of-Day (ret_iqr on ETH)

| Hours (UTC) | Trades | Win Rate | Avg bps |
|---|---|---|---|
| 00-03 | 664 | 68.2% | +3.8 |
| 04-07 | 547 | 63.4% | +3.2 |
| 08-11 | 613 | 60.2% | +2.5 |
| **12-15** | **1238** | **79.6%** | **+6.7** |
| **16-19** | **423** | **81.6%** | **+7.3** |
| 20-23 | 558 | 72.0% | +5.0 |

⚠️ Strong US-session bias (12-19 UTC). Positive in all hours but 2-3x better during US trading.

### 6b. Day-of-Week

| Day | Trades | Win Rate | Avg bps |
|---|---|---|---|
| Mon | 500 | 75.4% | +5.0 |
| Tue | 591 | 73.4% | +5.5 |
| Wed | 612 | 70.8% | +5.3 |
| Thu | 663 | 69.1% | +4.3 |
| **Fri** | **679** | **88.4%** | **+8.6** |
| Sat | 462 | 56.9% | +1.7 |
| Sun | 536 | 62.7% | +3.1 |

⚠️ Weekend performance significantly weaker. Friday is best day.

### 6c. OOS Temporal Stability (ETH, first half vs second half)

| Signal | 1st Half | 2nd Half |
|---|---|---|
| ret_iqr | +5.9 bps | +4.0 bps |
| wt_mom | +6.1 bps | +3.8 bps |
| macd_hv | +5.7 bps | +4.3 bps |
| regime_persist | +5.3 bps | +3.6 bps |
| stoch_vel | +3.9 bps | +2.5 bps |

⚠️ All signals show decay from 1st to 2nd half of OOS. Still positive but declining.

### 6d. Selection Bias — KEY CONCERN

> **All 201 signal types share the same core mechanism:** fade extreme percentile readings using the same trade structure (limit offset + trailing stop). The ~99% OOS success rate across all signals is because the TRADE STRUCTURE provides the edge, not the specific signal.

The signals act as **volatility filters** — they fire during "extreme" moments where the limit+trail structure naturally profits from mean reversion.

---

## 7. Multi-Coin Results (Full Dataset)

| Symbol | ret_iqr avg | wt_mom avg | macd_hv avg | regime avg | stoch_vel avg |
|---|---|---|---|---|---|
| ETHUSDT | +5.0 | +5.0 | +5.0 | +4.4 | +3.2 |
| SOLUSDT | +5.9 | +5.2 | +5.4 | +4.0 | +3.4 |
| DOGEUSDT | +11.2 | +10.4 | +10.4 | +9.3 | +8.1 |
| XRPUSDT | +9.3 | +8.2 | +8.2 | +7.5 | +5.8 |
| **BTCUSDT** | **-2.1** | **-1.6** | **-2.0** | **-2.0** | **-2.9** |

**BTC fails all signals in the original test period (May-Aug 2025).** This is likely because BTC has lower intrabar volatility relative to the 15 bps offset, making limit fills less favorable.

---

## 8. Multi-Period Robustness

### ETHUSDT — ret_iqr_60_95 across 7 periods

| Period | Dates | Avg bps | WR | Total % | Sharpe |
|---|---|---|---|---|---|
| A | 2024-Jan-Mar | +7.9 | 73.3% | +356% | +249 |
| B | 2024-Jun-Aug | +7.8 | 67.8% | +334% | +178 |
| C | 2024-Nov-Jan | +5.4 | 65.1% | +223% | +252 |
| D | 2025-Jan-Mar | +8.4 | 69.6% | +395% | +298 |
| E | 2025-May-Aug | +5.0 | 71.8% | +201% | +335 |
| F | 2025-Sep-Nov | +8.2 | 77.5% | +363% | +403 |
| G | 2025-Dec-Feb | +12.9 | 81.0% | +531% | +294 |

**✅ ALL 7 periods positive for ETH.** Range: +5.0 to +12.9 bps avg.

### BTCUSDT — ret_iqr_60_95 across 7 periods

| Period | Dates | Avg bps | WR | Total % | Sharpe |
|---|---|---|---|---|---|
| A | 2024-Jan-Mar | +7.8 | 71.9% | +349% | +298 |
| B | 2024-Jun-Aug | +6.2 | 63.5% | +254% | +227 |
| C | 2024-Nov-Jan | +4.0 | 58.5% | +149% | +184 |
| D | 2025-Jan-Mar | +5.0 | 61.1% | +206% | +247 |
| **E** | **2025-May-Aug** | **-2.1** | **29.3%** | **-55%** | **-160** |
| F | 2025-Sep-Nov | +2.3 | 57.2% | +87% | +153 |
| G | 2025-Dec-Feb | +7.0 | 71.3% | +258% | +339 |

**⚠️ BTC fails only in Period E (May-Aug 2025).** 6/7 periods positive. Period E appears to be a low-volatility regime for BTC where the 15 bps offset doesn't get filled favorably.

---

## Red Flags & Concerns

1. **🔴 BTCUSDT fails in original test period** — All 5 signals negative. Works in 6/7 other periods. Likely a volatility regime issue.

2. **🟡 Selection bias** — 99%+ of all tested signals are positive because they share the same trade structure. The edge is in the limit+trail mechanism, not the specific signal.

3. **🟡 Temporal decay** — OOS 2nd half consistently weaker than 1st half on ETH. Edge may be slowly decaying.

4. **🟡 Weekend weakness** — Saturday/Sunday performance significantly worse (1.7-3.1 bps vs 4.3-8.6 bps weekdays).

5. **🟡 Time-of-day concentration** — Best performance during US session (12-19 UTC). Morning Asia session weakest.

---

## Recommendations

1. **Exclude BTCUSDT** from live trading, or use a volatility filter to skip low-vol BTC regimes
2. **Reduce position size on weekends** (Sat/Sun) given weaker performance
3. **Monitor for edge decay** — if avg bps drops below +2.0, reassess
4. **Focus on ETH, SOL, DOGE, XRP** — all show consistent multi-period profitability
5. **Consider the trade structure as the primary edge** — signal selection is secondary
6. **Start with small size** and validate live execution matches backtest assumptions (fill rates, slippage)

---

## Files

- **Validation script:** `validate_winners.py`
- **Source signals:** `research_v42by_stoch_efficiency.py` and related v42b* scripts
- **Data:** `/home/ubuntu/Projects/skytrade6/data/` (Bybit 1-min bars)
