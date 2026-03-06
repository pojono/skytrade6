# Post-Mortem: claude-2 Research Sprint

**Date:** 2026-03-06
**Duration:** ~3 sessions across Feb–Mar 2026
**Scope:** 8 signal ideas + 1 new cross-exchange signal, tested on Bybit + Binance data (2024-01 → 2026-03)

---

## What We Set Out To Do

Find a deployable short-term trading edge in crypto perpetual futures using:
- Open Interest, Long/Short ratios, Funding Rates
- Cross-symbol lead-lag (BTC → alts)
- Spot-futures divergence
- Cross-exchange price divergence (Bybit vs Binance)
- Orderbook L2 imbalance
- Volatility compression / OI breakout signals

Target: net-positive after 20 bps round-trip taker fees, validated out-of-sample.

---

## What We Found

### Signal-by-signal results

| # | Signal | Hypothesis | Verdict | Why it failed |
|---|--------|-----------|---------|---------------|
| 1 | OI + L/S crowding | Extreme positioning → liquidation cascade | ❌ Dead | No signals fired at chosen thresholds; relaxing thresholds → noise |
| 2 | Premium reversion | Spot-futures premium mean-reverts | ❌ Dead | Edge eaten by fees; too small and inconsistent |
| 3 | High implied FR | Premium momentum | ❌ Dead | -4 bps avg net; no directionality |
| 4 | BTC pump → long alts | BTC leads, alts follow with lag | ⚠️ Dead* | Worked in 2024 (+213 bps), dead in 2025-2026 without Oct outlier |
| 5 | Spot leads futures | Spot price divergence predicts futures | ❌ Dead | +16 bps raw, regime-filtered to +34 bps but too thin and inconsistent |
| 6 | Vol compression + OI | Coiled spring breakout | ❌ Dead | -18 bps avg; no directional edge |
| 7 | Combined signals | Ensemble of above | ❌ Dead | Combining dead signals doesn't create a live one |
| 8 | Combo ideas | Various combinations | ❌ Dead | Same as above |
| 9 | Cross-exchange divergence | Bybit vs Binance price gap | ❌ Dead | -20 bps; markets too efficient at 1m resolution |

**\*Idea 4 appeared to survive all audits at +269 bps avg. Removing a single October 2025 outlier day (+5944 bps) revealed the edge was dead since end of 2024.**

### Orderbook L2 attempt

Attempted to build an orderbook imbalance signal. Discovered that Bybit orderbook data is 99.8% delta updates with only 1 full snapshot per day — requires full book reconstruction from deltas. Deprioritized as the marginal value vs complexity was poor.

---

## What We Did Right

1. **Rigorous anti-bias framework**
   - T+1 entry delay on all signals (no same-bar lookahead)
   - 30-bar declustering to prevent signal clustering inflation
   - Daily portfolio aggregation instead of per-symbol averaging
   - Walk-forward regime filters with expanding window (min 180 days training)

2. **Honest OOS testing**
   - True OOS period (2024) was never touched during signal discovery
   - Bootstrap confidence intervals and shuffle tests
   - Bonferroni correction for multiple comparisons

3. **Multi-exchange validation**
   - Tested all signals across both Bybit and Binance data
   - Cross-exchange signal tested explicitly

4. **Tick-level validation**
   - Verified alt reaction curve at millisecond precision
   - Confirmed entry timing windows and per-alt reaction speeds
   - Provided actionable execution parameters (even though the signal is dead)

5. **Killed our darlings**
   - Did not declare victory prematurely
   - October outlier analysis was the right call — it exposed the truth

---

## What We Did Wrong

1. **Initially fooled by aggregation methodology**
   - Per-symbol averaging made Idea 4 look weak; daily aggregation made it look strong
   - Neither was "wrong" but the daily aggregation hid the fact that the edge was concentrated in a few extreme days
   - Lesson: always check both AND check robustness to outlier removal

2. **Didn't check outlier sensitivity early enough**
   - The October 2025 outlier (+5944 bps) inflated every metric by 2-3x
   - Should have run outlier-excluded analysis from the start
   - Lesson: **always report "without best N days" alongside headline numbers**

3. **Spent too long on signals before checking sample size**
   - 34 signal days across 26 months is not enough for statistical significance
   - Sharpe ~0.28 on 32 observations → p-value >0.3 → cannot reject null
   - Should have flagged "insufficient sample" earlier and moved on

4. **Undermodeled execution costs**
   - Only 20 bps taker fees. No spread, no slippage, no market impact
   - During the exact moments we need to trade (BTC pump = high vol), costs are highest
   - Lesson: for volatility-triggered strategies, model worst-case execution

5. **Survivorship bias in signal selection**
   - Tested 9 signals, 1 "survived" → classic multiple testing problem
   - Even with Bonferroni, 1/9 surviving at p=0.3 is expected by chance
   - Lesson: one surviving signal out of many is weaker evidence than it appears

---

## Key Lessons for Future Research

### Statistical rigor

- **N < 50 trades → don't bother with fancy analysis.** The sample is too small for any meaningful inference.
- **Always exclude top 3 outliers and re-evaluate.** If the signal dies → it was never real.
- **Report Sharpe on trade-level returns, not averages.** Mean ÷ std with small N is unreliable.
- **Multiple testing: 1/9 surviving at low significance ≈ noise.**

### Execution reality

- **Spread + slippage matter more than fees** for high-frequency signals during volatile events.
- **Maker orders are unrealistic** for momentum-following strategies — you're always a taker.
- **Simultaneous execution across 10 alts** has market impact that compounds with volatility.

### Signal design

- **Cross-asset lead-lag decays.** Markets get faster. What worked in 2024 doesn't work in 2026.
- **1-minute resolution is too coarse** for most microstructure signals (spreads already closed).
- **Crypto microstructure signals have short half-lives.** The market adapts within 6-12 months.
- **Low frequency ≠ robust.** 1-2 signals/month with high variance is an unmanageable strategy.

### Process

- **Kill signals faster.** If N < 50 after initial scan → move on immediately.
- **Check outlier robustness first.** Before any deep analysis, remove top 3 days and check if the signal still exists.
- **Model realistic costs from day one.** Not after the signal "passes" other tests.
- **Track edge decay over time.** Plot rolling 6-month performance to catch decay early.

---

## Final Honest Assessment

**Zero deployable edges found.** All 9 signals tested are dead or dying after realistic cost modeling and outlier removal.

The research was not wasted — it established:
- A rigorous testing framework (T+1, declustering, daily aggregation, walk-forward regime filters)
- Data infrastructure across Bybit + Binance (klines, tick trades, orderbooks, funding, OI)
- Clear lessons on what doesn't work and why

But the bottom line is: **no production-ready strategy emerged from this sprint.**

---

## What Would Be Worth Trying Next

1. **Higher frequency** — sub-second signals where execution edge matters (requires colocation)
2. **Market making** — the bid-ask spread IS the edge, not a cost
3. **Funding rate harvesting** — already explored in skytrade7, shows more promise
4. **Cross-exchange arbitrage** — requires maker execution on both venues (VIP tiers)
5. **Alternative data** — social sentiment, on-chain flows, whale wallet tracking
6. **Longer horizons** — daily/weekly signals where 20 bps fees are negligible

The common thread: the 1-minute-bar, taker-execution, single-exchange research space that claude-2 explored is likely mined out. Future work should change at least one of these constraints.
