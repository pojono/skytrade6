# FINDINGS v40: Regime-Switch Causal Chain Validation & Trading Strategy

## Overview

**Objective**: Validate the regime-switch causal chain (from v29 findings) on a much larger dataset (5 symbols × 88 days) and build a profitable trading strategy based on this logic.

**Data**: Tick-level trades, liquidations, OI, and funding rate for BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, DOGEUSDT from 2025-05-12 to 2025-08-07 (88 days each).

**Result**: 
- ✅ **Causal chain STRONGLY CONFIRMED** across all 5 symbols (5/6 elements confirmed on each)
- ❌ **No profitable trading strategy found** — regime switches are untradeable noise events at second-level resolution

---

## 1. Causal Chain Validation (All 5 Symbols)

### Summary Table

| Symbol   | Switches | Liq Lead | Vol Spike | Liq Peak | OI Drop      | OI Unwind     | FR Lift | Stab(s) | Score |
|----------|----------|----------|-----------|----------|--------------|---------------|---------|---------|-------|
| BTCUSDT  | 2,473    | 3.17x ✅ | 2.10x ✅  | 10.32x ✅| -$465K ✅    | -$1,286K ✅   | 1.11x ❌| 62s ✅  | 5/6   |
| ETHUSDT  | 1,994    | 3.14x ✅ | 2.12x ✅  | 19.05x ✅| -$513K ✅    | -$2,139K ✅   | 1.14x ❌| 92s ✅  | 5/6   |
| SOLUSDT  | 1,446    | 4.04x ✅ | 2.11x ✅  | 17.55x ✅| -$217K ✅    | -$395K ✅     | 1.13x ❌| 75s ✅  | 5/6   |
| XRPUSDT  | 1,568    | 3.41x ✅ | 2.18x ✅  | 16.39x ✅| -$81K ✅     | -$205K ✅     | 1.11x ❌| 69s ✅  | 5/6   |
| DOGEUSDT | 1,438    | 4.90x ✅ | 2.12x ✅  | 13.62x ✅| -$75K ✅     | -$76K ✅      | 1.20x ❌| 96s ✅  | 5/6   |

### Confirmed Causal Chain Sequence

The following sequence is **universally confirmed** across all 5 symbols and 88 days:

1. **Liquidation Lead (-30s)**: Liquidation count rises 3-5x above baseline ~30 seconds BEFORE the vol threshold crossing. This is the earliest detectable signal.

2. **Volatility Spike (0s)**: 60s realized volatility crosses 2x the rolling 1h median — this defines the "regime switch" moment.

3. **Liquidation Peak (+30s)**: Liquidation count peaks at 10-19x baseline approximately 30 seconds AFTER the switch. This is the cascade peak.

4. **OI Drop at Switch**: Open interest drops $75K-$513K at the switch moment (varies by symbol notional size). Confirms forced position closures.

5. **OI Unwind (+150s)**: Cumulative OI drop reaches $76K-$2.1M by +150 seconds. The cascade takes ~2.5 minutes to fully unwind.

6. **FR Clustering**: Funding rate 2-4h clustering shows only 1.11-1.20x lift — **NOT statistically significant**. FR timing is NOT a reliable predictor of regime switches.

7. **Stabilization**: Volatility returns to near-baseline levels within 62-96 seconds after the switch.

### Key Observations

- **OI was FALLING before switches** (⚠️ on all symbols) — contradicts the v29 hypothesis that OI rises before switches. On 88 days of data, OI is already declining before the cascade fires, suggesting the cascade is part of an ongoing deleveraging, not a sudden break from a buildup.

- **Liq lead is strongest on altcoins**: DOGEUSDT (4.90x) and SOLUSDT (4.04x) show the strongest liquidation lead signals, likely because altcoin cascades are more violent and predictable.

- **ETH has the largest OI unwind**: -$2.1M cumulative OI drop by +150s, reflecting its deep derivatives market.

- **Stabilization is fastest on BTC** (62s) and slowest on DOGE (96s) — larger, more liquid markets recover faster.

---

## 2. Trading Strategy Results

### Approach

Tested 17 strategy variants across 3 directional approaches:

**A) Momentum** — ride the cascade direction (sell-liqs → SHORT, buy-liqs → LONG)
**B) Reversion** — fade the cascade after peak (sell-liqs → LONG after delay)  
**C) Price Momentum** — use actual price change direction

Each with variations in:
- Entry delay: 5s, 90s, 120s, 180s, 300s
- Hold time: 60s, 120s, 180s, 300s, 480s
- Confirmations: none, liq P90, OI dropping, full (both)
- Stop loss: 3-4x recent vol
- Take profit: none or 2x vol

### Results: ALL STRATEGIES LOSE MONEY

| Symbol   | Best Variant          | Trades | WinR% | AvgBps | TotRet%  | Sharpe |
|----------|-----------------------|--------|-------|--------|----------|--------|
| BTCUSDT  | rev_90s_3min          | 1,989  | 15.6% | -3.6   | -72.30%  | -35.5  |
| ETHUSDT  | mom_full_5s_2min      | 488    | 32.0% | -2.5   | -12.09%  | -11.4  |
| SOLUSDT  | rev_oi_90s_3min       | 366    | 28.7% | -1.7   | -6.25%   | -5.8   |
| XRPUSDT  | mom_full_5s_2min      | 388    | 29.1% | -2.7   | -10.66%  | -13.3  |
| DOGEUSDT | mom_full_5s_3min      | 388    | 29.1% | -2.5   | -9.83%   | -10.8  |

### Walk-Forward OOS Results: ALSO ALL NEGATIVE

Every top-3 IS variant tested OOS produced negative returns. No variant showed any edge.

### Why No Strategy Works

1. **Regime switches are high-vol noise events**: At the moment of a switch, vol is 2x+ normal. Any directional bet faces enormous adverse selection — the stop loss gets hit 40-70% of the time.

2. **Average loss per trade ≈ -3.5 bps**: This is almost exactly the round-trip fee (4 bps). The strategies have ~zero gross edge — they're essentially random after accounting for the extreme vol environment.

3. **Win rates are 13-32%**: Even the best variants win less than 1/3 of trades. The cascade direction is unpredictable at the second level.

4. **Momentum and reversion both fail**: Momentum fails because the cascade direction reverses unpredictably. Reversion fails because the "overshoot" doesn't reliably bounce back within the hold period.

5. **Confirmations reduce trades but don't improve edge**: OI and liq confirmations filter out ~60-80% of trades but the remaining trades have the same negative expectancy.

---

## 3. Conclusions

### What the Causal Chain Tells Us

The regime-switch causal chain is a **real, robust market microstructure phenomenon**:
- Liquidations fire → vol spikes → more liquidations cascade → OI unwinds → market stabilizes
- This happens ~1,500-2,500 times per symbol over 88 days (~17-28 per day)
- The sequence is consistent across BTC, ETH, SOL, XRP, and DOGE

### Why It's Not Tradeable

The causal chain describes **what happens during a cascade**, not **which direction the cascade goes**. The key missing piece is **directional predictability**:

- Liquidation side (sell vs buy) is roughly balanced across all switches
- Price momentum at the switch is not predictive of continuation
- OI direction before the switch is not predictive
- FR timing is not predictive

The cascade is a **symmetric volatility event** — it's equally likely to go up or down, and the extreme vol environment makes any directional bet a coin flip with high transaction costs.

### Practical Implications

1. **Risk Management (Primary Value)**: The causal chain is most valuable for **avoiding** regime switches, not trading them. If you detect a liq spike (the 30s lead signal), you should:
   - Reduce position size
   - Widen stops
   - Avoid entering new positions for ~2-3 minutes

2. **Volatility Trading**: The predictable vol spike could be traded via options/vol products (buy vol when liq lead fires, sell vol after stabilization), but this requires options market access.

3. **Market Making**: Market makers should widen spreads when the liq lead signal fires and narrow them after stabilization (~60-90s).

4. **NOT for Directional Trading**: Simple directional bets (momentum or reversion) during regime switches have zero edge after fees across all tested variants and symbols.

### Revised Understanding vs v29

| Aspect | v29 (7 days, BTC only) | v40 (88 days, 5 symbols) |
|--------|------------------------|--------------------------|
| Causal chain | Confirmed | **Strongly confirmed** |
| OI rises before switch | Suggested | **NOT confirmed** — OI falls before switches |
| FR timing matters | Suggested | **NOT confirmed** — FR lift is only 1.1x |
| Liq as short-term signal | 30s lead | **Confirmed** — 3-5x lead across all symbols |
| Directional trading | Not tested | **Tested exhaustively — NO EDGE** |
| Best use case | Unclear | **Risk management / vol avoidance** |

---

## 4. Data Summary

- **Total regime switches detected**: 8,919 across 5 symbols
- **Total trades processed**: ~607M tick-level trades
- **Total seconds analyzed**: ~38M (5 × 7.6M)
- **Strategy variants tested**: 17 × 5 symbols = 85 backtests
- **Runtime**: ~50 minutes
