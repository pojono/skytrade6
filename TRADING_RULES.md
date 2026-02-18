# Trading Rules — Derived from v29–v42 Research

**Based on**: 1.6M 5-min bars, 5 symbols (BTC/ETH/SOL/DOGE/XRP), 3+ years (2023–2026), tick-level microstructure data, walk-forward validated.

---

## CORE PRINCIPLE

**You cannot predict direction. You CAN predict volatility.**

Direction prediction AUC = 0.50 (confirmed 6+ times). Vol expansion AUC = 0.71–0.81. Every rule below is about positioning for volatility, not direction.

---

## WHEN TO TRADE

### Rule 1: Trade the London-NY Overlap
- **Peak volatility**: 14:00 UTC (summer) / 15:00 UTC (winter)
- This is the NYSE open — proven by DST shift across all 5 symbols
- Range is **60% higher** during London-NY overlap (12:00–16:00 summer) vs single-session hours
- Best single hour to be active: the hour containing NYSE open

### Rule 2: Trade Weekdays, Not Weekends
- Weekday vol is **1.25–1.67x** weekend vol (BTC has largest gap)
- Saturday is the quietest day for all 5 symbols
- Walk-forward backtest: weekdays profitable, Fri-Sat losing
- **If you must trade weekends**: only during 00:00–02:00 UTC (residual Asian flow)

### Rule 3: Trade Monday Open Aggressively
- Monday 00:00–02:00 UTC is **13–31% more volatile** than Sunday night
- Institutional/algo flow restarts — a "pseudo-open" effect
- Weekend gap size predicts Monday vol (ρ = 0.33–0.47)
- **Larger weekend gap = more volatile Monday**

### Rule 4: Trade Post-Funding Windows
- First 30 min after funding (00:00, 08:00, 16:00 UTC) is **14–21% more volatile**
- The 16:00 UTC funding coincides with US session — double effect
- Pre-funding is NOT significantly different from mid-cycle (the spike is post, not pre)

---

## WHEN NOT TO TRADE

### Rule 5: Avoid Options Expiry Fridays
- Monthly expiry (last Friday of month): **7–17% less volatile**
- Quarterly expiry (last Fri of Mar/Jun/Sep/Dec): **16–22% less volatile**
- Options pinning / max-pain suppresses volatility
- **Pre-expiry Thursday** is also quiet (especially for XRP: 22% below normal)

### Rule 6: Avoid Quarter-End Weeks
- Last 5 days of Mar/Jun/Sep/Dec: **22–30% less volatile**
- All 5 symbols affected equally
- Combine with Rule 5: quarter-end expiry Friday is the quietest trading day of the year

### Rule 7: Avoid September
- September is the **quietest month** for all 5 symbols, every year
- Range is 1.65–2.23x lower than peak months (Mar/Nov)
- Reduce position sizes or widen entry thresholds in September

### Rule 8: Avoid Late-Month Weeks
- Weeks 4–5 (22nd–31st) are consistently quieter than weeks 1–2
- Best weeks to trade: **first two weeks of the month**

### Rule 9: Avoid Asia-Only Hours on Weekends
- Saturday 04:00–06:00 UTC is the absolute vol trough (BTC: 8.9 bps — **3.9x less** than Tuesday 14:00)
- If vol is below your minimum threshold, sit out entirely

---

## HOW TO TRADE

### Rule 10: Use 2:1 TP/SL Ratio (Symmetric Straddle)
- Place simultaneous long TP + short TP at 2:1 ratio
- **Conservative**: TP=6 bps, SL=3 bps — works on ALL symbols including BTC
- **Aggressive**: TP=10 bps, SL=5 bps — better for high-vol coins (DOGE/SOL/XRP)
- Time limit: 300 seconds (5 minutes)
- Edge comes from fat tails (excess kurtosis), not direction

### Rule 11: Use Volume Spikes as Entry Triggers
- Volume **Granger-causes** range (F-stats 2,900–16,000)
- A **3σ volume surge** predicts **6–9x normal volatility** in the next 5-min bar
- A **2σ volume surge** predicts **3–4x normal volatility**
- Even a **1σ surge** predicts **2x normal volatility**
- **Action**: When volume spikes, immediately enter straddle positions

### Rule 12: After a Vol Spike, Ride the Decay
- Vol decays as a **power law** (not exponential) — it persists longer than you think
- After a 3σ spike, vol stays at 50% of spike level for **~8 hours**
- BTC half-life: ~27 hours; DOGE: ~54 hours (altcoins cluster longer)
- **Action**: After a vol event, keep straddles active for hours, not minutes
- Gradually reduce position size as vol decays

### Rule 13: For BTC, Use ML Timing; For Alts, Just Trade
- BTC is low-vol and mean-reverting at 5-min scale — needs ML model (AUC 0.64) to time entries
- ETH/SOL/DOGE/XRP have enough vol that 2:1 TP/SL is naturally profitable
- **BTC**: ML P90 filter → trade only when model says "high vol likely"
- **Others**: Z-score vol filter (z > -0.5) → trade whenever vol isn't abnormally low

---

## POSITION SIZING

### Rule 14: Scale by Expected Volatility
- Use the temporal calendar to set base position size:
  - **Peak hours (London-NY overlap)**: full size
  - **Single-session hours**: 70% size
  - **Quiet hours (21:00–00:00 UTC)**: 50% size
  - **Weekend**: 30% size or skip

### Rule 15: Scale by Month
- **Mar, Nov**: peak months — full size
- **Jan, Feb, Oct, Dec**: above average — 90% size
- **Apr, May, Jun, Jul, Aug**: average — 80% size
- **Sep**: trough — **60% size or skip**

### Rule 16: Boost Size After Volume Surges
- 1σ volume surge: 1.5x normal size
- 2σ volume surge: 2x normal size
- 3σ volume surge: 3x normal size (max)
- Decay back to normal over 30–60 minutes

---

## CROSS-ASSET RULES

### Rule 17: All Crypto Vol Moves Together
- BTC-ETH vol correlation: **0.81**
- BTC-SOL: 0.68, BTC-DOGE: 0.64, BTC-XRP: 0.60
- No lead-lag at 5-min resolution — vol propagates in < 5 minutes
- **If one asset spikes, all will spike simultaneously** — don't wait for confirmation

### Rule 18: DOGE is the Best Vol Asset
- Highest EV per trade (+1.30 bps at 2:1 TP/SL)
- Highest fat-tail excess (3:1 ratio still profitable at +1.15 bps)
- But also highest absolute risk — size accordingly

### Rule 19: BTC is the Hardest to Trade
- Lowest vol, most mean-reverting at 5-min scale
- Only symbol where inverse ratio (1:2) is profitable without ML
- **Needs ML timing** or use tighter levels (TP=6/SL=3)

---

## RISK MANAGEMENT

### Rule 20: 1:1 TP/SL = Zero Edge (Always)
- Confirmed across all 5 symbols, all time periods
- 1:1 EV is exactly 0.00 — there is no directional edge to capture
- Never use symmetric TP/SL — always maintain asymmetry

### Rule 21: Don't Chase Direction
- Continuation prediction: AUC = 0.50 (dead)
- Straight-line move prediction: AUC = 0.50 (dead)
- Direction UP prediction: AUC = 0.50 (dead)
- **No feature set we tested can predict direction** — not vol, not OI, not funding, not liquidations

### Rule 22: VWAP Reversion is Real but Not Tradeable
- Price reverts to intraday VWAP (ρ = -0.02 to -0.05, all p ≈ 0)
- But the effect is **far too small** to overcome trading costs
- Don't build strategies around VWAP mean reversion at 5-min scale

---

## CURIOSITIES (Use with Caution)

### Rule 23: New Moon = More Volatile
- New moon is **8–23% more volatile** than full moon (all 5 symbols, p ≈ 0)
- KW H-stats 230–872 — statistically bulletproof
- **BUT**: likely confounded with month-end/expiry cycles (~29.5 day lunar cycle ≈ calendar month)
- Treat as supplementary signal, not primary

### Rule 24: Winter is More Volatile Than Summer
- Winter months show **1.2–1.5x higher volatility** across all hours
- This combines with the September trough to create a clear seasonal pattern
- Best trading months: **Nov–Mar** (winter + high-vol months)

---

## STRATEGY SUMMARY

```
ENTRY DECISION:
  1. Is it a weekday?                          → if no, reduce size 70%
  2. Is it London-NY overlap (12-16 UTC)?      → if yes, full size
  3. Is it post-funding (first 30 min)?        → if yes, boost 15%
  4. Is it options expiry day?                 → if yes, reduce size 20%
  5. Is it September or quarter-end week?      → if yes, reduce size 30%
  6. Did volume just spike 2σ+?               → if yes, ENTER NOW, boost size
  7. For BTC: does ML model say P90+?          → if no, skip BTC

EXECUTION:
  - Place symmetric straddle: long TP + short TP
  - BTC/ETH: TP=6, SL=3 (conservative)
  - SOL/DOGE/XRP: TP=10, SL=5 (aggressive)
  - Time limit: 300 seconds
  - Expected: ~280 trades/day/symbol, 62-72% win rate

EXIT:
  - TP or SL hit → done
  - Time limit hit → close at market
  - After vol spike: keep trading for 8+ hours (power-law decay)

EXPECTED PERFORMANCE (walk-forward validated):
  - Portfolio (5 symbols): ~1,400 trades/week
  - Avg EV: +0.84 bps/trade
  - Weekly PnL: ~+7,250 bps
  - Losing days: primarily Sat (low vol)
```

---

*Last updated: Feb 2026. Based on research v29–v42, validated on 2023–2026 data.*
