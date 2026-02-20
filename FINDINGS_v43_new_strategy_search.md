# FINDINGS v43: New Strategy Search (No Trailing Stop)

**Date:** 2026-02-20
**Constraint:** No trailing stop (inflates results without tick simulation), limit orders preferred, no HFT
**Fees:** maker 0.02%, taker 0.055%
**Data:** Bybit ticker (5-sec, 76 days), Bybit futures trades (tick, 1143 days), Bybit 1h OHLCV (1143 days)

---

## Summary: 8 Strategy Ideas Tested, ALL Failed

| ID | Strategy | Data | Timeframe | Result |
|----|----------|------|-----------|--------|
| v43a | Funding Settlement MR | Ticker 5s | 1h | Too few signals (3-21/week) |
| v43a | OI Squeeze MR | Ticker 5s | 1h | Too few signals |
| v43a | Spread Widening MR | Ticker 5s | 1h | Mostly negative, 1 marginal on 9 trades |
| v43b | Pure Price MR (1-min bars) | Ticker 5s | 1min | **ALL 165 configs negative**. Fee wall too high |
| v43b | Price MR + OI confirmation | Ticker 5s | 1min | Negative. OI filter doesn't help |
| v43b | Price MR + Spread confirmation | Ticker 5s | 1min | Negative |
| v43b | Price MR + Funding alignment | Ticker 5s | 1min | Negative |
| v43c | Funding Rate Contrarian (4h hold) | Ticker 5s | 1h/4h | **ALL negative** on SOL and ETH |
| v43c | Mark-Index Spread Contrarian | Ticker 5s | 1h/4h | Negative |
| v43c | Combined Funding + MIS | Ticker 5s | 1h/4h | Negative |
| v43d | Symmetric Grid (14 days) | Ticker 5s | 1min | **Positive IS** (Sharpe 3-12 on SOL) |
| v43e | Symmetric Grid (76 days, 5 symbols) | Ticker 5s | 1min | **Fails OOS** — 1/20 configs positive (BTC, n=26) |
| v43f | Volume Imbalance Momentum (30d) | Tick trades | 1h | Hints of signal (n=19, +37 bps OOS) but too few trades |
| v43h | Volume Imbalance Momentum (3yr) | 1h OHLCV | 1h | **ALL 12 configs × 3 symbols negative** over 3 years |
| v43h | Volume Imbalance Contrarian (3yr) | 1h OHLCV | 1h | Negative |
| v43h | Buy Ratio Momentum (3yr) | 1h OHLCV | 1h | Negative |
| v43h | Cumulative Imbalance (3yr) | 1h OHLCV | 1h | Negative |

---

## Key Lessons

### 1. The Fee Wall Is Real
Short-term MR on 1-min bars generates ~5-10 bps gross edge, but round-trip fees of 4-7.5 bps eat it all. **No 1-min MR strategy survives fees** with fixed TP/SL (no trailing stop).

### 2. Directional Signals Are Too Weak
Funding rate (IC=-0.12 on SOL, confirmed OOS) and mark-index spread (IC=-0.06) are real signals but **far too weak to generate profits after fees** as standalone strategies. The 4h holding period doesn't help — the signal magnitude is ~10-20 bps gross, fees are 7.5 bps.

### 3. Grid Strategy = Short Volatility
Symmetric grid is profitable in range-bound markets (14 days SOL: Sharpe 12) but **catastrophically fails in trends**. Timeout trades lose 60-170 bps, overwhelming TP profits. OOS validation (76 days, 5 symbols) shows 19/20 configs negative.

### 4. Volume Imbalance Has No Edge
Buy/sell volume imbalance on 1h bars — neither momentum nor contrarian — has any predictive power over 3 years (38 months). Positive months ≈ 45-50% = coin flip. This is true for SOL, ETH, and BTC.

### 5. The v43f "Promising Hint" Was Noise
The 30-day test showed +37 bps avg on 19 OOS trades. Expanding to 3 years (700+ trades) revealed this was pure noise — the signal averages -1.5 to -10.7 bps over 38 months.

---

## What This Means

The only strategies that have survived rigorous OOS testing in this repository are:
1. **Cascade MM** (liquidation-based) — proven in v41, v42
2. **Microstructure MR with trailing stop** — proven in v42s/v42t

Both rely on **trailing stop** for exit, which the user correctly identified as unreliable without tick-level simulation.

Without trailing stop, using only fixed TP/SL + timeout:
- **No directional signal** is strong enough to overcome fees
- **No MR signal** on short timeframes survives fees
- **Grid strategies** are short-vol and fail in trends
- **Volume flow signals** are noise on 1h+ timeframes

### Remaining Unexplored Directions
1. **Tick-level simulation** of cascade MM / micro MR (validate trailing stop accuracy)
2. **Options/vol arbitrage** (vol prediction R²=0.34 is the strongest confirmed signal)
3. **Cross-exchange arbitrage** (requires sub-second, violates no-HFT constraint)
4. **Longer holding periods** (daily/weekly) with stronger signals
5. **ML ensemble** on 1h OHLCV features with walk-forward (but v24e showed this fails on 3yr)

---

## Files

| File | Description |
|------|-------------|
| `research_v43_new_strategies.py` | v43a: Funding/OI/Spread MR |
| `research_v43b_mean_reversion.py` | v43b: Price MR with confirmations |
| `research_v43c_funding_directional.py` | v43c: Funding rate directional |
| `research_v43d_adaptive_grid.py` | v43d: Grid strategy prototype |
| `research_v43e_grid_validation.py` | v43e: Grid 76-day × 5-symbol validation |
| `research_v43f_volume_imbalance.py` | v43f: Volume imbalance 30-day test |
| `research_v43h_vol_imb_ohlcv.py` | v43h: Volume imbalance 3-year validation |
| `PLAN_v43_fresh_validation.md` | Original plan (superseded) |

---

**Research Status:** Complete ✅
**Verdict:** No new profitable strategy found without trailing stop. The fee wall + weak signals + regime changes make it extremely difficult to find edge with fixed TP/SL on crypto futures.
