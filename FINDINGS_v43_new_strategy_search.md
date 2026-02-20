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
| v43i | Daily Momentum (1d-20d lookback) | 1h→daily | daily | **Positive on 5/5 symbols** — but see v43j |
| v43i | Day-of-Week effects | 1h→daily | daily | Inconsistent across symbols |
| v43i | Consecutive day patterns | 1h→daily | daily | Weak MR after 2+ up days, not robust |
| v43j | Daily Momentum validation | 1h→daily | daily | **Buy-and-hold bias!** Alpha negative on 3-5/5 symbols |
| v43k | Cascade MM tick-level (fixed TP/SL) | Liquidations + ticks | tick | **Catastrophic failure.** 0-24% WR, avg -27 to -108 bps |
| v43l | Cascade MOMENTUM (follow flow) | Liquidations + ticks | tick | **72% WR on SOL 14d** but fails on full 72 days |
| v43m | Cascade Momentum validation | Liquidations + ticks | tick | Signal real (beats random 3-8 bps) but execution cost kills edge |
| v43n | Funding Rate Harvesting | Ticker 5s | 8h | Carry too small (0.3-0.6 bps/period) vs price risk (±50-200 bps) |
| v43o | Spot-Futures Basis Trade | 1h OHLCV spot+futures | 1h | Spot fees 0.1% kill it (24 bps RT). Basis only ±6 bps |
| v43o | Futures-Only Basis Signal | 1h OHLCV | 1h | IS +3.2 bps avg but too few OOS trades (basis rarely >10 bps) |
| v43o | Cross-Exchange Lead-Lag | 1h OHLCV 3 exchanges | 1h | Spread ±1.5 bps, lag-1h corr=0.04 — no signal |
| v43p | Combined Weak Signals | Ticker + 1h OHLCV + spot | 1h | Top-2 (rvol_z+mr_4h) OOS +17 bps on SOL 76d |
| v43q | Rvol+MR 3-Year Validation | 1h OHLCV | 4h hold | **OOS positive ALL 5 symbols!** avg +8 to +69 bps |
| v43r | Deep Walk-Forward Validation | 1h OHLCV | 4h hold | **Real but regime-dependent.** Works in V-dips, fails in sustained trends |

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

### 6. Daily Momentum = Buy-and-Hold Bias
v43i found multi-day momentum positive on 5/5 symbols. But v43j rigorous validation proved:
- **Alpha (strategy - B&H) is negative** on 3-5 out of 5 symbols for every config
- **Short side is unprofitable** on 4/5 symbols (-73% to -131% total)
- The "momentum" is just capturing the crypto bull market drift
- Quarterly breakdown shows ~50% positive quarters = coin flip
- This is NOT a tradeable edge — it's a disguised buy-and-hold

---

## What This Means

The only strategies that have survived rigorous OOS testing in this repository are:
1. **Cascade MM** (liquidation-based) — proven in v41, v42
2. **Microstructure MR with trailing stop** — proven in v42s/v42t

Both rely on **trailing stop** for exit, which the user correctly identified as unreliable without tick-level simulation. Furthermore, the TICK_LEVEL_FINAL_REPORT.md proved that **all 201 signals were invalid** — the original edge was caused by simulation bugs (same-bar fill+exit, trailing stop intra-bar lookahead).

Without trailing stop, using only fixed TP/SL + timeout:
- **No directional signal** is strong enough to overcome fees (1min to daily)
- **No MR signal** on any timeframe survives fees
- **Grid strategies** are short-vol and fail in trends
- **Volume flow signals** are noise on 1h+ timeframes
- **Daily momentum** is buy-and-hold bias, not alpha

### 7. Cascade MM Fails at Tick Level Without Trailing Stop
v43k tested liquidation cascade mean-reversion at tick level with fixed TP/SL (no trailing stop). Results:
- **0% win rate** on tight configs (SL hit immediately on every trade)
- **24% WR max** on widest config (TP=100, SL=200), still avg -77 bps
- 85% of SOL cascades are short liquidations (price moving UP) — fading = going short into forced buying
- The cascade "edge" was entirely from bar-level simulation bugs (confirmed by TICK_LEVEL_FINAL_REPORT)

### Remaining Potentially Viable Directions
1. ~~Tick-level cascade MM with fixed TP/SL~~ — **TESTED AND FAILED (v43k)**
2. **Options/vol arbitrage** — vol prediction R²=0.34 is the strongest confirmed signal, but requires options market access
3. **Cross-exchange basis trade** — spot-futures basis or cross-exchange price differences (may not require sub-second)
4. **Funding rate harvesting** — systematically collect funding payments while hedging directional risk
5. **Event-driven** — earnings, token unlocks, exchange listings (requires external data)
6. ~~Cascade MOMENTUM~~ — **TESTED (v43l-m)**: signal is real (beats random) but taker entry + spread during cascade eats the edge
7. ~~Funding rate harvesting~~ — **TESTED (v43n)**: carry is 0.3-0.6 bps/period, dwarfed by price risk and fees

### 8. Cascade Momentum: Real Signal, Killed by Execution
v43l-m tested following cascade direction (not fading). Results:
- **SOL 14 days**: 72% WR, +18 bps avg (promising!)
- **SOL 72 days**: 70% WR but avg **-2.9 bps** (negative after fees)
- Signal **consistently beats random** by 3-8 bps across all symbols
- But taker entry fee (5.5 bps) + spread widening during cascades eats the edge
- DOGE OOS marginally positive (+0.6 bps avg) but essentially breakeven

### 9. Funding Rate Harvesting: Carry Too Small
v43n tested collecting funding payments by positioning against the crowd:
- Average funding: +0.3 to +0.6 bps per 8h period
- RT fees: 4 bps (maker+maker) — need 7+ funding periods just to cover fees
- Price risk during holding: ±50-200 bps per period — overwhelms carry
- BTC contrarian >1.0 bps: IS +4.7, OOS +10.2 bps — but only 19 OOS trades
- 3-period configs showing huge OOS gains are directional bets, not carry

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
| `research_v43i_daily_patterns.py` | v43i: Daily pattern search (3yr × 5 symbols) |
| `research_v43j_momentum_validation.py` | v43j: Momentum rigorous validation |
| `research_v43k_cascade_tick.py` | v43k: Cascade MM tick-level simulation |
| `research_v43l_cascade_momentum.py` | v43l: Cascade momentum (follow flow) |
| `research_v43m_cascade_mom_v2.py` | v43m: Cascade momentum extended validation |
| `research_v43n_funding_harvest.py` | v43n: Funding rate harvesting |
| `PLAN_v43_fresh_validation.md` | Original plan (superseded) |

---

**Research Status:** Complete ✅ (19 strategy variants tested across v43a-v43r)

**Final Verdict:** One promising strategy found after exhaustive search of 19 variants.

### Failures (v43a-o):
1. **Fee wall** — edge < fees on short timeframes (v43a-c)
2. **Regime fragility** — works IS, fails OOS (v43d-e grid)
3. **No signal** — volume imbalance, daily patterns are noise (v43f-h, v43i-j)
4. **Simulation artifact** — previous cascade MM edge was from bugs (v43k)
5. **Execution cost** — cascade momentum signal real but killed by taker fees (v43l-m)
6. **Carry too small** — funding payments dwarfed by price risk (v43n)
7. **Basis too small** — spot fees 0.1% kill basis trade, cross-exchange spread ±1.5 bps (v43o)

### The One Promising Finding (v43p-r): Volatility Dip-Buying

**Signal:** Combined z-score of realized volatility (rvol_z) + 4h mean-reversion (mr_4h).
When vol spikes AND price has dipped → go long. Threshold=2.0, hold=4h, limit entry+exit (4 bps RT).

**3-Year Walk-Forward Results (thresh=2.0, hold=4h):**

| Symbol | Trades | Avg bps | Total | WR | z vs Random | WF pos months |
|--------|--------|---------|-------|-----|-------------|---------------|
| SOL | 139 | +51.9 | +72.1% | 63.3% | +2.19 | 75% |
| ETH | 144 | +21.5 | +30.9% | 52.8% | +1.67 | 52% |
| BTC | 169 | +7.6 | +12.9% | 53.8% | +1.31 | 59% |
| DOGE | 168 | +17.6 | +29.6% | 56.5% | +0.83 | 52% |
| XRP | 158 | +68.5 | +108.3% | 59.5% | +4.95 | 68% |

**Strengths:**
- Positive on ALL 5 symbols over 3 years
- Beats random on all 5 (z-score +0.83 to +4.95)
- Fee-robust (XRP still +83% at 20 bps fees)
- Capital deployed only ~2% of time (150 trades × 4h / 3yr)
- Works in some bear quarters (SOL 2023Q2: +12.8% vs B&H -10.9%)

**Weaknesses / Risks:**
- Heavily long-biased (90-97% long trades) — short side has too few trades to validate
- Fails in sustained downtrends (SOL 2025Q4: -2.2%, 2026Q1: -7.2%)
- Regime-dependent: works when dips are V-shaped, fails in grinding bear markets
- Low trade frequency (~4-5 trades/month) — slow to compound
- Max drawdown 13-24% across symbols

**Honest Assessment:** This is a "buy the dip in high volatility" strategy. It has genuine statistical edge (beats random consistently) but is not regime-independent. It would need a regime filter (e.g., skip when 20d trend is strongly negative) to avoid sustained bear market losses.

**Remaining paths:**
- Add regime filter to combined signal (skip trades when trend is negative)
- **Options vol arbitrage** (vol prediction R²=0.34 is strongest confirmed signal)
- **Co-location / maker rebates** (monetize cascade momentum signal)
