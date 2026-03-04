# Cross-Exchange Pattern Research — Findings

## Executive Summary

We investigated cross-exchange price dislocations between **Bybit and Binance** across **116 common USDT perpetual symbols** over **8 months** (Jul 2025 – Mar 2026) using 1-minute OHLCV, premium index, funding rate, open interest, and long/short ratio data.

**Bottom line: A structural edge exists in cross-exchange mean-reversion, but it's small and regime-dependent.**

### Best Strategy Found

**S2: Volatility-Conditioned Mean-Reversion (LONG-only)**

| Metric | Taker (20bps RT) | Maker (8bps RT) |
|--------|-------------------|-------------------|
| Total trades | 424 (282 ex-Oct) | 424 (282 ex-Oct) |
| Symbols traded | 113 | 113 |
| Win rate | 57.8% (55.0% ex-Oct) | 59.7% (57.8% ex-Oct) |
| Avg gross return | +769 bps | +769 bps |
| Avg net return | +749 bps (+28 bps ex-Oct) | +761 bps (+40 bps ex-Oct) |
| Profit factor | 3.19 (1.25 ex-Oct) | 3.26 (1.40 ex-Oct) |
| Positive months | 6/9 (5/8 ex-Oct) | 6/9 (6/8 ex-Oct) |
| Profitable symbols | 64/99 ex-Oct | — |

**Config**: `sig_base < -2.5` AND `rvol_ratio > 2.0`, hold up to 24 bars (2h), min hold 3 bars (15m).

---

## Research Pipeline

### Data

- **116 symbols** common to both Bybit and Binance
- **Date range**: 2025-07-01 to 2026-03-03 (~245 trading days)
- **Resolution**: 1-minute bars resampled to 5-minute
- **Data types**: OHLCV, mark price, premium index, funding rate, OI (5min), LS ratio (5min)
- **Total datapoints**: ~70K rows per symbol × 116 = ~8.1M rows

### Feature Engineering (53 features)

**Tier 1 — Price-based cross-exchange signals:**
- Price divergence: `(BB_close - BN_close) / midpoint` in bps
- Smoothed divergence: MA-6, MA-12, MA-36, MA-72
- Z-scored divergence: rolling 72-bar and 288-bar z-scores
- Return lead/lag: cumulative BB excess return over BN

**Tier 2 — Premium-based signals:**
- Premium spread: `BB_premium - BN_premium` in bps
- Premium z-scores (72-bar, 288-bar lookbacks)
- Premium change divergence

**Tier 3 — Volume/flow signals:**
- Volume ratio: `BB_turnover / BN_turnover` (log-scaled, z-scored)
- Binance taker buy imbalance (aggressive buyer fraction)
- Volume spike detection per exchange

**Tier 4 — Positioning signals:**
- OI divergence (rate of change difference between exchanges)
- LS ratio divergence
- Funding rate features

**Tier 5 — Composite signals:**
- Flow momentum: `price_div × vol_ratio_z`
- OI-premium composite

### Signal Discovery (116 symbols)

Top signals by correlation with 30-minute forward return (averaged across all 116 symbols):

| Signal | r(30m) | Edge (decile) | Consistency |
|--------|--------|---------------|-------------|
| `price_div_bps` | -0.045 | -6.1 bps | 22% positive |
| `price_div_ma6` | -0.035 | -5.8 bps | 19% positive |
| `oi_div` | -0.035 | -3.0 bps | 18% positive |
| `premium_spread_ma12` | -0.025 | -4.0 bps | 22% positive |
| `flow_momentum` | -0.020 | -0.7 bps | 37% positive |

**Key finding**: All top signals show NEGATIVE correlation — when BB price is "too high" vs BN, future returns are negative. This is **mean-reversion**.

---

## Strategy Iterations

### V1: Bidirectional Mean-Reversion

Composite z-score signal from weighted average of:
- Price divergence z-scores (weight 3.0, 2.0)
- Premium spread z-scores (weight 2.0, 1.5)
- Smoothed price divergence (weight 1.5)
- OI divergence (weight 1.0)
- Volume ratio z-score (weight 0.5)
- Return difference accumulation (weight 1.0)

**Result**: Works for LONG side only (composite < 0 → price too low → buy).
- LONG: 374 trades, WR=54.3%, avg_net=+755 bps
- SHORT: 756 trades, WR=45.8%, avg_net=-60 bps

**Walk-forward (50/50 split):**

| Threshold | Half | Trades | WR(tk) | PF(tk) |
|-----------|------|--------|--------|--------|
| 4.0 | IS | 395 | 44.8% | 2.24 |
| 4.0 | OOS | 347 | 56.2% | 1.17 |
| 3.5 | IS | 587 | 44.5% | 1.96 |
| 3.5 | OOS | 543 | 53.0% | 0.95 |

OOS PF=1.17 at thr=4.0 — edge exists but small. SHORT side has no edge → **LONG-only**.

### V2: LONG-Only Optimized

Best config: `sig_base_thr4.0_h24_ts0` (threshold 4.0, max 24 bars hold)
- 236 trades, 105 symbols, avg_net=+1503 bps (taker)
- OOS PF=13.8, OOS WR=73.7%

**CRITICAL PROBLEM**: October 2025 dominates (136/236 trades = 58%, contributes 101% of total PnL).

Monthly breakdown:
| Month | Trades | WR | Avg Net | Total |
|-------|--------|----|---------|-------|
| 2025-10 | 136 | 69.9% | +2645 | +359,677 |
| 2025-11 | 25 | 72.0% | +77 | +1,923 |
| All other months combined | 75 | ~42% | ~-60 | ~-4,809 |

### V3: Multi-Strategy with Ex-October Validation

Tested 5 strategy families:

| Family | Best ex-Oct PF | Description |
|--------|----------------|-------------|
| **S2: Vol-conditioned** | **1.25 (taker), 1.40 (maker)** | Mean-rev only when vol expanding |
| S1: Plain mean-rev | 1.13 (taker) | Original signal, higher thresholds |
| S5: Div breakout | 0.96 | Price divergence z-score breakout |
| S3: Momentum | 0.82 | Follow the leader exchange |
| S4: Taker flow | 0.63 | Binance taker buy/sell imbalance |

**Winner: S2_volcond_thr2.5_vol2.0_h24**

Entry: composite signal < -2.5 AND realized vol ratio (1h/6h) > 2.0
Exit: signal crosses 0 OR 2h max hold
Direction: LONG only

Ex-October performance (282 trades, taker fees):
- WR: 55.0%
- Avg net: +28 bps per trade
- PF: 1.25
- 5/8 months profitable (excl. Oct)

With maker fees (282 trades):
- WR: 57.8%
- Avg net: +40 bps per trade
- PF: 1.40
- 6/8 months profitable

---

## Honest Assessment

### What works
1. **Cross-exchange mean-reversion is real** — when one exchange's price diverges from the other, it reverts. This is structural (arbitrage pressure).
2. **LONG side has edge, SHORT doesn't** — buying when BB is "too cheap" vs BN works. Shorting when BB is "too expensive" doesn't. This suggests the dislocation is asymmetric (BB tends to lag in selloffs, then catches up).
3. **Volatility conditioning improves robustness** — only trading when vol is expanding (rvol_ratio > 2.0) filters out noise and catches real dislocations.
4. **The edge is strongest during volatile periods** — October 2025 (likely a crash/correction month) produced massive dislocations and massive profits.

### What doesn't work
1. **SHORT side** — no edge after fees. The short mean-reversion signal is consumed by fees and noise.
2. **Momentum (S3)** — following the leader exchange has no edge on 5m bars.
3. **Taker flow (S4)** — Binance taker buy/sell imbalance is NOT predictive of next-bar direction. PF < 0.65 across all configs.
4. **Low-threshold signals** — at thr < 2.0, the signal-to-noise ratio is too low to beat fees.

### Risks and caveats
1. **October dependency** — ~60% of high-threshold trades cluster in one volatile month. The strategy needs volatile markets to generate both trades and edge.
2. **Trade frequency** — only ~1-2 trades per day on average (ex-October), many days with 0 trades.
3. **Execution assumptions** — we assume midpoint execution. Real slippage on altcoins could eat 5-10 bps, especially during the volatile periods when the signal fires.
4. **Small ex-October edge** — +28 bps per trade (taker) is thin. A few bad fills or data issues could erase it.
5. **No out-of-sample period** — we tested on all 8 months. True OOS would require live forward testing.

### Estimated Daily P&L

Assuming $10K notional per trade:

| Scenario | Trades/day | Net bps/trade | Daily P&L |
|----------|-----------|---------------|-----------|
| Normal month (taker) | 1.2 | +28 | ~$3.36 |
| Normal month (maker) | 1.2 | +40 | ~$4.80 |
| Volatile month (taker) | 4.7 | +2182 | ~$1,025 |
| Volatile month (maker) | 4.7 | +2194 | ~$1,031 |

The strategy is essentially a **vol-event harvester**: small steady income in normal times, massive payoffs during market dislocations.

---

## Files

| File | Description |
|------|-------------|
| `PLAN.md` | Research plan and thesis |
| `load_data.py` | Unified cross-exchange data loader (116 symbols, 5m bars) |
| `features.py` | 53 cross-exchange features |
| `discover.py` | Signal discovery across all symbols |
| `backtest.py` | V1 bidirectional backtest with parameter sweep |
| `run_backtest.py` | Efficient sweep runner (loads data once) |
| `strategy_v2_fast.py` | V2 LONG-only sweep with OOS validation |
| `strategy_v3.py` | V3 multi-strategy with ex-October validation |
| `analyze_results.py` | Deep analysis of best config |
| `signal_discovery_results.csv` | Signal correlation table (116 symbols) |
| `backtest_sweep_results.csv` | V1 parameter sweep (50 configs) |
| `v2_sweep_results.csv` | V2 LONG-only sweep results |
| `v2_best_trades.csv` | V2 best config trades (236 trades) |
| `v3_sweep_results.csv` | V3 multi-strategy sweep results |
| `v3_best_trades.csv` | V3 best config trades (424 trades) |

## Next Steps

1. **Live forward test** — paper-trade the S2_volcond strategy for 1 month to validate
2. **Maker order execution** — if using limit orders (4bps/leg), PF jumps to 1.40. Worth building limit order logic.
3. **Multi-timeframe** — test on 15m and 1h bars for longer-horizon trades during vol events
4. **Expand to more exchanges** — add OKX as a third exchange for more dislocation signals
5. **Regime detection** — build a real-time vol regime classifier to dynamically adjust thresholds
6. **Position sizing** — scale position size proportional to signal strength (sig=7+ has 88% WR)
