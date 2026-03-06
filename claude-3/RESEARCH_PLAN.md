# Research Plan: Cross-Sectional Price Prediction

**Date:** 2026-03-06

---

## Motivation

With 131+ symbols × 2 years of 1m klines, funding rates, OI, L/S ratios, premium index, and tick data across Bybit and Binance, we have a large enough universe to test **cross-sectional** signals — ranking which coins will outperform others over the next N hours.

This is fundamentally different from single-coin direction prediction (confirmed dead in v1–v42). Cross-sectional prediction exploits **relative performance**, cancelling out common market moves.

---

## Fee Hurdles

| Execution | Per side | Round-trip | Edge needed |
|-----------|----------|------------|-------------|
| Taker | 10 bps | 20 bps | Gross > 20 bps/trade |
| Maker | 4 bps | 8 bps | Gross > 8 bps/trade |
| Mixed | 7 bps avg | 14 bps | Gross > 14 bps/trade |

**Note on turnover:** If only 60% of positions change per rebal, effective cost = 0.60 × round-trip. Low-turnover signals dramatically reduce the hurdle.

---

## Six Signal Ideas

### Signal A — Cross-Sectional Momentum
- **Data:** 1m klines × 131 coins (2 years Bybit)
- **Signal:** Coin's return over past W hours (W = 1h, 2h, 4h, 8h, 24h, 48h)
- **Direction:** Long top decile (recent winners), short bottom decile
- **Theory:** Cross-sectional crypto momentum is one of the most documented effects in literature
- **Fee target:** Maker 8 bps at 24h+ holding

### Signal B — Funding Rate Carry
- **Data:** `_funding_rate.csv` × 152 coins (2 years Bybit)
- **Signal:** Most recent funding rate (forward-filled from settlement)
- **Long:** 10 coins with most negative funding (shorts pay you)
- **Short:** 10 coins with most positive funding (you collect)
- **Theory:** You literally receive the funding regardless of direction. Carry IS the profit.
- **Fee target:** At normal funding (3 bps/settlement) × 3 = 9 bps > maker 8 bps ✓

### Signal C — BTC Lead-Lag (Beta Drift)
- **Data:** 1m klines all coins, rolling 30d beta to BTC
- **Signal:** Expected move (beta × BTC return) minus actual coin move
- **Theory:** BTC often leads alts; lagging coins catch up
- **Fee target:** Very short horizon — maker only, hardest to clear

### Signal D — OI–Price Divergence
- **Data:** `_open_interest_5min.csv` × 152 coins
- **Signal:** sign(price_ret_1h) × sign(OI_change_1h)
  - +1: aligned (momentum regime) / -1: diverging (mean reversion)
- **Fee target:** Maker 8 bps at 4h horizon

### Signal E — Spot–Futures Basis (Premium Index)
- **Data:** `_premium_index_kline_1m.csv` × 152 coins
- **Signal:** Premium z-score vs trailing 30d mean (deviation from normal basis)
- **Theory:** Futures rich vs spot = overpriced → short. Originally expected mean reversion.
- **Fee target:** Only trade when |basis| > 15 bps

### Signal F — L/S Ratio Cross-Sectional
- **Data:** `_long_short_ratio_5min.csv` × 152 Bybit; `_metrics.csv` × 140 Binance
- **Signal:** buyRatio z-score vs coin's own 30d rolling mean
- **Theory:** From v24, Binance top-trader L/S ratio had IC=0.20 on single coin. Cross-sectional may amplify.
- **Fee target:** 4h rebal, maker

---

## Phase Plan

```
Phase 1 — Signal IC Analysis (pure stats)
  For each signal: IC vs forward returns at 1h/4h/8h/24h horizons
  131 coins × 2yr data = 10,000+ cross-sections per bar
  → Rank signals, identify winners

Phase 2 — Portfolio Backtests
  Long top 10 / short bottom 10 per signal/combo
  Rebalance every 8h (funding settlement alignment)
  Walk-forward: 6mo train / 3mo OOS, rolling
  → Net Sharpe, max DD, turnover

Phase 3 — Execution Realism
  Rebalancing frequency: 8h vs 16h vs 24h
  Universe size: N=5/10/15/20/30 per leg
  Vol-scaled vs equal-weight sizing
  Maker fill rate sensitivity
  Capacity / market impact analysis
```

---

## Data Quality Issue Discovered

**2025-01-01 corruption:** Close prices for many symbols were zeroed out or near-zero (~0.001) in the raw kline CSV files. This caused `pct_change()` to produce `inf` and `-1.0` forward return values (~22% of rows had |fwd_8h| > 100%).

**Fix in `phase1_build_signals.py`:**
```python
price_median = df["close"].median()
bad_price = (df["close"] <= 0) | (df["close"] < price_median * 0.01)
df.loc[bad_price, "close"] = np.nan
df["close"] = df["close"].interpolate(method="time", limit=12)
```

All 131 signal parquets were rebuilt with this fix. Phase 1 IC results differ significantly between dirty and clean data — **always use clean data results**.
