# Strategy Specification: Cross-Sectional Funding + Momentum

**Date:** 2026-03-06
**Status:** Research complete — ready for live implementation design

---

## Executive Summary

A cross-sectional long/short strategy on perpetual futures, exploiting the combination of funding rate carry and short-term price momentum. The strategy is market-neutral by construction (equal dollar long and short), rebalances every 8 hours aligned with Bybit funding settlements, and requires maker-only execution.

**Full-period backtest (clean data, 2024–2026):**
- Net Sharpe: **3.27** (all 4 OOS windows positive, avg 2.60)
- Net return per 8h bar: **27.3 bps** (well above 8 bps RT maker hurdle)
- Annualized return: **~300%** (gross; actual depends on leverage/sizing)
- Maximum drawdown: **-46%** (on notional position value)

---

## Signal Construction

### Step 1: Funding Rate Signal

For each symbol `s` at time `t`:
```
funding_s_t = most recent funding rate (forward-filled from 8h settlements)
```
Source: Bybit `_funding_rate.csv`, 8h settlement frequency.

**Direction:** Long high-funding coins (longs pay you), short low/negative-funding coins.

No normalization required — raw funding rate is used directly for ranking.

### Step 2: Momentum Signal

For each symbol `s` at time `t`:
```
mom24h_s_t = (close_s_t / close_s_{t-24h}) - 1
```
Source: Bybit `_kline_1m.csv`, use 24h lookback on close prices.

**Direction:** Long positive momentum (recent winners), short negative momentum (recent losers).

### Step 3: Composite Signal

```
z_funding = zscore(funding, across universe at time t)
z_mom24h  = zscore(mom24h, across universe at time t)
composite = 0.5 * z_funding + 0.5 * z_mom24h
```

Equal weights. Do NOT use vol-scaling (tested: degrades Sharpe from 3.27 to 2.54).

---

## Portfolio Construction

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Universe | Top 130 symbols by 400+ day history | Liquid, established coins only |
| Rebal frequency | **Every 8 hours** | Funding settlement alignment (00:00, 08:00, 16:00 UTC) |
| Positions per leg | **10 long + 10 short** | Sweet spot (N=5 too volatile, N=15+ dilutes alpha) |
| Long selection | Top 10 by composite rank | Highest funding + momentum |
| Short selection | Bottom 10 by composite rank | Lowest funding + momentum |
| Sizing | **Equal-weight** | 1/20 of AUM per position |
| Dollar neutrality | Long notional = Short notional | Market-neutral design |

---

## Execution

| Parameter | Value |
|-----------|-------|
| Order type | **Limit orders (maker)** |
| Fee target | 4 bps/side |
| Trigger | 3h before funding settlement: compute signals, place limit orders |
| Deadline | At settlement time: cancel unfilled, fill remainder with taker |
| Max taker fraction | 50% (tested: still profitable at any mix) |
| Slippage tolerance | Limit price = mid ± 1 bps |

**Rebalancing flow:**
1. T-1h: Load latest funding rates and 1m close prices
2. T-30m: Compute composite scores, determine target portfolio
3. T-30m to T: Place limit orders at mid, adjust if not filled
4. T: Cancel remaining, fill stragglers at market (taker)
5. Collect funding payment at T (automatically credited)

---

## Risk Management

| Risk | Limit | Action |
|------|-------|--------|
| Single position | Max 15% of AUM per position | Reduce position if breached |
| Net market exposure | ±5% of AUM | Rebalance immediately |
| Sector concentration | Max 4 positions in same L1/L2 cluster | Excluded from selection |
| Max drawdown | -30% from peak (daily NAV) | Reduce position size by 50% |
| Daily loss limit | -5% of AUM | Pause until next session |
| Liquidation buffer | Maintain 20% margin headroom | Reduce gross exposure if needed |

---

## Expected Performance (Steady State)

*Based on walk-forward OOS windows Jan 2025 – Jan 2026:*

| Metric | OOS Average | Range |
|--------|------------|-------|
| Net bps/bar | ~22 bps | 5–41 bps |
| Ann. Sharpe | ~2.60 | 0.90–3.73 |
| Ann. Return (1× notional) | ~150–200% | varies |
| Max DD per quarter | ~25–35% | ~20–35% |
| Turnover per rebal | ~52% | position changes |
| Monthly trades (N=10) | ~20 × 3 rebal × 2 legs = 120 trades | — |

---

## Fee Math

At 8h rebalancing with 52% turnover per bar and maker execution:

```
Trades per bar     = 20 positions × 52% turnover = 10.4 round-trips
Fee per bar        = 10.4 × 8 bps RT = 83 bps notional
Gross alpha        = 31.5 bps per position × 20 = 630 bps notional
Net per bar        = 630 - 83 = 547 bps notional
Net per position   = 547 / 20 = 27.3 bps/position/bar  ✓
```

---

## Signals NOT Used

| Signal | Reason |
|--------|--------|
| prem_z (premium z-score) | IC ≈ 0 on clean data; dirty-data IC was spurious |
| mom_8h | Mean-reverting at 8h horizon (IC = -0.172) |
| mom_48h | Similar to mom_24h but adds no benefit; marginal |
| ls_z (L/S ratio) | IC ≈ 0 on clean data |
| oi_div (OI divergence) | Weak negative IC; not reliably actionable |
| BTC lead-lag | Not tested in final form; conceptually sound but execution harder |

---

## Known Risks and Limitations

**1. Regime sensitivity:** Weakest OOS quarter was Apr–Jul 2025 (Sharpe 0.90) during a broad market correction. The strategy is long/short but not perfectly delta-neutral in practice — funding tends to be positive for winners, which are correlated longs.

**2. Funding rate compression:** If markets enter a sustained negative-funding regime (persistent bearish sentiment), the funding carry signal flips sign. The composite would need monitoring for regime change.

**3. Data quality:** The 2025-01-01 corruption affected raw CSV data. Production data pipeline must validate close prices (detect zeros, extreme outliers) before computing signals.

**4. Two-year backtest:** While we have 2 years of Bybit data and 1 year of Binance, the strategy has only been OOS-validated over 4 quarterly windows. More OOS history would be ideal before scaling.

**5. Concentration in bull markets:** The strategy tends to be long high-momentum, high-funding coins. During alt-season rallies, this generates exceptional returns (+453% annualized in Oct–Jan 2026 window); during drawdowns, losses can be sharp.

---

## Implementation Checklist

- [ ] Live data feed: Bybit funding rate (8h) + 1m klines
- [ ] Signal computation module (Python, <5s latency)
- [ ] Order management: limit order placement + cancellation logic
- [ ] Position reconciliation: target vs actual at each rebal
- [ ] Risk monitor: continuous P&L, exposure, liquidation margin
- [ ] Data quality checks: zero price detection, outlier filtering
- [ ] Dry-run backtest against live data (paper trading) before deployment
- [ ] Exchange connectivity: Bybit REST + WebSocket API
