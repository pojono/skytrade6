# Implementation Guide: Cross-Sectional Funding + Momentum Strategy

**Version:** 1.0
**Research basis:** Phases 1–25 (Jan 2025 – Mar 2026)
**Exchange:** Bybit Perpetuals (USDT-margined)
**Status:** Ready for live implementation

---

## Table of Contents

1. [Strategy Overview](#1-strategy-overview)
2. [Data Requirements](#2-data-requirements)
3. [Universe Construction](#3-universe-construction)
4. [Signal Computation](#4-signal-computation)
5. [Portfolio Construction](#5-portfolio-construction)
6. [Entry and Exit Logic](#6-entry-and-exit-logic)
7. [Execution Model](#7-execution-model)
8. [Risk Management](#8-risk-management)
9. [Walk-Forward Optimization](#9-walk-forward-optimization)
10. [Live Monitoring](#10-live-monitoring)
11. [Expected Performance](#11-expected-performance)
12. [What Not to Do](#12-what-not-to-do)

---

## 1. Strategy Overview

### What it is

A systematic long/short market-neutral strategy on Bybit perpetual futures. At each Bybit funding settlement (every 8 hours), we rank all coins in the universe by a composite signal combining real-time predicted funding rate and 24h price momentum. We go long the top 10 coins and short the bottom 10 coins with equal-weight positions.

The core economic mechanism:
- **Predicted funding carry**: Bybit longs pay shorts a funding fee at each settlement. By predicting which coins will have the highest funding rate at the next settlement, we position to receive that payment.
- **Momentum reinforcement**: Coins with high funding tend to have driven by retail FOMO — momentum adds conviction that the trend continues to the next settlement.

The strategy is not a directional bet. When BTC drops 10%, our longs and shorts both drop, netting roughly zero. The P&L comes from the *spread* between top and bottom coins, not market direction.

### Key numbers (backtested Jan 2025 – Mar 2026, 2× leverage)

| Metric | Value |
|--------|-------|
| Sharpe ratio | **3.96** |
| Maximum drawdown | **-40.1%** |
| Annualised return | **1,983%** on notional |
| $10k starting capital | → **$351,824** in 15 months |
| Win rate | **49.9%** of 8h bars |
| Positive months | **11/15** (73%) |
| Walk-forward windows | **4/4 positive** |

### When it works and when it doesn't

Works in:
- High-funding regimes: retail speculative activity drives funding rates wide across the meme/AI coin universe (Sep–Dec 2025, Jan 2026 were exceptional).
- Trend months with dispersion: some coins pumping hard while others bleed.

Struggles in:
- Flat, low-volume markets where all coins have near-zero funding (May, Jul 2025 were -9%).
- Early in the meme coin cohort lifecycle (Jan–Feb 2025, newly listed coins with unstable price discovery): worst month was -18.6%.
- Bear markets with uniformly negative funding across the universe.

**Patience is required.** The strategy has burst-mode returns concentrated in a few months per year. An -8% month does not mean the strategy is broken.

---

## 2. Data Requirements

### 2.1 Data types needed

You need three data streams per coin, at 1-minute granularity:

| Data Stream | Bybit endpoint | Description |
|-------------|---------------|-------------|
| **Kline (price)** | `GET /v5/market/kline` | OHLCV candles. Use 1m for computation, resample to 1h. |
| **Premium index kline** | `GET /v5/market/premium-index-price-kline` | The premium of perp price over mark price, 1m bars. This is the core input for predicted_funding. |
| **Funding rate history** | `GET /v5/market/funding/history` | Settled funding rates (00:00, 08:00, 16:00 UTC). Used for historical backtesting only. |

### 2.2 Minimum history per coin

For a coin to participate in signal computation at time T, it must have **at least 14 days** of premium index kline history prior to T. This prevents newly-listed coins with unstable price discovery from contaminating the cross-section.

For walk-forward optimization (see Section 9), you need **at least 6 months** of clean history.

### 2.3 Data cadence

| Process | Frequency | Latency requirement |
|---------|-----------|-------------------|
| Premium index kline fetch | Every 1h (aligned to funding windows) | < 60s |
| Price kline fetch | Every 1h | < 60s |
| Signal computation | Every 1h (but act only at 8h settlement) | < 5s |
| Order placement | At T-30m before settlement | < 10s |

### 2.4 Historical data for initial setup

Download the following for all universe coins before going live:
- 1m premium index klines: minimum 6 months, ideally 12+ months
- 1m price klines: same period
- Funding rate history: same period (for backtest validation only)

Bybit historical data can be downloaded from:
`GET /v5/market/premium-index-price-kline` with `limit=1000` and pagination by `start` / `end` timestamps.

### 2.5 Storage schema

Store per-coin data in Parquet files or a time-series database (TimescaleDB, InfluxDB). Minimum columns:

**price_kline table:**
```
symbol      VARCHAR
ts          TIMESTAMP (UTC, 1m resolution)
open        FLOAT
high        FLOAT
low         FLOAT
close       FLOAT
volume      FLOAT
```

**premium_index_kline table:**
```
symbol      VARCHAR
ts          TIMESTAMP (UTC, 1m resolution)
open        FLOAT   -- premium index (perp - mark) / mark, in decimal
close       FLOAT
```

**funding_rate table (historical reference):**
```
symbol          VARCHAR
settle_ts       TIMESTAMP (UTC, exactly 00:00/08:00/16:00)
funding_rate    FLOAT   -- in decimal, e.g. 0.001 = 0.1%
```

---

## 3. Universe Construction

### 3.1 Starting universe

Begin with **all Bybit USDT-margined perpetuals** that have been listed for at least 6 months at your launch date. As of March 2026 this is approximately 150+ symbols.

### 3.2 Required exclusions (structural)

Remove the following 18 **Majors** permanently. This is not a data-driven decision — it is structural. Major coins have deep liquidity, tight spreads, and their price action is dominated by broad market flows, not the retail funding microstructure this strategy exploits. Phase 6 confirmed their combined contribution is -3,692 bps (Sharpe -1.86).

```
BTCUSDT  ETHUSDT  BNBUSDT  SOLUSDT  XRPUSDT  ADAUSDT
DOGEUSDT AVAXUSDT DOTUSDT  LTCUSDT  BCHUSDT  TRXUSDT
XLMUSDT  ETCUSDT  HBARUSDT ATOMUSDT ALGOUSDT EGLDUSDT
```

Add new coins to this exclusion list as they graduate to "Major" status (top-10 by market cap globally, heavily traded on all major exchanges).

### 3.3 Minimum listing age guard

At each 8h decision point, **exclude any coin with fewer than 14 calendar days** of premium index kline history. This is a soft guard for the live ramp-up period. It can be removed after the first live quarter once signal behaviour is validated.

Implementation:
```python
first_ts = premium_index_kline[symbol].index.min()
age_days = (current_ts - first_ts).days
if age_days < 14:
    exclude(symbol)
```

### 3.4 What NOT to do with the universe

Do not dynamically exclude coins based on recent performance. Phase 9 proved this definitively: a coin that contributes negatively in months 1–3 often recovers in months 4–6. Rolling exclusion based on trailing P&L attribution does not survive walk-forward and will cost you alpha in recovery months.

The only valid exclusion criteria are:
1. Structural (Major coin) — permanent
2. Observable at trade time (listing age, minimum data coverage) — time-varying but not look-ahead

### 3.5 Live universe size

After applying exclusions, expect approximately **100–120 active coins** at any given time as Bybit continues listing new perpetuals. The strategy is designed for this range. The cross-sectional z-score requires at least 30 valid coins per bar to be meaningful — if universe shrinks below 30, pause the strategy and investigate.

---

## 4. Signal Computation

All signal computation happens at each decision point: T = 00:00, 08:00, or 16:00 UTC (1 hour before the corresponding settlement).

### 4.1 Signal 1: Predicted Funding Rate

**This is the primary signal.** It directly predicts the funding rate that will be settled in approximately 60 minutes.

#### Background

Bybit's perpetual funding formula (public documentation):
```
funding_rate = clamp(TWAP(premium_index, [window_start, settlement]) + 0.0001, -0.0075, +0.0075)
```
Where:
- `premium_index` = (perp_mid_price − mark_price) / mark_price, sampled every minute
- `window_start` = start of the current 8h funding window (00:00, 08:00, or 16:00 UTC)
- `0.0001` = daily interest rate (3 funding periods × 0.0001/3 ≈ 0.000033 per period, but Bybit adds 0.0001 flat)
- Clamp: ±0.75% per settlement

We replicate this formula using the 1m premium index kline data:

#### Implementation

```python
INTEREST_RATE = 0.0001      # Bybit fixed interest component
FUNDING_CAP   = 0.0075      # ±0.75% per settlement

def compute_predicted_funding(premium_1m_series: pd.Series, current_ts: pd.Timestamp) -> float:
    """
    premium_1m_series: pd.Series indexed by UTC timestamp, values = premium index
    current_ts:        decision time (e.g., 07:00 UTC for the 08:00 settlement)
    Returns:           predicted funding rate for the next settlement
    """
    # Start of the current 8h window
    window_start = current_ts.floor("8h")

    # Select all 1m bars within [window_start, current_ts)
    mask = (premium_1m_series.index >= window_start) & (premium_1m_series.index < current_ts)
    window_data = premium_1m_series[mask]

    if len(window_data) == 0:
        return np.nan

    # Running TWAP = simple mean of all bars in the window
    twap = window_data.mean()

    # Apply Bybit formula
    predicted = np.clip(twap + INTEREST_RATE, -FUNDING_CAP, FUNDING_CAP)
    return predicted
```

Apply this function to every coin in the universe at each decision point to produce a panel of predicted funding rates.

#### Why this beats the lagged settled rate

The lagged settled rate (the last value from the funding history endpoint) updates only 3× per day. On volatile meme coins:
- Lagged rate correlation with next settlement: **0.01** (random)
- Predicted rate correlation with next settlement: **0.92** (near-certain at T-1h)

At T-1h before settlement, the TWAP window is 87.5% complete (7 of 8 hours sampled). The remaining 60 minutes can move it slightly, but the prediction is already highly accurate.

#### Direction

Long coins with high predicted funding (longs are paying, shorts receive).
Short coins with low or negative predicted funding (shorts are paying, longs receive — reversed carry).

### 4.2 Signal 2: 24h Momentum

```python
def compute_mom24h(close_1h_series: pd.Series, current_ts: pd.Timestamp) -> float:
    """
    close_1h_series: pd.Series of hourly close prices, indexed by UTC timestamp
    current_ts:      decision time
    Returns:         24h return
    """
    t_now  = close_1h_series.asof(current_ts)
    t_24h  = close_1h_series.asof(current_ts - pd.Timedelta(hours=24))

    if pd.isna(t_now) or pd.isna(t_24h) or t_24h == 0:
        return np.nan

    return (t_now / t_24h) - 1.0
```

#### Why momentum is included when its standalone IC is negative

On its own, 24h momentum has ICIR = -0.023 (slightly mean-reverting at the 8h horizon). But it is super-additive with funding: within the top-funding coins, the winners (high momentum) continue outperforming more than the mean-reverters. Momentum conditions the funding signal on self-reinforcing trends.

Phase 2 confirmed: funding alone Sharpe 2.81, funding + momentum Sharpe 3.27.

### 4.3 Cross-Sectional Z-Score

Both signals are converted to cross-sectional z-scores at each bar before combining.

```python
def cs_zscore(panel_row: pd.Series, min_valid: int = 15) -> pd.Series:
    """
    panel_row: pd.Series, one row of the signal panel (all coins at time T)
    min_valid: minimum number of non-NaN values to compute z-score
    Returns:   z-scored series, clipped to [-3, +3]
    """
    valid = panel_row.dropna()
    if len(valid) < min_valid:
        return pd.Series(np.nan, index=panel_row.index)

    mu  = valid.mean()
    sig = valid.std()

    if sig == 0:
        return pd.Series(0.0, index=panel_row.index)

    z = (panel_row - mu) / sig
    return z.clip(-3, 3)
```

Apply to the full panel (all coins × all time bars) row by row (i.e., cross-sectionally, not time-series).

### 4.4 Composite Score

```python
def compute_composite(z_funding: pd.Series, z_mom: pd.Series) -> pd.Series:
    """
    z_funding: cross-sectional z-score of predicted_funding at time T
    z_mom:     cross-sectional z-score of mom_24h at time T
    Returns:   composite score for ranking
    """
    # 2:1 weighting — funding dominates, momentum reinforces
    composite = (2.0 * z_funding + 1.0 * z_mom) / 3.0

    # If funding is unavailable for a coin, exclude it entirely
    composite[z_funding.isna()] = np.nan

    # If momentum is unavailable, fall back to funding alone
    # (z_mom NaN → treat as 0, which is neutral)
    z_mom_filled = z_mom.fillna(0.0)
    composite = (2.0 * z_funding + z_mom_filled) / 3.0
    composite[z_funding.isna()] = np.nan

    return composite
```

**Why 2:1 and not equal weight?** Phase 22 tested all combinations. `2×funding + 1×mom` (Sharpe 3.94) outperformed equal weight and every other tested ratio. The intuition: funding is the primary economic mechanism; momentum is a filter that selects the strongest carry coins.

**Why not include funding_trend?** Phase 22 confirmed that adding `funding_trend` (24h change in funding) to `predicted_funding` reduces Sharpe (3.94 → 2.96). The predicted_funding already implicitly encodes trend: if the TWAP is rising within the window, `predicted_funding` increases. The trend signal is redundant and adds noise.

---

## 5. Portfolio Construction

### 5.1 Ranking and selection

At each decision point T (one hour before settlement):

```python
composite = compute_composite(z_funding, z_mom)  # pd.Series, indexed by symbol

# Drop NaN coins (insufficient data)
valid = composite.dropna()

# Rank: highest composite = best long, lowest = best short
ranked = valid.sort_values(ascending=False)

N = 10  # positions per leg (10 long + 10 short)

longs  = ranked.index[:N].tolist()   # top N by composite
shorts = ranked.index[-N:].tolist()  # bottom N by composite
```

### 5.2 Why N=10?

Phase 24 swept N from 5 to 20. Results:

| N | Sharpe |
|---|--------|
| 5  | 3.18 |
| **10** | **3.96** |
| 15 | 3.71 |
| 20 | 3.52 |

N=5 has better signal concentration but excessive single-coin risk (MaxDD -52%). N=15+ dilutes the signal. N=10 is the optimal tradeoff.

### 5.3 Position sizing

Equal-weight across all positions. No volatility scaling, no Kelly sizing.

```python
gross_exposure = account_equity * leverage  # e.g., $10k × 2 = $20k gross
position_size  = gross_exposure / (2 * N)   # $20k / 20 = $1k per position

# Each long position: +$1k notional
# Each short position: -$1k notional
# Net exposure: $0 (market neutral)
```

**Why not volatility scaling?** Phase 12 tested volatility targeting. Result: it reduces Sharpe because the alpha months are the high-volatility months. Vol-scaling deliberately reduces exposure exactly when the strategy earns most. Equal weight keeps the alpha intact.

### 5.4 Leverage recommendation

| Leverage | MaxDD | Risk profile |
|----------|-------|--------------|
| 1× | ~-22% | Conservative, lower drawdown |
| **2×** | **-40%** | **Recommended: best risk/reward** |
| 3× | ~-60% | Aggressive, only for experienced operators |
| 5×+ | -80%+ | Not recommended |

Sharpe is invariant to leverage (both returns and fees scale proportionally). Choose leverage based purely on drawdown tolerance.

**Hard rule: never exceed 3× leverage.**

### 5.5 Rebalance delta

At each 8h rebalance, you do not need to rebuild the entire portfolio from scratch. Compute the diff:

```python
target_longs  = set(longs)
target_shorts = set(shorts)
current_longs = set(current_positions["long"])
current_shorts = set(current_positions["short"])

# Exits
close_longs  = current_longs  - target_longs
close_shorts = current_shorts - target_shorts

# Entries
open_longs   = target_longs  - current_longs
open_shorts  = target_shorts - current_shorts

# Unchanged (stay open, no action needed)
hold_longs   = current_longs  & target_longs
hold_shorts  = current_shorts & target_shorts
```

Typical turnover: ~52% of positions change per bar (measured from backtest). With N=10, expect ~5 new longs, ~5 new shorts, and ~5 closes on each leg per rebalance.

---

## 6. Entry and Exit Logic

### 6.1 Decision timeline (every 8 hours)

Bybit funding settles at exactly **00:00, 08:00, 16:00 UTC**. All times below are relative to the settlement time S.

```
S - 70m  Pull fresh 1m premium index klines for all symbols
S - 65m  Pull fresh 1m price klines for all symbols
S - 60m  Compute predicted_funding and mom_24h for all coins
S - 60m  Apply universe filters (age, non-NaN checks)
S - 60m  Compute composite scores, rank, select top/bottom N=10
S - 60m  Compute target portfolio diff (entries, exits, holds)
S - 30m  Place limit orders for all entries and exits
S - 5m   Check fill status
S + 0    Settlement occurs. Funding credited/debited automatically.
S + 5m   Cancel any unfilled limit orders, execute remainder at market (taker)
S + 10m  Reconcile actual positions vs target. Log any discrepancies.
```

### 6.2 Entry logic

Entry = open a new long or short position for a coin just added to the target set.

```
Order type:      Limit (post-only, maker)
Price:           Best bid (for longs) / Best ask (for shorts) at time of placement
                 Alternative: mid-price ± 0.5 bps
Validity:        Good-till-settlement (cancel at S+5m if unfilled)
Fallback:        Market order at S+5m for any unfilled portion
Max taker ratio: 50% of positions can be taker-filled (Sharpe remains 2.90 at 50/50)
```

**Do not place entries more than 30 minutes before settlement.** The cross-section can shift, and you want to be positioned based on the most current predicted funding.

### 6.3 Exit logic

Exit = close a position for a coin that dropped out of the target set.

```
Order type:      Limit (post-only, maker)
Price:           Best ask (to close longs) / Best bid (to close shorts)
Validity:        Good-till-settlement (cancel at S+5m if unfilled)
Fallback:        Market order at S+5m (prioritise exit over entry if capital constrained)
```

**Always exit before entering new positions.** This keeps gross exposure controlled and frees margin.

### 6.4 Hold logic

A coin that remains in the target set across consecutive rebalances needs no action (its position is already correct size). The only action is a size adjustment if account equity changed significantly (e.g., after a large gain, rebalance to maintain target notional).

Rebalance holds if: `|current_notional - target_notional| / target_notional > 5%`. Otherwise leave it.

### 6.5 When there are insufficient valid coins

If fewer than 30 coins pass the universe filter at decision time T:

- Do not trade that bar.
- Close any open positions if you have been below 30 for 3 consecutive bars.
- Log an alert: universe health degraded.

This threshold (30) ensures the cross-sectional z-score is meaningful. With 30 coins, you can reliably identify the top/bottom 10 as true outliers.

### 6.6 Flat periods

The strategy may go through weeks with modest losses or flat performance. **This is normal.** Do not:
- Override the signal with discretionary judgment
- Change N or leverage mid-regime
- Add signals that "worked recently"

These interventions are the primary way systematic strategies get destroyed.

---

## 7. Execution Model

### 7.1 Order types

**Primary:** Limit (post-only / maker). Bybit charges 0 or negative fees for maker orders.
**Fallback:** Market (taker). Bybit charges approximately 6 bps for taker.

Target mix: 70% maker, 30% taker or better. At 100% maker, effective fee ≈ 0 bps. At 50/50 mix, effective fee ≈ 3 bps/side. Even at 100% taker (6 bps/side), the strategy remains profitable (Sharpe ~2.9 from backtest with fully-loaded fees).

### 7.2 Fee math

| Fill type | Fee (Bybit USDT perp) |
|-----------|----------------------|
| Maker | -0.01% (rebate) or 0% |
| Taker | +0.06% |
| Blended target | ≤ 0.04%/side (4 bps) |

Per rebalance cost at target mix:
```
Turnover per rebal ≈ 52% of positions
Round-trip cost   ≈ 52% × 2 × 4 bps = 4.2 bps per position per bar
Gross alpha       ≈ 45–60 bps per position per active bar
Net after fees    ≈ 40–55 bps per position per active bar
```

### 7.3 Slippage

For $10k–$100k AUM, slippage is negligible on most universe coins. The limiting factor is the meme coin tail (5–10 coins with $1–5M daily volume).

Rule: **position size ≤ 1% of coin's 30-day average daily volume.**

At $5M AUM, each position ≈ $250k notional. For coins with <$25M average daily volume, reduce position size or exclude. At $10M+ AUM, implement the capacity model:
```python
impact_bps = 10 * sqrt(order_usd / avg_daily_vol_usd)
# Target: impact_bps < 3 bps per fill
# If impact > 3 bps: reduce position_size proportionally
```

### 7.4 API rate limits

Bybit V5 REST API: 120 requests/minute default, up to 600/minute with higher account tier.

With 113 coins and 3 data fetches per coin per hour, you need approximately 340 requests/hour = 6/minute. Well within limits. Use batch endpoints where available:
- `GET /v5/market/kline?symbol=X&limit=60` — one request per coin per data type
- Consider WebSocket subscriptions for real-time 1m kline streams to avoid polling

---

## 8. Risk Management

### 8.1 Position-level limits

| Limit | Threshold | Action |
|-------|-----------|--------|
| Max single position | 12% of AUM | Reduce to target 5% immediately |
| Net market exposure | ±5% of AUM | Emergency rebalance to neutral |
| Cluster concentration | Max 4 same cluster per leg | Exclude lowest-ranked duplicate |
| Coin minimum volume | < 1% of daily vol | Exclude from universe |

### 8.2 Account-level limits

| Limit | Threshold | Action |
|-------|-----------|--------|
| Drawdown from peak | -15% | Reduce gross exposure to 50% of normal |
| Drawdown from peak | -25% | Halt all new entries, close 50% of positions |
| Drawdown from peak | -35% | Full halt, close all positions, review strategy |
| Daily loss | -5% of AUM | Stop trading for remainder of calendar day |
| Margin utilisation | >75% | Reduce gross exposure immediately |

### 8.3 Signal health monitoring

At each rebalance, compute and log:

```python
# Cross-sectional dispersion of predicted_funding
signal_std = predicted_funding_panel.std(axis=1).mean()  # mean over recent 30 bars

# Alert thresholds (based on Phase 23 research — 2024 had near-zero dispersion)
if signal_std < 0.0002:
    alert("Signal dispersion critically low — edge may be gone")
elif signal_std < 0.0005:
    alert("Signal dispersion low — monitor closely")
```

**This is the primary "edge is alive" indicator.** When the meme coin ecosystem is active, predicted_funding std is typically 0.001–0.005 across the universe. If it stays below 0.0002 for more than 5 consecutive bars, the strategy has no edge that session.

### 8.4 Leverage control

At 2× leverage with $10k account, gross exposure = $20k. Bybit requires margin = notional / leverage. Maintain at least **25% free margin headroom** at all times:

```python
required_margin = gross_notional / leverage
free_margin     = account_balance - required_margin
headroom_pct    = free_margin / account_balance

if headroom_pct < 0.25:
    # Reduce gross exposure: close the N weakest-conviction positions
    reduce_positions(n=3)
```

### 8.5 Inverse scaling (anti-overheating)

When the strategy is clearly overheating or breaking down, reduce sizing:

```python
rolling_sharpe_30 = compute_rolling_sharpe(returns_8h, window=30)

if rolling_sharpe_30 > 5.0:
    # Overheating — extremely unusual returns, likely mean-revert
    scale_factor = 0.5
elif rolling_sharpe_30 < 0.0:
    # Strategy losing money — reduce exposure
    scale_factor = 0.5
else:
    scale_factor = 1.0

position_size *= scale_factor
```

Phase 18 confirmed: inverse scaling reduces MaxDD from -32% → -20% while maintaining or improving Sharpe. It is particularly effective at avoiding the post-monster-month snapback.

---

## 9. Walk-Forward Optimization

### 9.1 What gets optimized

The strategy has **very few free parameters** by design. Only optimize what is genuinely uncertain:

| Parameter | Fixed or optimized | Value |
|-----------|-------------------|-------|
| N (positions per leg) | **Fixed** | 10 |
| Signal weights (2:1) | **Fixed** | 2×funding, 1×momentum |
| Leverage | **Fixed** | 2× (operator choice) |
| Listing age guard | **Fixed** | 14 days |
| Inverse scaling thresholds | **Fixed** | Sharpe > 5 or < 0 → 0.5× |
| Regime filter thresholds θ₁, θ₂ | **Optimized** (if using regime filter) | WFO every 6 months |

**If you are not using the regime filter (recommended for initial deployment), there is nothing to optimize.** The strategy has no free parameters once you fix the above.

### 9.2 Regime filter walk-forward (optional)

The regime filter skips trading when cross-sectional signal dispersion and funding dispersion are below thresholds. These thresholds need to be calibrated per market regime.

Walk-forward protocol:
```
Training window: 6 months
OOS test window: 3 months
Step size:       3 months (rolling forward)

For each training window:
  Grid search θ₁ ∈ [10th, 90th percentile of signal_strength in window]
  Grid search θ₂ ∈ [10th, 90th percentile of funding_disp in window]
  Select (θ₁*, θ₂*) that maximize Sharpe on training window
  Apply (θ₁*, θ₂*) to next 3-month OOS window without re-fitting
```

**Important:** Do not refit thresholds more frequently than every 3 months. Overfitting to short windows produces thresholds that react to noise, not regime changes.

### 9.3 Universe walk-forward

The universe does not need optimization. The only update is:
- **Every 6 months:** Check if any new coin qualifies as a "Major" (top-10 global market cap, cross-listed on all major exchanges). If yes, add to the permanent exclusion list.
- **Never** remove coins from the trading universe based on trailing performance.

### 9.4 Signal weight walk-forward

Do not re-optimize signal weights (2:1 ratio) on a rolling basis. The 2:1 ratio was validated across 4 independent walk-forward windows covering 15 months. Re-optimizing on shorter windows will produce overfit weights. If you suspect the signal relationship has changed, paper trade for 1 quarter before acting.

### 9.5 How to detect strategy degradation

The strategy edge depends on retail speculative activity in the meme/AI coin universe. Monitor:

| Indicator | Healthy range | Alert range |
|-----------|--------------|-------------|
| Mean predicted_funding across universe | 0.0003 – 0.005 | < 0.0001 for 2+ weeks |
| Cross-sectional std of predicted_funding | 0.001 – 0.005 | < 0.0002 for 5+ bars |
| Trailing 30-bar Sharpe | > 0.5 | < 0 for 10+ bars |
| Number of coins with |funding| > 0.001 | > 30 | < 10 for 3+ days |

If all four indicators flash simultaneously, pause the strategy and investigate whether the meme/AI coin ecosystem has structurally shifted (e.g., Bybit changed funding mechanics, bear market with no retail).

---

## 10. Live Monitoring

### 10.1 Required dashboards

**Signal dashboard (update every 1h):**
- Cross-sectional distribution of predicted_funding: histogram, mean, std
- Top 10 and bottom 10 coins by predicted_funding at current bar
- 24h momentum distribution

**Position dashboard (update every 1h):**
- Current longs and shorts with sizes and P&L
- Net market exposure ($)
- Gross leverage (actual vs target)
- Margin utilisation %

**Performance dashboard (update every 8h):**
- Equity curve from strategy start
- Rolling 30-bar Sharpe (inverse scaling trigger)
- Drawdown from peak
- Monthly P&L table (running)
- Fee breakdown: maker vs taker actual

**Universe health dashboard (update daily):**
- Count of valid coins at each bar
- Signal dispersion (std of predicted_funding)
- Coins added/removed from universe (new listings, delistings)

### 10.2 Alerts to configure

| Event | Severity | Action |
|-------|----------|--------|
| Universe < 30 valid coins | Critical | Do not trade, investigate |
| Signal std < 0.0002 | Warning | Log, monitor 2+ bars |
| Drawdown > -15% from peak | Warning | Scale to 50% exposure |
| Drawdown > -25% from peak | Critical | Halt entries, close half |
| Daily loss > -5% | High | Stop for calendar day |
| Data feed failure (any coin) | High | Exclude coin, log |
| API error > 3 consecutive | Critical | Halt, alert operator |
| Position > 12% AUM | High | Immediate reduce |

### 10.3 Data quality checks (run before every bar)

```python
def validate_data(premium_1m, price_1h, symbol, current_ts):
    # 1. Check for stale data (last update > 10 minutes ago)
    if (current_ts - premium_1m.index[-1]) > pd.Timedelta(minutes=10):
        return False, "stale_data"

    # 2. Check for zero or negative prices
    if (price_1h <= 0).any():
        return False, "bad_price"

    # 3. Check for extreme 1h returns (data error, not real move)
    returns_1h = price_1h.pct_change()
    if (returns_1h.abs() > 0.5).any():
        return False, "extreme_return"

    # 4. Check minimum bar count in premium window
    window_start = current_ts.floor("8h")
    window_data  = premium_1m[premium_1m.index >= window_start]
    if len(window_data) < 5:
        return False, "insufficient_premium_data"

    return True, "ok"
```

Exclude coins that fail any check from the current bar's computation. Log all exclusions.

---

## 11. Expected Performance

### 11.1 Backtested results (Jan 2025 – Mar 2026)

| Configuration | Sharpe | MaxDD | $10k→ | WF windows |
|---------------|--------|-------|-------|------------|
| 1× leverage | 3.96 | -22% | $65,889 | 4/4 |
| **2× leverage (recommended)** | **3.96** | **-40%** | **$351,824** | **4/4** |
| 3× leverage | 3.96 | -60% | $1,545,162 | 4/4 |

Note: Sharpe is the same at all leverage levels because both returns and fees scale identically. MaxDD and absolute returns scale proportionally.

### 11.2 Monthly performance distribution

The monthly returns are highly skewed. Plan accordingly:

```
Monster months (>+50%): Sep 2025 (+70%), Oct 2025 (+268%), Nov 2025 (+52%), Dec 2025 (+64%), Jan 2026 (+81%)
Normal positive months: Mar-Apr 2025 (+19%, +30%), Feb-Mar 2026 (+30%, +10%)
Flat/small loss months: May, Jun 2025 (-9%, -6%)
Bad months: Jan 2025 (-19%), Feb 2025 (-9%), Jul 2025 (-9%), Aug 2025 (-3%)
```

Expect to have 3–5 bad months per year. The strategy recovers in the good months. **Do not change the strategy during bad months.**

### 11.3 Capacity limits

| AUM | Expected Sharpe degradation | Net Sharpe | Notes |
|-----|-----------------------------|-----------|-------|
| < $500k | < 5% | ~3.8 | Negligible market impact |
| $500k – $2M | 5–10% | 3.5–3.8 | Optimal range |
| $2M – $5M | 10–20% | 3.2–3.5 | Still excellent |
| $5M – $10M | 20–30% | 2.8–3.2 | Acceptable |
| $10M – $25M | 30–50% | 2.0–2.8 | Marginal |
| > $25M | > 50% | < 2.0 | Not recommended |

Constraint: top contributing coins (JELLYJELLY, COAI, PIPPIN, MYX) have $1–10M daily volume. At $5M AUM each position is $250k — already 2.5–25% of daily volume for some coins.

### 11.4 Statistical validation

From Phase 14 Monte Carlo analysis:
- **Permutation test p-value: 0.001** — probability of getting this alpha by random chance is 0.1%
- **Bootstrap 95% CI on Sharpe: [0.38, 4.26]** — lower bound is positive
- **Block bootstrap (autocorrelation-adjusted) 95% CI: [0.39, 4.26]** — same conclusion

The wide CI reflects only 15 months of OOS history. More data will tighten the interval. Paper trade for one full quarter before deploying significant capital.

---

## 12. What Not to Do

Based on 25 phases of research, these are the experiments that failed. Implement them and you will degrade the strategy.

### Never add these signals

| Signal | Why it fails |
|--------|-------------|
| **OI divergence** | Near-zero IC on clean data. The IC in early tests was a data artefact. |
| **8h momentum** | Mean-reverting at 8h horizon (ICIR -0.17). Opposite direction to the edge. |
| **BTC trend filter** | Uncorrelated with strategy quality. Adds a BTC directional bet you don't want. |
| **Mean-reversion layer** | Sharpe -1.5, $1k → $91. Catastrophic. The funding edge IS momentum reinforcement, not mean-reversion. |
| **L/S ratio (ls_z)** | Zero IC. Retail sentiment data is captured by funding — no additional information. |
| **Funding trend (for predicted_funding)** | Redundant. Predicted_funding already captures rising TWAP within the window. Adding it reduces Sharpe 3.94 → 2.96. |

### Never make these structural changes

| Change | Why it fails |
|--------|-------------|
| **Vol-targeting / dynamic leverage** | Cuts exposure in high-vol months — exactly when the strategy earns most. |
| **Dynamic coin exclusion by trailing P&L** | Bad coins in training recover OOS. This was proven in Phase 9. |
| **Regime filter with short OOS windows** | Overfits. Only use if fitting on 6-month windows and applying for 3 months minimum. |
| **Adaptive N** | Fixed N=10 outperforms all adaptive schemes tested. Complexity adds instability. |
| **Asymmetric (long-only or short-only) regime** | Makes MaxDD worse, not better. The long/short neutrality is the drawdown control. |
| **Funding gate (skip low-funding bars)** | High-funding bars earn MORE than average bars — a gate would skip the best sessions. |

### Never do these operationally

| Action | Why it's harmful |
|--------|----------------|
| **Override the signal after a bad month** | Curve-fitting in real time. The bad month is usually followed by recovery. |
| **Change leverage mid-drawdown** | Either you reduce too early (miss recovery) or too late (cuts into crisis). Set it once. |
| **Re-optimize weights on rolling 1-month windows** | 1-month windows are pure noise. Overfit weights are worse than fixed weights. |
| **Add discretionary positions alongside systematic** | Mixes two different processes. If discretionary is good, run it separately. |
| **Trade more than 3× leverage** | At 5×, a single 20% adverse month liquidates the account. |

---

## Appendix A: Complete Parameter Reference

| Parameter | Value | Source |
|-----------|-------|--------|
| Universe size | ~100–120 coins | All Bybit USDT perps minus Majors |
| Majors exclusion list | 18 coins | Structural (Phase 6) |
| Minimum listing age | 14 days | Soft guard (Phase 25) |
| Rebalance frequency | Every 8h | Funding settlement alignment (Phase 3) |
| Settlement times | 00:00, 08:00, 16:00 UTC | Bybit funding schedule |
| Positions per leg | N = 10 | Sharpe-optimal (Phase 24) |
| Long + short total | 20 positions | — |
| Position sizing | Equal-weight | Phase 3 (vol-scaling rejected Phase 12) |
| Signal 1 weight | 2× | predicted_funding (Phase 22) |
| Signal 2 weight | 1× | mom_24h (Phase 22) |
| Z-score clip | ±3σ | Standard (Phase 2) |
| Min valid coins for z-score | 15 | Phase 2 |
| Interest rate constant | 0.0001 | Bybit funding formula |
| Funding cap | ±0.0075 | Bybit clamp |
| Leverage (recommended) | 2× | Phase 24 |
| Inverse scaling: upper | Sharpe > 5 → 0.5× | Phase 18 |
| Inverse scaling: lower | Sharpe < 0 → 0.5× | Phase 18 |
| Inverse scaling window | 30 bars (= 10 days) | Phase 18 |
| Drawdown halt: hard | -35% from peak | Phase risk design |
| Drawdown halt: soft | -25% → 50% exposure | Phase risk design |
| Daily loss limit | -5% of AUM | Phase risk design |
| Target maker fill rate | ≥ 70% | Phase 3 execution |
| Max taker fill rate | 50% acceptable | Phase 3 |
| Capacity (optimal) | $1M – $5M AUM | Phase 15 |
| Capacity (hard limit) | $10M AUM | Phase 15 |

---

## Appendix B: Bybit API Endpoints Reference

| Data | Endpoint | Key parameters |
|------|----------|---------------|
| 1m kline (price) | `GET /v5/market/kline` | `symbol`, `interval=1`, `start`, `end`, `limit=1000` |
| 1m premium index kline | `GET /v5/market/premium-index-price-kline` | `symbol`, `interval=1`, `start`, `end`, `limit=1000` |
| Funding rate history | `GET /v5/market/funding/history` | `symbol`, `startTime`, `endTime`, `limit=200` |
| Current instruments | `GET /v5/market/instruments-info` | `category=linear` |
| Place order | `POST /v5/order/create` | `category`, `symbol`, `side`, `orderType`, `qty`, `price` |
| Cancel order | `POST /v5/order/cancel` | `category`, `symbol`, `orderId` |
| Get positions | `GET /v5/position/list` | `category=linear` |
| Account balance | `GET /v5/account/wallet-balance` | `accountType=UNIFIED` |

Use WebSocket `wss://stream.bybit.com/v5/public/linear` for real-time 1m kline streams if you want to eliminate polling latency.

---

## Appendix C: Recommended Implementation Stack

```
Data storage:     Parquet files (simple) or TimescaleDB (production)
Computation:      Python 3.10+, pandas, numpy
Scheduling:       APScheduler or cron (trigger at S-70m before each settlement)
Exchange API:     pybit (official Bybit Python SDK) or requests with HMAC auth
Logging:          structlog to JSON, rotate daily
Alerting:         Telegram bot API or PagerDuty webhook
Monitoring:       Grafana + InfluxDB (optional but recommended for production)
Paper trading:    Run full signal pipeline, log target portfolio, compute virtual PnL
                  before committing any real capital
```

---

*This document represents the complete output of 25 research phases covering Jan 2025 – Mar 2026. All performance figures are backtested. Paper trade for one full quarter before deploying live capital. The strategy depends on the continued existence of a speculative meme/AI coin ecosystem on Bybit with volatile funding rates — monitor signal dispersion as the primary edge indicator.*
