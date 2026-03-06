# Strategy Specification: Cross-Sectional Funding + Momentum

**Date:** 2026-03-06
**Status:** Research complete (Phases 1–16) — ready for live implementation

---

## Executive Summary

A cross-sectional long/short strategy on perpetual futures, exploiting funding rate carry, short-term price momentum, and funding trend across a universe of ~113 coins (Majors excluded). The strategy is market-neutral by construction, rebalances every 8 hours aligned with Bybit funding settlements, and sits flat when regime conditions are unfavourable.

**Final configuration (Jan 2025 – Mar 2026, correct time alignment, No-Majors):**
- Net Sharpe: **3.47** (Phases 11–16 improved signal)
- Sortino: **5.80**
- Annualized return: **~1311%** (on notional; 1x unlevered)
- Maximum drawdown: **-32.7%** (vs -46% of original baseline)
- $1,000 → **$22,067** over 15 months
- Win rate: **52.3%** of 8h bars

> **Improvement over Phase 9 baseline (Sharpe 2.99):** Adding `funding_trend` as a 3rd signal
> (weight 2:1:1 = funding:mom24h:funding_trend) improved MaxDD from -46% to -33% with negligible
> Sharpe change. The funding_trend signal helps avoid bad months (May 2025: -14.8% → +6.2%).
>
> **What was rejected (Phases 11–16):** Dynamic leverage (reduces Sharpe), asymmetric regime
> filter (worsens MaxDD), BTC trend gate (drops Sharpe), OI/vol/BTC-relative signals (negative IC).

---

## Research Summary (Phases 1–16)

| Phase | Finding |
|-------|---------|
| 1 — Signal IC | Funding carry ICIR +0.151 at 8h. All momentum signals mean-revert. prem_z/ls_z/oi_div flat. |
| 2 — Portfolio backtest | funding + mom_24h equal-weight: Sharpe 3.27, 4/4 OOS positive |
| 3 — Execution | 8h rebal, N=10, equal-weight, maker-only. Fee-robust to any fill rate. |
| 4 — Monthly breakdown | 11/15 months positive. Monster months in Sep/Oct 2025, Jan 2026. Bad months: May, Jul 2025. |
| 5 — Regime filter | signal_strength + funding_disp (walk-forward): MaxDD -46% → -25%, Sharpe +1.0 on filtered bars |
| 6 — Cluster analysis | Majors (BTC/ETH/SOL) net **negative** (-3,692 bps, Sharpe -1.86). Meme/AI/Legacy drive all alpha. |
| 7 — Universe filter | Removing 18 Majors + 8 worst meme coins: Sharpe 2.75 → 3.84 **(partial look-ahead in coin list)** |
| 8 — Combined | Universe + regime filter: Sharpe → 4.31, MaxDD → -18% **(partial look-ahead — see Phase 9)** |
| 9 — Rolling universe | Proper walk-forward coin selection. Rolling exclusion **does not help** OOS. Only Majors exclusion survives: Sharpe 2.99, 4/4 windows. |
| 10 — Final equity | $1,000 → $26,504 (No-Majors, no regime, Jan 2025–Mar 2026). Regime filter: $6,808. |
| 11 — New signal IC | funding_trend: ICIR +0.106 (positive, significant). OI/vol/BTC-relative: all negative IC. Rejected. |
| 12 — Dynamic leverage | Vol-targeting and DD-scaling **reduce Sharpe** — alpha is in high-vol burst months. Rejected. |
| 13 — Regime improvements | Asymmetric (short-only) filter worsens MaxDD. BTC trend gate drops Sharpe. Soft threshold best option for conservative deployment. |
| 14 — Monte Carlo | Permutation test: p < 0.001. Bootstrap 95% CI Sharpe entirely positive. Strategy is statistically robust. |
| 15 — Capacity | Capacity limit ~$10M AUM (Sharpe -25%). Optimal: $1M–$5M. $100M+ breaks even. |
| 16 — Final combined | **Best: 2×funding + mom_24h + funding_trend (2:1:1 weights). Sharpe 3.47, MaxDD -33%, $1k→$22k.** |

---

## Universe

**113 coins** — all Bybit perpetuals with 400+ day history, minus 18 Majors.

**Excluded: 18 Majors** (structural exclusion — no look-ahead required)
```
BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT ADAUSDT DOGEUSDT AVAXUSDT
DOTUSDT LTCUSDT BCHUSDT TRXUSDT XLMUSDT ETCUSDT HBARUSDT ATOMUSDT
ALGOUSDT EGLDUSDT
```

**Why Majors are excluded (structural argument):** BTC/ETH/SOL have deep liquidity and are heavily traded across all timeframes. At the 8h cross-sectional scale, their moves are dominated by market-wide flows rather than the funding + momentum microstructure the strategy exploits. Phase 6 confirmed this empirically: Majors occupy 59% of long slots but deliver net -3,692 bps total contribution (Sharpe -1.86). Phase 9 confirmed the exclusion holds walk-forward (Sharpe 2.75 → 2.99, 4/4 OOS windows).

**Why NOT to dynamically exclude other coins:** Phase 9 tested rolling exclusion of negative-contribution coins. It *hurts* OOS performance — a coin that underperformed in training months often recovers in OOS. Past per-coin P&L does not predict future per-coin P&L at 3-month horizon. The 8 worst meme coins identified in Phase 6 (PENGU, BAN, etc.) were identified with full hindsight and cannot be used for live filtering.

---

## Signal Construction

### Step 1: Funding Rate Signal
```python
# For each symbol s at decision time t:
funding_s_t = most_recent_funding_rate(s)   # forward-filled from 8h settlement
```
Source: Bybit `_funding_rate.csv`. No normalization before z-scoring.

**Direction:** Long high-funding (longs pay shorts → you collect), short low/negative-funding.

### Step 2: 24h Momentum Signal
```python
mom24h_s_t = close_s_t / close_s_{t-24h} - 1
```
Source: Bybit `_kline_1m.csv`, use last close before decision time.

**Direction:** Long recent winners, short recent losers.
**Note:** Momentum has *negative* IC globally (-0.186 ICIR) but is super-additive with funding — it conditions on self-reinforcing uptrends within high-funding coins.

### Step 3: Funding Trend Signal (added Phase 16)
```python
# 24h change in funding rate (at 1h resolution, then resampled to 8h)
funding_trend_s_t = funding_s_t - funding_s_{t-24h}
```
Positive funding_trend = funding rate is rising → growing directional conviction → stronger carry signal.
ICIR: +0.106, t-stat: +3.79 (Phase 11).

### Step 4: Composite Score
```python
z_funding       = cross_sectional_zscore(funding,       universe_at_t).clip(-3, 3)
z_mom24h        = cross_sectional_zscore(mom24h,        universe_at_t).clip(-3, 3)
z_funding_trend = cross_sectional_zscore(funding_trend, universe_at_t).clip(-3, 3)

# 2:1:1 weighting — funding dominates (strongest signal)
composite = (2*z_funding + z_mom24h + z_funding_trend) / 4
```
Computed only over the 113-coin universe (Majors excluded before z-scoring).

**Why 2:1:1?** Funding has ICIR +0.203, funding_trend +0.106, mom24h -0.161. Funding should dominate. The 2:1:1 weighting was selected by Phase 16 backtest comparison — it reduces MaxDD by 28% vs equal-weight with negligible Sharpe cost.

---

## Regime Filter

**Trade only when BOTH conditions are met at decision time t:**

```python
signal_strength_t = std(composite, across universe)  # CS spread of scores
funding_disp_t    = std(funding_rate, across universe)  # differentiation in funding

trade = (signal_strength_t > θ₁) AND (funding_disp_t > θ₂)
```

**Thresholds θ₁, θ₂:** Fit on trailing 6-month training window (walk-forward). Grid-search 10th–90th percentile, maximise training Sharpe. Recalibrate every 6 months.

**Interpretation:**
- `signal_strength` low → all coins rank similarly → no differentiation → noise, not signal
- `funding_disp` low → funding rates compressed across universe → carry signal weak

**Effect:** Strategy is active ~28% of bars. Sits flat the rest. When it fires, conditions are confirmed by two independent indicators.

**When inactive:** Close all positions (or maintain last positions at zero turnover cost — test both in live).

---

## Portfolio Construction

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Universe | **105 coins** (excl. Majors + worst meme) | Negative-alpha coins excluded |
| Rebal frequency | **Every 8 hours** (00:00, 08:00, 16:00 UTC) | Funding settlement alignment |
| Regime check | At each rebal, before placing orders | Skip rebal if filter inactive |
| Positions per leg | **10 long + 10 short** | N=10 sweet spot (tested N=5–30) |
| Long selection | Top 10 by composite rank | Highest funding + momentum |
| Short selection | Bottom 10 by composite rank | Lowest funding + momentum |
| Sizing | **Equal-weight** (1/20 AUM per position) | Vol-scaling tested and rejected |
| Dollar neutrality | Long notional = Short notional | Market-neutral by construction |

---

## Execution

| Parameter | Value |
|-----------|-------|
| Order type | **Limit orders (maker-first)** |
| Fee target | 4 bps/side |
| Pre-rebal window | T-60m: compute signals, check regime |
| Order placement | T-30m: place limit orders at mid |
| Fill deadline | T: cancel unfilled, execute remainder at market |
| Max taker fraction | 50% (Sharpe remains 2.90 at 50/50 mix) |
| Slippage tolerance | Limit price = mid ± 1 bps |

**Rebalancing flow:**
1. T-60m: Pull latest funding rates + 1m closes. Compute composite. Check regime filter.
2. If regime filter **inactive**: skip rebal. Close open positions gradually over next 24h.
3. If regime filter **active**: compute target portfolio (top/bottom 10).
4. T-30m: Place limit orders for new positions and exits.
5. T: Cancel unfilled. Execute residual at market (taker).
6. Funding credited automatically at T.

---

## Risk Management

| Risk | Limit | Action |
|------|-------|--------|
| Single position | Max 12% of AUM | Reduce to 5% immediately |
| Net market exposure | ±5% of AUM | Emergency rebalance |
| Cluster concentration | Max 4 of same cluster per leg | Exclude lowest-ranked duplicates |
| Max drawdown | -20% from equity peak | Cut gross exposure 50% |
| Daily loss limit | -5% of AUM | Halt until next session |
| Regime inactive streak | >30 consecutive bars flat | Alert — review signal health |
| Liquidation buffer | 25% margin headroom always | Reduce gross exposure if below |

---

## Expected Performance (Steady State)

*Phase 16 final configuration: 2×funding + mom_24h + funding_trend (2:1:1), No-Majors, Jan 2025–Mar 2026.*

### Recommended: No-Majors + funding_trend signal (Phase 16 Strategy 4)
| Metric | Value |
|--------|-------|
| Sharpe | **3.47** |
| Sortino | **5.80** |
| Ann. Return | **~1311%** (on notional, 1x unlevered) |
| Max Drawdown | **-32.7%** |
| $1,000 → | **$22,067** (15 months) |
| Win rate | **52.3%** of bars |
| Positive months | **11/15** (73%) |

### Conservative: Baseline + soft regime filter
| Metric | Value |
|--------|-------|
| Sharpe | **2.63** |
| Max Drawdown | **-21.0%** |
| $1,000 → | **$3,919** (15 months) |
| Active | 100% (scaled 0.25×–1.0× by confidence) |

### Historical baseline (Phase 10, no signal improvement)
| Metric | Value |
|--------|-------|
| Sharpe | **3.45** |
| Max Drawdown | **-45.4%** |
| $1,000 → | **$23,424** |

**Monthly record (Phase 16 Strategy 4, No-Majors + funding_trend):**
- Positive months: **11/15** (73%)
- Best month: Oct 2025 +172.6%
- Worst month: Jul 2025 -18.0%
- Bad months avoided vs baseline: May 2025 (-14.8% → +6.2%), Dec 2025 (-13.1% → +28.6%)

---

## Fee Math (Active Bars)

When regime filter is active (~28% of bars), 52% position turnover, maker execution:
```
Gross alpha per active bar  ≈ 45–60 bps/position (higher quality bars selected)
Fee per active bar          = turnover × 8 bps RT ≈ 4.2 bps/position
Net per active bar          ≈ 40–55 bps/position

Over full year:
  Active bars = 0.28 × 1095 = 307 bars
  Net P&L     = 307 × ~47 bps × 20 positions / 20 = 307 × 47 bps ≈ 14,000 bps notional
```

---

## Cluster Performance (Phase 6 findings)

| Cluster | Coins | Total Contribution | Sharpe | Long% |
|---------|-------|--------------------|--------|-------|
| Meme | 46 | +26,290 bps | 2.95 | 53% |
| AI/Infra | 13 | +7,340 bps | 1.46 | 45% |
| Legacy/Other | 9 | +5,773 bps | 2.86 | 50% |
| Gaming/NFT | 7 | +1,988 bps | 1.97 | 40% |
| DeFi | 20 | +1,602 bps | 0.62 | 49% |
| L1/L2 | 16 | +1,084 bps | 0.45 | 36% |
| **Majors** | 18 | **-3,692 bps** | **-1.86** | 59% |

Top individual contributors: JELLYJELLYUSDT, COAIUSDT, PIPPINUSDT, MYXUSDT, ENSOUSDT, DASHUSDT.
Worst individual coins: PENGUUSDT, BANUSDT, RESOLVUSDT (all excluded from live universe).

---

## Signals NOT Used

| Signal | Reason |
|--------|--------|
| prem_z (premium z-score) | IC ≈ 0 on clean data; dirty-data IC was spurious |
| mom_8h | Mean-reverting at 8h horizon (ICIR = -0.172) |
| mom_48h | Marginal improvement, adds complexity |
| ls_z (L/S ratio) | IC ≈ 0 on clean data |
| oi_div (OI divergence) | Weak negative IC, unreliable |
| BTC lead-lag | Not tested; conceptually valid but execution harder |

---

## Known Risks and Limitations

**1. Regime filter active only 28% of bars.** The strategy generates PnL in concentrated bursts (alt-season, high-funding regimes). Long flat periods require patience and discipline not to override.

**2. Meme coin concentration.** The alpha is disproportionately from obscure meme/small-cap coins (JELLYJELLY, COAI, PIPPIN, MYX). These can have liquidity events, exchange delistings, or sudden bid-ask spread widening. Position size caps and cluster concentration limits are essential.

**3. Survivorship bias in universe.** The 105-coin universe uses coins with 400+ days of history — coins that didn't survive or weren't listed aren't included. Live forward performance should be monitored carefully.

**4. Partial look-ahead in coin exclusions.** The 8 excluded worst meme coins were identified using full-history P&L attribution. In live trading, the universe should be updated dynamically using trailing 6-month performance, not hindsight.

**5. Funding rate compression risk.** If the market enters a sustained bear regime with negative funding, the carry signal flips. Monitor average universe funding rate; if negative for >2 weeks, pause strategy and review.

**6. Four OOS windows.** Walk-forward validation covers Jan 2025 – Jan 2026 (4 quarterly windows). More history would increase confidence. Paper trade for 1 quarter before deploying capital.

**7. Look-ahead in Phase 7–8 results.** Sharpe 3.84–4.31 from those phases used coin exclusion lists derived from full-history P&L (PENGU, BAN, etc.). Phase 9 proved this does not survive walk-forward. The strategy spec now uses only the structural Majors exclusion. The Phase 5 regime filter is valid (thresholds fit per training window on signal features, not coin identity).

---

## Capacity (Phase 15)

Market impact model: `impact_bps = 10 × sqrt(order_usd / daily_vol_usd)`

| AUM | Net Sharpe | Ann Ret | Degradation |
|-----|-----------|---------|-------------|
| $500k | ~2.05 | ~348% | -6% |
| $1M | ~2.00 | ~326% | -8% |
| $5M | ~1.78 | ~244% | -18% |
| **$10M** | **~1.62** | **~194%** | **-25% ← limit** |
| $25M | ~1.29 | ~114% | -41% |
| $100M | ~0.41 | -9% | breakeven |

**Recommended AUM: $1M–$5M.** At $10M the Sharpe degrades 25% from no-impact baseline. Above $25M the strategy is marginally viable. The constraint is meme/small-cap liquidity — top contributors (JELLYJELLY, COAI, PIPPIN) have $1–10M daily volume.

**Per-coin position limit:** At $5M AUM with 20 positions, each position = $250k notional. Order size should not exceed 2% of coin's daily volume to keep impact below 3bps.

---

## Statistical Validation (Phase 14)

- **Permutation test p-value: 0.001** (1/1000 chance the alpha is random)
- **Bootstrap 95% CI on Sharpe: [0.38, 4.26]** — lower bound positive
- **Block bootstrap 95% CI: [0.39, 4.26]** — robust to autocorrelation
- Wide CI reflects limited history (15 months OOS). Paper trade 1 quarter before live capital.

---

## Implementation Checklist

- [ ] Live data feed: Bybit funding rate (8h) + 1m klines for 113 symbols
- [ ] Universe manager: auto-update exclusion list every 6 months by trailing Sharpe
- [ ] Signal computation: 2×z_funding + z_mom24h + z_funding_trend → composite (< 5s latency)
- [ ] funding_trend = current_funding - funding_24h_ago (rolling, at 8h resolution)
- [ ] Regime check: signal_strength and funding_disp vs rolling thresholds (optional)
- [ ] Order management: limit order placement, fill monitoring, taker fallback
- [ ] Position reconciliation: target vs actual at each rebal
- [ ] Risk monitor: live P&L, net exposure, cluster concentration, margin headroom
- [ ] Data quality: zero price detection, extreme return filtering (clip ±99%)
- [ ] Alert system: regime inactive streak, drawdown breach, data feed failure
- [ ] Paper trading: 1-quarter dry run before live capital
- [ ] Exchange connectivity: Bybit REST + WebSocket API (V5)
