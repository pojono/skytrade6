# Big-Move Liquidation Strategy — Findings (v26h/v26i)

**Date:** Feb 2026  
**Symbols:** DOGEUSDT, SOLUSDT, ETHUSDT, XRPUSDT  
**Data:** 282 days of tick-level data, 1-minute OHLC bars  
**Fees:** maker=0.02%, taker=0.055% (round-trip: TP=0.04%, timeout=0.075%)

---

## The Problem

The microstructure research (v26g) showed that fading liquidation cascades produces a mean return of only +1-4 bps at 1 minute. With round-trip fees of 4-7.5 bps, **market-order entry cannot overcome fees** at any filter combination.

## v26h: Market-Order Fade — FAILED

Swept 6 dimensions: TP (10-50bp), cascade depth (1-5+), notional percentile (P95-P99), displacement filter (5-50bp), entry delay (0-120s), buy-side only.

**Result: Every single configuration is net negative after fees.** Best raw return was +3.88 bps (P99 events), but 4 bps maker+maker fees eat the entire edge.

## v26i: Limit-Order Entry + Microstructure Filters — PROFITABLE

The key insight: **limit orders placed INTO the cascade capture the spread as alpha.** The entry offset (e.g., 0.15% below market) means you buy at a discount. Combined with microstructure filters to select only the highest-quality cascades, this overcomes fees.

### Combined Filters (cascade 3+, P97 1+, displacement 10+ bps, 60min hold, no SL)

Best configurations per symbol (Sweep 6):

| Symbol   | Offset | TP    | Fills | Total Net % | Avg Net %/trade | WR    | TP%   | Sharpe |
|----------|--------|-------|-------|-------------|-----------------|-------|-------|--------|
| DOGEUSDT | 0.30%  | 0.20% | 152   | **+19.2%**  | +0.127%         | 95.4% | 95.4% | 10.0   |
| DOGEUSDT | 0.20%  | 0.20% | 135   | **+13.3%**  | +0.099%         | 95.6% | 95.6% | 3.9    |
| DOGEUSDT | 0.20%  | 0.12% | 135   | **+9.8%**   | +0.073%         | 99.3% | 99.3% | 11.7   |
| SOLUSDT  | 0.30%  | 0.30% | 219   | **+26.9%**  | +0.123%         | 85.8% | 83.6% | 4.9    |
| SOLUSDT  | 0.20%  | 0.20% | 242   | **+22.1%**  | +0.091%         | 92.1% | 91.3% | 6.0    |
| SOLUSDT  | 0.20%  | 0.15% | 242   | **+18.3%**  | +0.076%         | 95.0% | 94.6% | 8.4    |
| ETHUSDT  | 0.15%  | 0.15% | 435   | **+18.7%**  | +0.043%         | 94.3% | 94.3% | 3.3    |
| ETHUSDT  | 0.20%  | 0.15% | 408   | **+16.5%**  | +0.040%         | 93.4% | 93.4% | 3.1    |
| ETHUSDT  | 0.15%  | 0.12% | 435   | **+16.1%**  | +0.037%         | 96.3% | 96.3% | 3.7    |
| XRPUSDT  | 0.15%  | 0.30% | 179   | **+27.4%**  | +0.153%         | 90.5% | 89.9% | 5.7    |
| XRPUSDT  | 0.20%  | 0.30% | 165   | **+24.4%**  | +0.148%         | 89.1% | 87.9% | 5.8    |
| XRPUSDT  | 0.10%  | 0.20% | 185   | **+16.7%**  | +0.090%         | 93.0% | 92.4% | 4.6    |

### Universal Configuration (works across all 4 symbols)

**Offset=0.15%, TP=0.15%, no SL, 60min hold, cascade 3+, P97 1+, disp 10+:**

| Symbol   | Fills | Total Net % | Avg Net % | WR    | Sharpe |
|----------|-------|-------------|-----------|-------|--------|
| DOGEUSDT | 142   | +8.9%       | +0.063%   | 97.2% | 2.7    |
| SOLUSDT  | 253   | +17.4%      | +0.069%   | 96.0% | 5.3    |
| ETHUSDT  | 435   | +18.7%      | +0.043%   | 94.3% | 3.3    |
| XRPUSDT  | 179   | +14.0%      | +0.078%   | 97.2% | 5.5    |
| **TOTAL**| **1009**| **+58.9%** | **+0.058%** | **95.8%** | — |

**+58.9% combined net return over 282 days, 1009 trades, 95.8% win rate.**

---

## Why It Works

1. **Limit order offset = built-in alpha.** Placing a buy limit 0.15% below market during a cascade means you only fill if price drops further — and the cascade provides that drop. The offset itself is your edge.

2. **Microstructure filters select the best cascades.** Requiring 3+ events, at least one P97 event, and 10+ bps displacement ensures you only trade during genuine cascade events with enough momentum to fill your limit order AND bounce back.

3. **No stop loss eliminates taker fees on losers.** The R:R research showed that removing SL and using time-based exit is optimal. When TP hits, you pay maker+maker (0.04%). When timeout hits, you pay maker+taker (0.075%) — but 94-97% of trades hit TP.

4. **The cascade provides liquidity.** During a cascade, there's heavy selling pressure (for buy-side cascades). Your limit buy order provides liquidity to panicking sellers, earning you the maker rebate and a better fill price.

---

## Key Observations

### What matters most:
- **Entry offset** is the #1 driver of profitability. 0% offset = breakeven or negative. 0.10-0.20% offset = strongly profitable.
- **TP should be tight** (0.12-0.20%). Wider TPs (0.30-0.50%) have lower TP rates and more timeout exits (expensive).
- **No SL is optimal.** Every SL configuration tested was worse than no-SL.
- **60min hold > 30min hold.** More time = more TP fills.

### What doesn't matter:
- **Buy-side only filter** — marginal improvement, reduces trade count significantly.
- **Very wide offsets (0.30%)** — high per-trade return but fewer fills.

### Risk considerations:
- **~135-435 trades over 282 days** = 0.5-1.5 trades/day per symbol.
- **With 4 symbols: ~3.6 trades/day** — enough for meaningful returns.
- **95%+ win rate** means drawdowns are rare and shallow.
- **Average hold: 2-6 minutes** for TP exits — capital is deployed briefly.

---

## Strategy Parameters (Recommended)

```
Entry:
  - Trigger: cascade detected (3+ P95 events within 60s, at least 1 P97, displacement ≥10 bps)
  - Order: limit buy/sell at 0.15% offset from current price
  - Direction: fade the cascade (buy when longs liquidated, sell when shorts liquidated)

Exit:
  - Take profit: 0.15% from entry (limit order, maker fee)
  - Stop loss: none
  - Max hold: 60 minutes (market order, taker fee)
  - Cooldown: 5 minutes between trades

Fees:
  - Entry: 0.02% (maker)
  - TP exit: 0.02% (maker) → round-trip = 0.04%
  - Timeout exit: 0.055% (taker) → round-trip = 0.075%
```

---

## Files

- `liq_bigmove_strategy.py` — v26h market-order sweep (all negative)
- `liq_bigmove_limit.py` — v26i limit-order sweep (profitable)
- `results/liq_bigmove_strategy.txt` — v26h results
- `results/liq_bigmove_limit.txt` — v26i results
