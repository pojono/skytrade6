# Liquidation Cascade Market-Making — Trading Guide

**Strategy:** Place limit orders INTO liquidation cascades, collect the mean-reversion bounce.
**Edge:** Forced liquidations create temporary price dislocations; limit orders capture the snap-back.
**Backtest:** 282 days, 4 symbols, +91.4% combined (aggressive) / +72.1% (safe) with displacement filter.
**Fees:** Bybit maker=0.02%, taker=0.055%

> **v26k update:** Displacement ≥10 bps is the only filter needed. All other filters (bad hours,
> long-only, weekday-only) are removed — they over-filter and hurt total returns.

---

## 1. What You Need

### Data Feed
- **Bybit Linear Perpetual Liquidation WebSocket**
  - Endpoint: `wss://stream.bybit.com/v5/public/linear`
  - Subscribe: `{"op": "subscribe", "args": ["liquidation.ETHUSDT"]}`
  - Each message contains: side (`Buy`/`Sell`), price, qty, timestamp
  - This is the **trigger** — without it, you cannot detect cascades

### Exchange Account
- **Bybit Unified Trading Account** (preferred — 0% maker on VIP tiers)
- Alternatively any exchange with ≤2 bps maker fees
- API keys with order placement permissions
- Futures enabled, sufficient margin

### Symbols (ranked by total return with displacement filter)
| Priority | Symbol | Total Return | WR | Sharpe | Max DD |
|----------|--------|-------------|-----|--------|--------|
| 1 | **ETHUSDT** | **+28.3%** | 95.4% | +4.5 | 3.6% |
| 2 | **SOLUSDT** | **+24.5%** | 94.8% | +5.2 | 5.8% |
| 3 | **DOGEUSDT** | **+19.1%** | 96.8% | +2.9 | 5.4% |
| 4 | **XRPUSDT** | **+19.5%** | 95.3% | +5.3 | 2.5% |

*(Numbers shown for Config 2 AGGR: off=0.15%, TP=0.12%, no SL, 60min hold, displacement ≥10 bps)*

**Trade all 4 simultaneously.** Low cross-symbol correlation provides diversification.
**Skip BTC** — the edge is too thin after fees.

---

## 2. Cascade Detection (The Trigger)

### Definition
A **cascade** = 2 or more P95-large liquidations within 60 seconds of each other.

### Step-by-Step

1. **Maintain a rolling notional threshold** per symbol
   - Track the last ~1000 liquidation events
   - Compute the 95th percentile of `qty × price` (notional)
   - Update this threshold every hour
   - Typical values: BTC ~$200K, ETH ~$80K, SOL ~$15K, DOGE ~$10K

2. **When a liquidation arrives that exceeds P95:**
   - Start a 60-second window
   - Count subsequent P95+ liquidations in that window
   - Track cumulative buy-side vs sell-side notional

3. **When 2+ P95 events cluster within 60s → CASCADE DETECTED**
   - Determine dominant side:
     - `buy_notional > sell_notional` → **Buy-dominant** (longs liquidated, price dropped)
     - `sell_notional > buy_notional` → **Sell-dominant** (shorts liquidated, price spiked)

4. **Check displacement ≥10 bps (THE key filter)**
   - Compare price at cascade end vs price just before cascade start
   - `displacement_bps = abs(end_price - pre_price) / pre_price × 10000`
   - If displacement < 10 bps → **SKIP** (not a real cascade)
   - This single filter improves combined returns by +37-391% depending on config

### What the cascade tells you
| Cascade Type | What Happened | Your Action |
|-------------|---------------|-------------|
| **Buy-dominant** (55-70% of cascades) | Longs got force-closed, price dipped | Place **limit BUY** below market |
| **Sell-dominant** (30-45%) | Shorts got force-closed, price spiked | Place **limit SELL** above market |

### Cascade frequency (from backtest)
| Symbol | Cascades/day | Avg duration |
|--------|-------------|-------------|
| ETH | 9.4 | 12 seconds |
| BTC | 7.4 | 14 seconds |
| SOL | 6.9 | 9 seconds |
| XRP | 5.3 | 9 seconds |
| DOGE | 4.4 | 8 seconds |

**Expect 5-9 signals per day per symbol.** Not all will fill.

---

## 3. Order Placement (The Entry)

### Timing: How Fast?

**You have about 8-14 seconds.** That's the average cascade duration. Your order must be placed **during** the cascade, not after.

| Requirement | Target |
|------------|--------|
| Cascade detection latency | < 2 seconds after 2nd P95 event |
| Order placement latency | < 1 second after detection |
| **Total reaction time** | **< 3 seconds** |
| Co-location needed? | No, but < 50ms network to Bybit helps |

**Practical flow:**
```
T+0.0s   First P95 liquidation arrives via websocket
T+0.5s   (waiting for cluster confirmation)
T+1.0s   Second P95 liquidation arrives → CASCADE CONFIRMED
T+1.5s   Your code computes limit price and sends order
T+2.0s   Order acknowledged by exchange
T+2-14s  Cascade continues, price moves toward your limit
T+2-60s  Your limit order fills (or doesn't)
```

A simple Python script with `websockets` + Bybit API can achieve this. No HFT infrastructure needed — you're not racing other traders, you're providing liquidity into forced selling.

### Order Parameters

**Config 1 SAFE (recommended for first deployment):**

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| **Order type** | `Limit` with `PostOnly=true` | Guarantees maker fee, rejects if would cross |
| **Entry offset** | **0.15%** below/above current price | Far enough to capture dislocation, close enough to fill |
| **Take profit** | **0.15%** from fill price | Quick exit, captures the bounce |
| **Stop loss** | **0.50%** from fill price | Wide SL — minimizes expensive taker exits |
| **Max hold** | **60 minutes** | Cancel unfilled orders + close position if no TP/SL |
| **Time in force** | `GTC` (cancel manually after 60 min) | Or use exchange's built-in time-in-force |
| **Reduce only (TP/SL)** | Yes | Prevent accidental position increase |
| **Filter** | Displacement ≥10 bps | Skip cascades where price barely moved |

**Config 2 AGGRESSIVE (best total return, once proven live):**

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| **Entry offset** | **0.15%** | Same as safe |
| **Take profit** | **0.12%** from fill price | Tighter TP → higher fill rate on TP |
| **Stop loss** | **None** | No SL = no expensive taker exits on losers |
| **Max hold** | **60 minutes** | Timeout is the only exit besides TP |
| **Filter** | Displacement ≥10 bps | Same as safe |

### Concrete Example: Buy-Dominant Cascade on ETHUSDT (Config 1 SAFE)

```
Current ETH price: $2,500.00
Cascade detected: Buy-dominant (longs liquidated, price dropping)
Displacement: 15 bps (≥10 → PASS)

Your action: LIMIT BUY

  Limit price:  $2,500.00 × (1 - 0.0015) = $2,496.25
  TP price:     $2,496.25 × (1 + 0.0015) = $2,500.00  (basically back to pre-cascade)
  SL price:     $2,496.25 × (1 - 0.0050) = $2,483.77

  Risk:   $2,496.25 → $2,483.77 = -$12.48 per ETH  (0.50%)
  Reward: $2,496.25 → $2,500.00 = +$3.75 per ETH  (0.15%)
  R:R = 0.3:1 (but 87.7% win rate makes this very profitable)
```

### Concrete Example: Sell-Dominant Cascade on SOLUSDT (Config 2 AGGR)

```
Current SOL price: $180.00
Cascade detected: Sell-dominant (shorts liquidated, price spiking)
Displacement: 22 bps (≥10 → PASS)

Your action: LIMIT SELL (short)

  Limit price:  $180.00 × (1 + 0.0015) = $180.27
  TP price:     $180.27 × (1 - 0.0012) = $180.05
  SL price:     NONE (hold until TP or 60min timeout)

  Reward: $180.27 → $180.05 = +$0.22 per SOL  (0.12%)
  Risk:   Timeout at market after 60min (avg timeout loss: ~1.25%)
```

---

## 4. Position Management (After Fill)

### Expected Outcomes

**Config 1 SAFE (with displacement filter):**

| Exit Type | Frequency | What Happens |
|-----------|-----------|-------------|
| **Take Profit** | **~88%** of fills | Price bounces back 0.15% → you collect (maker fee) |
| **Stop Loss** | **~11%** of fills | Price continues against you → cut loss at 0.50% (taker fee) |
| **Timeout** | **~1%** of fills | 60 min expires, close at market (taker fee) |

**Config 2 AGGRESSIVE (with displacement filter):**

| Exit Type | Frequency | What Happens |
|-----------|-----------|-------------|
| **Take Profit** | **~95-97%** of fills | Price bounces back 0.12% → you collect (maker fee) |
| **Timeout** | **~3-5%** of fills | 60 min expires, close at market (taker fee, avg loss ~1.25%) |

### Hold Time
- **Average: 1-3 minutes** (most trades resolve very quickly)
- 88-97% of trades hit TP within the first few minutes
- If a trade hasn't hit TP in 10 minutes, it's more likely to timeout

### Rules

1. **One position per symbol at a time.** Don't stack cascades.
2. **5-minute cooldown** between trades on the same symbol.
3. **Cancel unfilled limit orders after 60 minutes.**
4. **If position open and new cascade in same direction** → do nothing (already positioned).
5. **If position open and new cascade in opposite direction** → do nothing (let TP/SL handle it).

### TP/SL Implementation

**Option A: Exchange-native TP/SL (recommended)**
- Place TP and SL as conditional orders immediately after fill confirmation
- Bybit supports `TakeProfit` and `StopLoss` on the position itself

**Option B: Monitor and send market orders**
- Poll position every 1-2 seconds
- Send market close when price hits TP or SL level
- Slightly worse execution but simpler

---

## 5. When to Trade

### Trade ALL Hours

The v26k filter comparison showed that hour-of-day and day-of-week filters **hurt** total returns by over-filtering. The displacement ≥10 bps filter already captures the quality signal.

| Metric | All Hours + Disp Filter | US Hours Only + All Filters |
|--------|------------------------|----------------------------|
| Total (4 sym, AGGR) | **+91.4%** | +10.2% |
| Trades | 2,394 | 354 |
| Avg Sharpe | **+4.5** | +1.9 |

**Trade 24/7.** The displacement filter handles quality control.

### Avoid
- **Major news events** (FOMC, CPI) — cascades may not revert
- **Exchange maintenance windows**

---

## 6. Position Sizing

### Per-Trade Size
| Account Size | Per-Trade Size | Leverage | Margin Used |
|-------------|---------------|----------|-------------|
| $10K | $2K-5K notional | 5-10x | $200-1,000 |
| $50K | $10K-25K notional | 5-10x | $1K-5K |
| $100K | $20K-50K notional | 5-10x | $2K-10K |

### Why Small
- Config 1: Max loss per trade = 0.50% of position (SL) = $10-250 on $2K-50K
- Config 2: Max loss per trade = timeout at market (avg ~1.25%, worst ~5%)
- With displacement filter, only 3-12% of trades are losers
- Daily P&L variance is very low due to high win rate

### Risk Limits
- **Max 1 position per symbol** at any time
- **Max 3-4 symbols simultaneously** (to manage attention)
- **Daily loss limit: 1% of account** → stop trading for the day
- **Weekly loss limit: 2% of account** → review and recalibrate

---

## 7. Expected Performance

### Backtest Results (282 days, displacement ≥10 bps filter)

| Metric | Config 1 SAFE | Config 2 AGGR |
|--------|--------------|---------------|
| Trades (4 sym) | 2,394 | 2,394 |
| Win rate | 87.8% | **95.6%** |
| Total return | +72.1% | **+91.4%** |
| Avg Sharpe | +3.8 | **+4.5** |
| Positive months | 19/40 | **20/40** |
| Avg return/trade | +0.030% | +0.038% |

### Conservative Live Estimates (50% fill rate assumption)

| Metric | Config 1 SAFE | Config 2 AGGR |
|--------|--------------|---------------|
| Trades/day (4 sym) | ~4-8 fills | ~4-8 fills |
| Daily return | +0.06-0.13% | +0.08-0.16% |
| Monthly return | +1.5-3.5% | +2-4.5% |
| Annual return | +18-45% | +25-55% |
| Max drawdown | 3-6% | 4-7% |

These are **after fees, after assuming 50% of backtest fills actually execute live**.

---

## 8. Technical Implementation Skeleton

```
LOOP forever:
  1. Receive liquidation event from websocket
  2. Is notional ≥ P95 threshold?
     NO  → skip
     YES → add to cascade tracker

  3. Does current cascade have ≥ 2 events within 60s?
     NO  → wait for more
     YES → CASCADE DETECTED

  4. Compute displacement: abs(current_price - pre_cascade_price) / pre_cascade_price
     < 10 bps → skip (not a real cascade)
     ≥ 10 bps → continue

  5. Is cooldown expired? (5 min since last trade on this symbol)
     NO  → skip
     YES → continue

  6. Is there already an open position on this symbol?
     YES → skip
     NO  → continue

  7. Determine direction:
     buy_dominant → LIMIT BUY at price × 0.9985
     sell_dominant → LIMIT SELL at price × 1.0015

  8. Place PostOnly limit order

  9. Start 60-minute timer:
     - If filled → place TP (±0.15% safe / ±0.12% aggr) and SL (∓0.50% safe / none aggr)
     - If not filled after 60 min → cancel order
     - If TP or SL hit → position closed, reset cooldown
     - If 60 min after fill and no TP/SL → close at market
```

### Key Libraries
- **WebSocket:** `websockets` or `aiohttp` for Bybit stream
- **REST API:** `pybit` (official Bybit SDK) or `ccxt`
- **Scheduling:** `asyncio` for concurrent symbol monitoring

---

## 9. Checklist Before Going Live

### Paper Trade First (1-2 weeks)
- [ ] Websocket connection stable for 24h+
- [ ] Cascade detection matches expected frequency (5-9/day/symbol)
- [ ] Displacement filter working (≥10 bps check)
- [ ] Orders placed within 3 seconds of cascade detection
- [ ] Fill rate is 50-80% (if much lower, offset may be too wide)
- [ ] Win rate is 85-95% on filled trades (with displacement filter)
- [ ] No bugs in TP/SL logic (check both long and short)
- [ ] Cooldown and position limits working correctly
- [ ] Logging captures every event for post-analysis

### Go Live (small size)
- [ ] Start with 1 symbol (DOGE or SOL)
- [ ] Minimum position size for 1 week
- [ ] Compare live fills vs paper fills
- [ ] Check actual maker fee charged (should be 0-2 bps)
- [ ] Verify TP/SL execution quality

### Scale Up
- [ ] Add 2nd symbol after 1 profitable week
- [ ] Add 3rd/4th after 2 profitable weeks
- [ ] Increase size only after 1 profitable month
- [ ] Never increase size after a losing streak (that's when you review)

---

## 10. What Can Go Wrong

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Cascade doesn't revert** (trend continues) | 4-12% of trades | -0.50% (SL) or timeout | SL/timeout handles this; it's priced in |
| **Order doesn't fill** | 20-50% of signals | Missed opportunity | Acceptable; unfilled = no risk |
| **Websocket disconnects** | Occasional | Miss cascades | Auto-reconnect, heartbeat monitoring |
| **Exchange API down** | Rare | Can't place/cancel orders | Position size limits cap exposure |
| **Bybit removes liq feed** | Low but possible | Strategy dies | Monitor announcements; have backup exchange |
| **Strategy gets crowded** | Medium-term risk | Fill rates drop, edge shrinks | Monitor fill rates monthly; if <40%, reduce size |
| **Flash crash / black swan** | Rare | SL may slip | Never use >10x leverage; daily loss limit |
| **Fee structure changes** | Possible | Edge may shrink | Strategy works at 0-2 bps maker; monitor fee tier |

---

## Quick Reference Card

```
┌──────────────────────────────────────────────────────┐
│  LIQUIDATION CASCADE MARKET-MAKING                    │
├──────────────────────────────────────────────────────┤
│  TRIGGER: 2+ P95 liquidations within 60 sec          │
│  FILTER:  Displacement ≥10 bps (THE key filter)       │
│  SPEED:   Place order within 3 seconds                │
│                                                        │
│  CONFIG 1 SAFE (start here):                          │
│    Entry offset:  0.15%    TP: 0.15%                  │
│    Stop loss:     0.50%    Max hold: 60 min            │
│    Expected: 87.8% WR, +72.1% combined (282d)         │
│                                                        │
│  CONFIG 2 AGGRESSIVE (once proven):                   │
│    Entry offset:  0.15%    TP: 0.12%                  │
│    Stop loss:     NONE     Max hold: 60 min            │
│    Expected: 95.6% WR, +91.4% combined (282d)         │
│                                                        │
│  SYMBOLS: ETH, SOL, DOGE, XRP (all 4)                │
│  DIRECTION:                                            │
│    Longs liquidated (Buy-dom) → LIMIT BUY below       │
│    Shorts liquidated (Sell-dom) → LIMIT SELL above    │
│  COOLDOWN: 5 min between trades per symbol             │
│  HOURS: ALL (displacement filter handles quality)      │
└──────────────────────────────────────────────────────┘
```
