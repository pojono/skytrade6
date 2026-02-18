# Liquidation Cascade Market-Making — Trading Guide

**Strategy:** Place limit orders INTO liquidation cascades, collect the mean-reversion bounce.
**Edge:** Forced liquidations create temporary price dislocations; limit orders capture the snap-back.
**Backtest:** 282 days, 5 symbols, +20-62% total return (0% maker), +20-33% (2 bps maker).

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

### Symbols (ranked by backtest Sharpe)
| Priority | Symbol | Why |
|----------|--------|-----|
| 1 | **DOGEUSDT** | Sharpe 172, max DD 1.4%, 75% WR |
| 2 | **SOLUSDT** | Sharpe 164, max DD 1.9%, 74% WR |
| 3 | **XRPUSDT** | Sharpe 144, max DD 2.0%, 72% WR |
| 4 | **ETHUSDT** | Sharpe 139, max DD 3.0%, 72% — highest absolute return |
| 5 | BTCUSDT | Sharpe 46, needs different config, thinner edge |

**Start with DOGE + SOL + ETH.** Add XRP once comfortable.
**Skip BTC** unless you have 0% maker — the edge is too thin.

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

**Use these exact settings (optimal for ETH/SOL/DOGE/XRP):**

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| **Order type** | `Limit` with `PostOnly=true` | Guarantees maker fee, rejects if would cross |
| **Entry offset** | **0.15%** below/above current price | Far enough to capture dislocation, close enough to fill |
| **Take profit** | **0.15%** from fill price | Quick exit, captures the bounce |
| **Stop loss** | **0.25%** from fill price | Wider than TP — lets losers breathe |
| **Max hold** | **30 minutes** | Cancel unfilled orders + close position if no TP/SL |
| **Time in force** | `GTC` (cancel manually after 30 min) | Or use exchange's built-in time-in-force |
| **Reduce only (TP/SL)** | Yes | Prevent accidental position increase |

### BTC Exception
BTC needs wider parameters due to larger cascade dynamics:

| Parameter | BTC Value |
|-----------|-----------|
| Entry offset | **0.10%** |
| Take profit | **0.50%** |
| Stop loss | **0.25%** |

### Concrete Example: Buy-Dominant Cascade on ETHUSDT

```
Current ETH price: $2,500.00
Cascade detected: Buy-dominant (longs liquidated, price dropping)

Your action: LIMIT BUY

  Limit price:  $2,500.00 × (1 - 0.0015) = $2,496.25
  TP price:     $2,496.25 × (1 + 0.0015) = $2,500.00  (basically back to pre-cascade)
  SL price:     $2,496.25 × (1 - 0.0025) = $2,490.01

  Risk:   $2,496.25 → $2,490.01 = -$6.24 per ETH  (0.25%)
  Reward: $2,496.25 → $2,500.00 = +$3.75 per ETH  (0.15%)
  R:R = 0.6:1 (but 73% win rate makes this very profitable)
```

### Concrete Example: Sell-Dominant Cascade on SOLUSDT

```
Current SOL price: $180.00
Cascade detected: Sell-dominant (shorts liquidated, price spiking)

Your action: LIMIT SELL (short)

  Limit price:  $180.00 × (1 + 0.0015) = $180.27
  TP price:     $180.27 × (1 - 0.0015) = $180.00
  SL price:     $180.27 × (1 + 0.0025) = $180.72

  Risk:   $180.27 → $180.72 = -$0.45 per SOL  (0.25%)
  Reward: $180.27 → $180.00 = +$0.27 per SOL  (0.15%)
```

---

## 4. Position Management (After Fill)

### Expected Outcomes

| Exit Type | Frequency | What Happens |
|-----------|-----------|-------------|
| **Take Profit** | **~73%** of fills | Price bounces back 0.15% → you collect |
| **Stop Loss** | **~25%** of fills | Price continues against you → cut loss at 0.25% |
| **Timeout** | **~2%** of fills | 30 min expires, close at market |

### Hold Time
- **Average: 1-3 minutes** (most trades resolve very quickly)
- 73% of trades hit TP within the first few minutes
- If a trade hasn't hit TP in 5 minutes, it's more likely to hit SL or timeout

### Rules

1. **One position per symbol at a time.** Don't stack cascades.
2. **5-minute cooldown** between trades on the same symbol.
3. **Cancel unfilled limit orders after 30 minutes.**
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

### Best Hours: 13:00-18:00 UTC (US Session)

| Metric | All Hours | US Hours Only |
|--------|-----------|---------------|
| Sharpe (DOGE) | 172 | **185** |
| Sharpe (SOL) | 164 | **159** |
| Max DD (DOGE) | 1.4% | **0.9%** |
| Max DD (SOL) | 1.9% | **1.7%** |
| Trades/day | 5-9 | 2-3 |

US hours have the **highest quality** cascades (more volume, cleaner reversions). If you can only trade part of the day, trade 13-18 UTC.

### Avoid
- **Low-volume weekends** — fewer cascades, wider spreads
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
- Max loss per trade = 0.25% of position = $5-125 on $2K-50K
- With 25% loss rate, expect ~1-2 losses per day per symbol
- Daily P&L variance is very low due to small per-trade risk

### Risk Limits
- **Max 1 position per symbol** at any time
- **Max 3-4 symbols simultaneously** (to manage attention)
- **Daily loss limit: 1% of account** → stop trading for the day
- **Weekly loss limit: 2% of account** → review and recalibrate

---

## 7. Expected Performance

### Conservative Estimates (with 2 bps maker fee, 50% fill rate assumption)

| Metric | Per Symbol | 4-Symbol Portfolio |
|--------|-----------|-------------------|
| Trades/day | 2-4 fills | 8-16 fills |
| Win rate | 70-75% | 70-75% |
| Avg return/trade | +0.02-0.03% | +0.02-0.03% |
| Daily return | +0.04-0.12% | +0.16-0.48% |
| Monthly return | +1-3% | +4-10% |
| Annual return | +12-40% | +50-120% |
| Max drawdown | 2-4% | 3-6% |
| Sharpe (annual) | 2-5 | 3-8 |

These are **after fees, after assuming 50% of backtest fills actually execute**.

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

  4. Is cooldown expired? (5 min since last trade on this symbol)
     NO  → skip
     YES → continue

  5. Is there already an open position on this symbol?
     YES → skip
     NO  → continue

  6. Determine direction:
     buy_dominant → LIMIT BUY at price × 0.9985
     sell_dominant → LIMIT SELL at price × 1.0015

  7. Place PostOnly limit order

  8. Start 30-minute timer:
     - If filled → place TP (±0.15%) and SL (∓0.25%) orders
     - If not filled after 30 min → cancel order
     - If TP or SL hit → position closed, reset cooldown
     - If 30 min after fill and no TP/SL → close at market
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
- [ ] Orders placed within 3 seconds of cascade detection
- [ ] Fill rate is 50-80% (if much lower, offset may be too wide)
- [ ] Win rate is 65-75% on filled trades
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
| **Cascade doesn't revert** (trend continues) | 25% of trades | -0.25% per trade | SL handles this; it's priced in |
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
┌─────────────────────────────────────────────────┐
│  LIQUIDATION CASCADE MARKET-MAKING              │
├─────────────────────────────────────────────────┤
│  TRIGGER: 2+ P95 liquidations within 60 sec     │
│  SPEED:   Place order within 3 seconds           │
│                                                   │
│  ETH/SOL/DOGE/XRP:                               │
│    Entry offset:  0.15%                           │
│    Take profit:   0.15%                           │
│    Stop loss:     0.25%                           │
│    Max hold:      30 min                          │
│    Cooldown:      5 min                           │
│                                                   │
│  BTC (if 0% maker):                              │
│    Entry offset:  0.10%                           │
│    Take profit:   0.50%                           │
│    Stop loss:     0.25%                           │
│                                                   │
│  DIRECTION:                                       │
│    Longs liquidated (Buy-dom) → LIMIT BUY below  │
│    Shorts liquidated (Sell-dom) → LIMIT SELL above│
│                                                   │
│  EXPECTED:                                        │
│    Win rate:  ~73%                                │
│    Avg hold:  1-3 min                             │
│    Trades:    5-9 signals/day/symbol              │
│    Fills:     50-80% of signals                   │
│    Best hours: 13-18 UTC                          │
└─────────────────────────────────────────────────┘
```
