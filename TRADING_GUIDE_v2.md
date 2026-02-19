# Liquidation Cascade Market-Making — Trading Guide v2

**Last updated:** Feb 19, 2026  
**Data:** 282 days (2025-05-11 to 2026-02-17), 4 symbols  
**Fees:** Bybit maker=0.02%, taker=0.055%  
**Verified:** Numbers independently confirmed via `liq_verify.py`

---

## The Strategy in One Sentence

Place limit orders INTO liquidation cascades that moved price ≥10 bps, collect the mean-reversion bounce.

---

## 1. How It Works

Forced liquidations create temporary price dislocations. When enough traders get liquidated in a short window (a "cascade"), price overshoots — then snaps back. You place a limit order to catch the overshoot and profit from the snap-back.

**Your edge:** You're providing liquidity into forced selling. The cascade participants are forced to exit at any price — you get to name yours.

---

## 2. What You Need

- **Bybit Linear Perpetual account** with API keys (limit order capability)
- **Liquidation WebSocket feed:** `wss://stream.bybit.com/v5/public/linear`
  - Subscribe: `{"op": "subscribe", "args": ["liquidation.DOGEUSDT"]}`
- **Ticker/price feed** for computing displacement
- **Symbols:** ETHUSDT, SOLUSDT, DOGEUSDT, XRPUSDT (all 4)

---

## 3. Signal Detection

### Step 1: Track Large Liquidations

Maintain a rolling P95 notional threshold per symbol (from last ~1000 liquidation events). When a liquidation exceeds this threshold, it's "large."

### Step 2: Detect Signal

A single P95 liquidation event is enough to trigger. You don't need to wait for a cluster.

If multiple P95 events fire within 60 seconds, group them and use the last one as your signal timestamp. Track which side dominates:
- `buy_notional > sell_notional` → **Buy-dominant** (longs liquidated, price dropped)
- `sell_notional > buy_notional` → **Sell-dominant** (shorts liquidated, price spiked)
- Single event → use that event's side

### Step 3: Check Displacement ≥10 bps

```
displacement = abs(price_now - price_before_cascade) / price_before_cascade × 10000
```

- **≥10 bps → TRADE** (real cascade with genuine forced selling)
- **<10 bps → SKIP** (noise, not worth trading)

This single filter is the most important part of the strategy. It improves returns by +37% to +391% depending on config.

### Signal Frequency

| Symbol | P95 signals/day | After ≥10 bps filter |
|--------|----------------|---------------------|
| ETH | ~12 | ~3.0 |
| SOL | ~8 | ~2.4 |
| XRP | ~6 | ~1.7 |
| DOGE | ~5 | ~1.9 |

Expect ~2-3 actionable signals per day per symbol. Not all will fill.

---

## 4. Order Placement

### Speed Required

Act on the first P95 event. Don't wait for a second one.

```
T+0.0s   P95 liquidation arrives
T+0.5s   Compute displacement (≥10 bps?)
T+1.0s   Place limit order
T+1-60s  Wait for fill
```

No HFT infrastructure needed. A Python script with `websockets` + Bybit API is sufficient.

### Direction

| Cascade Type | What Happened | Your Action |
|-------------|---------------|-------------|
| **Buy-dominant** | Longs liquidated, price dropped | **LIMIT BUY** below market |
| **Sell-dominant** | Shorts liquidated, price spiked | **LIMIT SELL** above market |

### Order Parameters

| Parameter | Config 1: SAFE | Config 2: AGGRESSIVE |
|-----------|---------------|---------------------|
| Order type | `Limit` + `PostOnly=true` | `Limit` + `PostOnly=true` |
| Entry offset | **0.15%** from market | **0.15%** from market |
| Take profit | **0.15%** from entry | **0.12%** from entry |
| Stop loss | **0.50%** from entry | **None** |
| Max hold | 60 minutes | 60 minutes |
| Cooldown | 5 min per symbol | 5 min per symbol |

**Start with Config 1 SAFE.** Switch to Config 2 after proving the edge live.

### Example: Buy-Dominant Cascade on ETHUSDT

```
ETH price: $2,500.00
Cascade: Buy-dominant, displacement = 15 bps (≥10 → TRADE)

LIMIT BUY at $2,500 × 0.9985 = $2,496.25
TP at $2,496.25 × 1.0015 = $2,500.00
SL at $2,496.25 × 0.9950 = $2,483.77  (Config 1 only)

Win: +$3.75/ETH (0.15%)   → 88% of the time
Loss: -$12.48/ETH (0.50%) → 11% of the time
```

### Example: Sell-Dominant Cascade on SOLUSDT

```
SOL price: $180.00
Cascade: Sell-dominant, displacement = 22 bps (≥10 → TRADE)

LIMIT SELL at $180.00 × 1.0015 = $180.27
TP at $180.27 × 0.9988 = $180.05
No SL (Config 2)

Win: +$0.22/SOL (0.12%)   → 95% of the time
Loss: timeout at market after 60min → 5% of the time
```

---

## 5. After Fill — Position Management

### Exit Outcomes

**Config 1 SAFE:**

| Exit | Frequency | Fee | Net |
|------|-----------|-----|-----|
| Take Profit | **88%** | maker+maker = 0.04% | **+0.11%** |
| Stop Loss | **11%** | maker+taker = 0.075% | **-0.575%** |
| Timeout | **1%** | maker+taker = 0.075% | varies |

**Config 2 AGGRESSIVE:**

| Exit | Frequency | Fee | Net |
|------|-----------|-----|-----|
| Take Profit | **95-97%** | maker+maker = 0.04% | **+0.08%** |
| Timeout | **3-5%** | maker+taker = 0.075% | avg **-1.25%** |

### Rules

1. **One position per symbol at a time**
2. **5-minute cooldown** between trades on same symbol
3. **Cancel unfilled orders after 60 minutes**
4. If already positioned and new cascade fires → do nothing

### Hold Time

Most trades resolve in **1-3 minutes**. If not hit TP in 10 minutes, it's likely heading to timeout.

---

## 6. When to Trade

**Trade 24/7.** The displacement filter handles quality control.

We tested hour-of-day, day-of-week, and US-session-only filters. They all reduce total returns by cutting too many profitable trades:

| Approach | Combined Return (AGGR) | Trades |
|----------|----------------------|--------|
| **Displacement only (24/7)** | **+91.4%** | 2,394 |
| All filters combined | +31.3% | 1,064 |
| US-hours only + all filters | +10.2% | 354 |

---

## 7. Backtest Performance

### Config 1 SAFE — 282 days, displacement ≥10 bps

| Symbol | Trades | WR | Total Return | Sharpe | Max DD |
|--------|--------|----|-------------|--------|--------|
| DOGE | 539 | 88.7% | +18.9% | +4.3 | 2.2% |
| SOL | 670 | 88.2% | +22.2% | +4.5 | 2.5% |
| ETH | 852 | 87.8% | +25.0% | +4.4 | 4.7% |
| XRP | 477 | 87.8% | +15.0% | +3.4 | 5.0% |
| **Combined** | **2,538** | **88.1%** | **+81.0%** | **+4.2** | — |

### Config 2 AGGRESSIVE — 282 days, displacement ≥10 bps

| Symbol | Trades | WR | Total Return | Sharpe | Max DD |
|--------|--------|----|-------------|--------|--------|
| DOGE | 539 | 97.0% | +21.8% | +3.2 | 5.4% |
| SOL | 670 | 95.1% | +27.4% | +5.5 | 5.8% |
| ETH | 852 | 95.4% | +29.2% | +4.6 | 3.6% |
| XRP | 477 | 95.6% | +21.9% | +5.6 | 2.5% |
| **Combined** | **2,538** | **95.8%** | **+100.3%** | **+4.7** | — |

### Conservative Live Estimates (assume 50% of backtest fills)

| Metric | Config 1 SAFE | Config 2 AGGRESSIVE |
|--------|--------------|---------------------|
| Fills/day (4 symbols) | ~4-9 | ~4-9 |
| Daily return | +0.07-0.14% | +0.09-0.18% |
| Monthly return | +2-4% | +2.5-5% |
| Annual return | +20-50% | +30-60% |

---

## 8. Position Sizing

| Account | Position Size | Leverage | Max Loss/Trade |
|---------|-------------|----------|---------------|
| $10K | $2K-5K | 5-10x | $10-25 (Config 1) |
| $50K | $10K-25K | 5-10x | $50-125 |
| $100K | $20K-50K | 5-10x | $100-250 |

- **Config 1/3:** Max loss = 0.50% of position (SL)
- **Config 2:** Max loss = timeout at market (avg ~1.25%, worst case ~5%)
- **Max 1 position per symbol** at any time
- **Max 4 symbols simultaneously**
- **Never >5% of account in concurrent open positions**

---

## 9. Kill Switches

| Trigger | Action |
|---------|--------|
| Daily loss > 2% | Stop trading for the day |
| Weekly loss > 5% | Stop trading, review |
| Fill rate < 50% for 3 days | Pause, investigate |
| Win rate < 70% over 50+ trades | Pause, investigate |
| Websocket down > 5 min | Alert, auto-reconnect |

---

## 10. Implementation

```
LOOP forever:
  1. Receive liquidation event from websocket
  2. Is notional ≥ P95 threshold?  → NO: skip
  3. Is displacement ≥10 bps?  → NO: skip
  4. Is cooldown expired (5 min)?  → NO: skip
  5. Is position already open?  → YES: skip
  6. Place PostOnly limit order:
     - Buy-dominant → LIMIT BUY at price × 0.9985
     - Sell-dominant → LIMIT SELL at price × 1.0015
  7. On fill → place TP (and SL if Config 1)
  8. After 60 min → cancel unfilled order or close position at market
```

### Libraries
- **WebSocket:** `websockets` or `aiohttp`
- **REST API:** `pybit` (official Bybit SDK) or `ccxt`
- **Async:** `asyncio` for concurrent symbol monitoring

---

## 11. Go-Live Checklist

### Phase 1: Paper Trade (2 weeks)
- [ ] Websocket stable 24h+
- [ ] Cascade detection: ~5-9 signals/day/symbol
- [ ] Displacement filter working (≥10 bps)
- [ ] Orders placed within 3 seconds
- [ ] Fill rate 50-80%
- [ ] Win rate 85-95%
- [ ] TP/SL logic correct for both long and short
- [ ] Logging captures every event

### Phase 2: Live — Small Size (2 weeks)
- [ ] Start with 1 symbol (DOGE or ETH)
- [ ] Minimum position size
- [ ] Compare live vs paper fills
- [ ] Verify actual maker fee (should be ≤2 bps)

### Phase 3: Scale Up
- [ ] Add 2nd symbol after 1 profitable week
- [ ] Add 3rd/4th after 2 profitable weeks
- [ ] Increase size after 1 profitable month
- [ ] Never increase after a losing streak

---

## 12. Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Cascade doesn't revert | 4-12% of trades | SL (Config 1) or timeout handles it |
| Order doesn't fill | 20-50% of signals | No risk — unfilled = no exposure |
| Websocket disconnects | Occasional | Auto-reconnect + heartbeat |
| Bybit removes liq feed | Low | Monitor announcements |
| Strategy gets crowded | Medium-term | Monitor fill rates monthly |
| Flash crash | Rare | Daily loss limit + max leverage 10x |
| Fee structure changes | Possible | Works at 0-2 bps maker; monitor |

---

## Quick Reference

```
TRIGGER:  Single P95 liquidation (no need to wait for 2nd)
FILTER:   Displacement ≥10 bps
ENTRY:    Limit order at 0.15% offset (PostOnly)
TP:       0.15% (safe) / 0.12% (aggressive)
SL:       0.50% (safe) / none (aggressive)
HOLD:     60 min max
COOLDOWN: 5 min per symbol
SYMBOLS:  ETH, SOL, DOGE, XRP
HOURS:    24/7
```

| | Safe | Aggressive |
|---|---|---|
| WR | 88.1% | 95.8% |
| Return (282d) | +81.0% | +100.3% |
| Sharpe | +4.2 | +4.7 |

---

*Source: `FINDINGS_v26k_filter_comparison.md`, `ACTIONABLE_TOP3_CONFIGS.md`, `RESEARCH_INDEX_liquidation_strategy.md`*
