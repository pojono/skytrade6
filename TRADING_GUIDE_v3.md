# Liquidation Cascade Strategy — Step-by-Step Execution Guide v3

**Updated:** Feb 19, 2026 (includes trailing stop research)  
**Exchange:** Bybit Linear Perpetual  
**Symbols:** DOGEUSDT, SOLUSDT, ETHUSDT, XRPUSDT

---

## The Idea in 30 Seconds

Large forced liquidations temporarily push price too far. You place a limit order to catch the overshoot, then exit when price snaps back. A trailing stop locks in profit as soon as the bounce starts fading.

---

## What You Need Running

1. **WebSocket connection** to `wss://stream.bybit.com/v5/public/linear`
   - Subscribe to liquidation stream for each symbol
   - Subscribe to ticker stream for each symbol (real-time price)
2. **Bybit API keys** with futures trading permission
3. **State per symbol:**
   - Rolling buffer of last ~1000 liquidation notionals → compute P95 threshold
   - Current best bid/ask price (from ticker WS)
   - Last trade timestamp (for cooldown)
   - Open position flag

---

## Step-by-Step: What Happens on Each Trade

### STEP 1 — Detect a Large Liquidation (continuous)

```
Every liquidation event from WebSocket:
  notional = qty × price
  if notional >= P95_threshold:
    → this is a "large" liquidation, go to STEP 2
  else:
    → ignore
```

**P95 threshold** = 95th percentile of the last ~1000 liquidation notionals for this symbol. Update hourly.

Typical P95 values: ETH ~$80K, SOL ~$15K, DOGE ~$10K, XRP ~$12K.

---

### STEP 2 — Check Displacement (instant)

As soon as a P95 liquidation fires, check how much price has moved:

```
displacement_bps = abs(current_price - price_1min_ago) / price_1min_ago × 10000

if displacement_bps >= 10:
  → real cascade, go to STEP 3
else:
  → noise, SKIP
```

This is the single most important filter. It separates real forced-selling events from routine liquidations.

**You do NOT need to wait for a second liquidation.** A single P95 event with ≥10 bps displacement is enough.

---

### STEP 3 — Pre-flight Checks (instant)

```
if position_already_open on this symbol:  → SKIP
if last_trade on this symbol < 5 min ago: → SKIP (cooldown)
→ all clear, go to STEP 4
```

---

### STEP 4 — Determine Direction & Place Limit Order

**Timeline: T+0s to T+1s after detection**

```
Liquidation side = "Buy"  → longs were liquidated → price dropped
  YOUR ACTION: LIMIT BUY at current_price × 0.9985  (0.15% below market)

Liquidation side = "Sell" → shorts were liquidated → price spiked
  YOUR ACTION: LIMIT SELL at current_price × 1.0015  (0.15% above market)
```

**Order parameters:**
| Field | Value |
|-------|-------|
| Order type | `Limit` |
| PostOnly | `true` (guarantees maker fee, rejects if would cross book) |
| Time in force | `GTC` |
| Qty | your position size |

**Example — ETHUSDT, longs liquidated, price dropped to $2,500:**
```
LIMIT BUY at $2,500 × 0.9985 = $2,496.25
```

**Do NOT place TP/SL yet.** Wait for fill first.

---

### STEP 5 — Wait for Fill

**Timeline: T+1s to T+60s**

The cascade is still pushing price. Your limit order sits in the book. Price may or may not reach it.

```
if filled:
  → record fill_price, go to STEP 6
if 60 seconds pass without fill:
  → CANCEL the order, done
```

Most fills happen within 2-15 seconds. If not filled in 60s, the cascade wasn't strong enough.

---

### STEP 6 — Start Trailing Stop (immediately after fill)

**Timeline: T+fill onwards**

This is the key exit mechanism. No fixed take-profit. Instead, track the best price since fill and exit when price reverses by `trail_width`.

```
trail_width = 5 bps  (0.05%, conservative for live)
peak_price  = fill_price  (initialize)

LOOP every tick (or every 100-500ms):
  
  if LONG position:
    peak_price = max(peak_price, current_bid)
    trail_level = peak_price × (1 - trail_width/10000)
    if current_bid <= trail_level:
      → EXIT: send MARKET SELL, go to STEP 7

  if SHORT position:
    peak_price = min(peak_price, current_ask)  
    trail_level = peak_price × (1 + trail_width/10000)
    if current_ask >= trail_level:
      → EXIT: send MARKET BUY, go to STEP 7
```

**What this does in practice:**

```
Example: LONG ETH from $2,496.25, trail = 5 bps ($1.25)

T+0s   fill at $2,496.25  peak=$2,496.25  trail=$2,495.00
T+2s   price=$2,497.50    peak=$2,497.50  trail=$2,496.25  (trail moves up)
T+5s   price=$2,498.75    peak=$2,498.75  trail=$2,497.50  (trail moves up)
T+8s   price=$2,498.00    peak=$2,498.75  trail=$2,497.50  (price dips, trail stays)
T+10s  price=$2,497.40    peak=$2,498.75  trail=$2,497.50  (approaching trail)
T+11s  price=$2,497.30    → TRAIL HIT → MARKET SELL at ~$2,497.30

Profit: $2,497.30 - $2,496.25 = +$1.05 (+0.042%)
Fees:   maker entry + taker exit = 0.02% + 0.055% = 0.075%
Net:    +0.042% - 0.075% = -0.033%  ← small loss on this one

But when the bounce is bigger:
T+5s   price=$2,499.00    peak=$2,499.00  trail=$2,497.75
T+15s  price=$2,497.60    → TRAIL HIT → MARKET SELL at ~$2,497.60

Profit: $2,497.60 - $2,496.25 = +$1.35 (+0.054%)
Net:    +0.054% - 0.075% = -0.021%  ← still small loss

And when the bounce is strong (most trades):
T+3s   price=$2,500.00    peak=$2,500.00  trail=$2,498.75
T+8s   price=$2,498.50    → TRAIL HIT → MARKET SELL at ~$2,498.50

Profit: $2,498.50 - $2,496.25 = +$2.25 (+0.090%)
Net:    +0.090% - 0.075% = +0.015% net profit
```

---

### STEP 7 — After Exit

```
record: fill_price, exit_price, pnl, exit_reason, hold_time
set cooldown = now + 5 minutes for this symbol
set position_open = false
```

---

### STEP 8 — Safety Timeout (parallel to Step 6)

Even with trailing stop, keep a maximum hold time as a safety net:

```
if 60 minutes since fill AND position still open:
  → EXIT at market immediately
  → log as "timeout" (should be very rare with trailing stop)
```

With a 5 bps trail, timeouts should be near-zero. They only happen if price goes against you immediately after fill and never bounces even 5 bps.

---

## Timeline Summary — One Complete Trade

```
TIME        EVENT                           ACTION
─────────── ─────────────────────────────── ──────────────────────────────
T+0.0s      P95 liquidation arrives on WS   Check displacement
T+0.1s      Displacement ≥10 bps confirmed  Check cooldown & position
T+0.2s      All clear                       Compute limit price
T+0.5s      Send limit order to Bybit       Wait for fill
T+2-15s     Order fills                     Start trailing stop
T+2-300s    Trailing stop tracking           Update peak, check trail
T+5-300s    Price reverses by 5 bps         Market exit → DONE
─────────── ─────────────────────────────── ──────────────────────────────
Total: typically 10 seconds to 5 minutes per trade
```

---

## Trail Width Selection

| Trail Width | Backtest Return | Sharpe | Max DD | Win Rate | Notes |
|-------------|----------------|--------|--------|----------|-------|
| 3 bps | +227.6% | +34.1 | 0.18% | 91.6% | Theoretical best, may be too tight for live |
| **5 bps** | **+180.2%** | **+27.0** | **0.23%** | **88.2%** | **Recommended for live start** |
| 8 bps | +109.7% | +16.5 | 0.45% | 69.5% | Conservative, room for slippage |
| 10 bps | +64.2% | +9.7 | 1.32% | 51.3% | Very conservative |
| No trail (old: TP=12bps) | +159.7% | +28.2 | 1.62% | 98.1% | Old approach, timeout losses |

**Start with 5 bps.** If you see consistent slippage > 2 bps on exits, widen to 8 bps.

---

## Position Sizing

| Account Size | Position Size (notional) | Leverage | Worst-Case Loss/Trade |
|-------------|-------------------------|----------|----------------------|
| $5K | $1K–2.5K | 5–10x | $1–4 |
| $25K | $5K–12K | 5–10x | $5–20 |
| $100K | $20K–50K | 5–10x | $20–80 |

- **Max 1 position per symbol** at any time
- **Max 4 positions total** (one per symbol)
- **Never more than 20% of account in open positions**

With trailing stop, worst-case single trade is ~0.17% (vs 1.6% without trail). This is much safer.

---

## Kill Switches

| Trigger | Action |
|---------|--------|
| Daily loss > 1% of account | Stop trading for the day |
| Weekly loss > 3% of account | Stop trading, review logs |
| Fill rate < 40% for 3 days | Widen offset or pause |
| Win rate < 60% over 100 trades | Pause, investigate |
| WebSocket down > 2 min | Alert, auto-reconnect |
| 5 consecutive timeout exits | Something is wrong, pause |

---

## Order of Operations — Building the Bot

### Phase 0: Data Collection (1 day)
1. Connect to liquidation WS for all 4 symbols
2. Log every event to disk
3. Verify you're receiving data (~5-12 events/min per symbol)
4. Connect to ticker WS, log prices
5. Verify price updates are flowing (~10/sec per symbol)

### Phase 1: Signal Detection Only (3 days)
1. Implement P95 threshold calculation (rolling last 1000 events)
2. Implement displacement check (compare price now vs 1 min ago)
3. Log every signal: timestamp, symbol, side, displacement, notional
4. Verify: ~2-3 signals/day/symbol that pass the ≥10 bps filter
5. **Do NOT place any orders yet**

### Phase 2: Paper Trading (2 weeks)
1. When signal fires, compute limit price (0.15% offset)
2. Log the hypothetical order (don't send to exchange)
3. Track if price would have reached your limit (using WS ticker)
4. Track hypothetical trailing stop exits
5. Compare your paper results to backtest expectations:
   - Fill rate: expect 50-80%
   - Win rate: expect 80-90%
   - Avg PnL/trade: expect +0.04-0.08%

### Phase 3: Live — Minimum Size (2 weeks)
1. Start with **1 symbol** (ETHUSDT — most liquid, most trades)
2. **Minimum position size** ($100-500 notional)
3. Place real limit orders with PostOnly
4. Implement trailing stop on real fills
5. Log everything: fill price, peak price, trail exit price, slippage
6. **Key metric to watch:** actual slippage on trail exits (should be < 3 bps)

### Phase 4: Scale (ongoing)
1. Add 2nd symbol after 1 profitable week
2. Add 3rd/4th after 2 profitable weeks
3. Increase position size after 1 profitable month
4. **Never increase after a losing streak**

---

## Pseudocode — Complete Bot Logic

```python
# State per symbol
state = {
    'p95_threshold': 0,
    'liq_buffer': [],        # last 1000 liquidation notionals
    'last_trade_time': 0,
    'position': None,        # None, 'long', or 'short'
    'fill_price': 0,
    'peak_price': 0,
    'fill_time': 0,
    'pending_order_id': None,
}

TRAIL_WIDTH_BPS = 5
ENTRY_OFFSET = 0.0015       # 0.15%
COOLDOWN_SEC = 300           # 5 minutes
MAX_HOLD_SEC = 3600          # 60 minutes
DISP_THRESHOLD_BPS = 10

# ── On every liquidation event ──
def on_liquidation(symbol, side, qty, price):
    s = state[symbol]
    notional = qty * price
    
    # Update P95 threshold
    s['liq_buffer'].append(notional)
    if len(s['liq_buffer']) > 1000:
        s['liq_buffer'].pop(0)
    s['p95_threshold'] = percentile(s['liq_buffer'], 95)
    
    # Is this a large liquidation?
    if notional < s['p95_threshold']:
        return
    
    # Check displacement
    price_1min_ago = get_price_1min_ago(symbol)
    disp_bps = abs(price - price_1min_ago) / price_1min_ago * 10000
    if disp_bps < DISP_THRESHOLD_BPS:
        return
    
    # Pre-flight checks
    if s['position'] is not None:
        return
    if time.time() - s['last_trade_time'] < COOLDOWN_SEC:
        return
    
    # Place limit order
    current_price = get_current_price(symbol)
    if side == 'Buy':   # longs liquidated, price dropped
        limit_price = current_price * (1 - ENTRY_OFFSET)
        order_side = 'Buy'
    else:                # shorts liquidated, price spiked
        limit_price = current_price * (1 + ENTRY_OFFSET)
        order_side = 'Sell'
    
    order_id = place_limit_order(symbol, order_side, limit_price, post_only=True)
    s['pending_order_id'] = order_id
    
    # Cancel if not filled in 60 seconds
    schedule_cancel(order_id, delay=60)


# ── On order fill ──
def on_fill(symbol, fill_price, side):
    s = state[symbol]
    s['position'] = 'long' if side == 'Buy' else 'short'
    s['fill_price'] = fill_price
    s['peak_price'] = fill_price
    s['fill_time'] = time.time()
    s['pending_order_id'] = None


# ── On every price tick (while in position) ──
def on_tick(symbol, bid, ask):
    s = state[symbol]
    if s['position'] is None:
        return
    
    # Safety timeout
    if time.time() - s['fill_time'] > MAX_HOLD_SEC:
        close_at_market(symbol)
        s['position'] = None
        s['last_trade_time'] = time.time()
        return
    
    # Trailing stop logic
    if s['position'] == 'long':
        s['peak_price'] = max(s['peak_price'], bid)
        trail_level = s['peak_price'] * (1 - TRAIL_WIDTH_BPS / 10000)
        if bid <= trail_level:
            close_at_market(symbol)  # MARKET SELL
            s['position'] = None
            s['last_trade_time'] = time.time()
    
    elif s['position'] == 'short':
        s['peak_price'] = min(s['peak_price'], ask)
        trail_level = s['peak_price'] * (1 + TRAIL_WIDTH_BPS / 10000)
        if ask >= trail_level:
            close_at_market(symbol)  # MARKET BUY
            s['position'] = None
            s['last_trade_time'] = time.time()
```

---

## Expected Performance (Realistic)

| Metric | Backtest (5bps trail) | Realistic Live Estimate |
|--------|----------------------|------------------------|
| Trades/day (4 symbols) | ~8-10 signals | ~4-7 fills (50-70% fill rate) |
| Win rate | 88% | 75-85% |
| Avg PnL/trade | +0.076% | +0.03-0.06% |
| Max drawdown | 0.23% | 1-3% |
| Monthly return | ~6-8% | ~2-5% |
| Worst single trade | ~-0.17% | ~-0.5% |

**Expect 50-70% of backtest performance live.** The gap comes from: slippage on trail exits, partial fills, wider spreads during cascades, and order queue position.

---

## Quick Reference Card

```
DETECT:    P95 liquidation + displacement ≥ 10 bps
ENTRY:     Limit order 0.15% into the cascade (PostOnly)
EXIT:      Trailing stop 5 bps from peak (market order)
SAFETY:    60 min timeout (should almost never trigger)
COOLDOWN:  5 min per symbol
SYMBOLS:   ETH, SOL, DOGE, XRP
HOURS:     24/7
SIZING:    Max 1 position per symbol, max 20% of account exposed
```

---

*Sources: `liq_trailing_stop_research.py`, `liq_stress_test.py`, `FINDINGS_trailing_stop.md`, `CONFIDENCE_ASSESSMENT.md`*
