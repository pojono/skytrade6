# Production Implementation Guide: Volatility Dip-Buying Strategy

## What This Strategy Does (Plain English)

This strategy watches 9 crypto perpetual futures contracts on Bybit. Every hour, it checks if volatility has spiked abnormally AND price has moved sharply in one direction. When both conditions are met (which happens ~31 times/month across all 9 coins), it bets that price will **partially revert** over the next 4 hours. It enters a position (long or short), holds for exactly 4 hours, then exits. No trailing stops, no take-profits — just a fixed 4-hour hold.

**Why it works:** After a volatility spike + sharp move, crypto prices tend to overshoot and then partially revert. This is a well-known mean-reversion effect that persists because it's driven by liquidation cascades, stop-loss hunting, and emotional overreaction — structural features of leveraged crypto markets.

---

## Strategy Parameters (DO NOT CHANGE)

These parameters were validated across 50 symbols over 32 months. They are **fixed** — do not optimize them.

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Timeframe** | 1h candles | Use Bybit 1h klines |
| **Signal threshold** | 2.0 | Enter when \|combined\| > 2.0 |
| **Hold period** | 4 hours (4 bars) | Exit exactly 4 bars after entry |
| **Cooldown** | 4 hours (4 bars) | No new trade within 4 bars of last exit |
| **Fees budget** | 4 bps round-trip | Use limit orders (maker fees) both sides |
| **Leverage** | 1x–2x | 2x recommended for experienced traders |
| **Position sizing** | Equal weight (1/N) | Each symbol gets 1/9 of capital |

---

## Step 1: Choose Your Symbols

### Recommended Portfolio: Top 9 (Best Risk-Adjusted)

| # | Symbol | Sharpe | CAGR | Avg bps/trade | Why included |
|---|--------|--------|------|---------------|--------------|
| 1 | **ONDOUSDT** | +2.08 | +42.2% | +94.9 | Strongest Sharpe, high avg bps |
| 2 | **TAOUSDT** | +2.00 | +34.4% | +100.6 | Highest avg bps of any symbol |
| 3 | **SOLUSDT** | +1.67 | +22.5% | +50.8 | Longest track record, very consistent |
| 4 | **HBARUSDT** | +1.58 | +35.4% | +74.9 | Strong edge, 32 months of data |
| 5 | **SEIUSDT** | +1.54 | +32.3% | +73.8 | High avg bps, low correlation |
| 6 | **ADAUSDT** | +1.44 | +33.4% | +72.4 | Consistent performer |
| 7 | **BNBUSDT** | +1.37 | +14.4% | +30.2 | Lowest drawdown of all, stabilizer |
| 8 | **XRPUSDT** | +1.37 | +38.6% | +76.7 | High CAGR, lots of trades |
| 9 | **AAVEUSDT** | +1.32 | +26.7% | +53.5 | Good diversifier, 32 months data |

### Expected Portfolio Performance

| Metric | 1x Leverage | 2x Leverage |
|--------|-------------|-------------|
| **CAGR** | +30.0% | +66.5% |
| **Monthly Sharpe** | +2.50 | +2.50 |
| **Max Drawdown** | 3.1% | 6.2% |
| **Calmar Ratio** | 9.67 | 10.70 |
| **Trades/month** | ~31 | ~31 |
| **Positive months** | 78% | 78% |
| **Worst month** | -2.14% | -4.28% |

---

## Step 2: Signal Computation

### The Two Components

The signal has two parts that are averaged together:

#### Component 1: Realized Volatility Z-Score (`rvol_z`)

Measures whether current volatility is abnormally high compared to recent history.

```
1. Compute hourly returns in basis points:
   ret[i] = (close[i] - close[i-1]) / close[i-1] × 10000

2. Compute 24-hour realized volatility:
   rvol = rolling_std(ret, window=24, min_periods=8)

3. Compute trailing mean and std of rvol over 168 hours (1 week):
   rvol_mean = rolling_mean(rvol, window=168, min_periods=48)
   rvol_std  = rolling_std(rvol, window=168, min_periods=48)
   rvol_std  = max(rvol_std, 1e-8)  # prevent division by zero

4. Z-score:
   rvol_z = (rvol - rvol_mean) / rvol_std
```

**Interpretation:** `rvol_z > 2` means volatility is 2 standard deviations above its recent average — a volatility spike.

#### Component 2: Mean-Reversion Signal (`mr_4h`)

Measures whether price has moved sharply over the last 4 hours and bets on reversion.

```
1. Compute 4-hour cumulative return:
   r4 = rolling_sum(ret, window=4)

2. Compute trailing mean and std of hourly returns over 48 hours:
   r4_mean = rolling_mean(ret, window=48, min_periods=12) × 4
   r4_std  = rolling_std(ret, window=48, min_periods=12) × 2
   r4_std  = max(r4_std, 1e-8)

3. Z-score (INVERTED for mean-reversion):
   mr_4h = -((r4 - r4_mean) / r4_std)
```

**Interpretation:** If price dropped sharply (r4 very negative), `mr_4h` becomes positive → go long (bet on bounce). If price pumped sharply, `mr_4h` becomes negative → go short (bet on pullback).

#### Combined Signal

```
combined = (rvol_z + mr_4h) / 2
```

### Entry Rules

```
IF |combined| > 2.0 AND no position open AND cooldown expired:
    IF combined > 0: OPEN LONG
    IF combined < 0: OPEN SHORT
```

### Exit Rules

```
After exactly 4 hours (4 candle closes): CLOSE POSITION
```

No stop-loss. No take-profit. Just a fixed 4-hour hold.

---

## Step 3: Reference Implementation (Python)

This is the exact signal computation used in backtesting. Port this to your bot.

```python
import numpy as np
import pandas as pd

def compute_signal(closes: np.ndarray) -> float:
    """
    Compute the combined signal from an array of recent 1h close prices.
    
    Args:
        closes: Array of at least 200 recent 1h close prices (most recent last).
                More history = more stable signal. 200 is minimum.
    
    Returns:
        Combined signal value. Trade if |signal| > 2.0.
        Positive = go long, Negative = go short.
    """
    c = np.array(closes, dtype=np.float64)
    n = len(c)
    
    # Hourly returns in bps
    ret = np.zeros(n)
    ret[1:] = (c[1:] - c[:-1]) / c[:-1] * 10000
    ret_s = pd.Series(ret)
    
    # Component 1: Realized volatility z-score
    rvol = ret_s.rolling(24, min_periods=8).std()
    rvol_mean = rvol.rolling(168, min_periods=48).mean()
    rvol_std = rvol.rolling(168, min_periods=48).std().clip(lower=1e-8)
    rvol_z = ((rvol - rvol_mean) / rvol_std).iloc[-1]
    
    # Component 2: Mean-reversion 4h
    r4 = ret_s.rolling(4).sum()
    r4_mean = ret_s.rolling(48, min_periods=12).mean() * 4
    r4_std = ret_s.rolling(48, min_periods=12).std().clip(lower=1e-8) * 2
    mr_4h = -((r4 - r4_mean) / r4_std).iloc[-1]
    
    # Combined
    if np.isnan(rvol_z) or np.isnan(mr_4h):
        return 0.0
    
    return (rvol_z + mr_4h) / 2
```

---

## Step 4: Bot Architecture

### High-Level Flow

```
Every hour (on candle close):
│
├─ For each of 9 symbols:
│   │
│   ├─ Fetch latest 200+ hourly candles
│   ├─ Compute combined signal
│   │
│   ├─ IF position is open AND held for 4 bars:
│   │   └─ CLOSE position (limit order at mid-price)
│   │
│   ├─ IF no position AND cooldown expired AND |signal| > 2.0:
│   │   ├─ signal > 0 → OPEN LONG
│   │   └─ signal < 0 → OPEN SHORT
│   │   └─ Size = (total_equity / 9) × leverage
│   │
│   └─ Log: timestamp, symbol, signal, action, PnL
│
└─ Sleep until next hourly candle close
```

### Timing

- **Run at:** Every hour, ~5-10 seconds after the candle close (e.g., 00:00:05, 01:00:05, ...)
- **Why 5-10 seconds after:** Ensures the 1h candle is finalized on the exchange
- **Use cron or scheduler:** `0 * * * *` with a 5-second sleep at start

### Order Execution

| Action | Order Type | Notes |
|--------|-----------|-------|
| **Entry** | Limit order at best bid/ask | Post-only to guarantee maker fee |
| **Exit** | Limit order at best bid/ask | Post-only. If not filled in 30s, use market |
| **Fallback** | Market order | Only if limit not filled within tolerance |

**Fee budget:** The strategy assumes 4 bps round-trip (2 bps entry + 2 bps exit). Bybit maker fee is typically 1 bps, so you have margin. If you use market orders (taker = 5.5 bps), your edge shrinks significantly.

### Position Sizing

```
position_size_usd = (total_equity / 9) × leverage

Example with $10,000 at 2x:
  Per-symbol allocation = $10,000 / 9 = $1,111
  Position size = $1,111 × 2 = $2,222 per trade
```

**Important:** Only 1 position per symbol at a time. The strategy is NOT pyramiding.

---

## Step 5: Bybit API Integration

### Required API Endpoints

| Purpose | Endpoint | Method |
|---------|----------|--------|
| Get klines | `/v5/market/kline` | GET (public) |
| Get ticker | `/v5/market/tickers` | GET (public) |
| Place order | `/v5/order/create` | POST (auth) |
| Cancel order | `/v5/order/cancel` | POST (auth) |
| Get positions | `/v5/position/list` | GET (auth) |
| Get wallet | `/v5/account/wallet-balance` | GET (auth) |

### Fetching Klines

```python
import requests

def get_klines(symbol: str, limit: int = 200) -> list:
    """Fetch recent 1h klines from Bybit."""
    resp = requests.get("https://api.bybit.com/v5/market/kline", params={
        "category": "linear",
        "symbol": symbol,
        "interval": "60",
        "limit": limit,
    })
    data = resp.json()
    # Returns newest first, so reverse
    rows = data["result"]["list"][::-1]
    # Each row: [startTime, open, high, low, close, volume, turnover]
    closes = [float(r[4]) for r in rows]
    return closes
```

### Placing a Limit Order

```python
import hmac, hashlib, time, json, requests

API_KEY = "your_api_key"
API_SECRET = "your_api_secret"

def place_order(symbol: str, side: str, qty: float, price: float):
    """Place a limit order on Bybit."""
    timestamp = str(int(time.time() * 1000))
    params = {
        "category": "linear",
        "symbol": symbol,
        "side": side,        # "Buy" or "Sell"
        "orderType": "Limit",
        "qty": str(qty),
        "price": str(price),
        "timeInForce": "PostOnly",  # Maker only
    }
    
    param_str = timestamp + API_KEY + "5000" + json.dumps(params)
    sign = hmac.new(API_SECRET.encode(), param_str.encode(), hashlib.sha256).hexdigest()
    
    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-SIGN": sign,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": "5000",
        "Content-Type": "application/json",
    }
    
    resp = requests.post("https://api.bybit.com/v5/order/create",
                         headers=headers, json=params)
    return resp.json()
```

---

## Step 6: State Management

Your bot needs to track:

```python
# Per-symbol state
state = {
    "ONDOUSDT": {
        "position": None,       # None, "long", or "short"
        "entry_bar": None,      # Bar number when entered
        "entry_price": None,    # Entry price
        "last_exit_bar": 0,     # Bar number of last exit (for cooldown)
    },
    # ... repeat for all 9 symbols
}

# Global state
current_bar = 0  # Increments every hour
```

### State Persistence

Save state to a JSON file after every action. On restart, load from file.

```python
import json

STATE_FILE = "bot_state.json"

def save_state(state, current_bar):
    with open(STATE_FILE, 'w') as f:
        json.dump({"state": state, "bar": current_bar}, f)

def load_state():
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        return data["state"], data["bar"]
    except FileNotFoundError:
        return initialize_state(), 0
```

---

## Step 7: Main Loop

```python
import time
from datetime import datetime, timezone

SYMBOLS = [
    "ONDOUSDT", "TAOUSDT", "SOLUSDT", "HBARUSDT", "SEIUSDT",
    "ADAUSDT", "BNBUSDT", "XRPUSDT", "AAVEUSDT",
]
THRESHOLD = 2.0
HOLD_BARS = 4
COOLDOWN_BARS = 4
LEVERAGE = 2  # 1 or 2

def run_bot():
    state, current_bar = load_state()
    
    while True:
        # Wait for next hourly candle close
        now = datetime.now(timezone.utc)
        next_hour = now.replace(minute=0, second=10, microsecond=0)
        if next_hour <= now:
            next_hour += timedelta(hours=1)
        sleep_seconds = (next_hour - now).total_seconds()
        print(f"Sleeping {sleep_seconds:.0f}s until {next_hour}")
        time.sleep(sleep_seconds)
        
        current_bar += 1
        equity = get_account_equity()
        per_symbol_usd = equity / len(SYMBOLS)
        
        for symbol in SYMBOLS:
            try:
                process_symbol(symbol, state[symbol], current_bar,
                               per_symbol_usd, LEVERAGE)
            except Exception as e:
                print(f"ERROR {symbol}: {e}")
                # Log but don't crash — other symbols still need processing
        
        save_state(state, current_bar)
        print(f"Bar {current_bar} complete. Equity: ${equity:,.2f}")


def process_symbol(symbol, sym_state, current_bar, alloc_usd, leverage):
    """Process one symbol: check exits, then check entries."""
    
    # --- CHECK EXIT ---
    if sym_state["position"] is not None:
        bars_held = current_bar - sym_state["entry_bar"]
        if bars_held >= HOLD_BARS:
            close_position(symbol, sym_state)
            sym_state["last_exit_bar"] = current_bar
            sym_state["position"] = None
            sym_state["entry_bar"] = None
            sym_state["entry_price"] = None
            print(f"  {symbol}: CLOSED after {bars_held} bars")
            return  # Don't enter same bar as exit
    
    # --- CHECK ENTRY ---
    if sym_state["position"] is not None:
        return  # Already in a position
    
    if current_bar < sym_state["last_exit_bar"] + COOLDOWN_BARS:
        return  # Cooldown active
    
    # Compute signal
    closes = get_klines(symbol, limit=200)
    signal = compute_signal(closes)
    
    if abs(signal) <= THRESHOLD:
        return  # No signal
    
    # Enter trade
    side = "long" if signal > 0 else "short"
    size_usd = alloc_usd * leverage
    current_price = closes[-1]
    qty = size_usd / current_price
    
    open_position(symbol, side, qty, current_price)
    sym_state["position"] = side
    sym_state["entry_bar"] = current_bar
    sym_state["entry_price"] = current_price
    print(f"  {symbol}: OPENED {side.upper()} signal={signal:.2f} "
          f"price={current_price} size=${size_usd:.0f}")
```

---

## Step 8: Risk Management & Kill Switches

### Hard Rules

| Rule | Action |
|------|--------|
| **Max 1 position per symbol** | Never pyramid or add to positions |
| **Max portfolio drawdown > 10%** | Pause all trading for 24 hours |
| **Single trade loss > 3%** | Log warning (this is unusual) |
| **API errors > 3 in a row** | Pause symbol for 1 hour |
| **Equity < 80% of starting** | STOP bot, manual review required |

### Kill Switch Implementation

```python
STARTING_EQUITY = None  # Set on first run
MAX_DD_PCT = 10.0
EMERGENCY_DD_PCT = 20.0

def check_kill_switches(equity):
    global STARTING_EQUITY
    if STARTING_EQUITY is None:
        STARTING_EQUITY = equity
    
    dd = (1 - equity / STARTING_EQUITY) * 100
    
    if dd > EMERGENCY_DD_PCT:
        close_all_positions()
        raise SystemExit(f"EMERGENCY STOP: {dd:.1f}% drawdown")
    
    if dd > MAX_DD_PCT:
        print(f"WARNING: {dd:.1f}% drawdown — pausing 24h")
        time.sleep(86400)
```

### What Can Go Wrong

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| **Signal stops working** | Medium (regime change) | Monitor monthly Sharpe. If < 0 for 3 consecutive months, pause. |
| **Exchange downtime** | Low | Miss a trade — acceptable. Don't chase. |
| **Slippage on limit orders** | Medium | Use PostOnly. If not filled in 30s, market order. Budget 4 bps. |
| **Liquidation (at 2x)** | Very low | Max single-trade move needed: ~50%. Never happens in 4h. |
| **API key compromised** | Low | Use IP whitelist. Withdraw-disabled API key. |
| **Correlated crash (all 9 down)** | Low | Max portfolio loss in worst month was -2.14% at 1x. |

---

## Step 9: Monitoring & Logging

### What to Log Every Hour

```
2026-02-20 14:00:10 | Bar 1234 | Equity: $10,523.40
  ONDOUSDT: signal=-0.43 (no trade)
  TAOUSDT:  signal=+2.31 → LONG @ $178.50, size=$2,340
  SOLUSDT:  HOLDING long bar 2/4, entry=$84.20, unrealized +$12.30
  HBARUSDT: signal=+1.87 (below threshold)
  ...
```

### Weekly Health Check

Review these metrics every week:

1. **Win rate** — Should be 50-60%. Below 45% for 2 weeks = investigate.
2. **Avg bps/trade** — Should be +40-80 bps. Below +20 = investigate.
3. **Trades/week** — Should be ~7-8. Below 3 = market is quiet (normal). Above 15 = unusual.
4. **Fill rate** — What % of limit orders fill? Should be >90%. Below 80% = adjust price offset.
5. **Actual vs expected fees** — Should be ~2 bps per side. If higher, you're hitting taker.

### Monthly Review

| Check | Action if failing |
|-------|-------------------|
| Monthly return < -3% | Review trades. Is the signal still working? |
| 2 consecutive losing months | Reduce leverage to 1x |
| 3 consecutive losing months | Pause strategy, full review |
| Sharpe < 1.0 over trailing 6 months | Consider reducing symbols or pausing |

---

## Step 10: Deployment Checklist

### Before Going Live

- [ ] **Paper trade for 2 weeks** — Run the bot with logging but no real orders
- [ ] **Verify signal matches backtest** — Compare your bot's signals vs the research scripts on historical data
- [ ] **Test order execution** — Place and cancel a few limit orders manually
- [ ] **Set up API key** — Read-only + trade permissions only. No withdraw. IP whitelist.
- [ ] **Set up monitoring** — Alerts for errors, daily PnL summary
- [ ] **Start with 1x leverage** — Only move to 2x after 1 month of live results matching backtest
- [ ] **Start with small capital** — $1,000-$5,000 first. Scale up after 1-2 months.

### Infrastructure

| Component | Recommendation |
|-----------|---------------|
| **Server** | VPS close to Bybit (Singapore or Tokyo), $5-10/month |
| **Language** | Python 3.10+ with `requests`, `numpy`, `pandas` |
| **Scheduler** | `cron` or `APScheduler` — run every hour |
| **Logging** | File-based + optional Telegram alerts |
| **State** | JSON file (simple) or SQLite (robust) |
| **Monitoring** | Telegram bot for trade alerts and daily summary |

### Telegram Alert Example

```python
import requests

TELEGRAM_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"

def send_alert(message: str):
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                  json={"chat_id": TELEGRAM_CHAT_ID, "text": message})

# Usage
send_alert(f"🟢 LONG TAOUSDT @ $178.50\nSignal: +2.31\nSize: $2,340")
send_alert(f"📊 Daily PnL: +$45.20 (+0.43%)\nOpen positions: 2")
```

---

## Step 11: Projected Returns by Capital

### At 1x Leverage (Conservative)

| Starting Capital | Monthly | Yearly | Max DD $ |
|-----------------|---------|--------|----------|
| $5,000 | $125 | $1,500 | $155 |
| $10,000 | $250 | $3,000 | $310 |
| $25,000 | $625 | $7,500 | $776 |
| $50,000 | $1,250 | $15,000 | $1,552 |
| $100,000 | $2,500 | $30,000 | $3,104 |

### At 2x Leverage (Recommended)

| Starting Capital | Monthly | Yearly | Max DD $ |
|-----------------|---------|--------|----------|
| $5,000 | $277 | $3,323 | $310 |
| $10,000 | $554 | $6,646 | $621 |
| $25,000 | $1,384 | $16,614 | $1,552 |
| $50,000 | $2,769 | $33,228 | $3,104 |
| $100,000 | $5,538 | $66,455 | $6,209 |

**Note:** These are based on backtested performance. Live performance will likely be 20-30% lower due to slippage, missed fills, and market regime changes. Plan conservatively.

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────┐
│  VOL DIP-BUYING STRATEGY — QUICK REFERENCE          │
├─────────────────────────────────────────────────────┤
│                                                     │
│  SYMBOLS: ONDO TAO SOL HBAR SEI ADA BNB XRP AAVE   │
│  EXCHANGE: Bybit Linear Perpetual (USDT)            │
│                                                     │
│  SIGNAL: combined = (rvol_z + mr_4h) / 2            │
│  ENTRY:  |combined| > 2.0                           │
│  DIRECTION: positive → long, negative → short       │
│  HOLD:   4 hours (4 candles)                        │
│  COOLDOWN: 4 hours after exit                       │
│  SIZING: equal weight (1/9 per symbol)              │
│  LEVERAGE: 1x (safe) or 2x (recommended)            │
│  ORDERS: limit (PostOnly) for maker fees            │
│                                                     │
│  EXPECTED: ~31 trades/month, 78% positive months    │
│  CAGR: +30% (1x) or +67% (2x)                      │
│  MAX DD: 3.1% (1x) or 6.2% (2x)                    │
│                                                     │
│  KILL SWITCH: Stop if DD > 10% or 3 losing months   │
└─────────────────────────────────────────────────────┘
```

---

## Files in This Strategy Folder

| File | Purpose |
|------|---------|
| `PRODUCTION_GUIDE.md` | This file — step-by-step implementation guide |
| `SELF_AUDIT.md` | Honest pros/cons assessment |
| `FINDINGS_COMPREHENSIVE.md` | Detailed backtest results and analysis |
| `BROAD_VALIDATION_20COINS.md` | Original 20-coin validation |
| `top50_validation.py` | Walk-forward validation across 50 symbols |
| `final_portfolio_analysis.py` | Portfolio comparison: 6 vs 9 vs 12 vs 14 coins |
| `equity_curves_analysis.py` | Equity curves, drawdowns, heatmaps |
| `leverage_analysis.py` | Impact of 1x–10x leverage |
| `trade_frequency_analysis.py` | Trades per day/week/month analysis |
| `compounding_analysis.py` | More symbols vs compounding tradeoff |
| `charts/` | All generated charts and visualizations |
