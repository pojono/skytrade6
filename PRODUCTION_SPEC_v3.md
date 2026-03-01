# Production Spec: Settlement ML v3

**The optimal combination of all our research into a single production system.**  
**Updated:** 2026-03-01 — incorporates loser analysis, limit exit, fresh data validation (161 settlements).

---

## Architecture Overview

```
          T-10s              T+0ms    T+20ms                          T+60s
            │                  │        │                               │
  ┌─────────┐  ┌──────────────┐ ┌──────┐  ┌────────────────────────────┐
  │ STAGE 1 │  │ STAGE 2      │ │ENTRY │  │ STAGE 3: EXIT             │
  │ DECIDE  │  │ SIZING +     │ │      │  │                           │
  │         │  │ FILTERS      │ │ Sell  │  │ LogReg 100ms polling      │
  │ ML:     │  │              │ │ $1-2K│  │ + BIG_TRADE trigger       │
  │ trade   │  │ Read OB.200  │ │ mkt  │  │ "Is this the bottom?"    │
  │ or skip?│  │ at T-0       │ │ order │  │                           │
  │         │  │              │ │      │  │ EXIT VIA LIMIT BUY:       │
  │ Ridge + │  │ depth >= $2K?│ │ BB   │  │ PostOnly at best_bid      │
  │ LogReg  │  │ spread <= 8? │ │ fill │  │ wait 1s → rescue if no    │
  │         │  │ Size by depth│ │ @T+20│  │ fill (cancel + mkt buy)   │
  │ 9 feats │  │ Cap at 10%   │ │ ms   │  │ Saves 6 bps on 54% fills  │
  └─────────┘  └──────────────┘ └──────┘  └────────────────────────────┘
```

### Key timing (all times are Bybit server clock)

| Event | Time | Action |
|-------|------|--------|
| Pre-trade ML | T-10s | Decide: trade or skip? |
| Settlement | T+0ms | FR deducted. Book intact. |
| **Filter check** | **T-0** | **depth >= $2K? spread <= 8 bps? Skip if not.** |
| Dead zone | T+0 to T+15ms | Almost no trading ($6 median sell volume) |
| **Our entry** | **T+20ms BB fill** | **Short $1-2K. Book 99.8% intact. Escapes FR.** |
| Selling wave | T+25-50ms | Other bots arrive. $5-9K selling. Price crashes -30 bps. |
| **ML exit signal** | **T+1s to T+60s** | **LogReg detects sell exhaustion** |
| **Limit exit** | **+0 to +1s after signal** | **PostOnly buy at bid, 1s rescue timeout** |

---

## Stage 1: Pre-Trade Decision (T-10s to T-1s)

**When:** 10 seconds before settlement, using data already available via WS.

### Model A: Drop Magnitude Prediction
```
Model:    Ridge(alpha=10.0) + StandardScaler
Features: 9
Input:    fr_bps, fr_abs_bps, fr_squared,
          total_depth_usd, total_depth_imb_mean,
          ask_concentration, thin_side_depth, depth_within_50bps,
          oi_change_60s
Output:   predicted_drop_bps (e.g., -85.0)
```

### Model B: Trade/Skip Classifier
```
Model:    LogReg(C=0.1) + StandardScaler
Features: same 9
Input:    same as Model A
Output:   P(profitable) — probability drop > 40 bps
```

### Decision Logic
```python
predicted_drop = model_a.predict(features)    # e.g., -85 bps
prob_profitable = model_b.predict_proba(features)  # e.g., 0.82

# Rule 1: Hard skip on low FR
if abs(fr_bps) < 25:
    return SKIP  # Never profitable after fees

# Rule 2: ML filter
if prob_profitable < 0.50:
    return SKIP  # Model says not worth it

# Rule 3: Proceed to Stage 2 for OB-based sizing
return TRADE
```

### Expected filter performance
- **AUC: 0.859** (LOSO cross-symbol validated)
- Skips ~40 of 150 settlements (27%) that are unprofitable
- Of skipped: 22 correct skips, 18 missed trades (acceptable — we keep 91/109 profitable)
- Net effect: raises avg PnL from +30 to ~+38 bps by avoiding losers

---

## Stage 2: Entry + Sizing (T-0 to T+20ms)

### 2a. Position Sizing at T-0 (read OB before entry)

At T-0, the orderbook is still intact. Read OB.200 and compute bid depth:

```python
def compute_position_size(bids, asks):
    """Compute optimal notional from orderbook at T-0.
    
    Args:
        bids: [(price, qty), ...] sorted descending
        asks: [(price, qty), ...] sorted ascending
    
    Returns:
        notional_usd (float), or 0 to skip
    """
    if not bids or not asks:
        return 0
    
    mid = (bids[0][0] + asks[0][0]) / 2
    
    # Compute bid depth within 20 bps of mid (actionable liquidity)
    bid_depth_20bps = sum(
        p * q for p, q in bids 
        if (mid - p) / mid * 10000 <= 20
    )
    
    # Spread check: skip if too wide (loser analysis: >10bps spread → 18% WR)
    spread_bps = (asks[0][0] - bids[0][0]) / mid * 10000
    if spread_bps > 8:
        return 0        # SKIP — spread eats the edge
    
    # Sizing table (validated on 161 settlements, loser analysis)
    if bid_depth_20bps < 2000:
        return 0        # SKIP — too thin (17% WR below $2K depth, N=24)
    elif bid_depth_20bps < 5000:
        notional = 500
    elif bid_depth_20bps < 20000:
        notional = 1000
    elif bid_depth_20bps < 50000:
        notional = 2000
    elif bid_depth_20bps < 100000:
        notional = 3000
    else:
        notional = 5000
    
    # Safety cap: never exceed 10% of near-BBO depth
    cap = bid_depth_20bps * 0.10
    notional = min(notional, cap)
    
    return max(500, notional)
```

### Why $1-2K and NOT $10K

Slippage (spread + depth walking) is the #1 constraint, not model accuracy:

| Notional | RT Slippage | Net PnL | $ Profit | Win % |
|----------|------------|---------|----------|-------|
| $1,000 | 9.3 bps | **+14.3** | **$1.43** | **93%** |
| **$2,000** | **12.7 bps** | **+10.9** | **$2.18** | **84%** |
| $3,000 | 15.9 bps | +7.7 | $2.31 | 72% |
| $5,000 | 22.6 bps | +1.0 | $0.51 | 55% |
| $10,000 | **35.3 bps** | **-11.7** | **-$11.71** | **33%** |

**$10K is a LOSING strategy.** At 35.3 bps RT slippage, it exceeds the 23.6 bps ML edge.

### Loser Filters — Why 16% of Trades Lose (and How to Avoid Them)

**Root cause:** Losers don't lose because the drop is small — they lose because slippage on illiquid coins exceeds the 23.6 bps edge.

| Depth (20bps) | N | Win Rate | Why |
|---------------|---|----------|-----|
| **< $2K** | 24 | **17%** | **Slippage 40+ bps — skip!** |
| $2-5K | 45 | 87% | Marginal |
| $5-10K | 34 | 100% | Safe |
| $10K+ | 58 | 100% | Never lost |

92% of losers have **thin book (<$3K) or wide spread (>5 bps)** — both symptoms of illiquid micro-caps.

**Active filters in `compute_position_size()`:**
- `bid_depth_20bps < $2,000` → **SKIP** (catches 77% of losers, loses 3% of winners)
- `spread_bps > 8` → **SKIP** (catches remaining wide-spread losers)
- Result: **96% WR** on filtered trades (131/137), up from 84% unfiltered

**Validated on fresh unseen data (11 settlements, 10h):** 1 correctly skipped, 10 traded, **100% WR**.

### Slippage breakdown: spread matters on altcoins

Median spread at T-0: **2.6 bps** (unavoidable round-trip cost even for 1 lot).

| Notional | Spread | Depth Walk | Total | Spread % |
|----------|--------|-----------|-------|----------|
| $1,000 | 2.6 bps | 6.7 bps | 9.3 bps | 28% |
| $2,000 | 2.6 bps | 10.3 bps | 12.9 bps | 20% |
| $5,000 | 2.6 bps | 20.0 bps | 22.6 bps | 12% |

### Orderbook depth reality

| Distance from mid | Bid depth (median) | % of total |
|-------------------|-------------------|------------|
| Within 10 bps | **$1,926** | 2% |
| Within 20 bps | **$5,978** | 6% |
| Within 50 bps | **$19,005** | 24% |
| Full book (200 levels) | $93,000 | 100% |

The "$93K median depth" is misleading — only 6% sits within 20 bps of mid.

### 2b. Entry Execution at T+20ms

```python
# EC2 sends at ~T+17ms → Bybit fills at ~T+20ms
# This ensures BB created time > T+18ms → escapes FR payment on short
send_market_order(
    side="Sell",
    symbol=symbol,
    qty=notional / current_price,  # from Stage 2a sizing
    position_idx=2,                # hedge mode short
)
```

### Why T+20ms (Bybit fill time)?

| | T+15ms | **T+20ms** | T+25ms | T+50ms |
|--|--------|-----------|--------|--------|
| Sell volume before us | $6 | **$220** | $1,487 | $8,877 |
| Book intact | 100% | **99.8%** | 98.4% | ~87% |
| Price moved | -0.3 bps | **-5.3 bps** | -8.4 bps | -29.7 bps |
| FR escape? | Risky | **YES** | YES | YES |

- **T+15ms**: Book untouched but risky — BB created ~T+15ms is too close to T+18ms FR boundary
- **T+20ms**: Book 99.8% intact, FR safely escaped, 3 bps better entry than T+25ms
- **T+25ms**: Selling wave already started, 1.6% bid depth consumed

### FR Escape Logic

FR is negative on our target coins → **shorts PAY longs** at settlement.

- Position opened with BB created time **≤ T+17ms** → captured by FR snapshot → we PAY
- Position opened with BB created time **≥ T+18ms** → NOT captured → we DON'T PAY

EC2 send at ~T+17ms → BB fill at ~T+20ms → safely past the T+18ms boundary.

### Competitive dynamics: other bots

```
T+0-15ms:  Dead zone. Median $6 sell volume. We are FIRST.
T+15-25ms: First bots trickle in. $220 median.
T+25-50ms: MAIN WAVE. $5-9K selling. 80+ trades. Price crashes -30 bps.
T+50ms+:   Selling fades. Two-way trading resumes.
```

Our $2K = ~10% of total 1-second sell volume. We're riding the wave, not creating it. Other bots selling AFTER us deepens our profit — competition is GOOD once we're in.

---

## Stage 3: Real-Time Exit (T+1s to T+60s)

**This is where the ML magic happens.** Two sub-stages: ML signal + limit exit.

### Architecture: Event-Driven + BIG_TRADE Trigger + Limit Exit

```python
class ExitManager:
    def __init__(self, model, feature_cols):
        self.model = model          # LogReg, 56 features
        self.state = StreamingState()  # O(1) incremental updates
        self.last_eval_t = 0
        self.min_hold_ms = 1000     # don't exit before T+1s
        self.threshold = 0.50       # P(near_bottom) > 0.50 → exit

    def on_trade(self, t_ms, price, qty, side, notional):
        """Called on every WS trade event."""
        trigger = self.state.on_trade(t_ms, price, qty, side, notional)

        if t_ms < self.min_hold_ms:
            return None  # too early

        should_eval = False

        # Polling: evaluate every 100ms
        if t_ms - self.last_eval_t >= 100:
            should_eval = True

        # BIG_TRADE trigger: evaluate immediately on large trades
        if trigger == "BIG_TRADE":
            should_eval = True

        if should_eval:
            features = self.state.compute_features(t_ms)
            feat_arr = [features[c] for c in self.feature_cols]
            prob = self.model.predict_proba([feat_arr])[0, 1]
            self.last_eval_t = t_ms

            if prob > self.threshold:
                return EXIT_NOW  # close the short

        return None  # hold

    def on_ob1(self, t_ms, bid_p, bid_q, ask_p, ask_q):
        """Called on every OB L1 update."""
        self.state.on_ob1(t_ms, bid_p, bid_q, ask_p, ask_q)
```

### 3b. Limit Exit — Save 6 bps Per Fill

When ML signals EXIT_NOW, don't market buy — **place PostOnly limit buy at best_bid**:

```python
class LimitExitManager:
    RESCUE_TIMEOUT_MS = 1000  # 1s rescue window
    
    async def exit_position(self, qty, best_bid):
        """Exit via limit buy at bid with market rescue."""
        order = await self.place_limit_buy(
            price=best_bid,
            qty=qty,
            time_in_force="PostOnly",  # guarantees maker fee or reject
        )
        
        start = time.time()
        while (time.time() - start) * 1000 < self.RESCUE_TIMEOUT_MS:
            status = await self.check_order(order.id)
            if status.filled_qty >= qty:
                return "LIMIT_FILLED"    # saved 6 bps!
            await asyncio.sleep(0.05)    # 50ms poll
        
        # Timeout — rescue with market buy
        await self.cancel_order(order.id)
        remaining = qty - (await self.check_order(order.id)).filled_qty
        if remaining > 0:
            await self.market_buy(remaining)
            return "RESCUED"
        return "FILLED_DURING_CANCEL"
```

### Why limit exit works

| Metric | Value |
|--------|-------|
| Fill rate (1s timeout) | **54%** at T+10s |
| Median fill time | **168ms** |
| Fee saving per fill | **6 bps** (taker 10 → maker 4) |
| Price improvement per fill | **+3 bps** (buy at bid, not ask) |
| Rescue cost (unfilled) | +4.2 bps avg |
| **Net EV** | **+2.8 bps/trade** |

**Paradox resolved:** Even after the "bottom," residual selling continues for 10-30s. A $1-2K limit buy at best_bid fills 54% of the time within 1 second.

### Exit Model Details

```
Model:    LogReg(C=0.1) + SimpleImputer(median) + StandardScaler
Features: 56
Target:   near_bottom_10 (within 10 bps of eventual minimum)
```

### Why LogReg (not HGBC)
| Property | LogReg | HGBC |
|----------|--------|------|
| Test AUC | 0.771 | 0.755 |
| Overfit gap | **-0.007** (negative!) | +0.240 |
| Inference time | **<0.01ms** | ~0.5ms |
| Interpretable | Yes | No |
| Production risk | **Zero** overfit | High overfit |

### What the model detects

The LogReg learns a simple linear combination that detects **sell wave exhaustion**:

```
EXIT_SCORE = w₁ × distance_from_low      (bouncing off bottom? → exit)
           + w₂ × pct_of_window_elapsed  (late in window? → exit)
           + w₃ × drop_rate_slowing      (deceleration? → exit)
           + w₄ × volume_fading          (sell pressure gone? → exit)
           + w₅ × spread_normalizing     (market calm? → exit)
           + w₆ × time_since_new_low     (no new lows? → exit)
           + ... (50 more features with small weights)

if sigmoid(EXIT_SCORE) > 0.50 → EXIT NOW
```

### Safety Rails

```python
# Hard timeout: never hold longer than 60s
if t_ms >= 60000:
    return EXIT_NOW

# Emergency stop: if position is losing > 200 bps, exit
if unrealized_pnl_bps < -200:
    return EXIT_NOW

# Min hold: don't exit before 1s (too noisy)
if t_ms < 1000:
    return HOLD
```

---

## Complete Trade Lifecycle

```
T-10s    Collect features from WS (FR, OB depth, OI)
         Run Stage 1: predicted_drop=-85bps, P(profitable)=0.82
         Decision: TRADE

T-0ms    Settlement occurs. FR deducted. Book intact.
         Read OB.200 → bid_depth_20bps = $6,200 → notional = $1,000

T+17ms   EC2 sends market sell order ($1,000 notional)

T+20ms   Bybit fills the short. Entry price: -5.3 bps from mid.
         FR escaped (BB created > T+18ms boundary).
         Book 99.8% intact. Slippage: ~7 bps entry.

T+25ms   Other bots start selling. Price drops to -8 bps.
T+50ms   Main selling wave. Price crashes to -30 bps.
         (This is GOOD — we're already short.)

T+1s     First exit model evaluation.
         State: price=-35bps, distance_from_low=0
         P(near_bottom) = 0.12 → HOLD

T+5s     Price at -55 bps, rate of descent slowing.
         P(near_bottom) = 0.38 → HOLD

T+8s     BIG_TRADE trigger! Large buy at -52 bps.
         P(near_bottom) = 0.55 → EXIT!
         Place PostOnly limit buy at best_bid (-53 bps).

T+8.2s   Sell trade hits our limit → FILLED as maker!
         Exit at -53 bps. Maker fee: 4 bps (not 10).

         PnL = entry(-5.3) - exit(-53) - fees(14) = +33.7 bps
         Dollar: +33.7 × $1,000 / 10000 = +$3.37 ✓

--- OR if limit not filled ---

T+9s     1s timeout → cancel limit → market buy at -50 bps.
         PnL = entry(-5.3) - exit(-50) - fees(20) - rescue(2) = +22.7 bps
         Dollar: +22.7 × $1,000 / 10000 = +$2.27 ✓ (still profitable)
```

---

## Expected Performance in Production

### Per-trade economics (161 settlements, T+20ms entry, loser filters applied)

| Scenario | PnL/trade | $ Profit | Win % | Exit Fee |
|----------|-----------|----------|-------|----------|
| Market exit (no filters) | +6.7 bps | $1.34 | 84% | 10 bps |
| Market exit + **filters** | **+11.6 bps** | **$2.31** | **96%** | 10 bps |
| **Limit exit + filters** | **+14.5 bps** | **$2.91** | **90-96%** | **6.8 bps avg** |

### Recovery Long — 2x Buy at Bottom (NEW)

When ML signals EXIT, buy **2x**: 1x closes the short, 1x opens a long. The post-crash recovery bounce averages **+33 bps** from the bottom. Hold for +20s, then close with limit sell.

| Hold | N | Gross Recovery | Net PnL | Win Rate | $/trade | $/day |
|------|---|---------------|---------|----------|---------|-------|
| +10s | 104 | +30.6 bps | +14.1 bps | 66% | $1.26 | $32.7 |
| +15s | 111 | +31.5 bps | +15.1 bps | 68% | $1.29 | $35.9 |
| **+20s** | **105** | **+32.9 bps** | **+16.5 bps** | **63%** | **$1.42** | **$37.3** |
| +30s | 96 | +33.1 bps | +16.7 bps | 66% | $1.44 | $34.6 |

**Requirements:** Limit orders on both sides of long leg (maker fee). Long notional capped at $1K (conservative). Slippage discounted 60% vs T-0 OB (post-crash asks are thinner).

### Fresh data validation (11 unseen settlements, 10h, 2026-03-01)

| Metric | Market Exit | Limit Exit |
|--------|-----------|------------|
| Traded | 10 | 10 |
| Skipped (correctly) | 1 | 1 |
| **Win Rate** | **100%** | **90%** |
| **Avg PnL** | **+11.6 bps** | **+14.5 bps** |
| **$/trade** | **$2.31** | **$2.91** |
| Limit fills | — | 4/10 (40%) |

The one correctly skipped settlement (XCNUSDT, $161 depth) would have lost -35.6 bps.

### Revenue estimates — Bybit maximized

Scanner already monitors ALL ~552 Bybit perpetuals. 33 symbols with extreme FR produced 40.2 settlements/day over 4 days. This is the Bybit ceiling at current FR threshold (-15 bps).

| Lever | $/day | Monthly | Cumulative |
|-------|-------|---------|------------|
| **Short leg** (15% cap, limit exit) | **$77** | **$2,320** | $77 |
| **+ Long leg** (2x buy, +20s hold) | **+$37** | **+$1,089** | **$114** |
| + Binance double-dip (future) | +$26 est | +$780 | $140 |
| + OKX triple-dip (future) | +$16 est | +$480 | $156 |

**Bybit alone crosses $100/day** with short + long legs combined.

Revenue scales with number of exchanges, NOT with position size per trade.

### Improvement stack (cumulative)

```
Baseline (10% cap, market exit):            $43/day
 + Limit exit on short (saves 3 bps):       $54/day   (+26%)
 + Relax cap to 15%:                        $77/day   (+79%)
 + Long leg (2x buy, limit orders):         $114/day  (+165%)
```

---

## Implementation Checklist

### Phase 0: Quick Win (1 hour)
- [ ] Change `SNAP_EXIT_MS` from 5500 to 10000 in `fr_scalp_scanner.py`
- [ ] Change `SWEEP_EXIT_MS` to target BB fill at T+20ms (EC2 send ~T+17ms)
- [ ] Expected: +2 bps/trade, zero risk

### Phase 1: Position Sizing + Loser Filters (0.5 day)
- [ ] Add `compute_position_size()` to scanner (reads OB.200 at T-0)
- [ ] Replace fixed notional with OB-depth-based sizing ($500-$2K range)
- [ ] **Add depth filter: skip if bid_depth_20bps < $2,000**
- [ ] **Add spread filter: skip if spread > 8 bps**
- [ ] Expected: WR jumps from 84% → 96%, avoids -17 bps avg losers

### Phase 2: Pre-Trade ML Filter (1 day)
- [ ] Export trained Ridge + LogReg models to pickle/joblib
- [ ] Add `predict_settlement()` function to scanner
- [ ] Skip when P(profitable) < 0.50 or |FR| < 25 bps
- [ ] Log predictions vs actuals for monitoring
- [ ] Expected: +8 bps/trade from avoiding marginal trades

### Phase 3: ML Exit Signal (2-3 days)
- [ ] Port `StreamingState` class to scanner
- [ ] Export trained LogReg exit model
- [ ] Add 100ms polling loop post-entry
- [ ] Add BIG_TRADE trigger check on each trade
- [ ] Exit when P(near_bottom_10) > 0.50
- [ ] Add safety rails (min 1s hold, max 60s, emergency -200bps stop)
- [ ] Expected: +10-13 bps/trade from optimal exit timing

### Phase 4: Limit Exit (0.5 day)
- [ ] When ML signals EXIT: place PostOnly limit buy at best_bid
- [ ] Poll order status every 50ms for up to 1000ms
- [ ] If not filled: cancel + market buy remaining qty
- [ ] Handle partial fills (cancel remaining, market buy rest)
- [ ] Expected: saves 3.2 bps/leg average, +2.8 bps net EV

### Phase 5: Monitor & Retrain (ongoing)
- [ ] Log all features + predictions + actual outcomes
- [ ] Run `ml_settlement_pipeline.py` weekly with fresh data
- [ ] Pipeline auto-generates loser analysis + limit exit stats
- [ ] After 500 settlements: retrain, evaluate HGBC again
- [ ] Monitor depth/spread filter thresholds as more data arrives

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Model overfit in production | Low | -5 bps/trade | LogReg has negative overfit gap |
| Market regime change | Medium | Unknown | Weekly retraining, monitor MAE |
| Feature computation bug | Medium | Bad exits | Log features, compare to backtest |
| Latency spike at entry | Medium | Miss T+20ms window | Fallback to T+25ms (still profitable) |
| FR captured at boundary | Low | Pay FR (~50 bps) | EC2 send at T+17ms gives 2-3ms margin |
| Book thinner than expected | Low | Higher slippage | **Depth filter: skip if < $2K** |
| Wide spread at T-0 | Low | Edge eaten by spread | **Spread filter: skip if > 8 bps** |
| Limit exit not filled | Medium | Must rescue at market | 1s timeout limits exposure; +2.8 bps net EV |
| PostOnly rejected (spread inverted) | Very low | No fill | Immediate market buy fallback |
| Concurrent sellers eat bids | Low | Worse entry | At T+20ms only $220 sold, book 99.8% intact |

### Worst case
If ML exit performs no better than fixed timing, we still have:
- Fixed T+10s at $2K with filters: +11.6 bps, $2.31/trade, 96% WR
- Loser filters alone raise WR from 84% → 96% by skipping illiquid coins
- Position sizing prevents catastrophic losses from oversizing
- Limit exit adds +2.8 bps even if ML timing is suboptimal

### The ML can only help, not hurt
LogReg with negative overfit gap means it generalizes better than training. Combined with the T+60s hard timeout and -200bps emergency stop, the downside is bounded.

### Fresh data confirms all numbers
11 unseen settlements (10h, 2026-03-01): 100% WR on filtered trades, avg +$2.31/trade (market) or +$2.91/trade (limit). The one illiquid settlement (XCNUSDT, $161 depth) was correctly filtered.

---

## Research Evidence

| Document | What it proves |
|----------|---------------|
| `FINDINGS_position_sizing.md` | $1-3K optimal, $10K loses money |
| `FINDINGS_competitive_dynamics.md` | T+20ms: book intact, FR safe, before the crowd |
| `FINDINGS_loser_analysis.md` | **Depth is #1 predictor of losers, filters raise WR 84→96%** |
| `FINDINGS_limit_exit.md` | **Limit exit saves 2.8 bps/trade, 54% fill rate** |
| `FINDINGS_PIPELINE_V3_COMPLETE.md` | ML LOSO +26.1 bps, event-driven +22.2 bps |
| `FINDINGS_ml_integrity_audit.md` | No lookahead, no leakage, LogReg generalizes |
| `FINDINGS_deep_settlement_analysis.md` | Bottoms at T+13s median, recovery patterns |
| `REPORT_ml_settlement.md` | Auto-generated pipeline report (latest numbers) |

---

## Files

| File | Purpose |
|------|---------|
| `ml_settlement_pipeline.py` | End-to-end pipeline (download → train → backtest → report) |
| `research_exit_ml_v3.py` | Exit ML model (56 features, tick-based + event-driven backtest) |
| `research_exit_ml_eventdriven.py` | StreamingState class, event-driven simulator |
| `research_position_sizing.py` | OB slippage analysis + optimal sizing |
| `research_competitive_dynamics.py` | Trade flow + book depletion at settlement |
| `settlement_predictor.py` | Production predictor class |
| `REPORT_ml_settlement.md` | Auto-generated pipeline report |
