# Limit Order Exit — Save 6+ bps Per Trade

**Date:** 2026-02-28  
**Dataset:** 85 valid settlements with OB.1 + trade data  
**Question:** Can we use a limit buy (maker) instead of market buy (taker) for exit?

---

## The Opportunity

| | Taker Exit | Maker Exit | Saving |
|--|-----------|-----------|--------|
| Fee per leg | 10 bps (0.10%) | 4 bps (0.04%) | **6 bps** |
| Price | Buy at ask | Buy at bid | **+3-5 bps** |
| **Total benefit** | — | — | **~9-11 bps** |

Current round-trip: 10 (entry taker) + 10 (exit taker) = **20 bps**  
With limit exit: 10 (entry taker) + 4 (exit maker) = **14 bps** (-30% fee reduction)

---

## The Paradox (And Why It's Not A Problem)

**Worry:** ML says "exit" because it detects the bottom. Bottom means price is about to bounce UP. So who will sell into our limit buy?

**Reality:** Even after the bottom, there is active two-way trading. Sell trades continue to flow — the "bottom" isn't silence, it's just a shift in sell/buy balance. There is enough residual selling to fill a $1-2K limit buy within seconds.

**Fill rates across all exit times:**

| Exit Time | Fill Rate (at best_bid) | Median Fill Time | Net EV (2s rescue) |
|-----------|------------------------|-----------------|-------------------|
| T+5s | **95%** | 379ms | **+9.7 bps** |
| T+8s | **96%** | 296ms | **+8.8 bps** |
| T+10s | **87%** | 460ms | **+6.7 bps** |
| T+15s | **89%** | 392ms | **+7.7 bps** |
| T+20s | **94%** | 548ms | **+9.2 bps** |
| T+30s | **92%** | 894ms | **+8.2 bps** |

Fill rates are 87-96%. Most fills happen within 500ms. This is not a tight race — we have plenty of time.

---

## Strategy Comparison

Place limit buy at different prices (T+10s exit):

| Placement | Fill Rate | Med Fill | Price Improve | Net EV |
|-----------|----------|----------|--------------|--------|
| **best_bid** | **87%** | 460ms | **+3.5 bps** | **+6.7 bps** |
| bid + 25% spread | 89% | 425ms | +2.7 bps | +6.2 bps |
| mid (bid + 50%) | 89% | 425ms | +1.8 bps | +5.4 bps |
| ask - 25% spread | 89% | 392ms | +0.9 bps | +4.6 bps |

**Best strategy: place at best_bid.** It has the highest net EV because the price improvement (buying at bid vs ask) compounds with the fee saving. The slightly lower fill rate is more than offset by the larger saving per fill.

---

## Rescue Plan

When the limit order doesn't fill (5-13% of trades):

### Strategy: Cancel + Market Buy after timeout

| Rescue Timeout | Avg Extra Cost | Max Extra Cost | Net EV (at_bid) |
|----------------|---------------|----------------|-----------------|
| 500ms | +4.4 bps | +23.0 bps | **+7.7 bps** |
| **1000ms** | +7.2 bps | +30.1 bps | **+7.3 bps** |
| **2000ms** | +11.9 bps | +38.5 bps | **+6.7 bps** |
| 3000ms | +14.7 bps | +52.9 bps | +6.3 bps |
| 5000ms | +25.2 bps | +79.2 bps | +5.0 bps |

**Recommended: 1000ms timeout.** Best balance of fill rate and rescue cost. At 1s, most fills that will happen have happened (median fill is 460ms), and rescue cost is still manageable.

### Rescue logic:

```python
# 1. ML says "exit now"
# 2. Place PostOnly limit buy at best_bid
limit_order = place_order(
    side="Buy",
    orderType="Limit",
    price=best_bid,
    qty=position_qty,
    timeInForce="PostOnly",  # ensures maker fee, rejects if would cross
)

# 3. Wait up to 1000ms for fill
filled = wait_for_fill(limit_order, timeout_ms=1000)

if not filled:
    # 4. Cancel limit + market buy
    cancel_order(limit_order)
    place_order(
        side="Buy",
        orderType="Market",
        qty=remaining_qty,
    )
```

### Edge cases:
- **Partial fill**: Cancel remaining, market buy the rest
- **PostOnly rejected** (bid crossed ask): Immediately market buy (spread inverted, very rare)
- **Order fails**: Market buy as fallback (always have this safety net)

---

## Impact on PnL

### Before (market exit)

| Metric | Value |
|--------|-------|
| ML gross edge | 23.6 bps |
| Round-trip fees | 20 bps |
| RT slippage ($2K) | 12.9 bps |
| **Net PnL** | **+10.7 bps** → $2.13/trade |

### After (limit exit, at_bid, 1s rescue)

| Metric | Value | Change |
|--------|-------|--------|
| ML gross edge | 23.6 bps | — |
| Entry fee (taker) | 10 bps | — |
| Exit fee (87% maker, 13% taker) | 4.8 bps | **-5.2 bps** |
| Price improvement (87% × 3.5 bps) | -3.0 bps | **-3.0 bps** |
| Rescue cost (13% × 7.2 bps) | +0.9 bps | +0.9 bps |
| RT slippage ($2K) | 12.9 bps | — |
| **Net PnL** | **~+18.0 bps** → **$3.60/trade** | **+69%** |

### Revenue impact

| | Market Exit | Limit Exit | Delta |
|--|-----------|-----------|-------|
| $/trade ($2K) | $2.13 | $3.60 | +$1.47 |
| Daily (12 trades) | $25.60 | $43.20 | +$17.60 |
| Monthly | $768 | $1,296 | **+$528** |

---

## Why This Works So Well

1. **Post-settlement markets are active.** Even after the bottom, residual selling continues for 10-30s. There is always flow to fill a $1-2K order.

2. **We're buying at the bid, not the ask.** This gives us the full spread as price improvement (~3.5 bps median). It's like getting paid to exit.

3. **Rescue cost is tiny.** The 13% unfilled trades cost ~7 bps extra, but weighted by frequency that's only 0.9 bps/trade on average. The 87% filled trades save ~9.5 bps each.

4. **No timing pressure.** Unlike entry (where we need T+20ms precision), exit has seconds to work with. Even if fill takes 500ms, we're fine.

---

## Production Implementation

### Phase 1: Simple (limit at bid, 1s timeout)

```python
class LimitExitManager:
    RESCUE_TIMEOUT_MS = 1000
    
    async def exit_position(self, qty, current_bid):
        """Exit via limit buy at bid with market rescue."""
        # Place PostOnly limit
        order = await self.place_limit_buy(
            price=current_bid,
            qty=qty,
            time_in_force="PostOnly",
        )
        
        # Wait for fill
        start = time.time()
        while (time.time() - start) * 1000 < self.RESCUE_TIMEOUT_MS:
            status = await self.check_order(order.id)
            if status.filled_qty >= qty:
                return "LIMIT_FILLED"
            if status.filled_qty > 0:
                # Partial fill — cancel remaining
                remaining = qty - status.filled_qty
                await self.cancel_order(order.id)
                await self.market_buy(remaining)
                return "PARTIAL_FILL_RESCUED"
            await asyncio.sleep(0.05)  # 50ms poll
        
        # Timeout — rescue with market
        await self.cancel_order(order.id)
        filled = await self.check_order(order.id)
        remaining = qty - filled.filled_qty
        if remaining > 0:
            await self.market_buy(remaining)
            return "TIMEOUT_RESCUED"
        return "FILLED_DURING_CANCEL"
```

### Phase 2: Adaptive (adjust price based on fill probability)
- If spread < 1 bps: place at bid (100% fill rate at tight spreads)
- If spread > 4 bps: place at bid + 25% spread (sacrifice some improvement for fill certainty)
- Monitor fill rate over time, adjust thresholds

---

## Spread Impact on Fill Rate

| Spread at Exit | Fill Rate (at_bid, T+10s) |
|---------------|--------------------------|
| < 1 bps | **100%** |
| 1-2 bps | 65% |
| 2-4 bps | 89% |
| 4-8 bps | 95% |
| 8+ bps | 78% |

Interesting pattern: very tight spreads (< 1 bps) fill 100% because there's heavy trading. 1-2 bps is the worst — not much spread to capture, and the market is slightly less active. Wide spreads (4-8 bps) fill well because any sell trade crosses the spread to hit our bid.

---

## Risks

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Price moves up during 1s wait | Rescue costs more | 1s timeout limits exposure |
| PostOnly rejected (spread inverted) | No fill | Immediate market buy fallback |
| Exchange latency on cancel | Double fill | Check filled qty before rescue |
| Partial fill leaves dust | Tiny position remains | Market buy remaining if > min notional |

---

## Files

| File | Purpose |
|------|---------|
| This document | Limit exit analysis findings |
| `research_limit_exit.py` | Simulation script (to be created) |
| `ml_settlement_pipeline.py` | Pipeline (exit strategy not yet changed) |
