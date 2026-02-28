# Real-Time Exit Optimization via Microstructure ML

**Date:** 2026-02-28
**Status:** Research in progress

---

## Problem

Currently we exit at a fixed T+5.5s after settlement. But:
- 60% of bottoms happen AFTER T+5s (median T+11.4s)
- For |FR|>80, holding to T+30s gives +181 bps vs +140 at T+5s
- The bottom is only knowable in hindsight — can ML detect it in real-time?

## Hypothesis

Post-settlement microstructure is highly predictable because:
1. The sell wave is driven by FR arbitrageurs closing positions simultaneously
2. The orderbook thins out predictably (market makers widen)
3. Sell pressure decays monotonically (69% → 54% over 60s)
4. Volume surge is 30x normal — signal-to-noise ratio is very high
5. The entire event is time-bounded (everyone knows when settlement happened)

**If we can predict "will price drop further in the next 1s?" at 100ms intervals,
we can exit at the optimal moment.**

## Approach: Rolling Microstructure Features

At every 100ms tick after settlement, compute real-time features from the stream:

### Price features
- `price_bps`: current price relative to ref (pre-settlement last trade)
- `price_velocity`: price change in last 500ms
- `price_accel`: change in velocity
- `new_low`: 1 if current price is a new post-settlement low
- `distance_from_low`: current price - running minimum (bps)
- `time_since_last_new_low`: ms since price last made a new low

### Trade flow features  
- `sell_ratio_500ms`: sell volume / total volume in last 500ms
- `sell_ratio_1s`: same for 1s window
- `trade_rate_500ms`: number of trades in last 500ms
- `trade_rate_accel`: change in trade rate
- `avg_trade_size_500ms`: mean trade size in last 500ms
- `large_sell_count_1s`: trades > 2x median in last 1s
- `buy_volume_surge`: buy vol / avg_sell_vol (buyers stepping in?)

### Orderbook features (from OB.1 stream, ~100ms updates)
- `spread_bps`: current bid-ask spread
- `spread_change`: spread widening/narrowing
- `bid_qty_change`: bid quantity change from previous tick
- `ask_qty_change`: ask quantity change
- `imbalance`: (bid_qty - ask_qty) / total

### Time features
- `t_ms`: time since settlement
- `t_bucket`: which phase we're in (0-1s, 1-5s, 5-10s, 10-30s, 30-60s)

### Static features (known at entry)
- `fr_bps`: funding rate
- `fr_abs_bps`: |FR|
- `pre_depth_usd`: pre-settlement orderbook depth

## Target

At each 100ms tick, the target is:
```
future_min_1s = min price in next 1 second
further_drop = 1 if (future_min_1s < current_price - 5bps) else 0
```

This means: "will price drop at least 5 more bps in the next 1 second?"

If the model says NO with high confidence → EXIT.

## Exit Strategy

```python
for each 100ms tick after settlement:
    features = compute_microstructure_features(state)
    p_further_drop = model.predict_proba(features)
    
    if p_further_drop < 0.3:  # confident no more drop
        EXIT
    elif trailing_stop_triggered(15bps):  # safety net
        EXIT
    elif t > max_hold_time(fr_bps):  # hard timeout
        EXIT
```

## Validation Plan

1. **Build dataset**: For each settlement, create time-series of 100ms ticks with features + targets
2. **Train/test split**: Leave-One-Settlement-Out CV (no peeking at same event)
3. **Metrics**: AUC for binary prediction + simulated PnL
4. **Backtest**: Compare PnL of ML exit vs fixed exit vs trailing stop on all 131 recordings
5. **Sanity checks**: Does the model use future information? (check feature timestamps)

## Latency Budget

Total decision loop: ~5-15ms (plenty fast for 100ms ticks)
- WS receive: 3-5ms
- Feature computation: <0.1ms  
- Inference (Ridge/LGBM): <1ms
- Order send: 1-2ms
- Fill: 1-5ms

## Risk

- Only 131 recordings → need to be very careful about overfitting
- Each recording produces ~300-600 ticks → total ~40-80K samples (much better than 131!)
- Still need LOSO validation to ensure cross-symbol generalization
