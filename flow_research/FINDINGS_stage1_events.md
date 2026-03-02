# Stage-1 Forced Flow Event Detector — Findings

## Overview
Detects "stress events" (t0 moments) where the market transitions from normal state to
disequilibrium: one-sided taker flow overwhelming visible orderbook liquidity.

**Symbol:** DOGEUSDT  
**Period:** 2025-09-01 → 2025-09-30 (30 days)  
**Data:** Bybit trades CSV + ob200 JSONL (100ms snapshots, 200 levels)

## Results Summary

| Metric | Value |
|---|---|
| Total events | 8,669 |
| Events/day | 196–417 (mean 289, median 284) |
| Direction split | BUY 4,263 / SELL 4,406 (ratio 0.97) |
| FlowImpact | median 1.51, p95 8.59 |
| DepthDrop <0.7 | 80.6% of events |
| Book age median | 58.3ms |
| Book stale share | 0.00% |

## Configuration (Final)

| Parameter | Value | Notes |
|---|---|---|
| WINDOW_S | 15s | Rolling window for metrics |
| EVAL_INTERVAL_S | 0.5s | Time-based eval grid (not trade-based) |
| OB_LEVELS_K | 10 | Top 10 bid+ask levels for depth |
| TH_IMPACT | 1.0 | FlowImpact >= 1.0 (flow > depth) |
| TH_IMB | 0.7 | Imbalance >= 0.7 |
| TH_SAMESIDE | 0.75 | SameSideShare >= 0.75 |
| TH_MIN_NOTIONAL | $50,000 | Minimum AggTotal USD |
| TH_TRADES | adaptive q80(1h) | Prevents low-trade-count noise |
| COOLDOWN_S | 30s | Debounce between triggers |
| BASELINE_WINDOW_S | 900s (15min) | Rolling baseline for DepthDrop/SpreadRatio |

## Threshold Tuning History

Initial run (v1): TH_IMPACT=0.5, TH_IMB=0.6, COOLDOWN=15s, no notional filter
→ 17,568 events (586/day) — too many

**Root causes of over-triggering:**
1. 25% of events had FlowImpact 0.5–0.6 (barely above threshold)
2. Imbalance 0.6 = 80/20 split, common in any trending 15s window
3. 15s cooldown → 2,733 back-to-back triggers (same shock counted twice)
4. No notional floor → $3K micro-flow triggered events

Final run (v2): TH_IMPACT=1.0, TH_IMB=0.7, COOLDOWN=30s, MIN_NOTIONAL=$50K
→ 8,669 events (289/day) — **51% reduction**, higher quality

## Acceptance Criteria (Section 9)

1. **Semantic validity**: Direction balanced (0.97 ratio), 80.6% show depth vacuum
2. **Stable frequency**: 196–417/day (std=53, CV=18%) — no order-of-magnitude jumps
3. **No trade-rate dependence**: Timer-based 500ms eval grid, adaptive trade count q80
4. **Synchronous orderbook**: book_age median 58ms, stale share 0.00%

## Output Files

- `output/{SYMBOL}/events_stage1.parquet` — all events with metrics
- `output/{SYMBOL}/events_stage1.csv` — same, CSV for inspection
- `output/{SYMBOL}/daily_summary.csv` — per-day aggregates
- `output/{SYMBOL}/sanity.json` — distributions, events/hour

## Event Fields

| Field | Description |
|---|---|
| event_id | `{SYMBOL}_{DATE}_{NNNN}` |
| t0 / t0_iso | Unix timestamp / ISO string |
| direction | BUY or SELL (dominant flow) |
| price_at_t0 | Mid price at trigger |
| agg_buy / agg_sell / agg_total | Notional volumes in window |
| net_agg | agg_buy - agg_sell |
| flow_impact | agg_total / top_depth |
| imbalance | |net_agg| / agg_total |
| same_side_share | max(buy,sell) / total |
| agg_trades_count | Number of trades in window |
| top_depth | Top-K bid+ask notional |
| spread / mid | From orderbook |
| book_ts / book_age_ms / book_stale | Orderbook freshness |
| depth_drop | current_depth / median_depth (15min baseline) |
| spread_ratio | current_spread / median_spread (15min baseline) |
| flow_impact_peak | Peak FlowImpact during cooldown window |
| price_extreme | Most extreme price during cooldown |

## Performance

- ~110s per day on single core (1.1M trades + 860K OB snapshots)
- 8 cores parallel: 30 days in 838s (14 min)
- RAM: ~100MB per worker (OB snapshots in memory, trades streamed)

## Next Steps

- Visual validation on random events (overlay on price chart)
- Stage-2: classify events as FOLLOW vs FADE
- Feature engineering: post-event price moves at various horizons
