# Settlement Trading Pipeline — Report

**Generated:** 2026-03-01 07:40 UTC

## Executive Summary

| Metric | Value |
|--------|-------|
| **Best strategy** | ml_exit |
| **Daily revenue** | **$125.6/day** |
| Short leg | $72.5/day (100% WR) |
| Long leg | $53.1/day (60% WR) |
| Settlements traded | 127 / 160 |
| Long trades taken | 83 / 127 |
| Data period | 4 days |

## Strategy Comparison

| Strategy | Short $/day | Long $/day | **Total $/day** | Long WR |
|----------|------------|-----------|----------------|---------|
| short_only | $72.5 | $0.0 | **$72.5** | 0% |
| fixed_exit | $72.5 | $42.1 | **$114.5** | 55% |
| ml_exit | $72.5 | $53.1 | **$125.6** | 60% | **← best**

## Configuration

| Parameter | Value |
|-----------|-------|
| Taker fee | 10 bps/leg |
| Maker fee | 4 bps/leg |
| Limit fill rate | 54% |
| Position cap | 15% of depth_20 |
| Short gross edge | 23.6 bps (ML LOSO) |
| Long entry rule | bottom T ≤ 15s |
| Long exit ML threshold | p ≥ 0.6 |
| Long fixed hold | +20s |

## Outcome Distribution (ml_exit)

| Outcome | Count | % | Avg $ |
|---------|-------|---|-------|
| both_win | 50 | 39% | $+7.38 |
| short_only_win | 44 | 35% | $+2.15 |
| short_win_long_lose | 33 | 26% | $+1.17 |

## Per-Symbol Performance (ml_exit)

| Symbol | N | Short WR | Long WR | Avg $/trade |
|--------|---|----------|---------|-------------|
| SAHARAUSDT | 35 | 100% | 71% | $+7.00 |
| NEWTUSDT | 1 | 100% | 100% | $+5.64 |
| BARDUSDT | 9 | 100% | 67% | $+5.21 |
| ENSOUSDT | 20 | 100% | 54% | $+4.08 |
| POWERUSDT | 17 | 100% | 60% | $+3.46 |
| HOLOUSDT | 3 | 100% | 67% | $+2.51 |
| BIRBUSDT | 2 | 100% | 100% | $+2.37 |
| ATHUSDT | 4 | 100% | 50% | $+2.33 |
| ESPUSDT | 1 | 100% | 0% | $+2.23 |
| WETUSDT | 3 | 100% | 100% | $+1.71 |
| MOVEUSDT | 1 | 100% | 0% | $+1.40 |
| STEEMUSDT | 12 | 100% | 40% | $+1.37 |
| SOLAYERUSDT | 7 | 100% | 50% | $+1.29 |
| STABLEUSDT | 3 | 100% | 0% | $+0.99 |
| ROBOUSDT | 2 | 100% | 100% | $+0.93 |
| ZKCUSDT | 1 | 100% | 100% | $+0.88 |
| SOPHUSDT | 1 | 100% | 0% | $+0.83 |
| FLOWUSDT | 1 | 100% | 0% | $+0.76 |
| MIRAUSDT | 2 | 100% | 50% | $+0.68 |
| ALICEUSDT | 1 | 100% | 0% | $+0.52 |

## Worst Combined Trades (ml_exit)

| File | Symbol | Short $ | Long $ | Combined $ | Drop | Exit |
|------|--------|---------|--------|-----------|------|------|
| SAHARAUSDT_20260227_090000.jsonl | SAHARAUSDT | $+0.81 | $-2.06 | $-1.25 | +49bps | ml |
| BARDUSDT_20260226_230000.jsonl | BARDUSDT | $+1.47 | $-2.57 | $-1.10 | +14bps | ml |
| KERNELUSDT_20260227_080000.jsonl | KERNELUSDT | $+0.68 | $-1.16 | $-0.48 | +9bps | ml |
| ATHUSDT_20260228_160000.jsonl | ATHUSDT | $+2.03 | $-2.39 | $-0.37 | +32bps | ml |
| HOLOUSDT_20260301_040000.jsonl | HOLOUSDT | $+0.90 | $-1.15 | $-0.25 | +8bps | ml |
| SOLAYERUSDT_20260228_110000.jsonl | SOLAYERUSDT | $+1.36 | $-1.52 | $-0.16 | +21bps | ml |
| POWERUSDT_20260226_210000.jsonl | POWERUSDT | $+1.04 | $-1.04 | $-0.00 | +46bps | ml |
| MIRAUSDT_20260227_040000.jsonl | MIRAUSDT | $+0.86 | $-0.68 | $+0.18 | +22bps | ml |
| STEEMUSDT_20260228_040000.jsonl | STEEMUSDT | $+1.19 | $-0.91 | $+0.28 | +12bps | ml |
| STEEMUSDT_20260228_160000.jsonl | STEEMUSDT | $+1.04 | $-0.75 | $+0.28 | +13bps | ml |

## Production Rules

```python
# 1. FILTERS (skip if fails)
if depth_20 < 2000 or spread_bps > 8:
    skip()

# 2. SHORT LEG (always)
notional = adaptive_size(depth_20, cap=0.15)
short_entry = market_sell(notional)  # taker
short_exit = limit_buy(notional)     # maker if fills, taker rescue at 1s

# 3. LONG ENTRY DECISION (at short exit moment)
if ml_exit_time <= 15.0:  # seconds since settlement
    buy_qty = 2 * notional  # 1x close short + 1x open long
else:
    buy_qty = 1 * notional  # just close short

# 4. LONG EXIT (if long taken)
# Poll recovery ticks every 100ms
# LogReg predicts p(near_peak_10)
if pred_prob >= 0.6:
    limit_sell(long_notional)  # recovery peaking
elif time_since_bottom >= 30s:
    limit_sell(long_notional)  # forced timeout
```
