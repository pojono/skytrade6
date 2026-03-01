# Settlement Trading Pipeline — Report

**Generated:** 2026-03-01 07:30 UTC

## Executive Summary

| Metric | Value |
|--------|-------|
| **Best strategy** | ml_exit |
| **Daily revenue** | **$121.4/day** |
| Short leg | $72.5/day (100% WR) |
| Long leg | $48.9/day (58% WR) |
| Settlements traded | 127 / 160 |
| Long trades taken | 83 / 127 |
| Data period | 4 days |

## Strategy Comparison

| Strategy | Short $/day | Long $/day | **Total $/day** | Long WR |
|----------|------------|-----------|----------------|---------|
| short_only | $72.5 | $0.0 | **$72.5** | 0% |
| fixed_exit | $72.5 | $39.7 | **$112.2** | 53% |
| ml_exit | $72.5 | $48.9 | **$121.4** | 58% | **← best**

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
| both_win | 48 | 38% | $+7.23 |
| short_only_win | 44 | 35% | $+2.15 |
| short_win_long_lose | 35 | 28% | $+1.26 |

## Per-Symbol Performance (ml_exit)

| Symbol | N | Short WR | Long WR | Avg $/trade |
|--------|---|----------|---------|-------------|
| SAHARAUSDT | 35 | 100% | 67% | $+6.84 |
| NEWTUSDT | 1 | 100% | 100% | $+5.74 |
| BARDUSDT | 9 | 100% | 67% | $+5.38 |
| ENSOUSDT | 20 | 100% | 54% | $+3.90 |
| POWERUSDT | 17 | 100% | 50% | $+3.28 |
| BIRBUSDT | 2 | 100% | 100% | $+2.34 |
| ESPUSDT | 1 | 100% | 0% | $+2.23 |
| WETUSDT | 3 | 100% | 100% | $+2.09 |
| ATHUSDT | 4 | 100% | 50% | $+1.86 |
| HOLOUSDT | 3 | 100% | 67% | $+1.52 |
| MOVEUSDT | 1 | 100% | 0% | $+1.40 |
| SOLAYERUSDT | 7 | 100% | 50% | $+1.25 |
| ROBOUSDT | 2 | 100% | 100% | $+1.14 |
| STEEMUSDT | 12 | 100% | 40% | $+1.12 |
| STABLEUSDT | 3 | 100% | 0% | $+0.97 |
| MIRAUSDT | 2 | 100% | 50% | $+0.84 |
| SOPHUSDT | 1 | 100% | 0% | $+0.83 |
| ZKCUSDT | 1 | 100% | 100% | $+0.82 |
| FLOWUSDT | 1 | 100% | 0% | $+0.76 |
| ALICEUSDT | 1 | 100% | 0% | $+0.52 |

## Worst Combined Trades (ml_exit)

| File | Symbol | Short $ | Long $ | Combined $ | Drop | Exit |
|------|--------|---------|--------|-----------|------|------|
| KERNELUSDT_20260227_080000.jsonl | KERNELUSDT | $+0.68 | $-1.17 | $-0.49 | +9bps | ml |
| SOLAYERUSDT_20260228_110000.jsonl | SOLAYERUSDT | $+1.36 | $-1.66 | $-0.29 | +21bps | ml |
| HOLOUSDT_20260301_040000.jsonl | HOLOUSDT | $+0.90 | $-1.16 | $-0.26 | +8bps | ml |
| STEEMUSDT_20260228_160000.jsonl | STEEMUSDT | $+1.04 | $-1.25 | $-0.21 | +13bps | ml |
| BARDUSDT_20260226_230000.jsonl | BARDUSDT | $+1.47 | $-1.64 | $-0.17 | +14bps | ml |
| ATHUSDT_20260227_200000.jsonl | ATHUSDT | $+0.69 | $-0.63 | $+0.06 | +72bps | ml |
| ATHUSDT_20260228_160000.jsonl | ATHUSDT | $+2.03 | $-1.83 | $+0.20 | +32bps | ml |
| POWERUSDT_20260226_210000.jsonl | POWERUSDT | $+1.04 | $-0.78 | $+0.25 | +46bps | ml |
| SAHARAUSDT_20260227_090000.jsonl | SAHARAUSDT | $+0.81 | $-0.53 | $+0.28 | +49bps | ml |
| STEEMUSDT_20260227_160000.jsonl | STEEMUSDT | $+1.18 | $-0.83 | $+0.35 | +5bps | ml |

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
