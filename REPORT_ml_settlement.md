# Settlement Trading Pipeline — Report

**Generated:** 2026-03-01 07:21 UTC

## Executive Summary

| Metric | Value |
|--------|-------|
| **Best strategy** | ml_exit |
| **Daily revenue** | **$116.2/day** |
| Short leg | $72.5/day (100% WR) |
| Long leg | $43.7/day (66% WR) |
| Settlements traded | 127 / 160 |
| Long trades taken | 83 / 127 |
| Data period | 4 days |

## Strategy Comparison

| Strategy | Short $/day | Long $/day | **Total $/day** | Long WR |
|----------|------------|-----------|----------------|---------|
| short_only | $72.5 | $0.0 | **$72.5** | 0% |
| fixed_exit | $72.5 | $37.1 | **$109.5** | 58% |
| ml_exit | $72.5 | $43.7 | **$116.2** | 66% | **← best**

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
| both_win | 55 | 43% | $+5.91 |
| short_only_win | 44 | 35% | $+2.15 |
| short_win_long_lose | 28 | 22% | $+1.62 |

## Per-Symbol Performance (ml_exit)

| Symbol | N | Short WR | Long WR | Avg $/trade |
|--------|---|----------|---------|-------------|
| NEWTUSDT | 1 | 100% | 100% | $+10.41 |
| SAHARAUSDT | 35 | 100% | 79% | $+5.94 |
| BARDUSDT | 9 | 100% | 67% | $+4.94 |
| ENSOUSDT | 20 | 100% | 54% | $+4.06 |
| POWERUSDT | 17 | 100% | 80% | $+3.31 |
| BIRBUSDT | 2 | 100% | 100% | $+2.43 |
| ESPUSDT | 1 | 100% | 0% | $+2.23 |
| ATHUSDT | 4 | 100% | 75% | $+2.16 |
| HOLOUSDT | 3 | 100% | 67% | $+1.99 |
| WETUSDT | 3 | 100% | 100% | $+1.78 |
| SOLAYERUSDT | 7 | 100% | 50% | $+1.47 |
| MOVEUSDT | 1 | 100% | 0% | $+1.40 |
| ROBOUSDT | 2 | 100% | 100% | $+1.33 |
| STEEMUSDT | 12 | 100% | 40% | $+1.27 |
| ZKCUSDT | 1 | 100% | 100% | $+1.12 |
| STABLEUSDT | 3 | 100% | 0% | $+0.96 |
| MIRAUSDT | 2 | 100% | 50% | $+0.93 |
| SOPHUSDT | 1 | 100% | 0% | $+0.83 |
| FLOWUSDT | 1 | 100% | 0% | $+0.76 |
| ALICEUSDT | 1 | 100% | 0% | $+0.52 |

## Worst Combined Trades (ml_exit)

| File | Symbol | Short $ | Long $ | Combined $ | Drop | Exit |
|------|--------|---------|--------|-----------|------|------|
| KERNELUSDT_20260227_080000.jsonl | KERNELUSDT | $+0.68 | $-0.77 | $-0.09 | +9bps | ml |
| MIRAUSDT_20260227_040000.jsonl | MIRAUSDT | $+0.86 | $-0.49 | $+0.37 | +22bps | ml |
| HOLOUSDT_20260301_040000.jsonl | HOLOUSDT | $+0.90 | $-0.52 | $+0.38 | +8bps | ml |
| SOLAYERUSDT_20260228_110000.jsonl | SOLAYERUSDT | $+1.36 | $-0.96 | $+0.40 | +21bps | ml |
| STEEMUSDT_20260228_160000.jsonl | STEEMUSDT | $+1.04 | $-0.62 | $+0.42 | +13bps | ml |
| POWERUSDT_20260226_210000.jsonl | POWERUSDT | $+1.04 | $-0.60 | $+0.44 | +46bps | ml |
| STEEMUSDT_20260227_160000.jsonl | STEEMUSDT | $+1.18 | $-0.74 | $+0.44 | +5bps | ml |
| STEEMUSDT_20260228_040000.jsonl | STEEMUSDT | $+1.19 | $-0.74 | $+0.45 | +12bps | ml |
| ALICEUSDT_20260228_080000.jsonl | ALICEUSDT | $+0.52 | $+0.00 | $+0.52 | +133bps | none |
| BARDUSDT_20260226_230000.jsonl | BARDUSDT | $+1.47 | $-0.92 | $+0.55 | +14bps | ml |

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
