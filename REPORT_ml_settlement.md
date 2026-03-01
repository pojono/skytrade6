# Settlement Trading Pipeline — Report

**Generated:** 2026-03-01 10:17 UTC

## Executive Summary

| Metric | Value |
|--------|-------|
| **Best strategy** | short_only |
| **Daily revenue** | **$137.7/day** |
| Short leg | $137.7/day (76% WR) |
| Long leg | $0.0/day (0% WR) |
| Settlements traded | 127 / 160 |
| Long trades taken | 0 / 127 |
| Data period | 4 days |

## Strategy Comparison

| Strategy | Short $/day | Long $/day | **Total $/day** | Long WR |
|----------|------------|-----------|----------------|---------|
| short_only | $137.7 | $0.0 | **$137.7** | 0% | **← best**
| fixed_exit | $137.7 | $-22.3 | **$115.4** | 19% |
| ml_exit | $137.7 | $-17.4 | **$120.2** | 23% |

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

## Outcome Distribution (short_only)

| Outcome | Count | % | Avg $ |
|---------|-------|---|-------|
| short_only_lose | 30 | 24% | $-2.12 |
| short_only_win | 97 | 76% | $+6.33 |

## Per-Symbol Performance (short_only)

| Symbol | N | Short WR | Long WR | Avg $/trade |
|--------|---|----------|---------|-------------|
| BARDUSDT | 9 | 67% | 0% | $+11.58 |
| POWERUSDT | 17 | 76% | 0% | $+9.12 |
| MIRAUSDT | 2 | 50% | 0% | $+7.42 |
| SAHARAUSDT | 35 | 74% | 0% | $+3.97 |
| ATHUSDT | 4 | 100% | 0% | $+3.10 |
| ENSOUSDT | 20 | 90% | 0% | $+3.06 |
| STEEMUSDT | 12 | 83% | 0% | $+2.88 |
| SOLAYERUSDT | 7 | 86% | 0% | $+2.46 |
| HOLOUSDT | 3 | 100% | 0% | $+2.46 |
| ALICEUSDT | 1 | 100% | 0% | $+1.34 |
| WETUSDT | 3 | 67% | 0% | $+0.98 |
| BIRBUSDT | 2 | 100% | 0% | $+0.97 |
| ESPUSDT | 1 | 100% | 0% | $+0.62 |
| NEWTUSDT | 1 | 100% | 0% | $+0.29 |
| ZKCUSDT | 1 | 100% | 0% | $+0.14 |
| STABLEUSDT | 3 | 67% | 0% | $+0.07 |
| SOPHUSDT | 1 | 0% | 0% | $-0.14 |
| ROBOUSDT | 2 | 0% | 0% | $-0.35 |
| KERNELUSDT | 1 | 0% | 0% | $-0.39 |
| FLOWUSDT | 1 | 0% | 0% | $-0.55 |

## Worst Combined Trades (short_only)

| File | Symbol | Short $ | Long $ | Combined $ | Drop | Exit |
|------|--------|---------|--------|-----------|------|------|
| SAHARAUSDT_20260227_130000.jsonl | SAHARAUSDT | $-15.29 | $+0.00 | $-15.29 | -43bps | none |
| SAHARAUSDT_20260227_180000.jsonl | SAHARAUSDT | $-11.45 | $+0.00 | $-11.45 | -34bps | none |
| SAHARAUSDT_20260301_010000.jsonl | SAHARAUSDT | $-7.34 | $+0.00 | $-7.34 | +38bps | none |
| SAHARAUSDT_20260228_170000.jsonl | SAHARAUSDT | $-6.88 | $+0.00 | $-6.88 | -5bps | none |
| SAHARAUSDT_20260228_020000.jsonl | SAHARAUSDT | $-5.33 | $+0.00 | $-5.33 | -34bps | none |
| ENSOUSDT_20260227_030000.jsonl | ENSOUSDT | $-3.22 | $+0.00 | $-3.22 | +4bps | none |
| BARDUSDT_20260228_060000.jsonl | BARDUSDT | $-2.75 | $+0.00 | $-2.75 | -3bps | none |
| POWERUSDT_20260227_040000.jsonl | POWERUSDT | $-1.26 | $+0.00 | $-1.26 | +2bps | none |
| SAHARAUSDT_20260227_150000.jsonl | SAHARAUSDT | $-1.24 | $+0.00 | $-1.24 | +19bps | none |
| BARDUSDT_20260226_230000.jsonl | BARDUSDT | $-1.24 | $+0.00 | $-1.24 | +12bps | none |

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
