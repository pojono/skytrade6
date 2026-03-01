# Settlement Trading Pipeline — Report (Short-Only)

**Generated:** 2026-03-01 19:47 UTC

## Executive Summary

| Metric | Value |
|--------|-------|
| **Strategy** | short-only (ML-timed exit) |
| **Daily revenue (in-sample)** | **$137.7/day** |
| **LOSO conservative** | **$50–$75/day** |
| Win rate | 76% |
| Avg $/trade | $4.34 |
| Settlements traded | 127 / 160 |
| Data period | 4 days |

## Strategy Comparison (research)

| Strategy | Short $/day | Long $/day | **Total $/day** | Short WR | Long WR |
|----------|------------|-----------|----------------|----------|---------|
| short_only | $137.7 | $0.0 | **$137.7** | 76% | 0% | **← production**
| fixed_exit | $137.7 | $-22.3 | **$115.4** | 76% | 19% |
| ml_exit | $137.7 | $-17.4 | **$120.2** | 76% | 23% |

> **Note:** Long leg is unprofitable without look-ahead bias. Short-only is the production strategy.

## Configuration

| Parameter | Value |
|-----------|-------|
| Taker fee | 10 bps/leg |
| Maker fee | 4 bps/leg |
| Limit fill rate | 54% |
| Position cap | 15% of depth_20 |
| Short gross edge (LOSO avg) | 23.6 bps |
| Short exit ML threshold | p(near_bottom) ≥ 0.4 |
| Short exit timeout | 55s |

## Outcome Distribution (short-only)

| Outcome | Count | % | Avg $ |
|---------|-------|---|-------|
| short_only_lose | 30 | 24% | $-2.12 |
| short_only_win | 97 | 76% | $+6.33 |

## Per-Symbol Performance (short-only)

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

## Worst Trades (short-only)

| File | Symbol | Short $ | Gross bps | Slip bps | Depth $20 | Spread |
|------|--------|---------|-----------|----------|-----------|--------|
| SAHARAUSDT_20260227_130000.jsonl | SAHARAUSDT | $-15.29 | -70.9 | 8.8 | $19,139 | 4.2 |
| SAHARAUSDT_20260227_180000.jsonl | SAHARAUSDT | $-11.45 | -52.2 | 8.3 | $13,974 | 2.5 |
| SAHARAUSDT_20260301_010000.jsonl | SAHARAUSDT | $-7.34 | -29.9 | 10.0 | $13,768 | 0.5 |
| SAHARAUSDT_20260228_170000.jsonl | SAHARAUSDT | $-6.88 | -17.8 | 8.4 | $39,596 | 0.5 |
| SAHARAUSDT_20260228_020000.jsonl | SAHARAUSDT | $-5.33 | -51.9 | 4.6 | $11,235 | 0.5 |
| ENSOUSDT_20260227_030000.jsonl | ENSOUSDT | $-3.22 | -28.5 | 7.0 | $7,929 | 0.6 |
| BARDUSDT_20260228_060000.jsonl | BARDUSDT | $-2.75 | -6.6 | 10.3 | $18,265 | 0.9 |
| POWERUSDT_20260227_040000.jsonl | POWERUSDT | $-1.26 | +1.5 | 11.1 | $14,984 | 0.0 |
| SAHARAUSDT_20260227_150000.jsonl | SAHARAUSDT | $-1.24 | -1.7 | 7.7 | $15,857 | 2.6 |
| BARDUSDT_20260226_230000.jsonl | BARDUSDT | $-1.24 | -3.5 | 12.1 | $7,669 | 2.3 |

## Production Rules (Short-Only)

```python
# 1. FILTERS (skip if fails)
if depth_20 < 2000 or spread_bps > 8:
    skip()

# 2. SIZE
notional = adaptive_size(depth_20, cap=0.15)

# 3. SHORT ENTRY (at settlement T=0)
short_entry = market_sell(notional)  # taker

# 4. SHORT EXIT (ML-timed)
# Poll tick features every 100ms
# short_exit_logreg predicts p(near_bottom_10)
if pred_prob >= 0.4:
    limit_buy(notional)  # near bottom, cover short
elif time_since_settlement >= 55s:
    market_buy(notional)  # forced timeout
```
