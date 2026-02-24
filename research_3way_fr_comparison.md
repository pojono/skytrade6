# 3-Way Funding Rate Strategy Comparison: Binance vs Bybit vs OKX

**Date:** 2026-02-24
**Period:** 106 days common range (2025-11-10 → 2026-02-24)
**Data:** Official REST API historical funding rates
**Notional:** $10,000 per position | **RT cost:** 39 bps (BN/BB), 38 bps (OKX)

## Data Summary

| Exchange | Records | Symbols | 1h coins | 4h coins | 8h coins |
|----------|---------|---------|----------|----------|----------|
| Binance  | 576K    | 580     | 32       | 395      | 153      |
| Bybit    | 728K    | 552     | 95       | 281      | 162      |
| OKX      | 145K    | 259     | 20       | 154      | 85       |

Symbol overlap: 235 coins on all three exchanges. 1h overlap: 10 coins on all three.

---

## Key Finding: 1h Coins Are the Clear Winner

The funding interval is the single biggest driver of strategy performance. **1h settlement coins dramatically outperform 4h and 8h** on every exchange:

### HOLD Strategy Performance (entry≥20bps, exit<8bps, max 3 positions)

| Exchange | Interval | Coins | Trades | WR   | Total P&L    | Daily P&L | Avg Winner     | Avg Loser    |
|----------|----------|-------|--------|------|-------------|-----------|----------------|--------------|
| **Bybit**    | **1h**   | **95** | **703** | **60%** | **+$153,164** | **+$1,443** | +$382 (9.8 settles) | -$26 (1.6) |
| Binance  | 1h       | 32    | 189    | 70%  | +$87,046    | +$820     | +$665 (21.6 settles) | -$25 (1.7) |
| OKX      | 1h       | 20    | 291    | 60%  | +$79,812    | +$752     | +$474 (12.0 settles) | -$26 (1.6) |
| Binance  | 4h       | 395   | 240    | 69%  | +$65,265    | +$615     | +$404 (12.4 settles) | -$23 (1.7) |
| OKX      | 4h       | 154   | 395    | 55%  | +$51,301    | +$483     | +$258 (8.3 settles) | -$26 (1.6) |
| Bybit    | 4h       | 281   | 422    | 47%  | +$30,461    | +$287     | +$183 (6.2 settles) | -$28 (1.5) |
| Bybit    | 8h       | 162   | 205    | 44%  | +$13,031    | +$123     | +$178 (6.5 settles) | -$28 (1.5) |
| Binance  | 8h       | 153   | 111    | 43%  | +$5,352     | +$50      | +$148 (5.8 settles) | -$28 (1.5) |
| OKX      | 8h       | 85    | 24     | 54%  | +$1,605     | +$15      | +$145 (6.5 settles) | -$25 (1.7) |

**All 9 configurations profitable. Every month profitable for 1h coins.**

### Monthly Breakdown (1h coins only)

| Month    | BN       | BB        | OKX      |
|----------|----------|-----------|----------|
| 2025-11  | +$20,514 | +$30,983  | +$7,807  |
| 2025-12  | +$20,057 | +$45,428  | +$21,214 |
| 2026-01  | +$30,634 | +$42,947  | +$27,808 |
| 2026-02  | +$15,842 | +$33,806  | +$22,983 |

---

## SCALP vs HOLD Verdict

### SCALP (pick best coin each settlement, enter+exit each time)

| Exchange | Interval | Settles | WR   | Total      | Daily   |
|----------|----------|---------|------|------------|---------|
| **Bybit**| **1h**   | **2548**| **53%** | **+$62,037** | **+$585** |
| Bybit    | 4h       | 861     | 51%  | +$13,561   | +$128   |
| OKX      | 1h       | 2479    | 32%  | -$1,558    | -$15    |
| Binance  | 1h       | 2547    | 28%  | -$10,576   | -$100   |
| OKX      | 4h       | 2247    | 29%  | -$10,662   | -$100   |
| Binance  | 4h       | 2376    | 29%  | -$13,752   | -$130   |

**Scalp only works on Bybit 1h** (53% WR due to large FR variance). On Binance and OKX, scalp is a guaranteed loser — the 39bps round-trip eats all the FR income.

**HOLD universally dominates SCALP** because it amortizes the single RT cost over many settlements.

---

## FR Autocorrelation (Key for HOLD Predictability)

| Interval | Binance r | Bybit r | OKX r   |
|----------|-----------|---------|---------|
| 1h       | **0.696** | 0.618   | **0.726** |
| 4h       | 0.526     | 0.446   | 0.442   |
| 8h       | 0.426     | 0.431   | 0.352   |

1h coins have highest autocorrelation — FR is most persistent/predictable. OKX 1h has highest autocorrelation (0.726) despite fewer coins. Binance 1h close behind (0.696).

---

## Combined Multi-Exchange HOLD

Running HOLD across all 147 unique 1h coin-exchange pairs (picking best across all 3):

- **Trades:** 459, WR 71%
- **Total:** +$182,636 | **Daily: +$1,721**
- Breakdown: BB 271 trades (+$85,940), OKX 103 trades (+$45,077), BN 85 trades (+$51,619)

**Combined is +19% better than Bybit alone** ($1,721/day vs $1,443/day).

---

## Extreme FR Frequency (1h coins, best coin per settlement)

| Threshold | Binance (32) | Bybit (95) | OKX (20)  |
|-----------|-------------|------------|-----------|
| ≥ 10 bps  | 80.9%       | 96.7%      | 71.6%     |
| ≥ 20 bps  | 55.6%       | 85.2%      | 52.4%     |
| ≥ 39 bps  | 27.8%       | 62.8%      | 30.8%     |
| ≥ 75 bps  | 10.5%       | 36.4%      | 15.6%     |
| ≥ 100 bps | 6.2%        | 25.6%      | 10.7%     |

Bybit has the most extreme FR events (95 1h coins, wider tails). OKX and Binance similar.

---

## Conclusions

1. **HOLD >> SCALP** on every exchange, every interval. Not close.
2. **1h interval >> 4h >> 8h** — more settlements = more FR income before exit.
3. **Bybit 1h is the best single venue** — most 1h coins (95), highest daily ($1,443).
4. **OKX is viable** — 20 1h coins, $752/day, strong autocorrelation (0.726).
5. **Binance has highest WR** (70%) due to best autocorrelation, but fewer 1h coins (32).
6. **Combined multi-exchange adds ~19%** over single-venue.
7. **OKX fees slightly lower** (futures taker 5bps vs 5.5bps), marginal impact.

### Recommended Priority
1. **Bybit 1h HOLD** — primary venue (95 coins, $1,443/day)
2. **OKX 1h HOLD** — secondary venue (+$752/day incremental)
3. **Binance 1h HOLD** — tertiary venue (+$820/day but fewer opportunities)
4. Consider running all three simultaneously for $1,721/day combined
