# FINDINGS v24e: LS Ratio EWMA Momentum — 3-Year Validation

**Date:** Feb 2026
**Symbol:** SOLUSDT
**Period:** Jan 2023 – Jan 2026 (3 years, 1,127 days, 324,576 bars)
**Data:** Binance data warehouse metrics + Binance 5m klines

---

## Motivation

v24d found that simple EWMA z-score momentum on SOL was the only signal that survived OOS:
- EWMA(4h) z>1.5 momentum: +5.5 bps OOS (May-Oct 2025)
- EWMA(4h) z>2.0 momentum: +8.1 bps OOS

This study extends the test to 3 full years to determine if this is a real edge or a period-specific artifact.

---

## Key Result: The Momentum Signal Fails Over 3 Years

### Every momentum configuration loses money

| Z-score | Threshold | Trades | Avg (bps) | WR | Sharpe |
|---------|-----------|--------|-----------|-----|--------|
| EWMA(4h) | 0.5 | 220,871 | **-9.2** | 48.2% | -1.84 |
| EWMA(4h) | 1.0 | 106,339 | **-7.5** | 48.7% | -1.51 |
| EWMA(4h) | 1.5 | 35,024 | **-10.3** | 48.3% | -1.91 |
| EWMA(4h) | 2.0 | 9,616 | **-15.2** | 47.4% | -2.78 |
| EWMA(8h) | 2.0 | 10,150 | **-19.6** | 45.5% | -3.73 |
| EWMA(24h) | 2.0 | 11,796 | **-14.1** | 46.4% | -2.80 |
| EWMA(48h) | 2.5 | 2,385 | **-32.1** | 45.7% | -6.91 |

**0 out of 30 momentum configurations are profitable.** Sharpe ranges from -1.20 to -6.91.

### The v24d "survivor" was a period artifact

| Signal | v24d 9-month | 3-Year |
|--------|-------------|--------|
| EWMA(4h) z>1.5 | +5.5 bps | **-10.3 bps** |
| EWMA(4h) z>2.0 | +8.1 bps | **-15.2 bps** |
| EWMA(24h) z>2.0 | +8.4 bps | **-14.1 bps** |

The signal that appeared to work on 9 months (May 2025 – Jan 2026) is catastrophically negative over 3 years. **The May-Jan period was an anomaly, not a pattern.**

---

## Contrarian Signals: Two Marginal Survivors

| Z-score | Threshold | Trades | Avg (bps) | WR | Sharpe |
|---------|-----------|--------|-----------|-----|--------|
| EWMA(8h) | 2.0 | 10,150 | **+5.6** | 50.4% | **+1.06** |
| EWMA(48h) | 2.5 | 2,385 | **+18.1** | 50.0% | **+3.90** |

Only 2 out of 30 contrarian configurations are positive:
- **EWMA(8h) z>2.0 contrarian**: +5.6 bps avg, Sharpe 1.06 — marginal but real
- **EWMA(48h) z>2.5 contrarian**: +18.1 bps avg, Sharpe 3.90 — but only 2,385 trades over 3 years (~2/day)

The contrarian interpretation ("crowd is wrong at extremes") has weak support, but the trade frequency is very low.

---

## LS Ratio Distribution: Massively Non-Stationary

| Quarter | LS Mean | LS Std | SOL Price |
|---------|---------|--------|-----------|
| 2023-Q1 | 1.76 | 0.47 | $21 |
| 2023-Q2 | 2.29 | 0.70 | $19 |
| 2023-Q3 | 1.70 | 0.44 | $21 |
| 2023-Q4 | 1.45 | 0.46 | $102 |
| 2024-Q1 | 2.77 | 0.65 | $203 |
| 2024-Q2 | 2.89 | 0.67 | $147 |
| 2024-Q3 | 2.51 | 0.64 | $152 |
| 2024-Q4 | 3.06 | 1.54 | $189 |
| 2025-Q1 | 4.08 | 0.93 | $125 |
| 2025-Q2 | 2.47 | 0.68 | $155 |
| 2025-Q3 | 2.70 | 0.69 | $209 |
| 2025-Q4 | 3.89 | 0.77 | $125 |
| 2026-Q1 | 3.65 | 0.99 | $106 |

The LS ratio mean ranges from **1.45 to 4.08** across quarters — a 2.8× range. There is no stable relationship between LS ratio level and future returns. The ratio appears to be driven by market structure changes (exchange user composition, leverage availability, etc.) rather than by predictive positioning.

---

## Monthly IC: Coin Flip

- **Positive IC months:** 18/37 (49%)
- **Average monthly IC:** +0.009
- **IC range:** -0.110 to +0.233

The IC is essentially zero on average. The two highest months (Nov 2025: +0.18, Dec 2025: +0.23) are the exact period that drove the original v24 finding. Without those two months, the average IC would be negative.

---

## Conclusions

### The LS ratio has NO tradeable directional signal on SOL over 3 years

1. **Momentum fails completely** — 0/30 configs profitable, all negative Sharpe
2. **Contrarian has marginal edge at extreme thresholds only** — 2/30 configs positive, but very low frequency
3. **The v24d "survivor" was a 9-month anomaly** — extending to 3 years reverses the sign
4. **Monthly IC is a coin flip** (49% positive) with average +0.009
5. **The LS ratio distribution is massively non-stationary** (mean ranges 1.45–4.08)

### This definitively closes the LS ratio research line

| Study | Finding | Verdict |
|-------|---------|---------|
| v24 (Dec 2025, 31 days) | IC=0.20, Sharpe 9+ | Period-specific |
| v24b (May-Aug, Bybit) | No LS data available | Inconclusive |
| v24c (May-Oct vs Nov-Jan) | IC drops 92% OOS | Signal doesn't replicate |
| v24d (Online learning, 9mo) | ML fails, EWMA +5.5 bps OOS | Weak survivor |
| **v24e (3 years)** | **All momentum negative** | **Signal is dead** |

### The only remaining use case
The EWMA(8h) z>2.0 contrarian signal (+5.6 bps, Sharpe 1.06) could serve as a very low-frequency filter (~10k trades over 3 years). But it's too weak and too rare to build a strategy around.

---

## Files

| File | Description |
|------|-------------|
| `ls_ratio_ewma_3yr.py` | 3-year EWMA momentum test script |
| `results/ls_ratio_ewma_3yr.txt` | Complete experiment output |
| `FINDINGS_v24e_ewma_3yr.md` | This document |

---

**Research Status:** Complete ✅
**Verdict:** LS ratio signal is definitively dead. No tradeable edge over 3 years.
**LS ratio research line: CLOSED.**
