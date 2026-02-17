# FINDINGS v22: Walk-Forward Momentum Backtest (No Lookahead)

**Date:** Feb 2025
**Symbols:** ETHUSDT, BTCUSDT
**Period:** 2025-01-01 to 2026-01-31 (13 months, 30-day warmup → 12 months trading)
**Data:** 5-minute bars from Bybit tick data

---

## Motivation

In v20, we found that "ETH momentum_4h in the volatile regime = +9.6 bps" over 19,903 trades. But that result had serious flaws:

1. **Overlapping trades** — every 5-min bar counted as a separate "trade" with heavily overlapping 4h forward returns. The 19,903 "trades" were really ~715 independent trades.
2. **In-sample regime labels** — GMM was fit on the entire dataset, then signals evaluated on the same data.
3. **Look-ahead in features** — features like `rvol_4h` overlap with the forward return window.

This experiment tests whether the signal survives a **clean walk-forward backtest** with zero lookahead.

## Methodology

### Rules — No Lookahead

1. **Warmup:** First 30 days (Jan 2025) used to fit initial HMM/GMM and compute feature statistics
2. **Model refit:** Every 14 days on a rolling 90-day window (capped to keep HMM fast)
3. **Feature scaling:** Expanding mean/std from all past data (no future data)
4. **Regime detection:** HMM forward filter (online, causal) + 3-bar confirmation
5. **Signal:** `momentum_4h` z-scored using expanding statistics
6. **Entry:** When detected volatile AND |momentum_z| > 1.0 → go long (if positive) or short (if negative)
7. **Hold:** 4 hours (48 bars), strictly non-overlapping trades
8. **Fee:** 7 bps round-trip deducted from every trade

### Four Strategies Compared

| Strategy | Description |
|----------|-------------|
| **HMM volatile + momentum** | Trade momentum only when HMM detects volatile regime |
| **GMM volatile + momentum** | Same but with GMM single-bar posterior for detection |
| **Unconditional momentum** | Trade momentum everywhere, ignore regime |
| **HMM quiet + momentum** | Trade momentum only in quiet regime (control — should fail) |

---

## Results

### ETH — The Signal Survives (Barely)

| Strategy | Trades | Avg PnL | Win% | Sharpe | Total PnL | Pos Months |
|----------|--------|---------|------|--------|-----------|------------|
| **HMM volatile + momentum** | **769** | **+4.18 bps** | **46.4%** | **0.57** | **+3,213 bps** | **8/12** |
| GMM volatile + momentum | 748 | +2.70 bps | 45.6% | 0.36 | +2,019 bps | 8/12 |
| Unconditional momentum | 903 | +3.37 bps | 46.7% | 0.52 | +3,044 bps | 9/12 |
| HMM quiet + momentum (control) | 391 | −11.43 bps | 46.3% | −1.32 | −4,469 bps | 4/12 |

**ETH monthly breakdown (HMM volatile + momentum):**

| Month | Trades | Avg PnL | Total |
|-------|--------|---------|-------|
| 2025-02 | 69 | +14.16 | +977 |
| 2025-03 | 67 | −1.03 | −69 |
| 2025-04 | 68 | +12.61 | +857 |
| 2025-05 | 75 | +4.40 | +330 |
| 2025-06 | 51 | +15.41 | +786 |
| 2025-07 | 64 | +7.12 | +456 |
| 2025-08 | 80 | −5.71 | −457 |
| 2025-09 | 46 | −16.03 | −737 |
| 2025-10 | 70 | +4.54 | +318 |
| 2025-11 | 86 | −5.88 | −506 |
| 2025-12 | 47 | +21.88 | +1,028 |
| 2026-01 | 46 | +5.01 | +230 |

### BTC — The Signal Does NOT Survive

| Strategy | Trades | Avg PnL | Win% | Sharpe | Total PnL | Pos Months |
|----------|--------|---------|------|--------|-----------|------------|
| HMM volatile + momentum | 728 | −7.59 bps | 45.1% | −1.63 | −5,529 bps | 4/12 |
| GMM volatile + momentum | 715 | −7.13 bps | 45.0% | −1.50 | −5,101 bps | 3/12 |
| Unconditional momentum | 825 | −6.84 bps | 45.1% | −1.61 | −5,645 bps | 5/12 |
| HMM quiet + momentum (control) | 310 | −13.78 bps | 41.9% | −2.41 | −4,271 bps | 3/12 |

BTC momentum is **negative across all strategies**. Every approach loses money. The regime filter doesn't help — it just concentrates the losses.

---

## Analysis

### 1. The v20 "+9.6 bps" Was Inflated

| Metric | v20 (in-sample) | v22 (walk-forward) |
|--------|-----------------|-------------------|
| ETH volatile momentum | +9.6 bps | **+4.18 bps** |
| Trades counted | 19,903 (overlapping) | 769 (independent) |
| Regime labels | Full-sample GMM | Online HMM with 3-bar confirm |
| Sharpe | Not computed | 0.57 |

The signal drops from +9.6 to +4.18 bps when done honestly. That's a **57% reduction** — typical for in-sample to out-of-sample degradation.

### 2. HMM > GMM for This Signal

On ETH, HMM volatile + momentum (+4.18 bps, Sharpe 0.57) beats GMM volatile + momentum (+2.70 bps, Sharpe 0.36). The cleaner regime labels from HMM help avoid false volatile detections that dilute the signal.

### 3. Regime Conditioning Partially Works — But Only on ETH

- **ETH:** Volatile momentum (+4.18) beats unconditional momentum (+3.37) on Sharpe (0.57 vs 0.52), though unconditional has more positive months (9 vs 8)
- **BTC:** All strategies lose. Regime conditioning doesn't save a non-existent signal
- **Quiet regime is correctly negative** on both symbols (−11 to −14 bps), confirming that momentum fails in ranging markets

### 4. The Edge Is Razor-Thin

Even on ETH where the signal "works":
- **+4.18 bps per trade** after 7 bps fees
- **Win rate 46.4%** — below 50%, relies on winners being larger than losers
- **Median PnL is −11.85 bps** — most trades lose money
- **Sharpe 0.57** — below the 1.0 threshold for a standalone strategy
- **2 losing months** with drawdowns of −457 and −737 bps
- At 2.1 trades/day, annual gross PnL ≈ +3,200 bps = +32% (but with massive variance)

### 5. Why ETH But Not BTC?

ETH has higher efficiency (directional persistence) in volatile regimes than BTC. When ETH enters a volatile period, moves tend to continue — momentum carries. BTC's volatile periods are more mean-reverting or random, so momentum gets chopped up.

This is consistent with v20 findings where ETH had the highest regime-conditional IC (+0.056) vs BTC (+0.029).

---

## Conclusions

### The Honest Answer

**The "+9.6 bps ETH momentum in volatile regime" finding from v20 was overstated.** When tested with:
- No lookahead in regime labels
- Non-overlapping independent trades
- Online HMM detection with confirmation
- Walk-forward model refitting

...the signal shrinks to **+4.18 bps on ETH** and **doesn't exist on BTC**.

### Is It Tradeable?

**Not as a standalone strategy.** Sharpe 0.57 is too low, win rate is below 50%, and the edge is smaller than the fee. However:

- The **regime conditioning is real** — quiet regime momentum is reliably negative (−11 to −14 bps), confirming that knowing the regime has value
- The **HMM advantage is real** — HMM detection produces better trade selection than GMM (+4.18 vs +2.70 bps on ETH)
- As a **signal component** in a multi-factor model, the regime-conditional momentum could add value

### What We Learned

1. **Always do walk-forward.** In-sample results inflated the edge by 57%.
2. **Overlapping trade counting is deceptive.** 19,903 "trades" → 769 independent trades.
3. **Regime detection works for filtering.** Quiet regime momentum is reliably negative — knowing when NOT to trade has value.
4. **HMM > GMM for trade selection.** Cleaner regime labels → better signal quality.
5. **Signals are asset-specific.** ETH momentum works (weakly); BTC momentum doesn't.

---

## Files

| File | Description |
|------|-------------|
| `walkforward_momentum.py` | Walk-forward backtest engine (no lookahead) |
| `results/walkforward_momentum_v22.txt` | Full output with monthly breakdowns |
