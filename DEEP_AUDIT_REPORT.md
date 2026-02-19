# Deep Self-Audit Report — ALL SIGNALS ARE INVALID

**Date:** 2025-08-07 (updated 2026-02-19)  
**Script:** `deep_audit.py` (8.3 min runtime)  
**Verdict: 🔴 THE ENTIRE EDGE IS A SIMULATION BUG. NO REAL TRADING SIGNAL EXISTS.**

---

## Executive Summary

The previous validation report (`VALIDATION_REPORT.md`) concluded signals were real. **That conclusion was wrong.** A deep self-audit reveals two critical simulation bugs that together create a phantom edge of +3 to +8 bps per trade where none exists.

| Test | Original sim | Fixed sim | Verdict |
|---|---|---|---|
| Real signal (ETH) | **+3.2 bps** | **-4.9 bps** | 🔴 Bug, not edge |
| Random signal (ETH) | **+3.2 bps** | **-4.6 bps** | 🔴 Signal is meaningless |
| Inverted signal (ETH) | **+3.4 bps** | **-4.2 bps** | 🔴 Direction doesn't matter |
| Real signal (DOGE) | **+7.9 bps** | **-2.1 bps** | 🔴 Bug, not edge |
| Random signal (DOGE) | **+7.9 bps** | **-2.2 bps** | 🔴 Signal is meaningless |

**When the simulation bugs are fixed, every strategy is negative on every coin.**

---

## Bug #1: Same-Bar Fill + Exit (THE KILLER BUG)

### The Problem

In `sim_trade()`, the exit detection loop starts from `fi` (the fill bar):

```python
for k in range(fi, min(fi+max_hold, len(bars))):  # BUG: starts at fi
```

This means on the **same 1-minute bar** where the limit order fills, the code also checks for trailing stop activation and exit. In a single bar:

1. Price hits the limit → order fills (e.g., at the bar's low)
2. Price hits the high → trailing stop activates and ratchets up
3. Price comes back down → trailing stop triggers exit at a profit

**This is impossible in reality.** With 1-minute bars, you don't know the intra-bar price sequence. The low might happen after the high, not before. You cannot fill at the low AND exit at the high of the same bar.

### The Fix

Exit detection must start from `fi + 1` (the bar **after** the fill):

```python
for k in range(fi + 1, min(fi+max_hold, len(bars))):  # FIXED
```

### Impact

| Symbol | Original | Fixed | Delta |
|---|---|---|---|
| ETHUSDT | +3.2 bps | **-4.9 bps** | **-8.1 bps** |
| DOGEUSDT | +7.9 bps | **-2.1 bps** | **-10.0 bps** |

---

## Bug #2: Trailing Stop Intra-Bar Lookahead

### The Problem

Even on bars after the fill, the trailing stop logic has an ordering problem within each bar:

```python
# For a LONG trade on bar k:
cp = (b['high'] - lim) / lim           # 1. Use HIGH to compute profit
if cp > best_profit: best_profit = cp   # 2. Update best profit from HIGH
# ... activate trail, update trail stop using HIGH ...
if b['low'] <= current_sl:             # 3. Check if LOW hits the stop
    ep = current_sl; break              # 4. Exit at the stop price
```

This assumes price goes to the **high first** (setting the trail stop higher), then comes back to the **low** (triggering exit). In reality, the low might happen before the high, meaning the trail stop would never have been set that high.

### Impact

This bug inflates trail exit profitability. In the original sim:
- **99.3%** of all exits are trail exits (ETH: 9103/9166)
- Trail exits average **+3.6 bps** (ETH) and **+8.1 bps** (DOGE)
- SL exits: only 48 trades on ETH, 33 on DOGE

The trail stop is doing almost all the work, and it's systematically biased to be too profitable.

---

## Proof: Signals Are Meaningless

### Random Direction Test

Using the exact same entry timing (signal fires) but **random long/short direction**:

| | Real Signal | Random Direction | Difference |
|---|---|---|---|
| ETH avg bps | +3.2 | +3.2 | **0.0** |
| DOGE avg bps | +7.9 | +7.9 | **0.0** |

**The signal direction contributes exactly zero edge.**

### Inverted Signal Test

Flipping long↔short (doing the opposite of what the signal says):

| | Real Signal | Inverted Signal | Difference |
|---|---|---|---|
| ETH avg bps | +3.2 | +3.4 | **+0.2** (inverted is BETTER!) |
| DOGE avg bps | +7.9 | +7.8 | **-0.1** |

**The inverted signal is equally profitable.** The signal is pure noise.

### Every-Bar Entry Test

Entering every 30 bars with no signal at all:

| Direction | ETH avg bps | DOGE avg bps |
|---|---|---|
| Alternating | +3.4 | +7.9 |
| Long only | +3.4 | +8.0 |
| Short only | +2.6 | +7.8 |

**You don't even need a signal. The buggy trade structure profits on every bar.**

---

## Why the Bug Creates Phantom Profits

The mechanism is:

1. **Limit offset** places order 15 bps below close (for longs)
2. Price dips to fill the order on bar `fi`
3. **On that same bar**, the code sees the high of the bar
4. Since the fill was at the low and the high is above, the trailing stop activates
5. The trail stop is set just below the high
6. The bar's low (which already happened!) "triggers" the trail stop
7. Net result: you "bought at the low and sold near the high" of the same bar

This is a **perfect intra-bar round-trip** that's impossible to execute in practice.

---

## Offset Analysis (Confirms the Bug)

| Offset | Fill Rate | Avg bps | Notes |
|---|---|---|---|
| 0.00% | 100% | **-1.9** | No offset → no "buy below market" → negative |
| 0.05% | 93% | **-0.8** | Small offset → small phantom edge |
| 0.10% | 82% | **+1.7** | Larger offset → larger phantom edge |
| 0.15% | 73% | **+3.2** | Our default → our "edge" |
| 0.20% | 63% | **+4.4** | Even larger → even more phantom edge |
| 0.30% | 47% | **+6.4** | Huge offset → huge phantom edge |

The "edge" scales linearly with offset. This is because a larger offset means a bigger gap between fill price and bar high, creating a larger phantom intra-bar profit.

---

## Without Trailing Stop

Removing the trailing stop (TP/SL only):

| Symbol | With Trail | Without Trail |
|---|---|---|
| ETH | +3.2 bps, 63% WR | **+0.3 bps, 80% WR** |
| DOGE | +7.9 bps, 84% WR | **+4.3 bps, 89% WR** |

Without the trail, the edge drops dramatically. The remaining small positive is from the same-bar TP bug (TP can also trigger on the fill bar).

---

## Conclusion

**All 201 signals discovered across the v42 research series are invalid.** The apparent edge was entirely produced by two simulation bugs:

1. **Same-bar fill + exit**: The exit loop starts on the fill bar, allowing impossible intra-bar round-trips
2. **Intra-bar trail lookahead**: The trailing stop uses the bar's high to set the stop, then the bar's low to trigger it, assuming a favorable price sequence

When these bugs are fixed:
- **Every strategy is negative** on both ETH and DOGE
- **Random signals perform identically** to real signals
- **Inverted signals perform identically** to real signals
- **No signal filter is needed** — the bug profits on every bar

### What This Means

- The previous `VALIDATION_REPORT.md` conclusions are **all invalidated**
- The 8-point validation passed because it tested the buggy sim against itself
- The multi-period, multi-coin robustness was real — but it was robustness of the **bug**, not of any trading edge
- BTC "failed" because BTC has tighter spreads/lower volatility, making the intra-bar phantom profit smaller relative to fees

### Files

- **Deep audit script:** `deep_audit.py`
- **Invalidated report:** `VALIDATION_REPORT.md`
- **Buggy sim function:** `sim_trade()` in all `research_v42b*.py` scripts
