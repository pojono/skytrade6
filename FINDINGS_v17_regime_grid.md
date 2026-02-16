# FINDINGS v17 — Regime-Filtered Grid Bot

## Motivation

v15/v16 reduced grid bot losses significantly but never reached positive PnL.
The core problem: grid bots profit from mean-reversion but bleed in trends.
SOL went $95→$294, BTC $74K→$126K, ETH $1.4K→$5K — strong uptrends.

**Key insight from v8:** Markets are ranging 85-94% of the time over 13 months.
Trending is only 1-2.5% of bars — but those rare bars cause nearly ALL the losses.

**Strategy:** Pause the grid during dangerous regimes (high vol, trending),
only run when conditions favor mean-reversion. Combine with:
- N5b informed rebalance (v16 winner)
- Short rebalance intervals (minimize inventory holding time)
- Force-close inventory when transitioning to paused state

## Method

- **Vol pause:** Pause when Ridge-predicted vol > threshold × median
- **Efficiency pause:** Pause when backward-looking efficiency_4h > threshold (trending)
- **ADX pause:** Pause when ADX_4h > 0.30 (strong trend)
- **Parkvol pause:** Pause when parkvol_1h > 1.5× median (no ML, pure backward)
- **Informed rebalance:** Rebalance within 1h when composite informed flow z > 1.5
- **Short rebalance:** 30min, 1h, 2h base rebalance intervals
- **Force-close on pause:** When grid transitions to paused, close all inventory at market

## Results — SOLUSDT (387 days, $95→$294)

### Progression of improvements

| Strategy | PnL | PnL/day | Fills | vs Baseline |
|----------|-----|---------|-------|-------------|
| S0: Fix 1% (24h) | **-$290,060** | -$749 | 169K | — |
| R2: Vol pause (1.5x) | -$21,337 | -$55 | 40K | 93% better |
| C1: VolPause+InfRebal | -$3,937 | -$10 | 12K | 98.6% better |
| U2: 1%+1hR+VolP | -$1,499 | -$3.87 | 7K | 99.5% better |
| **V1: 1%+30mR+VolP** | **-$959** | **-$2.48** | 3K | **99.7% better** |

### Best strategies (top 5)

| Strategy | PnL | PnL/day | Fills | MaxDD |
|----------|-----|---------|-------|-------|
| V1: 1%+30mR+VolP | -$959 | -$2.48 | 3,158 | -$962 |
| V3: 1%+30mR+V\|EP | -$1,043 | -$2.69 | 3,038 | -$1,044 |
| V5: 1%+30mR+WideVP | -$1,261 | -$3.26 | 1,728 | -$1,261 |
| V4: 1%+1hR+VolP+Inf | -$1,294 | -$3.34 | 5,002 | -$1,296 |
| V2: 1%+1hR+V\|EP | -$1,299 | -$3.35 | 6,398 | -$1,300 |

## Results — BTCUSDT (387 days, $74K→$126K)

| Strategy | PnL | PnL/day | Fills | vs Baseline |
|----------|-----|---------|-------|-------------|
| S0: Fix 1% (24h) | **-$39.0M** | -$101K | 75K | — |
| V1: 1%+30mR+VolP | -$190K | -$491 | 378 | 99.5% better |
| U2: 1%+1hR+VolP | -$202K | -$521 | 488 | 99.5% better |
| C1: VolPause+InfRebal | -$304K | -$785 | 1,586 | 99.2% better |

BTC still loses ~$190-300K with best strategies. The 1% grid on BTC = $1,000 spacing,
which is too wide for 30min windows — only 378 fills in 387 days (< 1/day).

## Results — ETHUSDT (387 days, $1.4K→$5K)

| Strategy | PnL | PnL/day | Fills | vs Baseline |
|----------|-----|---------|-------|-------------|
| S0: Fix 1% (24h) | **-$3.73M** | -$9,629 | 138K | — |
| V1: 1%+30mR+VolP | -$14,558 | -$37.59 | 1,818 | 99.6% better |
| U2: 1%+1hR+VolP | -$18,947 | -$48.92 | 3,718 | 99.5% better |
| V4: 1%+1hR+VolP+Inf | -$16,574 | -$42.79 | 3,074 | 99.6% better |

## Key Findings

### 1. Regime filtering + short rebalance reduces losses by 99.5-99.7%

The combination of:
- **Pause during high vol** (predicted vol > 1.5× median, ~16% of bars)
- **Short rebalance** (30min-1h instead of 24h)
- **Force-close on pause** (don't hold inventory through dangerous periods)

Reduces losses from hundreds of thousands to low single-digit thousands on SOL.

### 2. Rebalance frequency is the most powerful lever

| Rebalance | SOL PnL | SOL PnL/day |
|-----------|---------|-------------|
| 24h + informed | -$3,937 | -$10.17 |
| 2h | -$3,063 | -$7.91 |
| 1h | -$1,499 | -$3.87 |
| 30min | -$959 | -$2.48 |

Shorter rebalance = less inventory drift = less loss. But diminishing returns —
going from 1h to 30min only saves $1.40/day.

### 3. Still not profitable — fundamental limitation

Even with 99.7% loss reduction, the grid bot is still slightly negative.
The remaining ~$1K/year loss on SOL comes from:
- **Rebalance cost:** Each rebalance crystallizes a small directional loss
- **Asymmetric fills:** In an uptrend, more buy fills than sell fills complete
- **Fee drag:** Even at 2bps maker, fees add up with frequent rebalancing

### 4. The grid bot needs a fundamentally different market

The 13-month test period (Jan 2025 - Jan 2026) was a **strong bull market**.
Grid bots are structurally short gamma — they sell rallies and buy dips.
In a sustained uptrend, this is a losing proposition regardless of regime filtering.

A grid bot would likely be profitable in a **sideways/ranging market** where:
- Price oscillates within a range for extended periods
- No sustained directional trend
- Volatility is moderate (enough fills, not too much drift)

### 5. Cross-asset comparison

| Symbol | Best PnL | Best PnL/day | Loss Reduction |
|--------|----------|-------------|----------------|
| SOL | -$959 | -$2.48 | 99.7% |
| ETH | -$14,558 | -$37.59 | 99.6% |
| BTC | -$190,185 | -$491 | 99.5% |

SOL is the most grid-friendly (highest fill rate, smallest losses).
BTC is the worst (too expensive per level, too few fills).

## Conclusion

We've pushed the grid bot to its theoretical limit for this market regime:
- **v15 baseline:** -$290K on SOL
- **v16 informed rebalance:** -$28K (90% better)
- **v17 regime filter + short rebalance:** -$959 (99.7% better)

The remaining loss is structural — the grid bot is fundamentally mismatched
with a trending market. To achieve positive PnL, we would need either:
1. A ranging market period to test on
2. A trend-following overlay (shift grid center with the trend)
3. A completely different strategy architecture (not a grid bot)

## Files

| File | Description |
|------|-------------|
| `grid_bot_v17.py` | Regime-filtered grid bot with 30+ strategies |
| `results/grid_v17_SOL.txt` | SOL results (30+ strategies) |
| `results/grid_v17_BTC.txt` | BTC results |
| `results/grid_v17_ETH.txt` | ETH results |
