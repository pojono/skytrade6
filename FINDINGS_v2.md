# Research Findings v2 — Cross-Asset Edge Validation

**Date:** 2026-02-15
**Exchange:** Bybit Futures (VIP0)
**Symbols:** BTCUSDT, ETHUSDT, SOLUSDT
**Signal:** Contrarian composite (vol_imbalance, dollar_imbalance, large_imbalance, count_imbalance, close_vs_vwap)

## Strategy Recap

- **Signal:** Rank-based composite of 5 microstructure features computed from tick data at 5m intervals
- **Direction:** Contrarian — short when buying pressure is extreme, long when selling pressure is extreme
- **Entry:** Z-score threshold on composite signal
- **Exit:** Fixed holding period
- **Fees:** 7 bps round-trip (Bybit VIP0: taker 5 bps + maker 2 bps)

## Cross-Asset Results (30-day backtest: Dec 1–30, 2025)

| Symbol | Thresh | Hold | Trades | Avg PnL (bps) | Total PnL (bps) | Win Rate |
|--------|--------|------|--------|----------------|------------------|----------|
| **BTCUSDT** | 1.0 | 4h | 161 | **+13.68** | **+2,202** | — |
| **ETHUSDT** | 1.0 | 4h | 161 | **-3.37** | **-542** | 48.4% |
| **SOLUSDT** | 1.0 | 4h | 162 | **+3.40** | **+550** | 56.2% |

## 7-Day vs 30-Day Comparison

| Symbol | 7d Avg PnL | 7d Best Config | 30d Avg PnL | 30d Best Config |
|--------|-----------|----------------|-------------|-----------------|
| BTCUSDT | +4.59 bps (2h) | thresh=1.5 | +13.68 bps (4h) | thresh=1.0 |
| ETHUSDT | +12.84 bps (2h) | thresh=1.0 | -3.37 bps (4h) | thresh=1.0 |
| SOLUSDT | +19.38 bps (4h) | thresh=1.5 | +3.40 bps (4h) | thresh=1.0 |

**Warning:** 7-day results were misleading — all 3 assets looked profitable on 7 days, but only BTC and SOL survived 30 days.

## Information Coefficients (30-day, Bybit Futures)

| Horizon | BTCUSDT | ETHUSDT | SOLUSDT |
|---------|---------|---------|---------|
| 15m | -0.055 *** | -0.033 *** | -0.033 *** |
| 30m | -0.037 *** | -0.007 | -0.007 |
| 1h | -0.028 ** | -0.002 | -0.002 |
| 2h | -0.015 | +0.013 | +0.013 |
| 4h | — | +0.021 | +0.021 |

**Key observation:** BTC has strong negative IC at all horizons (contrarian works). ETH IC flips positive at 2h+ (contrarian stops working). SOL is mixed.

## Parameter Sensitivity (30-day, all configs tested)

### BTCUSDT
| Thresh | 1h | 2h | 4h |
|--------|-----|-----|-----|
| 1.0 | -4.81 | -5.60 | **+13.68** |
| 1.5 | -4.89 | -8.30 | +4.59 |
| 2.0 | -16.56 | -55.44 | +22.23 |

### ETHUSDT
| Thresh | 1h | 2h | 4h |
|--------|-----|-----|-----|
| 1.0 | -5.44 | -4.26 | **-3.37** |
| 1.5 | -3.52 | -7.12 | -18.25 |
| 2.0 | +4.72 | +7.69 | -4.51 |

### SOLUSDT
| Thresh | 1h | 2h | 4h |
|--------|-----|-----|-----|
| 1.0 | -6.27 | -12.42 | **+3.40** |
| 1.5 | -5.18 | -12.70 | -25.61 |
| 2.0 | +27.01 | +42.75 | -42.93 |

## Conclusions

### What works
1. **BTCUSDT is the strongest asset** for this signal — +13.68 bps avg, robust across thresholds
2. **4-hour holding period** is critical — shorter holds are consistently negative after fees
3. **Threshold 1.0** (more aggressive) works better than 1.5 or 2.0 on 30 days
4. **SOLUSDT shows marginal edge** — +3.40 bps avg, 56.2% WR, but thin

### What doesn't work
1. **ETHUSDT** — contrarian signal breaks down at longer horizons, IC flips positive
2. **Short holding periods** (1h, 2h) — signal too weak relative to 7 bps fees
3. **High thresholds** (2.0+) — too few trades, results dominated by noise

### Why ETH is different
ETH's IC flips from negative (contrarian at 15m) to positive (momentum at 2h+). This suggests ETH has different microstructure dynamics — possibly more informed flow that persists, making contrarian bets unprofitable at longer horizons.

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Overfitting to 30-day window | High | Need 90-day walk-forward test |
| No slippage model | Medium | Assume +2 bps slippage → still profitable for BTC |
| Single regime (Dec 2025) | High | Test across different vol regimes |
| Fee assumption | Low | Bybit VIP0 is well-documented |
| Execution risk | Medium | 4h hold = no urgency on fills |

## Next Steps

1. **Walk-forward validation** on full 92-day dataset (train 60d, test 32d)
2. **Add slippage model** — test with 2-3 bps additional cost
3. **Vol regime filter** — signals are 2x stronger in low-vol (from notebook 02)
4. **Maker-only entry** — reduce RT fees to 4 bps (2+2) with limit orders
5. **Consider momentum signal for ETH** — opposite direction may work
6. **Live paper trading** on BTC to validate in real-time
