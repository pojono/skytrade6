# Research Findings — BTCUSDT Microstructure Signal

**Date:** 2026-02-15
**Symbol:** BTCUSDT
**Exchange:** Bybit Futures (VIP0)
**Period:** 2025-12-01 → 2025-12-30 (30 days)

## Data Overview

- **1.46 billion trades** across 6 sources (Bybit, Binance, OKX × futures + spot)
- 92 days of daily-partitioned parquet files
- Microsecond timestamp resolution

## Key Statistical Properties (from 01_data_profiling)

| Property | Value |
|----------|-------|
| 1m return kurtosis | 22–43 (extremely fat tails) |
| 1m return skew | ~0 (symmetric) |
| 1m return std | ~6.5 bps |
| Volatility clustering | Very persistent (ACF significant at 120+ min lags) |
| Cross-exchange price corr | 0.999999 |
| Cross-exchange return corr | 0.994–0.996 at 1m |
| Lead-lag at 1m | Symmetric — no clear leader at 1m resolution |

## Microstructure Features (from 02_signal_research)

17 features computed from raw tick data per 5-minute window:

**Top features by Information Coefficient (15m forward return):**

| Feature | IC (Binance) | IC (Bybit) | IC (OKX) | Direction |
|---------|-------------|-----------|---------|-----------|
| count_imbalance | -0.032 | -0.021 | -0.034 | Contrarian |
| vol_imbalance | -0.029 | -0.037 | -0.030 | Contrarian |
| dollar_imbalance | -0.029 | -0.037 | -0.030 | Contrarian |
| large_imbalance | -0.026 | -0.036 | -0.029 | Contrarian |
| close_vs_vwap | -0.022 | -0.022 | -0.022 | Contrarian |
| vol_profile_skew | -0.022 | -0.023 | -0.016 | Contrarian |

**All imbalance signals are contrarian** — buying pressure predicts negative forward returns.
Consistent across all 3 exchanges.

## Strategy: Contrarian Mean-Reversion

**Signal:** Composite rank-based score from 5 features:
- vol_imbalance, dollar_imbalance, large_imbalance, count_imbalance, close_vs_vwap

**Rules:**
- Entry: z-score > 1.0 → SHORT (contrarian to buying pressure)
- Entry: z-score < -1.0 → LONG (contrarian to selling pressure)
- Exit: fixed 4-hour holding period
- Fees: 7 bps round-trip (Bybit VIP0: taker 5 bps + maker 2 bps)

## Results (30-day backtest, Bybit Futures)

| Metric | Value |
|--------|-------|
| Total trades | 161 |
| Trades per day | 5.4 |
| Avg PnL per trade | **+13.68 bps** (net of fees) |
| Total PnL (30 days) | **+2,202 bps** |
| Holding period | 4 hours |
| Entry threshold | 1.0 z-score |

### Parameter sensitivity

Shorter holding periods (15m, 30m, 1h) were consistently negative after fees.
The 4h holding period was the only consistently profitable configuration.

### Regime dependence

Signals are ~2x stronger in low-volatility regimes (IC doubles).
High-vol regimes weaken the signal.

## Caveats

1. **No walk-forward validation yet** — results are in-sample on 30 days
2. **No slippage model** — assumes perfect fills at close price
3. **Single asset** — needs cross-asset validation (ETH, SOL)
4. **Short test period** — 30 days may not capture all market regimes
5. **Annualized projections are extrapolations** — not guarantees

## Next Steps

1. Validate on ETHUSDT and SOLUSDT
2. Walk-forward test (train on first 60 days, test on last 32)
3. Add vol-regime filter (skip high-vol periods)
4. Model slippage and partial fills
5. Test with maker-only entry (2+2 = 4 bps round-trip) using limit orders
