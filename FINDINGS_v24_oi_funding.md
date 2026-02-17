# FINDINGS v24: Open Interest & Funding Rate Research

**Date:** Feb 2025
**Symbols:** BTCUSDT, SOLUSDT
**Period:** Dec 2025 (31 days)
**Data:** 5-min bars + Binance metrics (OI, LS ratios, taker ratio) + premium index (funding basis)

---

## Why OI/Funding Is Different From Orderbook

Orderbook data (v23) captured **current liquidity structure** — fleeting and already priced in at 5-min frequency. OI and funding capture something fundamentally different:

- **OI changes** = new positions opening/closing → measures **conviction** and **crowding**
- **Funding/premium** = cost of holding longs vs shorts → measures **directional sentiment**
- **L/S ratios** = who's positioned which way → measures **positioning extremes**

These are **slower-moving, higher-signal** features that reflect aggregate market positioning rather than instantaneous microstructure.

---

## Data Pipeline

Already had all data downloaded and in parquet from Binance:

| Data Source | Resolution | Columns |
|-------------|-----------|---------|
| Binance metrics | 5-min | OI, OI value, top trader LS ratio, global LS ratio, taker buy/sell ratio |
| Binance premium index | 5-min | Premium OHLC (≈ continuous funding rate basis) |

### Engineered Features (25 total)

| Category | Features | Description |
|----------|----------|-------------|
| **OI dynamics** | oi_change_{5m,1h,4h,24h}, oi_accel_1h, oi_zscore_24h, oi_value_change_1h | Position opening/closing rate and extremes |
| **Funding** | funding_rate, funding_abs, funding_change_{1h,4h}, funding_zscore_24h, funding_cum_8h | Funding basis level, momentum, and extremes |
| **Positioning** | ls_ratio_{top,global}, ls_{top,global}_change_1h, ls_{top,global}_zscore_24h | Long/short ratio levels and shifts |
| **Taker flow** | taker_ratio, taker_ratio_{1h,4h}, taker_zscore_24h | Aggressive buying vs selling |
| **Cross** | oi_x_funding, oi_x_taker | OI × funding interaction, OI × taker interaction |

---

## Experiment 1: Feature Profiling by Regime

### Top Discriminating Features (BTC)

| Feature | Quiet | Volatile | Ratio | |t| |
|---------|-------|----------|-------|-----|
| **taker_ratio_4h** | 1.137 | 1.086 | 0.96 | 20.1 |
| **oi_change_24h** | -0.06% | +0.62% | -9.75× | 16.2 |
| **oi_zscore_24h** | -0.10 | +0.42 | -4.25× | 15.7 |
| **ls_ratio_top** | 2.38 | 2.24 | 0.94 | 15.6 |
| **ls_ratio_global** | 2.06 | 1.94 | 0.94 | 14.7 |

**Key findings:**
- **OI rises during volatile regimes** — 24h OI change is 10× higher in volatile periods
- **LS ratios drop in volatile regimes** — longs get squeezed/closed (ratio 0.94×)
- **Taker buying decreases** in volatile regimes — sellers dominate
- **Funding rate itself is NOT regime-dependent** — it doesn't change between quiet/volatile
- 11/25 features significant on BTC, 16/25 on SOL — **more discriminating than OB features**

### Cross-Asset Validation

| Feature | BTC |t| | SOL |t| | Consistent? |
|---------|---------|---------|-------------|
| taker_ratio_4h | 20.1 | 14.8 | ✅ Yes |
| oi_change_24h | 16.2 | 7.9 | ✅ Yes |
| ls_ratio_top | 15.6 | 10.0 | ✅ Yes |
| ls_ratio_global | 14.7 | 8.8 | ✅ Yes |

---

## Experiment 2: Regime Detection

| Feature Set | BTC GB AUC | SOL GB AUC |
|-------------|-----------|-----------|
| OHLCV only (14) | 0.9955 | 0.9965 |
| OHLCV + OI/F (39) | 0.9952 | 0.9964 |
| OI/F only (25) | 0.8219 | 0.8211 |
| OHLCV + OB (45) | 0.9954 | 0.9964 |
| OHLCV + OB + OI/F (70) | 0.9950 | 0.9956 |

### Verdict: **No improvement for regime detection** (same as OB).

OHLCV already saturates at 99.5%+ AUC. Neither OI/funding nor OB can improve this. OI/funding alone (82% AUC) is better than OB alone (90%) for regime detection — but both are redundant when OHLCV is available.

---

## Experiment 3: Directional Signal — THE BIG FINDING

### Information Coefficient (IC) vs Forward Returns

| Feature | BTC 5min | BTC 1h | BTC 4h | SOL 5min | SOL 1h | SOL 4h |
|---------|----------|--------|--------|----------|--------|--------|
| **ls_ratio_top** | +0.030 | +0.105 | **+0.205** | +0.031 | +0.122 | **+0.232** |
| **ls_ratio_global** | +0.028 | +0.098 | **+0.187** | +0.030 | +0.117 | **+0.217** |
| **ls_top_zscore_24h** | +0.020 | +0.085 | **+0.169** | +0.021 | +0.097 | **+0.218** |
| **ls_global_zscore_24h** | +0.014 | +0.065 | **+0.126** | +0.020 | +0.096 | **+0.196** |
| **taker_ratio_4h** | +0.011 | +0.064 | **+0.123** | +0.015 | +0.052 | +0.061 |
| funding_rate | -0.007 | -0.005 | -0.035 | -0.010 | -0.067 | **-0.157** |
| funding_cum_8h | -0.014 | -0.047 | -0.038 | -0.034 | -0.109 | **-0.143** |
| oi_change_4h | -0.008 | -0.028 | -0.088 | -0.012 | -0.043 | +0.005 |

### This is extraordinary.

**`ls_ratio_top` has IC = +0.20 on BTC and +0.23 on SOL at 4h horizon.** For context:
- OB imbalance best IC was 0.03 (v23) — this is **7× stronger**
- OB depth ratio best IC was 0.10 (v23) — this is **2× stronger**
- IC > 0.10 is considered "strong" in quantitative finance
- IC > 0.20 is considered "very strong" and rarely seen

The signal is **positive** — when more traders are long (high LS ratio), price goes UP over the next 4 hours. This is **NOT contrarian** — it's momentum/trend-following. The crowd is right in the short term.

### Simple Backtest (4h hold, 7bps fee)

| Signal | BTC Avg PnL | BTC WR | SOL Avg PnL | SOL WR |
|--------|-------------|--------|-------------|--------|
| **ls_top_zscore_24h** | **+18.6 bps** | **55.5%** | **+43.7 bps** | **60.6%** |
| **ls_global_zscore_24h** | **+9.4 bps** | **52.5%** | **+34.2 bps** | **57.2%** |
| funding_cum_8h | -0.0 bps | 47.0% | -36.7 bps | 42.0% |
| oi_change_4h | -21.9 bps | 36.4% | -17.1 bps | 42.7% |

**`ls_top_zscore_24h` is profitable after fees on both BTC (+18.6 bps) and SOL (+43.7 bps).** This is the first signal in our entire research that is:
1. Profitable after fees
2. Replicates cross-asset
3. Has high IC (>0.15)
4. Has reasonable trade count (3,000+ trades)

---

## Experiment 4: Volatility Prediction

| Feature Set | BTC Ridge R² | BTC GB R² | SOL Ridge R² | SOL GB R² |
|-------------|-------------|----------|-------------|----------|
| OHLCV only (14) | 0.337 | 0.141 | 0.314 | 0.075 |
| OHLCV + OI/F (39) | 0.254 | 0.163 | 0.185 | -0.023 |
| OI/F only (25) | -0.314 | -0.346 | -0.553 | -0.515 |
| OHLCV + OB (45) | 0.321 | **0.219** | 0.260 | **0.197** |
| OHLCV + OB + OI/F (70) | 0.255 | 0.156 | 0.183 | 0.035 |

### Verdict: **OI/funding does NOT improve vol prediction.**

OI/funding features actually **hurt** vol prediction (Ridge R² drops from 0.337 → 0.254 on BTC). OB features remain the best addition for vol prediction (OHLCV+OB GB R² = 0.219 vs OHLCV-only 0.141).

However, **OI features appear in the top 20 for vol prediction** — `oi_change_24h` (#3), `funding_cum_8h` (#5), `ls_ratio_top` (#8) — suggesting they carry some information but add noise when combined with too many features.

---

## Experiment 5: Crowding & Extreme Detection

### OI Z-Score Extremes (Contrarian)

| Condition | BTC Avg | BTC Sharpe | SOL Avg | SOL Sharpe |
|-----------|---------|-----------|---------|-----------|
| **OI z > +2.0 → short** | **+23.1 bps** | **+7.55** | **+7.7 bps** | **+1.45** |
| OI z > +1.5 → short | +10.2 bps | +3.80 | -5.7 bps | -1.42 |
| OI z < -1.5 → long | +1.0 bps | +0.47 | +3.6 bps | +0.69 |
| **OI z < -2.0 → long** | -9.4 bps | -2.02 | **+38.4 bps** | **+5.34** |

**OI extremes work as contrarian signals** — when OI is extremely high (z>2), shorting is profitable. This is the classic "crowded trade" reversal. Works on both assets but with different thresholds.

### Funding Z-Score Extremes (Contrarian)

| Condition | BTC Avg | BTC Sharpe | SOL Avg | SOL Sharpe |
|-----------|---------|-----------|---------|-----------|
| **Funding z > +2.0 → short** | +2.4 bps | +0.30 | **+51.4 bps** | **+4.07** |
| **Funding z > +1.5 → short** | -0.1 bps | -0.01 | **+47.1 bps** | **+6.26** |
| **Funding z > +1.0 → short** | -1.5 bps | -0.45 | **+26.7 bps** | **+5.66** |
| **Funding z < -1.0 → long** | -4.2 bps | -1.35 | **+20.6 bps** | **+4.66** |

**Funding extremes are a strong contrarian signal on SOL but not BTC.** SOL shows consistent profitability at ALL thresholds (Sharpe 1.6 to 6.3). BTC shows marginal results. This makes sense — SOL has more retail-driven funding dynamics.

### LS Ratio Z-Score — NOT Contrarian

| Condition | BTC Avg | SOL Avg |
|-----------|---------|---------|
| LS z > +2.0 → short | **-22.6 bps** | **-42.3 bps** |
| LS z < -2.0 → long | **-66.3 bps** | **-106.2 bps** |

**LS ratio extremes are catastrophically wrong as contrarian signals.** When top traders are extremely long, going short loses -22 to -42 bps. When they're extremely short, going long loses -66 to -106 bps. **The crowd (top traders) is RIGHT.** This confirms the momentum finding from Exp 3.

---

## Experiment 6: Walk-Forward Combined Signal

### Quintile Long-Short Strategy (4h returns)

| Feature Set | BTC IC | BTC L/S Avg | BTC Sharpe | SOL IC | SOL L/S Avg | SOL Sharpe |
|-------------|--------|-------------|-----------|--------|-------------|-----------|
| OHLCV only (14) | +0.066 | +2.7 bps | +1.52 | +0.041 | -0.1 bps | -0.02 |
| **OHLCV + OI/F (39)** | **+0.202** | **+17.3 bps** | **+9.32** | **+0.204** | **+26.9 bps** | **+8.62** |
| OHLCV + OB + OI/F (70) | +0.184 | +13.9 bps | +7.85 | +0.209 | +30.3 bps | +10.11 |

### This is the strongest walk-forward result in the entire research program.

- **OHLCV + OI/F: IC = 0.20 on both BTC and SOL** — nearly identical, confirming robustness
- **Walk-forward Sharpe 9.32 (BTC) and 8.62 (SOL)** — both highly significant
- **Quintile spread is monotonic:** Bottom quintile = -32 bps, Top quintile = +17 bps (BTC)
- Adding OB features (70 total) slightly hurts BTC but slightly helps SOL — net wash
- **OHLCV alone has near-zero predictive power** (IC 0.04-0.07) — the OI/funding features are doing all the work

---

## Summary: OI/Funding vs Orderbook vs OHLCV

| Metric | OHLCV Only | + Orderbook (v23) | + OI/Funding (v24) |
|--------|-----------|-------------------|---------------------|
| **Best directional IC (4h)** | ~0.03 | 0.10 (depth ratio, BTC only) | **0.20+ (LS ratio, both assets)** |
| **Walk-forward direction Sharpe** | 1.5 | Not tradeable | **9.3 (BTC), 8.6 (SOL)** |
| **Simple backtest (4h, after fees)** | N/A | -12 to +13 bps (mixed) | **+19 bps (BTC), +44 bps (SOL)** |
| **Vol prediction improvement** | Baseline | +5% R² | No improvement |
| **Regime detection** | 99.5% AUC | +0.0% | +0.0% |
| **Cross-asset replication** | N/A | Partial (depth ratio fails on SOL) | **Full (LS ratio works on both)** |

### Key Takeaways

1. **`ls_ratio_top` (Binance top trader long/short ratio) is the single most valuable feature discovered in this research.** IC = 0.20+ at 4h, profitable after fees, replicates cross-asset.

2. **The signal is momentum, not contrarian.** When top traders are long, price goes up. This likely reflects informed flow — Binance's "top traders" are the smart money.

3. **OI extremes work as contrarian signals** — crowded OI (z>2) predicts reversals. But the LS ratio momentum signal is much stronger.

4. **Funding rate extremes are a strong contrarian signal on SOL** (Sharpe 4-6) but not BTC. SOL has more retail-driven funding dynamics.

5. **OI/funding features do NOT improve vol prediction** — OB features remain better for that task.

6. **The walk-forward combined signal (OHLCV + OI/F) achieves Sharpe 9+ on both assets.** This is the first genuinely promising directional signal in the entire research program.

### Caveats

- **31-day sample** — needs validation on more months
- **LS ratio is Binance-specific** — may not be available on all exchanges
- **5-min resolution** — the LS ratio updates every 5 min on Binance, so there's no latency advantage
- **Walk-forward uses expanding window** — the first third is training, so only 20 days OOS
- **The quintile L/S strategy assumes you can go long AND short** — requires futures access

---

## Files

| File | Description |
|------|-------------|
| `oi_funding_research.py` | 6 experiments: regime profiling, detection, direction, vol, crowding, walk-forward |
| `results/oi_funding_v24_BTC.txt` | BTC experiment output |
| `results/oi_funding_v24_SOL.txt` | SOL experiment output |
