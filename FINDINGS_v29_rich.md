# FINDINGS v29-rich: Exhaustive Feature Extraction & Multi-Horizon Analysis

## Experiment Design

**Goal**: Extract every possible feature from all 4 tick-level data streams at 6 time horizons, then systematically identify which features, horizons, and data streams actually matter for regime switch prediction.

**Data**: BTCUSDT, May 11-24 2025 (14 days, 1.2M seconds, 26.3M trades, 34K liquidations, 9.4M ticker updates)

**Horizons tested**: 10s, 30s, 60s, 300s (5m), 900s (15m), 3600s (1h)

**Target**: Regime switch (vol spike > 2× rolling median) within 300s (5 min)

**Pipeline**: 3-phase — (1) univariate screening, (2) redundancy removal, (3) ML on survivors

## Raw Data Streams

| Stream | Source | Fields Used | Update Rate |
|--------|--------|-------------|-------------|
| Trades | csv.gz | price, size, side, tickDirection | ~2.4M/day |
| Liquidations | jsonl.gz | timestamp, side, volume, price | ~2.4K/day |
| Ticker (OI) | jsonl.gz | openInterestValue | ~every 1.6s |
| Ticker (FR) | jsonl.gz | fundingRate | ~every 60s |
| Ticker (Book) | jsonl.gz | bid1Price, bid1Size, ask1Price, ask1Size | ~every 1.6s |
| Ticker (Basis) | jsonl.gz | markPrice, indexPrice | ~every 1.6s |

## Feature Inventory

**140 total features** generated (122 valid after NaN filtering):

| Category | Features per Horizon | Description |
|----------|---------------------|-------------|
| **Volatility** | vol | Realized vol (std of log returns) |
| **Trade activity** | trade_count, trade_notional, large_count | Volume, $ volume, large trades |
| **Trade microstructure** | buy_ratio, buy_not_ratio, avg_trade_size, tick_imbalance, vwap_dev | Aggressor side, size, tick direction, VWAP |
| **Liquidations** | liq_count, liq_notional, liq_buy_ratio, liq_avg_size | Count, $, side, avg size |
| **Open Interest** | oi_delta, oi_delta_pct, oi_accel | Change, % change, acceleration |
| **Funding Rate** | fr, fr_abs, fr_delta | Level, magnitude, change |
| **Spread/Book** | spread_bps, book_imbalance | Spread, bid/ask imbalance |
| **Basis** | basis_bps | Futures premium vs index |
| **Cross-stream** | liq_to_trade_ratio, oi_delta_to_trade | Relative intensity |
| **Clock** | hour_of_day, day_of_week, fr_time_to_funding | Time features |

## Phase 1: Univariate Screening

### Top 10 Features by Univariate AUC

| Rank | Feature | Uni AUC | Stream | Horizon |
|------|---------|---------|--------|---------|
| 1 | vol_900s | 0.5645 | trades | 900s |
| 2 | trade_count_900s | 0.5547 | trades | 900s |
| 3 | vol_3600s | 0.5505 | trades | 3600s |
| 4 | trade_notional_900s | 0.5500 | trades | 900s |
| 5 | trade_notional_3600s | 0.5490 | trades | 3600s |
| 6 | large_count_900s | 0.5488 | trades | 900s |
| 7 | vol_300s | 0.5483 | trades | 300s |
| 8 | trade_count_3600s | 0.5476 | trades | 3600s |
| 9 | large_count_3600s | 0.5468 | trades | 3600s |
| 10 | trade_count_300s | 0.5416 | trades | 300s |

### By Data Stream

| Stream | Best AUC | Mean AUC | # Features |
|--------|----------|----------|------------|
| **Trades** | **0.5645** | 0.5164 | 54 |
| **Liquidations** | **0.5387** | 0.5106 | 27 |
| Funding Rate | 0.5127 | 0.5062 | 13 |
| Open Interest | 0.5109 | 0.5048 | 20 |
| Book | 0.5075 | 0.5033 | 6 |
| Clock | 0.5070 | 0.5043 | 2 |

### By Horizon

| Horizon | Best AUC | Mean AUC | # Features |
|---------|----------|----------|------------|
| 10s | 0.5111 | 0.5048 | 18 |
| 30s | 0.5166 | 0.5053 | 19 |
| 60s | 0.5240 | 0.5065 | 21 |
| **300s** | **0.5483** | **0.5119** | 21 |
| **900s** | **0.5645** | **0.5198** | 21 |
| **3600s** | **0.5505** | **0.5195** | 19 |

**Key finding**: Short horizons (10s, 30s, 60s) have almost no univariate signal. The signal lives at 300s–3600s (5min–1hr).

## Phase 2: Redundancy Removal

Of 70 features with AUC > 0.505, **23 were removed** as redundant (|corr| > 0.90):

**Major redundancy clusters**:
- `fr_abs` at all horizons are 0.91-0.94 correlated → keep only `fr_abs_3600s`
- `trade_count` ≈ `trade_notional` ≈ `large_count` at same horizon (0.94-0.99) → keep `trade_count` or `vol`
- `oi_delta` ≈ `oi_delta_pct` (1.00 corr) → keep one
- `tick_imbalance` ≈ `buy_ratio` (1.00 corr) → keep one
- `liq_notional` ≈ `liq_count` at 3600s (0.91) → keep one

**47 non-redundant signal features survived.**

## Phase 3: ML Results (GBM, 200 trees)

### Overall: AUC = 0.6002 (47 features)

### Top 10 by GBM Importance

| Rank | Feature | GBM Imp | Uni AUC | Stream | Horizon |
|------|---------|---------|---------|--------|---------|
| 1 | **vol_900s** | **0.1048** | 0.5645 | trades | 900s |
| 2 | **fr_time_to_funding** | **0.0973** | 0.5073 | FR | clock |
| 3 | vol_3600s | 0.0733 | 0.5505 | trades | 3600s |
| 4 | fr_abs_3600s | 0.0620 | 0.5127 | FR | 3600s |
| 5 | avg_trade_size_3600s | 0.0540 | 0.5161 | trades | 3600s |
| 6 | trade_notional_3600s | 0.0483 | 0.5490 | trades | 3600s |
| 7 | trade_notional_900s | 0.0454 | 0.5500 | trades | 900s |
| 8 | liq_notional_3600s | 0.0443 | 0.5387 | liq | 3600s |
| 9 | liq_avg_size_3600s | 0.0412 | 0.5320 | liq | 3600s |
| 10 | oi_delta_3600s | 0.0374 | 0.5065 | OI | 3600s |

### GBM Importance by Data Stream

| Stream | Total Importance |
|--------|-----------------|
| **Trades** | **0.479** |
| **Liquidations** | **0.212** |
| **Funding Rate** | **0.159** |
| Open Interest | 0.099 |
| Book | 0.041 |
| Clock | 0.010 |

### GBM Importance by Horizon

| Horizon | Total Importance |
|---------|-----------------|
| 10s | 0.002 |
| 30s | 0.003 |
| 60s | 0.009 |
| **300s** | **0.119** |
| **900s** | **0.349** |
| **3600s** | **0.411** |
| clock | 0.107 |

### Ablation: How Many Features Do You Need?

| N Features | AUC | Notes |
|------------|-----|-------|
| 3 | 0.5862 | vol_900s, fr_time_to_funding, vol_3600s |
| 5 | 0.5812 | + fr_abs_3600s, avg_trade_size_3600s |
| 10 | 0.5935 | + liq and OI features |
| 15 | 0.5911 | diminishing returns |
| 20 | 0.5800 | noise starts hurting |
| **47 (all)** | **0.6003** | full model |

**Key finding**: Just 3 features get you to 0.586 AUC. Top 10 gets 0.594. More features help marginally but the core signal is concentrated.

### Feature Set Comparison

| Feature Set | # Features | AUC |
|-------------|-----------|-----|
| **Trades only** | **21** | **0.6004** |
| Trades + Liq | 36 | 0.5858 |
| Trades + Liq + OI | 42 | 0.5924 |
| All streams | 47 | 0.6002 |

**Surprising finding**: Trades-only features (21) achieve the same AUC as all 47 features. Adding liq/OI/FR doesn't help — and can even hurt slightly (noise).

### Per-Horizon Ablation (all streams, single horizon)

| Horizon | # Features | AUC |
|---------|-----------|-----|
| 10s | 5 | 0.5317 |
| 30s | 5 | 0.5371 |
| 60s | 4 | 0.5397 |
| 300s | 9 | 0.5597 |
| **900s** | **13** | **0.5961** |
| 3600s | 9 | 0.5708 |
| clock | 2 | 0.5394 |

**900s (15 min) is the single best horizon** — alone it achieves 0.596 AUC, nearly matching the full model.

## Key Conclusions

### 1. Horizon matters more than data stream
- **900s and 3600s horizons dominate** (76% of total GBM importance)
- 10s/30s/60s features are essentially noise for regime prediction
- The old v29 script using only 60s features was leaving most of the signal on the table

### 2. Trades dominate, other streams add little
- Trade-derived features alone match the full model (AUC 0.600)
- Liquidations have univariate signal (AUC 0.539) but don't improve ML when combined with trades
- OI and FR have weak univariate signal but FR timing (`fr_time_to_funding`) is #2 in GBM importance — it captures the funding cycle effect

### 3. Massive redundancy in raw features
- 140 → 122 (NaN filter) → 70 (signal filter) → 47 (redundancy removal)
- `trade_count ≈ trade_notional ≈ large_count` at same horizon (0.94-0.99 corr)
- `oi_delta ≈ oi_delta_pct` (1.00 corr)
- `tick_imbalance ≈ buy_ratio` (1.00 corr)

### 4. The "sweet spot" feature set
For production, use ~10 features at 900s/3600s horizons:
1. `vol_900s` — realized volatility 15min
2. `fr_time_to_funding` — seconds to next funding
3. `vol_3600s` — realized volatility 1hr
4. `fr_abs_3600s` — funding rate magnitude 1hr
5. `avg_trade_size_3600s` — mean trade size 1hr
6. `trade_notional_3600s` — dollar volume 1hr
7. `trade_notional_900s` — dollar volume 15min
8. `liq_notional_3600s` — liquidation $ 1hr
9. `liq_avg_size_3600s` — avg liq size 1hr
10. `oi_delta_3600s` — OI change 1hr

### 5. Next steps
- Validate these findings on other symbols (ETH, SOL, DOGE, XRP)
- Test on out-of-sample dates (Jun-Aug 2025)
- The 900s horizon finding suggests regime switches are predictable ~15min ahead
- Consider adding 1800s (30min) and 7200s (2hr) horizons to bracket the sweet spot
