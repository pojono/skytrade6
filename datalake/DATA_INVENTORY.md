# Data Inventory

*Updated: 2026-03-06*

**2.2 TB** across **Bybit** (1.8 TB, 154 symbols) and **Binance** (318 GB, 144 symbols).
144 symbols available on both exchanges.

---

## Bybit — 154 symbols, 1.8 TB

| Data Type | File Pattern | Resolution | Date Range | Symbols |
|-----------|-------------|------------|------------|---------|
| Klines (futures) | `_kline_1m.csv` | 1 min | 2024-01-01 → 2026-03-04 | 152 |
| Mark price klines | `_mark_price_kline_1m.csv` | 1 min | 2024-01-01 → 2026-03-04 | 152 |
| Premium index klines | `_premium_index_kline_1m.csv` | 1 min | 2024-01-01 → 2026-03-04 | 152 |
| Funding rate | `_funding_rate.csv` | per settlement | 2024-01-01 → 2026-03-04 | 152 |
| Open interest | `_open_interest_5min.csv` | 5 min | 2024-01-01 → 2026-03-04 | 152 |
| Long/short ratio | `_long_short_ratio_5min.csv` | 5 min | 2024-01-01 → 2026-03-04 | 152 |
| Trades (futures) | `_trades.csv.gz` | tick | 2025-01-01 → 2026-03-04 | 153 |
| Klines (spot) | `_kline_1m_spot.csv` | 1 min | 2025-01-01 → 2026-03-04 | 96 |
| Trades (spot) | `_trades_spot.csv.gz` | tick | 2025-01-01 → 2026-03-04 | 93 |
| Orderbook (futures) | `_orderbook.jsonl.gz` | L2 ob200 | 2025-08-21 → 2026-03-04 | 115 |
| Orderbook (spot) | `_orderbook_spot.jsonl.gz` | L2 ob200 | 2025-07-01 → 2026-03-04 | 93 |

**Per-day sizes (BTCUSDT):** orderbook 227 MB, orderbook spot 98 MB, trades 104 MB, trades spot 7.5 MB, klines ~93 KB, mark price ~68 KB, premium index ~89 KB, OI/LS ratio ~8 KB, funding rate ~100 B.

**Top symbols by size:** ETHUSDT 90 GB, BTCUSDT 89 GB, XRPUSDT 54 GB, SOLUSDT 49 GB, DOGEUSDT 46 GB, HYPEUSDT 35 GB, PIPPINUSDT 30 GB, SUIUSDT 29 GB, ZECUSDT 27 GB, FARTCOINUSDT 26 GB.

---

## Binance — 144 symbols, 318 GB

| Data Type | File Pattern | Resolution | Date Range | Symbols |
|-----------|-------------|------------|------------|---------|
| Klines (futures) | `_kline_1m.csv` | 1 min | 2025-01-01 → 2026-03-04 | 144 |
| Mark price klines | `_mark_price_kline_1m.csv` | 1 min | 2025-01-01 → 2026-03-04 | 144 |
| Premium index klines | `_premium_index_kline_1m.csv` | 1 min | 2025-01-01 → 2026-03-04 | 144 |
| Index price klines | `_index_price_kline_1m.csv` | 1 min | 2025-01-01 → 2026-03-04 | 144 |
| Metrics (FR, OI, LS ratio) | `_metrics.csv` | composite | 2025-01-01 → 2026-03-04 | 144 |
| Klines (spot) | `_kline_1m_spot.csv` | 1 min | 2025-01-01 → 2026-03-04 | 83 |
| Trades (futures) | `_trades.csv.gz` | tick | 2025-01-01 → 2026-03-04 | 105 |
| Trades (spot) | `_trades_spot.csv.gz` | tick | 2025-07-01 → 2026-03-04 | 80 |
| BookDepth | `_bookDepth.csv.gz` | L2 snapshots | 2025-07-01 → 2026-03-04 | 105 |

**Top symbols by size:** BTCUSDT 24 GB, ETHUSDT 7.7 GB, PIPPINUSDT 2.0 GB, SOLUSDT 1.9 GB, RIVERUSDT 1.8 GB, XRPUSDT 1.8 GB, ZECUSDT 1.4 GB, DOGEUSDT 1.3 GB, BNBUSDT 1.3 GB, ENSOUSDT 1.1 GB.

---

## Cross-Exchange Coverage

| Data | Bybit | Binance |
|------|-------|---------|
| Futures klines 1m | 152 sym, from Jan 2024 | 144 sym, from Jan 2025 |
| Spot klines 1m | 96 sym, from Jan 2025 | 83 sym, from Jan 2025 |
| Funding rate / metrics | 152 sym, from Jan 2024 | 144 sym, from Jan 2025 |
| OI + LS ratio | 152 sym, from Jan 2024 | 144 sym (in metrics), from Jan 2025 |
| Futures trades (tick) | 153 sym, from Jan 2025 | 105 sym, from Jan 2025 |
| Spot trades (tick) | 93 sym, from Jan 2025 | 80 sym, from Jul 2025 |
| Orderbook (futures) | 115 sym, from Aug 2025 | 105 sym, from Jul 2025 |
| Orderbook (spot) | 93 sym, from Jul 2025 | — |
