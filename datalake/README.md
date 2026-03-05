# Datalake

Historical market data from **Bybit**, **Binance**, and **OKX** — both **perpetual futures** and **spot** — downloaded daily at 1-minute resolution.

## Directory Structure

```
datalake/
├── README.md
├── list_symbols.py             # List common symbols (Bybit ∩ Binance ∩ OKX)
├── download_bybit_data.py      # Bybit downloader (REST API + bulk archives)
├── download_binance_data.py    # Binance downloader (data.binance.vision archives)
├── download_okx_data.py        # OKX downloader (REST API + bulk archives)
├── compress_orderbooks.py      # Compress existing uncompressed bulk files to .gz
├── bybit/
│   └── {SYMBOL}/
│       ├── {YYYY-MM-DD}_kline_1m.csv              # MetricsLinear
│       ├── {YYYY-MM-DD}_mark_price_kline_1m.csv
│       ├── {YYYY-MM-DD}_premium_index_kline_1m.csv
│       ├── {YYYY-MM-DD}_funding_rate.csv
│       ├── {YYYY-MM-DD}_open_interest_5min.csv
│       ├── {YYYY-MM-DD}_long_short_ratio_5min.csv
│       ├── {YYYY-MM-DD}_trades.csv.gz              # TradesLinear
│       ├── {YYYY-MM-DD}_orderbook.jsonl.gz          # OrderbookLinear
│       ├── {YYYY-MM-DD}_orderbook_spot.jsonl.gz     # OrderbookSpot
│       ├── {YYYY-MM-DD}_kline_1m_spot.csv           # MetricsSpot
│       └── {YYYY-MM-DD}_trades_spot.csv.gz          # TradesSpot
├── binance/
│   └── {SYMBOL}/
│       ├── {YYYY-MM-DD}_kline_1m.csv              # futures
│       ├── {YYYY-MM-DD}_mark_price_kline_1m.csv
│       ├── {YYYY-MM-DD}_premium_index_kline_1m.csv
│       ├── {YYYY-MM-DD}_index_price_kline_1m.csv
│       ├── {YYYY-MM-DD}_metrics.csv
│       ├── {YYYY-MM-DD}_trades.csv.gz
│       ├── {YYYY-MM-DD}_aggTrades.csv.gz
│       ├── {YYYY-MM-DD}_bookDepth.csv.gz
│       ├── {YYYY-MM-DD}_bookTicker.csv.gz
│       ├── {YYYY-MM-DD}_kline_1m_spot.csv         # spot
│       ├── {YYYY-MM-DD}_trades_spot.csv.gz
│       └── {YYYY-MM-DD}_aggTrades_spot.csv.gz
└── okx/
    └── {SYMBOL}/
        ├── {YYYY-MM-DD}_kline_1m.csv              # swap (futures)
        ├── {YYYY-MM-DD}_mark_price_kline_1m.csv
        ├── {YYYY-MM-DD}_funding_rate.csv
        ├── {YYYY-MM-DD}_open_interest_5min.csv
        ├── {YYYY-MM-DD}_long_short_ratio_5min.csv
        ├── {YYYY-MM-DD}_taker_volume_5min.csv
        ├── {YYYY-MM-DD}_premium_history.csv
        ├── {YYYY-MM-DD}_trades.csv.gz
        ├── {YYYY-MM-DD}_kline_1m_spot.csv         # spot
        └── {YYYY-MM-DD}_trades_spot.csv.gz
```

## list_symbols.py

Prints comma-separated USDT perpetual symbols common to all three exchanges, sorted by Bybit 24h turnover. Use as argument placeholder in download commands.

```bash
python3 list_symbols.py                              # all common symbols
python3 list_symbols.py --limit 50                   # top 50
python3 list_symbols.py --limit 15 --offset 5        # symbols ranked 6th–20th
python3 list_symbols.py --limit 20 -v                # with detailed table

# Pipe into downloaders:
python3 download_bybit_data.py $(python3 list_symbols.py -l 50) 2024-01-01 2026-03-04 -t MetricsLinear
python3 download_binance_data.py $(python3 list_symbols.py -l 50) 2024-01-01 2026-03-04
python3 download_okx_data.py $(python3 list_symbols.py -l 50) 2024-01-01 2026-03-04
```

| Flag | Short | Description |
|------|-------|-------------|
| `--limit N` | `-l` | Max number of symbols (default: all) |
| `--offset N` | `-o` | Skip first N symbols (default: 0) |
| `--verbose` | `-v` | Print a table with rank, turnover, exchange presence (to stderr) |

## Bybit — `download_bybit_data.py`

Six data types that can be downloaded separately or all together:

```
python3 download_bybit_data.py SYMBOL[,SYMBOL,...] START_DATE END_DATE [-t TYPES] [-c N]
```

| Type | Source | Output | Content |
|------|--------|--------|---------|
| `TradesLinear` | Bulk `public.bybit.com/trading/` | `_trades.csv.gz` | Tick-level trades |
| `TradesSpot` | Bulk `public.bybit.com/spot/` | `_trades_spot.csv.gz` | Spot trades |
| `OrderbookLinear` | Bulk `quote-saver.bycsi.com/orderbook/linear/` | `_orderbook.jsonl.gz` | L2 ob200 snapshots |
| `OrderbookSpot` | Bulk `quote-saver.bycsi.com/orderbook/spot/` | `_orderbook_spot.jsonl.gz` | Spot L2 ob200 |
| `MetricsLinear` | REST API v5 | `.csv` | Kline, mark price, premium index, funding, OI, LS ratio |
| `MetricsSpot` | REST API v5 | `_*_spot.csv` | Spot kline |

**Default** (no `-t`): `MetricsLinear`

Each type's first-available-date is detected independently via binary search. A single command requesting both `MetricsLinear` (available from 2024) and `OrderbookLinear` (available from 2025) will download each from its own start date.

```bash
python3 download_bybit_data.py BTCUSDT 2024-01-01 2026-03-04
python3 download_bybit_data.py BTCUSDT 2024-01-01 2026-03-04 -t TradesLinear
python3 download_bybit_data.py BTCUSDT 2024-01-01 2026-03-04 -t TradesLinear,OrderbookLinear,MetricsLinear
python3 download_bybit_data.py BTCUSDT 2024-01-01 2026-03-04 -t all -c 10
```

### MetricsLinear sub-types

| Metric | REST API endpoint | File suffix |
|--------|-------------------|-------------|
| Klines (1m) | `/v5/market/kline` | `_kline_1m.csv` |
| Mark Price Klines (1m) | `/v5/market/mark-price-kline` | `_mark_price_kline_1m.csv` |
| Premium Index Klines (1m) | `/v5/market/premium-index-price-kline` | `_premium_index_kline_1m.csv` |
| Funding Rate | `/v5/market/funding/history` | `_funding_rate.csv` |
| Open Interest (5min) | `/v5/market/open-interest` | `_open_interest_5min.csv` |
| Long/Short Ratio (5min) | `/v5/market/account-ratio` | `_long_short_ratio_5min.csv` |

## Binance — `download_binance_data.py`

Downloads from [data.binance.vision](https://data.binance.vision/?prefix=data/futures/um/daily/) bulk archives (daily `.zip` with SHA256 checksum verification).

```
python3 download_binance_data.py SYMBOL[,SYMBOL,...] START_DATE END_DATE [-m futures|spot] [-t TYPES] [-c N]
```

**Futures (default):** `trades`, `aggTrades`, `klines`, `markPriceKlines`, `premiumIndexKlines`, `indexPriceKlines`, `bookDepth`, `bookTicker`, `metrics`
**Futures defaults:** `klines`, `markPriceKlines`, `premiumIndexKlines`, `indexPriceKlines`, `metrics`

**Spot:** `spotKlines`, `spotTrades`, `spotAggTrades`
**Spot defaults:** `spotKlines`

```bash
python3 download_binance_data.py BTCUSDT 2025-07-01 2025-07-31
python3 download_binance_data.py BTCUSDT 2025-07-01 2025-07-31 -t klines,metrics
python3 download_binance_data.py BTCUSDT 2025-07-01 2025-07-31 -t all
python3 download_binance_data.py BTCUSDT 2025-07-01 2025-07-31 --market spot
```

| Data Type | File Suffix | Content |
|-----------|-------------|---------|
| `klines` | `_kline_1m.csv` | OHLCV 1-minute candles |
| `markPriceKlines` | `_mark_price_kline_1m.csv` | Mark price 1-min candles |
| `premiumIndexKlines` | `_premium_index_kline_1m.csv` | Premium index 1-min candles |
| `indexPriceKlines` | `_index_price_kline_1m.csv` | Index price 1-min candles |
| `metrics` | `_metrics.csv` | Composite: OI, funding rate, LS ratio, taker volume |
| `trades` | `_trades.csv.gz` | Individual trades (gzipped) |
| `aggTrades` | `_aggTrades.csv.gz` | Aggregated trades (gzipped) |
| `bookDepth` | `_bookDepth.csv.gz` | Order book depth snapshots (gzipped) |
| `bookTicker` | `_bookTicker.csv.gz` | Best bid/ask ticker snapshots (gzipped) |

## OKX — `download_okx_data.py`

Downloads via OKX REST API v5 (paginated) + bulk trade archives from `static.okx.com`. Symbols auto-convert from `BTCUSDT` to OKX format (`BTC-USDT-SWAP` / `BTC-USDT`).

```
python3 download_okx_data.py SYMBOL[,SYMBOL,...] START_DATE END_DATE [-m swap|spot] [-t TYPES] [-c N]
```

**Swap (default):** `trades`, `klines`, `markPriceKlines`, `fundingRate`, `openInterest`, `longShortRatio`, `takerVolume`, `premiumHistory`
**Swap defaults:** `klines`, `markPriceKlines`, `fundingRate`, `openInterest`, `longShortRatio`, `takerVolume`, `premiumHistory`

**Spot:** `spotKlines`, `spotTrades`
**Spot defaults:** `spotKlines`

```bash
python3 download_okx_data.py BTCUSDT 2025-07-01 2025-07-31
python3 download_okx_data.py BTCUSDT 2025-07-01 2025-07-31 -t klines,fundingRate
python3 download_okx_data.py BTCUSDT 2025-07-01 2025-07-31 -t all
python3 download_okx_data.py BTCUSDT 2025-07-01 2025-07-31 --market spot
```

| Data Type | File Suffix | Content |
|-----------|-------------|---------|
| `klines` | `_kline_1m.csv` | OHLCV 1-minute candles |
| `markPriceKlines` | `_mark_price_kline_1m.csv` | Mark price 1-min candles |
| `fundingRate` | `_funding_rate.csv` | Funding rate + realized rate |
| `openInterest` | `_open_interest_5min.csv` | OI (contracts, coin, USD) |
| `longShortRatio` | `_long_short_ratio_5min.csv` | Long/short account ratio |
| `takerVolume` | `_taker_volume_5min.csv` | Taker buy/sell volume |
| `premiumHistory` | `_premium_history.csv` | Premium index history |
| `trades` | `_trades.csv.gz` | Individual trades (gzipped) |

## Cross-Exchange Data Mapping

### Futures / Swap

| Data | Bybit | Binance | OKX |
|------|-------|---------|-----|
| OHLCV klines (1m) | `_kline_1m.csv` | `_kline_1m.csv` | `_kline_1m.csv` |
| Mark price (1m) | `_mark_price_kline_1m.csv` | `_mark_price_kline_1m.csv` | `_mark_price_kline_1m.csv` |
| Premium index (1m) | `_premium_index_kline_1m.csv` | `_premium_index_kline_1m.csv` | `_premium_history.csv` |
| Index price (1m) | — | `_index_price_kline_1m.csv` | — |
| Funding rate | `_funding_rate.csv` | `_metrics.csv` (embedded) | `_funding_rate.csv` |
| Open interest | `_open_interest_5min.csv` | `_metrics.csv` (embedded) | `_open_interest_5min.csv` |
| Long/short ratio | `_long_short_ratio_5min.csv` | `_metrics.csv` (embedded) | `_long_short_ratio_5min.csv` |
| Taker volume | — | — | `_taker_volume_5min.csv` |
| Trades | `_trades.csv.gz` | `_trades.csv.gz` | `_trades.csv.gz` |
| Orderbook | `_orderbook.jsonl.gz` | `_bookDepth.csv.gz` | — |

### Spot

| Data | Bybit | Binance | OKX |
|------|-------|---------|-----|
| OHLCV klines (1m) | `_kline_1m_spot.csv` | `_kline_1m_spot.csv` | `_kline_1m_spot.csv` |
| Trades | `_trades_spot.csv.gz` | `_trades_spot.csv.gz` | `_trades_spot.csv.gz` |
| Aggregated trades | — | `_aggTrades_spot.csv.gz` | — |
| Orderbook | `_orderbook_spot.jsonl.gz` | — | — |

## Behavior

- **Skip existing:** All scripts skip files that already exist and are non-empty. Safe to re-run.
- **Atomic writes:** Files are written to `.tmp` first, then renamed. No partial files on interrupt.
- **Cleanup on interrupt:** Ctrl-C removes any in-progress `.tmp` files.
- **404 handling:** Missing data is logged and skipped gracefully.
- **Checksum verification:** Binance downloads verify SHA256 checksums (disable with `--no-checksum`).
- **Per-type first-date detection:** Bybit binary-searches each data type's first available date independently (HEAD probes for bulk, kline API for metrics).
- **Rate-limit protection:** 403 responses trigger aggressive backoff; bulk requests are throttled to avoid CloudFront bans.
- **Gzip compression:** Bulk data stored as `.csv.gz` / `.jsonl.gz`. Use `compress_orderbooks.py --run` to compress older uncompressed files.

## Requirements

```
pip install aiohttp
```
