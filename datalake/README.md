# Datalake

Historical USDT-M perpetual futures data from **Bybit** and **Binance**, downloaded daily at 1-minute resolution.

## Directory Structure

```
datalake/
├── README.md
├── download_bybit_data.py      # Bybit downloader (REST API + bulk archives)
├── download_binance_data.py    # Binance downloader (data.binance.vision archives)
├── bybit/
│   └── {SYMBOL}/
│       ├── {YYYY-MM-DD}_kline_1m.csv
│       ├── {YYYY-MM-DD}_mark_price_kline_1m.csv
│       ├── {YYYY-MM-DD}_premium_index_kline_1m.csv
│       ├── {YYYY-MM-DD}_funding_rate.csv
│       ├── {YYYY-MM-DD}_open_interest_5min.csv
│       ├── {YYYY-MM-DD}_long_short_ratio_5min.csv
│       ├── {YYYY-MM-DD}_trades.csv           (bulk, optional)
│       └── {YYYY-MM-DD}_orderbook.jsonl      (bulk, optional)
└── binance/
    └── {SYMBOL}/
        ├── {YYYY-MM-DD}_kline_1m.csv
        ├── {YYYY-MM-DD}_mark_price_kline_1m.csv
        ├── {YYYY-MM-DD}_premium_index_kline_1m.csv
        ├── {YYYY-MM-DD}_index_price_kline_1m.csv
        ├── {YYYY-MM-DD}_metrics.csv
        ├── {YYYY-MM-DD}_trades.csv           (bulk, optional)
        ├── {YYYY-MM-DD}_aggTrades.csv        (bulk, optional)
        ├── {YYYY-MM-DD}_bookDepth.csv        (bulk, optional)
        └── {YYYY-MM-DD}_bookTicker.csv       (bulk, optional)
```

## Scripts

Both scripts share a **unified CLI interface**:

```
python3 download_{exchange}_data.py SYMBOL[,SYMBOL,...] START_DATE END_DATE [OPTIONS]
```

### Common Options

| Flag | Short | Description |
|------|-------|-------------|
| `--types TYPES` | `-t` | Comma-separated data types to download. Use `all` for everything. |
| `--concurrency N` | `-c` | Max concurrent downloads (default: 5) |

### Bybit — `download_bybit_data.py`

Downloads data via the Bybit REST API v5 (paginated) and optionally bulk archives for trades/orderbook.

**Available types:** `trades`, `orderbook`, `klines`, `markPriceKlines`, `premiumIndexKlines`, `fundingRate`, `openInterest`, `longShortRatio`

**Default types:** `klines`, `markPriceKlines`, `premiumIndexKlines`, `fundingRate`, `openInterest`, `longShortRatio`

```bash
# Default types for one symbol
python3 download_bybit_data.py BTCUSDT 2025-07-01 2025-07-31

# Multiple symbols, higher concurrency
python3 download_bybit_data.py BTCUSDT,ETHUSDT,SOLUSDT 2025-07-01 2025-07-31 -c 10

# Specific types only
python3 download_bybit_data.py BTCUSDT 2025-07-01 2025-07-31 -t klines,fundingRate

# Everything including bulk trades and orderbook
python3 download_bybit_data.py BTCUSDT 2025-07-01 2025-07-31 -t all
```

| Data Source | Method | Notes |
|-------------|--------|-------|
| Klines (1m) | REST API `/v5/market/kline` | OHLCV with volume + turnover |
| Mark Price Klines (1m) | REST API `/v5/market/mark-price-kline` | OHLC only |
| Premium Index Klines (1m) | REST API `/v5/market/premium-index-price-kline` | OHLC only |
| Funding Rate | REST API `/v5/market/funding/history` | Variable frequency (8h/4h) |
| Open Interest (5min) | REST API `/v5/market/open-interest` | Snapshots every 5 min |
| Long/Short Ratio (5min) | REST API `/v5/market/account-ratio` | Buy/sell ratio every 5 min |
| Trades | Bulk archive `public.bybit.com` | Large files, `.csv.gz` |
| Orderbook (ob200) | Bulk archive `quote-saver.bycsi.com` | L2 snapshots, `.zip` → JSONL |

### Binance — `download_binance_data.py`

Downloads data from [data.binance.vision](https://data.binance.vision/?prefix=data/futures/um/daily/) bulk archives (daily `.zip` files with SHA256 checksum verification).

**Available types:** `trades`, `aggTrades`, `klines`, `markPriceKlines`, `premiumIndexKlines`, `indexPriceKlines`, `bookDepth`, `bookTicker`, `metrics`

**Default types:** `klines`, `markPriceKlines`, `premiumIndexKlines`, `indexPriceKlines`, `metrics`

```bash
# Default types for one symbol
python3 download_binance_data.py BTCUSDT 2025-07-01 2025-07-31

# Multiple symbols, higher concurrency
python3 download_binance_data.py BTCUSDT,ETHUSDT,SOLUSDT 2025-07-01 2025-07-31 -c 10

# Specific types only
python3 download_binance_data.py BTCUSDT 2025-07-01 2025-07-31 -t klines,metrics

# Everything including trades, aggTrades, bookDepth, bookTicker
python3 download_binance_data.py BTCUSDT 2025-07-01 2025-07-31 -t all

# Skip checksum verification (faster)
python3 download_binance_data.py BTCUSDT 2025-07-01 2025-07-31 --no-checksum
```

| Data Type | File Suffix | Content |
|-----------|-------------|---------|
| `klines` | `_kline_1m.csv` | OHLCV 1-minute candles |
| `markPriceKlines` | `_mark_price_kline_1m.csv` | Mark price 1-min candles |
| `premiumIndexKlines` | `_premium_index_kline_1m.csv` | Premium index 1-min candles |
| `indexPriceKlines` | `_index_price_kline_1m.csv` | Index price 1-min candles |
| `metrics` | `_metrics.csv` | Composite: OI, funding rate, LS ratio, taker volume |
| `trades` | `_trades.csv` | Individual trades |
| `aggTrades` | `_aggTrades.csv` | Aggregated trades |
| `bookDepth` | `_bookDepth.csv` | Order book depth snapshots |
| `bookTicker` | `_bookTicker.csv` | Best bid/ask ticker snapshots |

## Data Type Mapping (Bybit ↔ Binance)

| Data | Bybit file | Binance file |
|------|-----------|--------------|
| OHLCV klines (1m) | `_kline_1m.csv` | `_kline_1m.csv` |
| Mark price klines (1m) | `_mark_price_kline_1m.csv` | `_mark_price_kline_1m.csv` |
| Premium index klines (1m) | `_premium_index_kline_1m.csv` | `_premium_index_kline_1m.csv` |
| Index price klines (1m) | — | `_index_price_kline_1m.csv` |
| Funding rate | `_funding_rate.csv` | `_metrics.csv` (embedded) |
| Open interest | `_open_interest_5min.csv` | `_metrics.csv` (embedded) |
| Long/short ratio | `_long_short_ratio_5min.csv` | `_metrics.csv` (embedded) |
| Trades | `_trades.csv` | `_trades.csv` |
| Orderbook | `_orderbook.jsonl` | `_bookDepth.csv` |

## Behavior

- **Skip existing:** Both scripts skip files that already exist and are non-empty. Safe to re-run.
- **Atomic writes:** All files are written to `.tmp` first, then renamed. No partial files on interrupt.
- **Cleanup on interrupt:** Ctrl-C removes any in-progress `.tmp` files.
- **404 handling:** Missing data (newer coins, recent dates not yet published) is logged and skipped gracefully.
- **Checksum verification:** Binance downloads verify SHA256 checksums when available (disable with `--no-checksum`).

## Requirements

```
pip install aiohttp
```
