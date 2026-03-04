# Datalake

Historical market data from **Bybit**, **Binance**, and **OKX** — both **perpetual futures** and **spot** — downloaded daily at 1-minute resolution.

## Directory Structure

```
datalake/
├── README.md
├── download_bybit_data.py      # Bybit downloader (REST API + bulk archives)
├── download_binance_data.py    # Binance downloader (data.binance.vision archives)
├── download_okx_data.py        # OKX downloader (REST API + bulk archives)
├── list_symbols.py             # Symbol listing + batch download orchestrator
├── bybit/
│   └── {SYMBOL}/
│       ├── {YYYY-MM-DD}_kline_1m.csv              # futures
│       ├── {YYYY-MM-DD}_mark_price_kline_1m.csv
│       ├── {YYYY-MM-DD}_premium_index_kline_1m.csv
│       ├── {YYYY-MM-DD}_funding_rate.csv
│       ├── {YYYY-MM-DD}_open_interest_5min.csv
│       ├── {YYYY-MM-DD}_long_short_ratio_5min.csv
│       ├── {YYYY-MM-DD}_trades.csv                (bulk, optional)
│       ├── {YYYY-MM-DD}_orderbook.jsonl           (bulk, optional)
│       ├── {YYYY-MM-DD}_kline_1m_spot.csv         # spot
│       └── {YYYY-MM-DD}_trades_spot.csv           (bulk, optional)
├── binance/
│   └── {SYMBOL}/
│       ├── {YYYY-MM-DD}_kline_1m.csv              # futures
│       ├── {YYYY-MM-DD}_mark_price_kline_1m.csv
│       ├── {YYYY-MM-DD}_premium_index_kline_1m.csv
│       ├── {YYYY-MM-DD}_index_price_kline_1m.csv
│       ├── {YYYY-MM-DD}_metrics.csv
│       ├── {YYYY-MM-DD}_trades.csv                (bulk, optional)
│       ├── {YYYY-MM-DD}_aggTrades.csv             (bulk, optional)
│       ├── {YYYY-MM-DD}_bookDepth.csv             (bulk, optional)
│       ├── {YYYY-MM-DD}_bookTicker.csv            (bulk, optional)
│       ├── {YYYY-MM-DD}_kline_1m_spot.csv         # spot
│       ├── {YYYY-MM-DD}_trades_spot.csv           (bulk, optional)
│       └── {YYYY-MM-DD}_aggTrades_spot.csv        (bulk, optional)
└── okx/
    └── {SYMBOL}/
        ├── {YYYY-MM-DD}_kline_1m.csv              # swap (futures)
        ├── {YYYY-MM-DD}_mark_price_kline_1m.csv
        ├── {YYYY-MM-DD}_funding_rate.csv
        ├── {YYYY-MM-DD}_open_interest_5min.csv
        ├── {YYYY-MM-DD}_long_short_ratio_5min.csv
        ├── {YYYY-MM-DD}_taker_volume_5min.csv
        ├── {YYYY-MM-DD}_premium_history.csv
        ├── {YYYY-MM-DD}_trades.csv                (bulk, optional)
        ├── {YYYY-MM-DD}_kline_1m_spot.csv         # spot
        └── {YYYY-MM-DD}_trades_spot.csv           (bulk, optional)
```

## Scripts

All three scripts share a **unified CLI interface**:

```
python3 download_{exchange}_data.py SYMBOL[,SYMBOL,...] START_DATE END_DATE [OPTIONS]
```

### Common Options

| Flag | Short | Description |
|------|-------|-------------|
| `--market TYPE` | `-m` | Market type: futures/spot (Bybit: `linear`/`spot`, Binance: `futures`/`spot`, OKX: `swap`/`spot`) |
| `--types TYPES` | `-t` | Comma-separated data types to download. Use `all` for everything. Defaults depend on `--market`. |
| `--concurrency N` | `-c` | Max concurrent downloads (default: 5) |

Spot data files are saved in the **same folder** as futures data, distinguished by a `_spot` postfix in the filename.

### Bybit — `download_bybit_data.py`

Downloads data via the Bybit REST API v5 (paginated) and optionally bulk archives for trades/orderbook.

**Linear (default):** `trades`, `orderbook`, `klines`, `markPriceKlines`, `premiumIndexKlines`, `fundingRate`, `openInterest`, `longShortRatio`
**Linear defaults:** `klines`, `markPriceKlines`, `premiumIndexKlines`, `fundingRate`, `openInterest`, `longShortRatio`

**Spot:** `spotKlines`, `spotTrades`
**Spot defaults:** `spotKlines`

```bash
# Linear (default)
python3 download_bybit_data.py BTCUSDT 2025-07-01 2025-07-31
python3 download_bybit_data.py BTCUSDT,ETHUSDT,SOLUSDT 2025-07-01 2025-07-31 -c 10
python3 download_bybit_data.py BTCUSDT 2025-07-01 2025-07-31 -t klines,fundingRate
python3 download_bybit_data.py BTCUSDT 2025-07-01 2025-07-31 -t all

# Spot
python3 download_bybit_data.py BTCUSDT 2025-07-01 2025-07-31 --market spot
python3 download_bybit_data.py BTCUSDT 2025-07-01 2025-07-31 --market spot -t all
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

**Futures (default):** `trades`, `aggTrades`, `klines`, `markPriceKlines`, `premiumIndexKlines`, `indexPriceKlines`, `bookDepth`, `bookTicker`, `metrics`
**Futures defaults:** `klines`, `markPriceKlines`, `premiumIndexKlines`, `indexPriceKlines`, `metrics`

**Spot:** `spotKlines`, `spotTrades`, `spotAggTrades`
**Spot defaults:** `spotKlines`

```bash
# Futures (default)
python3 download_binance_data.py BTCUSDT 2025-07-01 2025-07-31
python3 download_binance_data.py BTCUSDT,ETHUSDT,SOLUSDT 2025-07-01 2025-07-31 -c 10
python3 download_binance_data.py BTCUSDT 2025-07-01 2025-07-31 -t klines,metrics
python3 download_binance_data.py BTCUSDT 2025-07-01 2025-07-31 -t all
python3 download_binance_data.py BTCUSDT 2025-07-01 2025-07-31 --no-checksum

# Spot
python3 download_binance_data.py BTCUSDT 2025-07-01 2025-07-31 --market spot
python3 download_binance_data.py BTCUSDT 2025-07-01 2025-07-31 --market spot -t all
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

### OKX — `download_okx_data.py`

Downloads data via the OKX REST API v5 (paginated) and optionally bulk trade archives from `static.okx.com`. Symbols are automatically converted from `BTCUSDT` format to OKX instrument IDs (`BTC-USDT-SWAP` for swap, `BTC-USDT` for spot).

**Swap (default):** `trades`, `klines`, `markPriceKlines`, `fundingRate`, `openInterest`, `longShortRatio`, `takerVolume`, `premiumHistory`
**Swap defaults:** `klines`, `markPriceKlines`, `fundingRate`, `openInterest`, `longShortRatio`, `takerVolume`, `premiumHistory`

**Spot:** `spotKlines`, `spotTrades`
**Spot defaults:** `spotKlines`

```bash
# Swap (default)
python3 download_okx_data.py BTCUSDT 2025-07-01 2025-07-31
python3 download_okx_data.py BTCUSDT,ETHUSDT,SOLUSDT 2025-07-01 2025-07-31 -c 10
python3 download_okx_data.py BTCUSDT 2025-07-01 2025-07-31 -t klines,fundingRate
python3 download_okx_data.py BTCUSDT 2025-07-01 2025-07-31 -t all

# Spot
python3 download_okx_data.py BTCUSDT 2025-07-01 2025-07-31 --market spot
python3 download_okx_data.py BTCUSDT 2025-07-01 2025-07-31 --market spot -t all
```

| Data Source | Method | Notes |
|-------------|--------|-------|
| Klines (1m) | REST API `/api/v5/market/history-candles` | OHLCV + volCcy + volCcyQuote |
| Mark Price Klines (1m) | REST API `/api/v5/market/history-mark-price-candles` | OHLC only |
| Funding Rate | REST API `/api/v5/public/funding-rate-history` | Variable frequency (8h/4h/1h) |
| Open Interest (5min) | REST API `/api/v5/rubik/stat/contracts/open-interest-history` | OI in contracts, coin, USD |
| Long/Short Ratio (5min) | REST API `/api/v5/rubik/stat/contracts/long-short-account-ratio-contract` | Account ratio |
| Taker Volume (5min) | REST API `/api/v5/rubik/stat/taker-volume-contract` | Buy/sell volume |
| Premium History | REST API `/api/v5/public/premium-history` | Premium index |
| Trades | Bulk archive `static.okx.com` | Daily `.zip` files |

| Data Type | File Suffix | Content |
|-----------|-------------|---------|
| `klines` | `_kline_1m.csv` | OHLCV 1-minute candles |
| `markPriceKlines` | `_mark_price_kline_1m.csv` | Mark price 1-min candles |
| `fundingRate` | `_funding_rate.csv` | Funding rate + realized rate |
| `openInterest` | `_open_interest_5min.csv` | OI (contracts, coin, USD) |
| `longShortRatio` | `_long_short_ratio_5min.csv` | Long/short account ratio |
| `takerVolume` | `_taker_volume_5min.csv` | Taker buy/sell volume |
| `premiumHistory` | `_premium_history.csv` | Premium index history |
| `trades` | `_trades.csv` | Individual trades |

## Data Type Mapping (Bybit ↔ Binance ↔ OKX)

### Futures / Swap

| Data | Bybit file | Binance file | OKX file |
|------|-----------|--------------|----------|
| OHLCV klines (1m) | `_kline_1m.csv` | `_kline_1m.csv` | `_kline_1m.csv` |
| Mark price klines (1m) | `_mark_price_kline_1m.csv` | `_mark_price_kline_1m.csv` | `_mark_price_kline_1m.csv` |
| Premium index klines (1m) | `_premium_index_kline_1m.csv` | `_premium_index_kline_1m.csv` | `_premium_history.csv` |
| Index price klines (1m) | — | `_index_price_kline_1m.csv` | — |
| Funding rate | `_funding_rate.csv` | `_metrics.csv` (embedded) | `_funding_rate.csv` |
| Open interest | `_open_interest_5min.csv` | `_metrics.csv` (embedded) | `_open_interest_5min.csv` |
| Long/short ratio | `_long_short_ratio_5min.csv` | `_metrics.csv` (embedded) | `_long_short_ratio_5min.csv` |
| Taker volume | — | — | `_taker_volume_5min.csv` |
| Trades | `_trades.csv` | `_trades.csv` | `_trades.csv` |
| Orderbook | `_orderbook.jsonl` | `_bookDepth.csv` | — |

### Spot

| Data | Bybit file | Binance file | OKX file |
|------|-----------|--------------|----------|
| OHLCV klines (1m) | `_kline_1m_spot.csv` | `_kline_1m_spot.csv` | `_kline_1m_spot.csv` |
| Trades | `_trades_spot.csv` | `_trades_spot.csv` | `_trades_spot.csv` |
| Aggregated trades | — | `_aggTrades_spot.csv` | — |

## Behavior

- **Skip existing:** All scripts skip files that already exist and are non-empty. Safe to re-run.
- **Atomic writes:** All files are written to `.tmp` first, then renamed. No partial files on interrupt.
- **Cleanup on interrupt:** Ctrl-C removes any in-progress `.tmp` files.
- **404 handling:** Missing data (newer coins, recent dates not yet published) is logged and skipped gracefully.
- **Checksum verification:** Binance downloads verify SHA256 checksums when available (disable with `--no-checksum`).
- **First-available-date detection:** All scripts binary-search for the first date with data, skipping gaps automatically.

## Requirements

```
pip install aiohttp
```
