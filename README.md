# skytrade6

Multi-exchange crypto market data pipeline for ML research and backtesting.

Downloads raw trade data from **Bybit**, **Binance**, and **OKX** (futures + spot), then converts it into analysis-ready, daily-partitioned Parquet files with unified schemas and UTC-aligned timestamps.

## Requirements

```bash
pip install pandas pyarrow aiohttp
```

## Quick Start

Download 3 months of BTCUSDT data from all exchanges and convert to Parquet:

```bash
# 1. Download raw data
python download_market_data.py BTCUSDT 2025-11-01 2026-01-31

# 2. Convert to Parquet
python build_parquet.py BTCUSDT
```

## Step 1 — Download Raw Data

```bash
python download_market_data.py <SYMBOL> <START_DATE> <END_DATE> [options]
```

| Argument | Description |
|----------|-------------|
| `SYMBOL` | Trading pair, e.g. `BTCUSDT`, `ETHUSDT`, `SOLUSDT` |
| `START_DATE` | Inclusive start date (`YYYY-MM-DD`) |
| `END_DATE` | Inclusive end date (`YYYY-MM-DD`) |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | `./data` | Output root directory |
| `--sources`, `-s` | all 6 | Which sources to download |
| `--binance-futures-data-types` | trades, aggTrades, bookDepth, bookTicker, metrics, klines, indexPriceKlines, markPriceKlines, premiumIndexKlines | Binance futures data types |
| `--binance-spot-data-types` | trades, aggTrades, klines | Binance spot data types |
| `--kline-intervals` | 1m, 5m, 15m, 30m, 1h, 4h, 1d | Kline intervals |
| `--concurrency`, `-c` | 10 | Max concurrent downloads |

### Available Sources

`bybit_futures`, `bybit_spot`, `binance_futures`, `binance_spot`, `okx_futures`, `okx_spot`

### Examples

```bash
# Download all data for multiple symbols
python download_market_data.py BTCUSDT 2025-11-01 2026-01-31
python download_market_data.py ETHUSDT 2025-11-01 2026-01-31
python download_market_data.py SOLUSDT 2025-11-01 2026-01-31

# Download only futures data
python download_market_data.py BTCUSDT 2025-11-01 2026-01-31 -s bybit_futures binance_futures okx_futures

# Download only OKX
python download_market_data.py BTCUSDT 2025-11-01 2026-01-31 -s okx_futures okx_spot

# Download only trades (no klines/metrics/bookDepth)
python download_market_data.py BTCUSDT 2025-11-01 2026-01-31 --binance-futures-data-types trades --binance-spot-data-types trades
```

### Raw Data Structure

```
data/{SYMBOL}/
  bybit/
    futures/              # perpetual trades (.csv.gz)
    spot/                 # spot trades (.csv.gz)
  binance/
    futures/
      trades/             (.zip)
      aggTrades/          (.zip)
      bookDepth/          (.zip)
      metrics/            (.zip)
      klines/{interval}/  (.zip)
      indexPriceKlines/{interval}/
      markPriceKlines/{interval}/
      premiumIndexKlines/{interval}/
    spot/
      trades/             (.zip)
      aggTrades/          (.zip)
      klines/{interval}/  (.zip)
  okx/
    futures/              # perpetual SWAP trades (.zip)
    spot/                 # spot trades (.zip)
```

## Step 2 — Build Parquet Files

```bash
python build_parquet.py <SYMBOL> [options]
```

| Argument | Description |
|----------|-------------|
| `SYMBOL` | Trading pair, e.g. `BTCUSDT` |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input`, `-i` | `./data` | Raw data directory |
| `--output`, `-o` | `./parquet` | Parquet output directory |
| `--ohlcv-intervals` | 1m, 5m, 15m, 1h | OHLCV bar intervals to generate |

### Examples

```bash
# Build all parquet files for each symbol
python build_parquet.py BTCUSDT
python build_parquet.py ETHUSDT
python build_parquet.py SOLUSDT

# Custom output directory
python build_parquet.py BTCUSDT -o ./my_parquet

# Only 1m and 1h OHLCV bars
python build_parquet.py BTCUSDT --ohlcv-intervals 1m 1h
```

### Features

- **Incremental builds** — existing parquet files are skipped, so re-runs are fast
- **OKX UTC re-partitioning** — OKX raw files use UTC+8 day boundaries; the pipeline automatically re-partitions trades into correct UTC days
- **Per-file progress logging** — shows file-by-file progress during processing
- **Memory-efficient** — processes one file at a time (streaming), no OOM on large datasets

### Parquet Output Structure

```
parquet/{SYMBOL}/
  trades/{source}/{YYYY-MM-DD}.parquet
  ohlcv/{interval}/{source}/{YYYY-MM-DD}.parquet
  binance/
    metrics/{YYYY-MM-DD}.parquet
    book_depth/{YYYY-MM-DD}.parquet
    agg_trades_futures/{YYYY-MM-DD}.parquet
    agg_trades_spot/{YYYY-MM-DD}.parquet
    klines_futures/{interval}/{YYYY-MM-DD}.parquet
    klines_spot/{interval}/{YYYY-MM-DD}.parquet
    index_price_klines_futures/{interval}/{YYYY-MM-DD}.parquet
    mark_price_klines_futures/{interval}/{YYYY-MM-DD}.parquet
    premium_index_klines_futures/{interval}/{YYYY-MM-DD}.parquet
```

Where `{source}` is one of: `bybit_futures`, `bybit_spot`, `binance_futures`, `binance_spot`, `okx_futures`, `okx_spot`

### Unified Trades Schema

All trade parquet files share the same schema regardless of exchange:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp_us` | int64 | Microseconds since epoch (UTC) |
| `price` | float64 | Trade price |
| `quantity` | float64 | Trade quantity (base asset) |
| `quote_quantity` | float64 | Trade value (quote asset) |
| `side` | int8 | `1` = buy, `-1` = sell |
| `trade_id` | string | Exchange-specific trade ID |

### OHLCV Schema

| Column | Type | Description |
|--------|------|-------------|
| `timestamp_us` | int64 | Bar open time (UTC, microseconds) |
| `open` | float64 | Open price |
| `high` | float64 | High price |
| `low` | float64 | Low price |
| `close` | float64 | Close price |
| `volume` | float64 | Total volume |
| `buy_volume` | float64 | Buy-side volume |
| `sell_volume` | float64 | Sell-side volume |
| `trade_count` | int64 | Number of trades |
| `vwap` | float64 | Volume-weighted average price |

## Loading Data in Python

```python
import pandas as pd
from pathlib import Path

symbol = "BTCUSDT"
date = "2025-12-15"

# Load trades from one source
df = pd.read_parquet(f"parquet/{symbol}/trades/binance_futures/{date}.parquet")

# Load and merge 1h OHLCV across all exchanges
sources = ["bybit_futures", "bybit_spot", "binance_futures",
           "binance_spot", "okx_futures", "okx_spot"]
ohlcv = {}
for src in sources:
    ohlcv[src] = pd.read_parquet(
        f"parquet/{symbol}/ohlcv/1h/{src}/{date}.parquet"
    ).set_index("timestamp_us")["close"].rename(src)

merged = pd.concat(ohlcv.values(), axis=1)  # join on UTC timestamp

# Load a date range
dfs = []
for f in sorted(Path(f"parquet/{symbol}/ohlcv/1h/binance_futures").glob("*.parquet")):
    dfs.append(pd.read_parquet(f))
full_period = pd.concat(dfs, ignore_index=True)
```