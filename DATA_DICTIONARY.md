# Market Data Dictionary

Reference for all raw data files downloaded by `download_market_data.py`.

---

## Directory Structure

```
data/{SYMBOL}/
  bybit/
    futures/                          .csv.gz (has header row)
    spot/                             .csv.gz (has header row)
  binance/
    futures/
      trades/                         .zip → .csv (has header row)
      aggTrades/                      .zip → .csv (has header row)
      bookDepth/                      .zip → .csv (has header row)
      bookTicker/                     .zip → .csv (has header row, not all symbols)
      metrics/                        .zip → .csv (has header row)
      klines/{interval}/              .zip → .csv (has header row)
      indexPriceKlines/{interval}/    .zip → .csv (has header row)
      markPriceKlines/{interval}/     .zip → .csv (has header row)
      premiumIndexKlines/{interval}/  .zip → .csv (has header row)
    spot/
      trades/                         .zip → .csv (NO header row)
      aggTrades/                      .zip → .csv (NO header row)
      klines/{interval}/              .zip → .csv (NO header row)
  okx/
    futures/                          .zip → .csv (has header row)
    spot/                             .zip → .csv (has header row)
```

> **Note:** Binance spot CSVs have **no header row**. Column names listed below must be
> applied manually when reading. Binance futures and all Bybit files include headers.

---

## Bybit

### Bybit Futures (Perpetual Trades)

**Source:** `https://public.bybit.com/trading/{SYMBOL}/`
**File format:** `.csv.gz` — gzip-compressed CSV with header
**Filename pattern:** `{SYMBOL}{YYYY-MM-DD}.csv.gz`

| Column           | Type    | Description                                                                 |
|------------------|---------|-----------------------------------------------------------------------------|
| `timestamp`      | float   | Unix timestamp in seconds with sub-second decimal (e.g. `1767225600.6864`) |
| `symbol`         | string  | Trading pair (e.g. `SOLUSDT`)                                               |
| `side`           | string  | Trade direction: `Buy` or `Sell`                                            |
| `size`           | float   | Trade quantity in base asset                                                |
| `price`          | float   | Execution price                                                             |
| `tickDirection`  | string  | Tick direction: `PlusTick`, `ZeroPlusTick`, `MinusTick`, `ZeroMinusTick`   |
| `trdMatchID`     | string  | Unique trade match UUID                                                     |
| `grossValue`     | float   | Gross value in contract denomination (scaled integer-like)                  |
| `homeNotional`   | float   | Value in base asset                                                         |
| `foreignNotional`| float   | Value in quote asset (USD)                                                  |
| `RPI`            | int     | Retail price improvement flag (0 = no)                                      |

**Example:**
```
timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,foreignNotional,RPI
1767225600.6864,SOLUSDT,Buy,0.2,124.650,PlusTick,7df520b4-f4a1-5eb9-86c2-3ac7ebac1832,2.493e+09,0.2,24.93,0
```

---

### Bybit Spot (Trades)

**Source:** `https://public.bybit.com/spot/{SYMBOL}/`
**File format:** `.csv.gz` — gzip-compressed CSV with header
**Filename pattern:** `{SYMBOL}_{YYYY-MM-DD}.csv.gz`

| Column      | Type   | Description                                          |
|-------------|--------|------------------------------------------------------|
| `id`        | int    | Sequential trade ID (resets daily, starts at 1)      |
| `timestamp` | int    | Unix timestamp in milliseconds                       |
| `price`     | float  | Execution price                                      |
| `volume`    | float  | Trade quantity in base asset                          |
| `side`      | string | Trade direction: `buy` or `sell` (lowercase)         |
| `rpi`       | int    | Retail price improvement flag (0 = no)               |

**Example:**
```
id,timestamp,price,volume,side,rpi
1,1767225600884,124.67,0.1,buy,0
```

---

## Binance Futures (USD-M)

All files sourced from `https://data.binance.vision/data/futures/um/daily/`.

### Trades

**Path:** `binance/futures/trades/`
**Filename:** `{SYMBOL}-trades-{YYYY-MM-DD}.zip`

| Column          | Type   | Description                                |
|-----------------|--------|--------------------------------------------|
| `id`            | int    | Trade ID                                   |
| `price`         | float  | Execution price                            |
| `qty`           | float  | Quantity in base asset                     |
| `quote_qty`     | float  | Quantity in quote asset (price × qty)      |
| `time`          | int    | Unix timestamp in milliseconds             |
| `is_buyer_maker`| string | `true` if buyer was the maker              |

**Example:**
```
id,price,qty,quote_qty,time,is_buyer_maker
3047799741,124.58,3.22,401.1476,1767225601696,true
```

---

### Aggregate Trades (aggTrades)

**Path:** `binance/futures/aggTrades/`
**Filename:** `{SYMBOL}-aggTrades-{YYYY-MM-DD}.zip`

| Column          | Type   | Description                                |
|-----------------|--------|--------------------------------------------|
| `agg_trade_id`  | int    | Aggregate trade ID                         |
| `price`         | float  | Execution price                            |
| `quantity`       | float  | Total aggregated quantity                  |
| `first_trade_id`| int    | First individual trade ID in aggregate     |
| `last_trade_id` | int    | Last individual trade ID in aggregate      |
| `transact_time` | int    | Unix timestamp in milliseconds             |
| `is_buyer_maker`| string | `true` if buyer was the maker              |

**Example:**
```
agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker
1018572200,124.58,6.43,3047799741,3047799742,1767225601696,true
```

---

### Book Depth (bookDepth)

**Path:** `binance/futures/bookDepth/`
**Filename:** `{SYMBOL}-bookDepth-{YYYY-MM-DD}.zip`

Periodic snapshots of order book depth at various price percentage levels.

| Column       | Type   | Description                                              |
|--------------|--------|----------------------------------------------------------|
| `timestamp`  | string | UTC datetime (`YYYY-MM-DD HH:MM:SS`)                    |
| `percentage` | int    | Price level offset from mid-price (e.g. `-5`, `-4`, ... `+5`) |
| `depth`      | float  | Total quantity at this depth level                       |
| `notional`   | float  | Total notional value (USD) at this depth level           |

**Example:**
```
timestamp,percentage,depth,notional
2026-01-01 00:00:09,-5,620184.39000000,75729582.96890000
```

---

### Book Ticker (bookTicker)

**Path:** `binance/futures/bookTicker/`
**Filename:** `{SYMBOL}-bookTicker-{YYYY-MM-DD}.zip`

> **Availability:** Not available for all symbols. Downloads that return 404 are skipped.

Best bid/ask snapshots. Expected columns (when available):

| Column            | Type   | Description                          |
|-------------------|--------|--------------------------------------|
| `symbol`          | string | Trading pair                         |
| `best_bid_price`  | float  | Best bid price                       |
| `best_bid_qty`    | float  | Best bid quantity                    |
| `best_ask_price`  | float  | Best ask price                       |
| `best_ask_qty`    | float  | Best ask quantity                    |
| `transaction_time`| int    | Unix timestamp in milliseconds       |
| `event_time`      | int    | Event timestamp in milliseconds      |

---

### Metrics

**Path:** `binance/futures/metrics/`
**Filename:** `{SYMBOL}-metrics-{YYYY-MM-DD}.zip`

Aggregated futures market metrics at 5-minute intervals.

| Column                              | Type   | Description                                        |
|--------------------------------------|--------|----------------------------------------------------|
| `create_time`                        | string | UTC datetime (`YYYY-MM-DD HH:MM:SS`)              |
| `symbol`                             | string | Trading pair                                       |
| `sum_open_interest`                  | float  | Total open interest (contracts)                    |
| `sum_open_interest_value`            | float  | Total open interest value (USD)                    |
| `count_toptrader_long_short_ratio`   | float  | Top trader long/short ratio (by accounts)          |
| `sum_toptrader_long_short_ratio`     | float  | Top trader long/short ratio (by positions)         |
| `count_long_short_ratio`             | float  | Global long/short ratio (by accounts)              |
| `sum_taker_long_short_vol_ratio`     | float  | Taker buy/sell volume ratio                        |

**Example:**
```
create_time,symbol,sum_open_interest,sum_open_interest_value,...
2026-01-01 00:05:00,SOLUSDT,8914568.00,1112335669.28,4.807,2.164,4.270,2.333
```

---

### Klines (OHLCV Candlesticks)

**Path:** `binance/futures/klines/{interval}/`
**Filename:** `{SYMBOL}-{interval}-{YYYY-MM-DD}.zip`
**Intervals:** `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d` (configurable)

| Column                    | Type  | Description                                |
|---------------------------|-------|--------------------------------------------|
| `open_time`               | int   | Candle open time, Unix ms                  |
| `open`                    | float | Open price                                 |
| `high`                    | float | High price                                 |
| `low`                     | float | Low price                                  |
| `close`                   | float | Close price                                |
| `volume`                  | float | Volume in base asset                       |
| `close_time`              | int   | Candle close time, Unix ms                 |
| `quote_volume`            | float | Volume in quote asset (USD)                |
| `count`                   | int   | Number of trades                           |
| `taker_buy_volume`        | float | Taker buy volume (base)                    |
| `taker_buy_quote_volume`  | float | Taker buy volume (quote)                   |
| `ignore`                  | int   | Unused (always `0`)                        |

**Example:**
```
open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore
1767225600000,124.5800,125.1900,124.5200,125.0700,297564.54,1767229199999,37180319.9561,35183,163584.72,20438432.7301,0
```

---

### Index Price Klines (indexPriceKlines)

**Path:** `binance/futures/indexPriceKlines/{interval}/`
**Filename:** `{SYMBOL}-{interval}-{YYYY-MM-DD}.zip`

Same 12-column schema as [Klines](#klines-ohlcv-candlesticks), but OHLC values represent the **index price** (weighted average across exchanges). Volume-related fields are `0`.

---

### Mark Price Klines (markPriceKlines)

**Path:** `binance/futures/markPriceKlines/{interval}/`
**Filename:** `{SYMBOL}-{interval}-{YYYY-MM-DD}.zip`

Same 12-column schema as [Klines](#klines-ohlcv-candlesticks), but OHLC values represent the **mark price** (used for PnL and liquidation calculations). Volume-related fields are `0`.

---

### Premium Index Klines (premiumIndexKlines)

**Path:** `binance/futures/premiumIndexKlines/{interval}/`
**Filename:** `{SYMBOL}-{interval}-{YYYY-MM-DD}.zip`

Same 12-column schema as [Klines](#klines-ohlcv-candlesticks), but OHLC values represent the **premium index** (difference between futures and index price, as a ratio). Volume-related fields are `0`. Count reflects the number of samples, not trades.

---

## OKX

All files sourced from `https://static.okx.com/cdn/okex/traderecords/trades/daily/`.

### OKX Futures (Perpetual Swap Trades)

**Source:** `https://static.okx.com/cdn/okex/traderecords/trades/daily/{YYYYMMDD}/`
**File format:** `.zip` → `.csv` with header
**Filename pattern:** `{BASE}-{QUOTE}-SWAP-trades-{YYYY-MM-DD}.zip`

| Column            | Type   | Description                                          |
|-------------------|--------|------------------------------------------------------|
| `instrument_name` | string | Instrument identifier (e.g. `BTC-USDT-SWAP`)        |
| `trade_id`        | int    | Trade ID                                             |
| `side`            | string | Trade direction: `buy` or `sell` (lowercase)         |
| `price`           | float  | Execution price                                      |
| `size`            | float  | Trade quantity in contracts (base asset)             |
| `created_time`    | int    | Unix timestamp in milliseconds                       |

**Example:**
```
instrument_name,trade_id,side,price,size,created_time
BTC-USDT-SWAP,1566502282,sell,104503.0,30.14,1748707200008
```

---

### OKX Spot (Trades)

**Source:** `https://static.okx.com/cdn/okex/traderecords/trades/daily/{YYYYMMDD}/`
**File format:** `.zip` → `.csv` with header
**Filename pattern:** `{BASE}-{QUOTE}-trades-{YYYY-MM-DD}.zip`

| Column            | Type   | Description                                          |
|-------------------|--------|------------------------------------------------------|
| `instrument_name` | string | Instrument identifier (e.g. `BTC-USDT`)             |
| `trade_id`        | int    | Trade ID                                             |
| `side`            | string | Trade direction: `buy` or `sell` (lowercase)         |
| `price`           | float  | Execution price                                      |
| `size`            | float  | Trade quantity in base asset                         |
| `created_time`    | int    | Unix timestamp in milliseconds                       |

**Example:**
```
instrument_name,trade_id,side,price,size,created_time
BTC-USDT,747835547,buy,104554.2,1.912e-05,1748707200531
```

> **Note:** OKX spot `size` for BTC pairs can be very small (scientific notation).
> The `quote_quantity` (price × size) is not provided and must be computed.

---

## Binance Spot

All files sourced from `https://data.binance.vision/data/spot/daily/`.

> **⚠️ Binance spot CSVs have NO header row.** You must supply column names when reading.

### Trades

**Path:** `binance/spot/trades/`
**Filename:** `{SYMBOL}-trades-{YYYY-MM-DD}.zip`

| #  | Column          | Type   | Description                                          |
|----|-----------------|--------|------------------------------------------------------|
| 0  | `trade_id`      | int    | Trade ID                                             |
| 1  | `price`         | float  | Execution price                                      |
| 2  | `qty`           | float  | Quantity in base asset                               |
| 3  | `quote_qty`     | float  | Quantity in quote asset                              |
| 4  | `time`          | int    | Unix timestamp in **microseconds** (since 2025-01-01)|
| 5  | `is_buyer_maker`| string | `True` / `False`                                     |
| 6  | `is_best_match` | string | `True` / `False`                                     |

**Example (no header):**
```
1803839896,124.64000000,1.12700000,140.46928000,1767225600629785,True,True
```

**Reading with pandas:**
```python
pd.read_csv("SOLUSDT-trades-2026-01-01.zip",
    header=None,
    names=["trade_id","price","qty","quote_qty","time","is_buyer_maker","is_best_match"])
```

---

### Aggregate Trades (aggTrades)

**Path:** `binance/spot/aggTrades/`
**Filename:** `{SYMBOL}-aggTrades-{YYYY-MM-DD}.zip`

| #  | Column          | Type   | Description                                          |
|----|-----------------|--------|------------------------------------------------------|
| 0  | `agg_trade_id`  | int    | Aggregate trade ID                                   |
| 1  | `price`         | float  | Execution price                                      |
| 2  | `quantity`      | float  | Total aggregated quantity                            |
| 3  | `first_trade_id`| int    | First individual trade ID                            |
| 4  | `last_trade_id` | int    | Last individual trade ID                             |
| 5  | `transact_time` | int    | Unix timestamp in **microseconds** (since 2025-01-01)|
| 6  | `is_buyer_maker`| string | `True` / `False`                                     |
| 7  | `is_best_match` | string | `True` / `False`                                     |

**Example (no header):**
```
624159153,124.64000000,1.12700000,1803839896,1803839896,1767225600629785,True,True
```

---

### Klines (OHLCV Candlesticks)

**Path:** `binance/spot/klines/{interval}/`
**Filename:** `{SYMBOL}-{interval}-{YYYY-MM-DD}.zip`

| #  | Column                    | Type  | Description                                          |
|----|---------------------------|-------|------------------------------------------------------|
| 0  | `open_time`               | int   | Candle open time, Unix **microseconds** (since 2025-01-01) |
| 1  | `open`                    | float | Open price                                           |
| 2  | `high`                    | float | High price                                           |
| 3  | `low`                     | float | Low price                                            |
| 4  | `close`                   | float | Close price                                          |
| 5  | `volume`                  | float | Volume in base asset                                 |
| 6  | `close_time`              | int   | Candle close time, Unix **microseconds**             |
| 7  | `quote_volume`            | float | Volume in quote asset                                |
| 8  | `count`                   | int   | Number of trades                                     |
| 9  | `taker_buy_volume`        | float | Taker buy volume (base)                              |
| 10 | `taker_buy_quote_volume`  | float | Taker buy volume (quote)                             |
| 11 | `ignore`                  | int   | Unused (always `0`)                                  |

**Example (no header):**
```
1767225600000000,124.64000000,125.25000000,124.60000000,125.15000000,51890.42700000,1767229199999999,6488617.42421000,20451,22746.00100000,2843577.88951000,0
```

---

## Timestamp Reference

| Source              | Unit                                | Notes                              |
|---------------------|-------------------------------------|------------------------------------|
| Bybit futures       | Seconds (float, sub-second decimal) | e.g. `1767225600.6864`             |
| Bybit spot          | Milliseconds (int)                  | e.g. `1767225600884`               |
| Binance futures     | Milliseconds (int)                  | All data types                     |
| Binance spot        | **Microseconds** (int)              | Since 2025-01-01 per Binance docs  |
| Binance bookDepth   | UTC datetime string                 | `YYYY-MM-DD HH:MM:SS`             |
| Binance metrics     | UTC datetime string                 | `YYYY-MM-DD HH:MM:SS`             |
| OKX futures         | Milliseconds (int)                  | All data types                     |
| OKX spot            | Milliseconds (int)                  | All data types                     |

---

## Quick Start

```bash
# Download all available data for BTCUSDT, Jan 1-7 2026
python download_market_data.py BTCUSDT 2026-01-01 2026-01-07

# Only Bybit sources
python download_market_data.py BTCUSDT 2026-01-01 2026-01-07 -s bybit_futures bybit_spot

# Only Binance futures trades and metrics
python download_market_data.py BTCUSDT 2026-01-01 2026-01-07 -s binance_futures \
    --binance-futures-data-types trades metrics

# Custom kline intervals
python download_market_data.py BTCUSDT 2026-01-01 2026-01-07 --kline-intervals 1m 1h 1d
```

### Reading compressed files directly with pandas

```python
import pandas as pd

# Bybit (.csv.gz) — has header
df = pd.read_csv("data/SOLUSDT/bybit/futures/SOLUSDT2026-01-01.csv.gz")

# Binance futures (.zip) — has header
df = pd.read_csv("data/SOLUSDT/binance/futures/trades/SOLUSDT-trades-2026-01-01.zip")

# Binance spot (.zip) — NO header, must supply names
df = pd.read_csv("data/SOLUSDT/binance/spot/trades/SOLUSDT-trades-2026-01-01.zip",
    header=None,
    names=["trade_id","price","qty","quote_qty","time","is_buyer_maker","is_best_match"])
```
