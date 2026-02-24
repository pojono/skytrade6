# How to Download & Update Market Data from Dataminer

## Prerequisites

- SSH access to dataminer server: `ubuntu@13.251.79.76`
- SSH key: `~/.ssh/id_ed25519_remote`

---

## Step 1: Download raw JSONL data from dataminer

```bash
# Download ALL new data (incremental — skips already-downloaded files)
./download_ticker_all.sh

# Download specific date range only
./download_ticker_all.sh --start 2026-02-20 --end 2026-02-24

# Preview what would be downloaded (no actual download)
./download_ticker_all.sh --dry-run

# Check what dates are available on the server
./download_ticker_all.sh --list-dates

# Force re-download (overwrite existing files)
./download_ticker_all.sh --force
```

This downloads **symbol=ALL** data for all 3 exchanges and all streams:

| Exchange | Streams | ~Size/hr |
|---|---|---|
| Binance | linear/ticker, linear/fundingRate, spot/ticker | ~63 MB |
| Bybit | linear/ticker, spot/ticker | ~44 MB |
| OKX | linear/ticker, spot/ticker | ~12 MB |

Output goes to: `data_all/{exchange}/{market}/{stream}/{stream}_{date}_hr{HH}.jsonl.gz`

---

## Step 2: Convert JSONL to Parquet

```bash
# Incremental conversion (only processes new files, appends to existing parquet)
python3 build_ticker_all_parquet.py

# Convert a specific stream only
python3 build_ticker_all_parquet.py --stream binance/ticker
python3 build_ticker_all_parquet.py --stream binance/fundingRate
python3 build_ticker_all_parquet.py --stream bybit/ticker

# Full rebuild from scratch (ignore manifest, overwrite parquet)
python3 build_ticker_all_parquet.py --full
```

Output parquets: `data_all/{exchange}/{stream}.parquet`

The script tracks processed files in `.manifest.json` — so re-running it only converts new files.

---

## Step 3: Download historical funding rates (REST API, no dataminer needed)

These scripts pull directly from exchange APIs — no dataminer server required:

```bash
# Bybit + Binance historical FR (goes back ~200 days)
python3 download_historical_fr.py

# Bitget historical FR (goes back ~17 days only)
python3 download_bitget_fr.py
```

Output: `data_all/historical_fr/{exchange}_fr_history.parquet`

---

## Daily Update Routine

Run these 3 commands to keep everything up to date:

```bash
# 1. Download new raw data from dataminer (incremental, ~2-5 min)
./download_ticker_all.sh

# 2. Convert new JSONL to parquet (incremental, ~1-3 min)
python3 build_ticker_all_parquet.py

# 3. Update historical FR parquets from APIs (~1-2 min)
python3 download_historical_fr.py
python3 download_bitget_fr.py
```

Or as a one-liner:

```bash
./download_ticker_all.sh && python3 build_ticker_all_parquet.py && python3 download_historical_fr.py && python3 download_bitget_fr.py
```

---

## Data Layout

```
data_all/
├── binance/
│   ├── linear/
│   │   ├── ticker/          ← JSONL files (ticker_{date}_hr{HH}.jsonl.gz)
│   │   └── fundingRate/     ← JSONL files
│   ├── spot/
│   │   └── ticker/          ← JSONL files
│   ├── ticker.parquet       ← Converted parquet (all linear tickers)
│   └── fundingRate.parquet  ← Converted parquet (all funding rates)
├── bybit/
│   ├── linear/ticker/       ← JSONL files
│   ├── spot/ticker/         ← JSONL files
│   └── ticker.parquet       ← Converted parquet
├── okx/
│   ├── linear/ticker/       ← JSONL files
│   └── spot/ticker/         ← JSONL files
└── historical_fr/
    ├── binance_fr_history.parquet   ← REST API FR data (~200 days)
    ├── bybit_fr_history.parquet     ← REST API FR data (~200 days)
    ├── okx_fr_history.parquet       ← REST API FR data (~200 days)
    └── bitget_fr_history.parquet    ← REST API FR data (~17 days)
```

## Notes

- `download_ticker_all.sh` skips the current (partial) hour by default. Use `--include-partial` to override.
- `build_ticker_all_parquet.py` skips files with `.partial` markers.
- All scripts are **incremental by default** — safe to re-run without re-downloading or re-processing.
