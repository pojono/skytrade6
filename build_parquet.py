#!/usr/bin/env python3
"""
Convert raw downloaded market data into daily-partitioned parquet files.

Reads from:  data/{SYMBOL}/  (output of download_market_data.py)
Writes to:   parquet/{SYMBOL}/

Output structure (one file per day per source):
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

Sources for trades/ohlcv:
  bybit_futures, bybit_spot, binance_futures, binance_spot,
  okx_futures, okx_spot

All timestamps are normalized to microseconds (int64, UTC).
All parquet files are sorted by timestamp and use snappy compression.
Existing parquet files are skipped (incremental builds).

Usage:
  python build_parquet.py SOLUSDT
  python build_parquet.py SOLUSDT --input ./data --output ./parquet
  python build_parquet.py SOLUSDT --ohlcv-intervals 1m 5m 15m 1h
"""

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OHLCV_INTERVALS = ["1m", "5m", "15m", "1h"]

PARQUET_WRITE_OPTS = dict(compression="snappy", use_dictionary=True)

# Unified trades schema
TRADES_COLUMNS = [
    "timestamp_us",    # int64, microseconds since epoch (UTC)
    "price",           # float64
    "quantity",        # float64, base asset
    "quote_quantity",  # float64, quote asset (price * quantity)
    "side",            # int8: 1 = buy, -1 = sell
    "trade_id",        # string, source-specific ID
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Global date filter (set via CLI args)
_DATE_START: str | None = None
_DATE_END: str | None = None


def find_files(directory: Path, pattern: str) -> list[Path]:
    """Find files matching glob pattern, sorted by name. Optionally filtered by date."""
    files = sorted(directory.glob(pattern))
    if _DATE_START or _DATE_END:
        filtered = []
        for f in files:
            d = extract_date_from_filename(f.name)
            if d is None:
                filtered.append(f)
                continue
            if _DATE_START and d < _DATE_START:
                continue
            if _DATE_END and d > _DATE_END:
                continue
            filtered.append(f)
        return filtered
    return files


def read_csv_gz(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip", **kwargs)


def read_csv_zip(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, compression="zip", **kwargs)


def extract_date_from_filename(fname: str) -> str | None:
    """Extract YYYY-MM-DD date from a filename."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", fname)
    return m.group(1) if m else None


def interval_to_us(interval: str) -> int:
    unit = interval[-1]
    val = int(interval[:-1])
    multipliers = {"s": 1_000_000, "m": 60_000_000, "h": 3_600_000_000, "d": 86_400_000_000}
    return val * multipliers[unit]


def write_parquet(df: pd.DataFrame, path: Path, sort_col: str = "timestamp_us"):
    if df.empty:
        return False
    df = df.sort_values(sort_col).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, **PARQUET_WRITE_OPTS)
    return True


# ---------------------------------------------------------------------------
# Trade parsers — each returns (date_str, DataFrame) for a single file
# ---------------------------------------------------------------------------


def parse_bybit_futures_trade_file(path: Path) -> tuple[str, pd.DataFrame]:
    """Parse one Bybit futures trade CSV. Filename: {SYMBOL}{YYYY-MM-DD}.csv.gz"""
    date = extract_date_from_filename(path.name)
    df = read_csv_gz(path)
    parsed = pd.DataFrame({
        "timestamp_us": (df["timestamp"] * 1_000_000).astype(np.int64),
        "price": df["price"].astype(np.float64),
        "quantity": df["size"].astype(np.float64),
        "quote_quantity": df["foreignNotional"].astype(np.float64),
        "side": np.where(df["side"] == "Buy", np.int8(1), np.int8(-1)),
        "trade_id": df["trdMatchID"].astype(str),
    })
    return date, parsed


def parse_bybit_spot_trade_file(path: Path) -> tuple[str, pd.DataFrame]:
    """Parse one Bybit spot trade CSV. Filename: {SYMBOL}_{YYYY-MM-DD}.csv.gz"""
    date = extract_date_from_filename(path.name)
    df = read_csv_gz(path)
    parsed = pd.DataFrame({
        "timestamp_us": (df["timestamp"] * 1_000).astype(np.int64),
        "price": df["price"].astype(np.float64),
        "quantity": df["volume"].astype(np.float64),
        "quote_quantity": (df["price"].astype(np.float64) * df["volume"].astype(np.float64)),
        "side": np.where(df["side"] == "buy", np.int8(1), np.int8(-1)),
        "trade_id": df["id"].astype(str),
    })
    return date, parsed


def parse_binance_futures_trade_file(path: Path) -> tuple[str, pd.DataFrame]:
    """Parse one Binance futures trade ZIP. Filename: {SYMBOL}-trades-{YYYY-MM-DD}.zip"""
    date = extract_date_from_filename(path.name)
    df = read_csv_zip(path)
    parsed = pd.DataFrame({
        "timestamp_us": (df["time"] * 1_000).astype(np.int64),
        "price": df["price"].astype(np.float64),
        "quantity": df["qty"].astype(np.float64),
        "quote_quantity": df["quote_qty"].astype(np.float64),
        "side": np.where(df["is_buyer_maker"].astype(str).str.lower() == "true",
                         np.int8(-1), np.int8(1)),
        "trade_id": df["id"].astype(str),
    })
    return date, parsed


def parse_binance_spot_trade_file(path: Path) -> tuple[str, pd.DataFrame]:
    """Parse one Binance spot trade ZIP (no header)."""
    date = extract_date_from_filename(path.name)
    col_names = ["trade_id", "price", "qty", "quote_qty", "time",
                 "is_buyer_maker", "is_best_match"]
    df = read_csv_zip(path, header=None, names=col_names)
    parsed = pd.DataFrame({
        "timestamp_us": df["time"].astype(np.int64),
        "price": df["price"].astype(np.float64),
        "quantity": df["qty"].astype(np.float64),
        "quote_quantity": df["quote_qty"].astype(np.float64),
        "side": np.where(df["is_buyer_maker"].astype(str).str.strip() == "True",
                         np.int8(-1), np.int8(1)),
        "trade_id": df["trade_id"].astype(str),
    })
    return date, parsed


def parse_okx_trade_file(path: Path) -> tuple[str, pd.DataFrame]:
    """Parse one OKX trade ZIP. Filename: {INSTRUMENT}-trades-{YYYY-MM-DD}.zip"""
    date = extract_date_from_filename(path.name)
    df = read_csv_zip(path)
    price = df["price"].astype(np.float64)
    quantity = df["size"].astype(np.float64)
    parsed = pd.DataFrame({
        "timestamp_us": (df["created_time"] * 1_000).astype(np.int64),
        "price": price,
        "quantity": quantity,
        "quote_quantity": (price * quantity).astype(np.float64),
        "side": np.where(df["side"].str.lower() == "buy", np.int8(1), np.int8(-1)),
        "trade_id": df["trade_id"].astype(str),
    })
    return date, parsed


# ---------------------------------------------------------------------------
# OHLCV from trades (vectorized)
# ---------------------------------------------------------------------------


def trades_to_ohlcv_1m(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trades into 1-minute OHLCV bars. Fully vectorized."""
    if trades_df.empty:
        return pd.DataFrame()

    INTERVAL_1M = 60_000_000
    df = trades_df.sort_values("timestamp_us")
    bar_ts = (df["timestamp_us"].values // INTERVAL_1M) * INTERVAL_1M
    qty = df["quantity"].values
    side = df["side"].values

    agg_df = pd.DataFrame({
        "bar_ts": bar_ts,
        "price": df["price"].values,
        "quantity": qty,
        "quote_quantity": df["quote_quantity"].values,
        "buy_qty": np.where(side == 1, qty, 0.0),
        "sell_qty": np.where(side == -1, qty, 0.0),
    })

    grouped = agg_df.groupby("bar_ts", sort=True)

    ohlcv = pd.DataFrame({
        "timestamp_us": grouped["bar_ts"].first().index.astype(np.int64),
        "open": grouped["price"].first().values,
        "high": grouped["price"].max().values,
        "low": grouped["price"].min().values,
        "close": grouped["price"].last().values,
        "volume": grouped["quantity"].sum().values,
        "quote_volume": grouped["quote_quantity"].sum().values,
        "trade_count": grouped["price"].count().values.astype(np.int64),
        "buy_volume": grouped["buy_qty"].sum().values,
        "sell_volume": grouped["sell_qty"].sum().values,
    })

    ohlcv["vwap"] = np.where(
        ohlcv["volume"] > 0,
        ohlcv["quote_volume"] / ohlcv["volume"],
        ohlcv["close"],
    )
    return ohlcv


def resample_ohlcv(ohlcv_1m: pd.DataFrame, interval_us: int) -> pd.DataFrame:
    """Resample 1m OHLCV bars to a larger interval. Fully vectorized."""
    if ohlcv_1m.empty:
        return pd.DataFrame()

    df = ohlcv_1m.sort_values("timestamp_us")
    bar_ts = (df["timestamp_us"].values // interval_us) * interval_us

    agg_df = pd.DataFrame({
        "bar_ts": bar_ts,
        "open": df["open"].values,
        "high": df["high"].values,
        "low": df["low"].values,
        "close": df["close"].values,
        "volume": df["volume"].values,
        "quote_volume": df["quote_volume"].values,
        "trade_count": df["trade_count"].values,
        "buy_volume": df["buy_volume"].values,
        "sell_volume": df["sell_volume"].values,
    })

    grouped = agg_df.groupby("bar_ts", sort=True)

    result = pd.DataFrame({
        "timestamp_us": grouped["bar_ts"].first().index.astype(np.int64),
        "open": grouped["open"].first().values,
        "high": grouped["high"].max().values,
        "low": grouped["low"].min().values,
        "close": grouped["close"].last().values,
        "volume": grouped["volume"].sum().values,
        "quote_volume": grouped["quote_volume"].sum().values,
        "trade_count": grouped["trade_count"].sum().values,
        "buy_volume": grouped["buy_volume"].sum().values,
        "sell_volume": grouped["sell_volume"].sum().values,
    })

    result["vwap"] = np.where(
        result["volume"] > 0,
        result["quote_volume"] / result["volume"],
        result["close"],
    )
    return result


# ---------------------------------------------------------------------------
# Binance extras parsers (single-file)
# ---------------------------------------------------------------------------


def parse_binance_agg_trade_file(path: Path, market: str) -> tuple[str, pd.DataFrame]:
    """Parse one Binance aggregate trades file."""
    date = extract_date_from_filename(path.name)
    if market == "spot":
        col_names = ["agg_trade_id", "price", "quantity", "first_trade_id",
                     "last_trade_id", "transact_time", "is_buyer_maker", "is_best_match"]
        df = read_csv_zip(path, header=None, names=col_names)
    else:
        df = read_csv_zip(path)

    time_col = "transact_time"
    if market == "futures":
        df["timestamp_us"] = (df[time_col] * 1_000).astype(np.int64)
    else:
        df["timestamp_us"] = df[time_col].astype(np.int64)

    df["price"] = df["price"].astype(np.float64)
    df["quantity"] = df["quantity"].astype(np.float64)
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.strip().str.lower() == "true"

    keep = ["timestamp_us", "agg_trade_id", "price", "quantity",
            "first_trade_id", "last_trade_id", "is_buyer_maker"]
    return date, df[keep].sort_values("timestamp_us").reset_index(drop=True)


def parse_binance_metrics_file(path: Path) -> tuple[str, pd.DataFrame]:
    """Parse one Binance futures metrics file."""
    date = extract_date_from_filename(path.name)
    df = read_csv_zip(path)

    df["timestamp_us"] = (
        pd.to_datetime(df["create_time"], utc=True)
        .astype(np.int64) // 1_000
    )

    df = df.rename(columns={
        "sum_open_interest": "open_interest",
        "sum_open_interest_value": "open_interest_value",
        "count_toptrader_long_short_ratio": "top_trader_ls_ratio_accounts",
        "sum_toptrader_long_short_ratio": "top_trader_ls_ratio_positions",
        "count_long_short_ratio": "global_ls_ratio",
        "sum_taker_long_short_vol_ratio": "taker_buy_sell_ratio",
    })

    keep = ["timestamp_us", "open_interest", "open_interest_value",
            "top_trader_ls_ratio_accounts", "top_trader_ls_ratio_positions",
            "global_ls_ratio", "taker_buy_sell_ratio"]

    for c in keep[1:]:
        df[c] = df[c].astype(np.float64)

    return date, df[keep].sort_values("timestamp_us").reset_index(drop=True)


def parse_binance_book_depth_file(path: Path) -> tuple[str, pd.DataFrame]:
    """Parse one Binance futures book depth file."""
    date = extract_date_from_filename(path.name)
    df = read_csv_zip(path)

    df["timestamp_us"] = (
        pd.to_datetime(df["timestamp"], utc=True)
        .astype(np.int64) // 1_000
    )
    df["percentage"] = df["percentage"].astype(np.int8)
    df["depth"] = df["depth"].astype(np.float64)
    df["notional"] = df["notional"].astype(np.float64)

    keep = ["timestamp_us", "percentage", "depth", "notional"]
    return date, df[keep].sort_values(["timestamp_us", "percentage"]).reset_index(drop=True)


def parse_binance_kline_file(path: Path, market: str) -> tuple[str, pd.DataFrame]:
    """Parse one Binance kline file."""
    date = extract_date_from_filename(path.name)

    kline_cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ]

    if market == "spot":
        df = read_csv_zip(path, header=None, names=kline_cols)
    else:
        df = read_csv_zip(path)
        df.columns = kline_cols

    if market == "spot":
        df["timestamp_us"] = df["open_time"].astype(np.int64)
    else:
        df["timestamp_us"] = (df["open_time"] * 1_000).astype(np.int64)

    for c in ["open", "high", "low", "close", "volume", "quote_volume",
               "taker_buy_volume", "taker_buy_quote_volume"]:
        df[c] = df[c].astype(np.float64)
    df["count"] = df["count"].astype(np.int64)

    keep = ["timestamp_us", "open", "high", "low", "close", "volume",
            "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]
    return date, df[keep].sort_values("timestamp_us").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Source file discovery
# ---------------------------------------------------------------------------

TRADE_SOURCES = {
    "bybit_futures": {
        "subdir": "bybit/futures",
        "pattern": "*.csv.gz",
        "parser": parse_bybit_futures_trade_file,
        "utc_aligned": True,
    },
    "bybit_spot": {
        "subdir": "bybit/spot",
        "pattern": "*.csv.gz",
        "parser": parse_bybit_spot_trade_file,
        "utc_aligned": True,
    },
    "binance_futures": {
        "subdir": "binance/futures/trades",
        "pattern": "*.zip",
        "parser": parse_binance_futures_trade_file,
        "utc_aligned": True,
    },
    "binance_spot": {
        "subdir": "binance/spot/trades",
        "pattern": "*.zip",
        "parser": parse_binance_spot_trade_file,
        "utc_aligned": True,
    },
    "okx_futures": {
        "subdir": "okx/futures",
        "pattern": "*-SWAP-trades-*.zip",
        "parser": parse_okx_trade_file,
        "utc_aligned": False,  # OKX uses UTC+8 day boundaries
    },
    "okx_spot": {
        "subdir": "okx/spot",
        "pattern": "*-trades-*.zip",
        "parser": parse_okx_trade_file,
        "utc_aligned": False,  # OKX uses UTC+8 day boundaries
    },
}

ONE_DAY_US = 86_400_000_000  # 24h in microseconds


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def _write_day_trades_and_ohlcv(
    trades_df: pd.DataFrame, date: str, source: str,
    out_dir: Path, intervals: list[str],
) -> tuple[bool, bool]:
    """Write trades + OHLCV parquet for one UTC day. Returns (wrote_trades, skipped)."""
    trades_path = out_dir / "trades" / source / f"{date}.parquet"
    wrote = False
    skip = False

    if trades_path.exists():
        skip = True
    else:
        write_parquet(trades_df, trades_path)
        wrote = True

    # Build OHLCV
    ohlcv_1m = trades_to_ohlcv_1m(trades_df)
    for iv in intervals:
        ohlcv_path = out_dir / "ohlcv" / iv / source / f"{date}.parquet"
        if ohlcv_path.exists():
            continue
        if iv == "1m":
            write_parquet(ohlcv_1m, ohlcv_path)
        else:
            write_parquet(resample_ohlcv(ohlcv_1m, interval_to_us(iv)), ohlcv_path)

    return wrote, skip


def _process_utc_aligned(source, cfg, data_dir, out_dir, intervals):
    """Process a UTC-aligned source (Bybit, Binance): 1 file = 1 UTC day."""
    src_dir = data_dir / cfg["subdir"]
    files = find_files(src_dir, cfg["pattern"])
    if not files:
        print(f"  {source}: no files found")
        return 0, 0

    parser = cfg["parser"]
    written = 0
    skipped = 0
    n = len(files)

    print(f"  {source} ({n} files)...")
    for i, f in enumerate(files, 1):
        # Early skip: check if output already exists before parsing raw file
        date_from_name = extract_date_from_filename(f.name)
        if date_from_name:
            trades_path = out_dir / "trades" / source / f"{date_from_name}.parquet"
            ohlcv_paths = [out_dir / "ohlcv" / iv / source / f"{date_from_name}.parquet"
                           for iv in intervals]
            if trades_path.exists() and all(p.exists() for p in ohlcv_paths):
                skipped += 1
                continue

        try:
            date, trades_df = parser(f)
        except Exception as exc:
            print(f"    [{i}/{n}] ⚠ {f.name}: {exc}")
            continue

        if date is None or trades_df.empty:
            print(f"    [{i}/{n}] - {f.name} (empty)")
            continue

        wrote, skip = _write_day_trades_and_ohlcv(
            trades_df, date, source, out_dir, intervals)

        if skip:
            skipped += 1
            print(f"    [{i}/{n}] ⊘ {date}  (exists)")
        elif wrote:
            written += 1
            print(f"    [{i}/{n}] ✓ {date}  trades={len(trades_df):,}")

        del trades_df

    print(f"  {source}: {written} written, {skipped} skipped")
    return written, skipped


def _process_non_utc_aligned(source, cfg, data_dir, out_dir, intervals):
    """Process a non-UTC-aligned source (OKX: UTC+8 day boundaries).

    OKX files cover 16:00 UTC (prev day) → 15:59 UTC (file date).
    Each file contributes to exactly 2 UTC days.

    Strategy: process files one at a time, split each into UTC days,
    and use a carry-over buffer for the partial day that spans into
    the next file. Only 1 file + small buffer in memory at a time.
    """
    src_dir = data_dir / cfg["subdir"]
    files = find_files(src_dir, cfg["pattern"])
    if not files:
        print(f"  {source}: no files found")
        return 0, 0

    parser = cfg["parser"]
    n = len(files)
    print(f"  {source} ({n} files, re-partitioning to UTC days)...")

    # Fast pre-check: estimate expected UTC dates from raw file date range.
    # OKX files are UTC+8, so raw date D covers UTC days D-1 and D.
    # If ALL expected trades parquet files exist, skip entirely.
    first_date = extract_date_from_filename(files[0].name)
    last_date = extract_date_from_filename(files[-1].name)
    if first_date and last_date:
        expected_start = datetime.strptime(first_date, "%Y-%m-%d") - timedelta(days=1)
        expected_end = datetime.strptime(last_date, "%Y-%m-%d")
        expected_dates = pd.date_range(expected_start, expected_end)
        trades_out = out_dir / "trades" / source
        all_exist = all(
            (trades_out / f"{d.strftime('%Y-%m-%d')}.parquet").exists()
            for d in expected_dates
        )
        if all_exist:
            print(f"  {source}: all {len(expected_dates)} output files exist, skipping")
            return 0, len(expected_dates)

    written = 0
    skipped = 0
    # Buffer: trades from the latter part of the previous file (16:00→23:59 UTC)
    # that belong to the same UTC day as the early part of the next file
    carry: dict[str, list[pd.DataFrame]] = {}

    for i, f in enumerate(files, 1):
        try:
            _, trades_df = parser(f)
        except Exception as exc:
            print(f"    [{i}/{n}] ⚠ {f.name}: {exc}")
            continue

        if trades_df.empty:
            continue

        # Split trades by UTC day
        utc_day_ts = (trades_df["timestamp_us"].values // ONE_DAY_US) * ONE_DAY_US
        trades_df["_utc_date"] = pd.to_datetime(utc_day_ts, unit="us", utc=True).strftime("%Y-%m-%d")

        day_groups = {}
        for utc_date, grp in trades_df.groupby("_utc_date"):
            day_groups[utc_date] = grp.drop(columns=["_utc_date"])

        del trades_df

        # Merge carry-over into day_groups
        for d, chunks in carry.items():
            if d in day_groups:
                chunks.append(day_groups[d])
                day_groups[d] = pd.concat(chunks, ignore_index=True)
            else:
                day_groups[d] = pd.concat(chunks, ignore_index=True)
        carry.clear()

        # The latest UTC day in this file is incomplete (continues in next file)
        # unless this is the last file
        sorted_days = sorted(day_groups.keys())

        if i < n and len(sorted_days) > 1:
            # Keep the latest day as carry-over
            latest = sorted_days[-1]
            carry[latest] = [day_groups.pop(latest)]

        # Write all complete UTC days
        for utc_date in sorted(day_groups.keys()):
            df = day_groups[utc_date]
            wrote, skip = _write_day_trades_and_ohlcv(
                df, utc_date, source, out_dir, intervals)

            if skip:
                skipped += 1
                print(f"    [{i}/{n}] ⊘ {utc_date}  (exists)")
            elif wrote:
                written += 1
                print(f"    [{i}/{n}] ✓ {utc_date}  trades={len(df):,}")

        del day_groups

    # Flush any remaining carry-over (last file's latest day)
    for utc_date, chunks in carry.items():
        df = pd.concat(chunks, ignore_index=True)
        wrote, skip = _write_day_trades_and_ohlcv(
            df, utc_date, source, out_dir, intervals)
        if skip:
            skipped += 1
            print(f"    ⊘ {utc_date}  (exists)")
        elif wrote:
            written += 1
            print(f"    ✓ {utc_date}  trades={len(df):,}")

    print(f"  {source}: {written} written, {skipped} skipped")
    return written, skipped


def process_trades_and_ohlcv(data_dir: Path, out_dir: Path, intervals: list[str]):
    """Process all trade sources: write daily trades + OHLCV parquet files."""
    total_written = 0
    total_skipped = 0

    for source, cfg in TRADE_SOURCES.items():
        if cfg["utc_aligned"]:
            w, s = _process_utc_aligned(source, cfg, data_dir, out_dir, intervals)
        else:
            w, s = _process_non_utc_aligned(source, cfg, data_dir, out_dir, intervals)
        total_written += w
        total_skipped += s

    return total_written, total_skipped


def process_binance_extras(data_dir: Path, out_dir: Path):
    """Process Binance non-trade data types (metrics, bookDepth, aggTrades, klines)."""
    total_written = 0

    # --- Aggregate trades ---
    for market in ["futures", "spot"]:
        agg_dir = data_dir / "binance" / market / "aggTrades"
        files = find_files(agg_dir, "*.zip")
        if not files:
            continue
        written = 0
        n = len(files)
        print(f"  agg_trades_{market} ({n} files)...")
        for i, f in enumerate(files, 1):
            try:
                date, df = parse_binance_agg_trade_file(f, market)
            except Exception as exc:
                print(f"    [{i}/{n}] ⚠ {f.name}: {exc}")
                continue
            dest = out_dir / "binance" / f"agg_trades_{market}" / f"{date}.parquet"
            if dest.exists():
                print(f"    [{i}/{n}] ⊘ {date}  (exists)")
            elif write_parquet(df, dest):
                written += 1
                print(f"    [{i}/{n}] ✓ {date}  rows={len(df):,}")
        print(f"  agg_trades_{market}: {written} written")
        total_written += written

    # --- Metrics ---
    metrics_files = find_files(data_dir / "binance" / "futures" / "metrics", "*.zip")
    if metrics_files:
        written = 0
        n = len(metrics_files)
        print(f"  metrics ({n} files)...")
        for i, f in enumerate(metrics_files, 1):
            try:
                date, df = parse_binance_metrics_file(f)
            except Exception as exc:
                print(f"    [{i}/{n}] ⚠ {f.name}: {exc}")
                continue
            dest = out_dir / "binance" / "metrics" / f"{date}.parquet"
            if dest.exists():
                print(f"    [{i}/{n}] ⊘ {date}  (exists)")
            elif write_parquet(df, dest):
                written += 1
                print(f"    [{i}/{n}] ✓ {date}  rows={len(df):,}")
        print(f"  metrics: {written} written")
        total_written += written

    # --- Book depth ---
    bd_files = find_files(data_dir / "binance" / "futures" / "bookDepth", "*.zip")
    if bd_files:
        written = 0
        n = len(bd_files)
        print(f"  book_depth ({n} files)...")
        for i, f in enumerate(bd_files, 1):
            try:
                date, df = parse_binance_book_depth_file(f)
            except Exception as exc:
                print(f"    [{i}/{n}] ⚠ {f.name}: {exc}")
                continue
            dest = out_dir / "binance" / "book_depth" / f"{date}.parquet"
            if dest.exists():
                print(f"    [{i}/{n}] ⊘ {date}  (exists)")
            elif write_parquet(df, dest):
                written += 1
                print(f"    [{i}/{n}] ✓ {date}  rows={len(df):,}")
        print(f"  book_depth: {written} written")
        total_written += written

    # --- Klines (all types) ---
    kline_types = {
        "futures": ["klines", "indexPriceKlines", "markPriceKlines", "premiumIndexKlines"],
        "spot": ["klines"],
    }
    dir_name_map = {
        "klines": "klines",
        "indexPriceKlines": "index_price_klines",
        "markPriceKlines": "mark_price_klines",
        "premiumIndexKlines": "premium_index_klines",
    }

    for market, dtypes in kline_types.items():
        for dtype in dtypes:
            clean_name = dir_name_map[dtype]
            base = data_dir / "binance" / market / dtype
            if not base.exists():
                continue

            for interval_dir in sorted(base.iterdir()):
                if not interval_dir.is_dir():
                    continue
                interval = interval_dir.name
                files = find_files(interval_dir, "*.zip")
                if not files:
                    continue
                written = 0
                n = len(files)
                print(f"  {clean_name}_{market}/{interval} ({n} files)...")
                for i, f in enumerate(files, 1):
                    try:
                        date, df = parse_binance_kline_file(f, market)
                    except Exception as exc:
                        print(f"    [{i}/{n}] ⚠ {f.name}: {exc}")
                        continue
                    dest = out_dir / "binance" / f"{clean_name}_{market}" / interval / f"{date}.parquet"
                    if dest.exists():
                        print(f"    [{i}/{n}] ⊘ {date}  (exists)")
                    elif write_parquet(df, dest):
                        written += 1
                        print(f"    [{i}/{n}] ✓ {date}  rows={len(df):,}")
                print(f"  {clean_name}_{market}/{interval}: {written} written")
                total_written += written

    return total_written


def run(args):
    symbol = args.symbol.upper()
    data_dir = Path(args.input) / symbol
    out_dir = Path(args.output) / symbol
    intervals = args.ohlcv_intervals

    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Symbol:      {symbol}")
    print(f"Input:       {data_dir.resolve()}")
    print(f"Output:      {out_dir.resolve()}")
    print(f"Intervals:   {', '.join(intervals)}")
    print("=" * 60)

    # Step 1: Trades + OHLCV
    print("\n[1/3] Processing trades & OHLCV...")
    t_written, t_skipped = process_trades_and_ohlcv(data_dir, out_dir, intervals)
    print(f"  → {t_written} trade files written, {t_skipped} skipped")

    # Step 2: Binance extras
    print("\n[2/3] Processing Binance extras...")
    b_written = process_binance_extras(data_dir, out_dir)
    print(f"  → {b_written} extra files written")

    # Step 3: Summary
    print("\n[3/3] Summary")
    total_files = len(list(out_dir.rglob("*.parquet")))
    total_size = sum(f.stat().st_size for f in out_dir.rglob("*.parquet"))
    print(f"  Total: {total_files} parquet files, {total_size / (1024*1024):.1f} MB")
    print(f"  Output: {out_dir.resolve()}")
    print("=" * 60)
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw market data to daily-partitioned parquet files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("symbol", help="Trading pair symbol, e.g. SOLUSDT")
    parser.add_argument("--input", "-i", default="./data",
                        help="Raw data directory (default: ./data)")
    parser.add_argument("--output", "-o", default="./parquet",
                        help="Parquet output directory (default: ./parquet)")
    parser.add_argument("--ohlcv-intervals", nargs="+", default=DEFAULT_OHLCV_INTERVALS,
                        help=f"OHLCV bar intervals (default: {DEFAULT_OHLCV_INTERVALS})")
    parser.add_argument("--start", default=None,
                        help="Start date filter YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", default=None,
                        help="End date filter YYYY-MM-DD (inclusive)")

    args = parser.parse_args()
    global _DATE_START, _DATE_END
    _DATE_START = args.start
    _DATE_END = args.end
    run(args)


if __name__ == "__main__":
    main()
