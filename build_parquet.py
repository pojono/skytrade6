#!/usr/bin/env python3
"""
Convert raw downloaded market data into analysis-ready parquet files.

Reads from:  data/{SYMBOL}/  (output of download_market_data.py)
Writes to:   parquet/{SYMBOL}/

Output structure:
  parquet/{SYMBOL}/
    trades/
      bybit_futures.parquet       # normalized raw trades
      bybit_spot.parquet
      binance_futures.parquet
      binance_spot.parquet
    ohlcv/
      {interval}/                 # 1m, 5m, 15m, 1h  (computed from raw trades)
        bybit_futures.parquet
        bybit_spot.parquet
        binance_futures.parquet
        binance_spot.parquet
    binance/
      metrics.parquet             # open interest, funding, long/short ratios
      book_depth.parquet          # order book depth snapshots
      agg_trades_futures.parquet  # aggregate trades (futures)
      agg_trades_spot.parquet     # aggregate trades (spot)
      klines_futures/             # pre-computed klines from Binance (for cross-check)
        {interval}.parquet
      klines_spot/
        {interval}.parquet
      index_price_klines/
        {interval}.parquet
      mark_price_klines/
        {interval}.parquet
      premium_index_klines/
        {interval}.parquet

All timestamps are normalized to microseconds (int64, UTC).
All parquet files are sorted by timestamp and use snappy compression.

Usage:
  python build_parquet.py SOLUSDT
  python build_parquet.py SOLUSDT --input ./data --output ./parquet
  python build_parquet.py SOLUSDT --ohlcv-intervals 1m 5m 15m 1h
"""

import argparse
import glob
import gzip
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OHLCV_INTERVALS = ["1m", "5m", "15m", "1h"]

PARQUET_WRITE_OPTS = dict(compression="snappy", use_dictionary=True)

# Unified trades schema: all sources normalize to these columns
TRADES_COLUMNS = [
    "timestamp_us",   # int64, microseconds since epoch (UTC)
    "price",          # float64
    "quantity",       # float64, base asset
    "quote_quantity",  # float64, quote asset (price * quantity)
    "side",           # int8: 1 = buy, -1 = sell
    "trade_id",       # string, source-specific ID
]

OHLCV_COLUMNS = [
    "timestamp_us",        # int64, bar open time (microseconds)
    "open",                # float64
    "high",                # float64
    "low",                 # float64
    "close",               # float64
    "volume",              # float64, base asset
    "quote_volume",        # float64, quote asset
    "trade_count",         # int64
    "buy_volume",          # float64, taker buy volume (base)
    "sell_volume",         # float64, taker sell volume (base)
    "vwap",                # float64, volume-weighted average price
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_files(directory: Path, pattern: str) -> list[Path]:
    """Find files matching glob pattern, sorted by name."""
    return sorted(directory.glob(pattern))


def read_csv_gz(path: Path, **kwargs) -> pd.DataFrame:
    """Read a gzip-compressed CSV."""
    return pd.read_csv(path, compression="gzip", **kwargs)


def read_csv_zip(path: Path, **kwargs) -> pd.DataFrame:
    """Read a zip-compressed CSV (first file in archive)."""
    return pd.read_csv(path, compression="zip", **kwargs)


def interval_to_us(interval: str) -> int:
    """Convert interval string to microseconds."""
    unit = interval[-1]
    val = int(interval[:-1])
    multipliers = {"s": 1_000_000, "m": 60_000_000, "h": 3_600_000_000, "d": 86_400_000_000}
    return val * multipliers[unit]


def write_parquet(df: pd.DataFrame, path: Path, sort_col: str = "timestamp_us"):
    """Write a DataFrame to parquet, sorted by timestamp."""
    if df.empty:
        print(f"    SKIP (empty): {path}")
        return
    df = df.sort_values(sort_col).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, **PARQUET_WRITE_OPTS)
    rows = len(df)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"    WROTE: {path}  ({rows:,} rows, {size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Trade parsers — one per source, all return unified schema
# ---------------------------------------------------------------------------


def parse_bybit_futures_trades(data_dir: Path) -> pd.DataFrame:
    """Parse Bybit futures trade CSVs into unified format."""
    files = find_files(data_dir / "bybit" / "futures", "*.csv.gz")
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = read_csv_gz(f)
        # timestamp is seconds float → convert to microseconds int
        parsed = pd.DataFrame({
            "timestamp_us": (df["timestamp"] * 1_000_000).astype(np.int64),
            "price": df["price"].astype(np.float64),
            "quantity": df["size"].astype(np.float64),
            "quote_quantity": df["foreignNotional"].astype(np.float64),
            "side": np.where(df["side"] == "Buy", np.int8(1), np.int8(-1)),
            "trade_id": df["trdMatchID"].astype(str),
        })
        dfs.append(parsed)

    return pd.concat(dfs, ignore_index=True)


def parse_bybit_spot_trades(data_dir: Path) -> pd.DataFrame:
    """Parse Bybit spot trade CSVs into unified format."""
    files = find_files(data_dir / "bybit" / "spot", "*.csv.gz")
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = read_csv_gz(f)
        # timestamp is milliseconds int → convert to microseconds
        parsed = pd.DataFrame({
            "timestamp_us": (df["timestamp"] * 1_000).astype(np.int64),
            "price": df["price"].astype(np.float64),
            "quantity": df["volume"].astype(np.float64),
            "quote_quantity": (df["price"].astype(np.float64) * df["volume"].astype(np.float64)),
            "side": np.where(df["side"] == "buy", np.int8(1), np.int8(-1)),
            "trade_id": df["id"].astype(str),
        })
        dfs.append(parsed)

    return pd.concat(dfs, ignore_index=True)


def parse_binance_futures_trades(data_dir: Path) -> pd.DataFrame:
    """Parse Binance futures trade CSVs into unified format."""
    files = find_files(data_dir / "binance" / "futures" / "trades", "*.zip")
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = read_csv_zip(f)
        # time is milliseconds → convert to microseconds
        parsed = pd.DataFrame({
            "timestamp_us": (df["time"] * 1_000).astype(np.int64),
            "price": df["price"].astype(np.float64),
            "quantity": df["qty"].astype(np.float64),
            "quote_quantity": df["quote_qty"].astype(np.float64),
            "side": np.where(df["is_buyer_maker"].astype(str).str.lower() == "true",
                             np.int8(-1), np.int8(1)),
            "trade_id": df["id"].astype(str),
        })
        dfs.append(parsed)

    return pd.concat(dfs, ignore_index=True)


def parse_binance_spot_trades(data_dir: Path) -> pd.DataFrame:
    """Parse Binance spot trade CSVs (no header) into unified format."""
    files = find_files(data_dir / "binance" / "spot" / "trades", "*.zip")
    if not files:
        return pd.DataFrame()

    col_names = ["trade_id", "price", "qty", "quote_qty", "time",
                 "is_buyer_maker", "is_best_match"]
    dfs = []
    for f in files:
        df = read_csv_zip(f, header=None, names=col_names)
        # time is microseconds since 2025-01-01
        parsed = pd.DataFrame({
            "timestamp_us": df["time"].astype(np.int64),
            "price": df["price"].astype(np.float64),
            "quantity": df["qty"].astype(np.float64),
            "quote_quantity": df["quote_qty"].astype(np.float64),
            "side": np.where(df["is_buyer_maker"].astype(str).str.strip() == "True",
                             np.int8(-1), np.int8(1)),
            "trade_id": df["trade_id"].astype(str),
        })
        dfs.append(parsed)

    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# OHLCV aggregation from trades
# ---------------------------------------------------------------------------


def trades_to_ohlcv(trades_df: pd.DataFrame, interval_us: int) -> pd.DataFrame:
    """Aggregate trades into OHLCV bars at the given interval (microseconds)."""
    if trades_df.empty:
        return pd.DataFrame()

    df = trades_df.sort_values("timestamp_us").copy()

    # Compute bar open timestamp (floor to interval boundary)
    df["bar_ts"] = (df["timestamp_us"] // interval_us) * interval_us

    grouped = df.groupby("bar_ts", sort=True)

    ohlcv = pd.DataFrame({
        "timestamp_us": grouped["timestamp_us"].first().index.astype(np.int64),
        "open": grouped["price"].first().values,
        "high": grouped["price"].max().values,
        "low": grouped["price"].min().values,
        "close": grouped["price"].last().values,
        "volume": grouped["quantity"].sum().values,
        "quote_volume": grouped["quote_quantity"].sum().values,
        "trade_count": grouped["price"].count().values.astype(np.int64),
        "buy_volume": grouped.apply(
            lambda g: g.loc[g["side"] == 1, "quantity"].sum(), include_groups=False
        ).values,
        "sell_volume": grouped.apply(
            lambda g: g.loc[g["side"] == -1, "quantity"].sum(), include_groups=False
        ).values,
    })

    ohlcv["vwap"] = np.where(
        ohlcv["volume"] > 0,
        ohlcv["quote_volume"] / ohlcv["volume"],
        ohlcv["close"],
    )

    return ohlcv


# ---------------------------------------------------------------------------
# Binance extras parsers
# ---------------------------------------------------------------------------


def parse_binance_agg_trades(data_dir: Path, market: str) -> pd.DataFrame:
    """Parse Binance aggregate trades. market = 'futures' or 'spot'."""
    files = find_files(data_dir / "binance" / market / "aggTrades", "*.zip")
    if not files:
        return pd.DataFrame()

    if market == "spot":
        col_names = ["agg_trade_id", "price", "quantity", "first_trade_id",
                     "last_trade_id", "transact_time", "is_buyer_maker", "is_best_match"]
        dfs = []
        for f in files:
            df = read_csv_zip(f, header=None, names=col_names)
            dfs.append(df)
    else:
        dfs = [read_csv_zip(f) for f in files]

    df = pd.concat(dfs, ignore_index=True)

    # Normalize timestamp
    time_col = "transact_time"
    if market == "futures":
        # milliseconds → microseconds
        df["timestamp_us"] = (df[time_col] * 1_000).astype(np.int64)
    else:
        # already microseconds
        df["timestamp_us"] = df[time_col].astype(np.int64)

    df["price"] = df["price"].astype(np.float64)
    df["quantity"] = df["quantity"].astype(np.float64)
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.strip().str.lower() == "true"

    keep = ["timestamp_us", "agg_trade_id", "price", "quantity",
            "first_trade_id", "last_trade_id", "is_buyer_maker"]
    return df[keep].sort_values("timestamp_us").reset_index(drop=True)


def parse_binance_metrics(data_dir: Path) -> pd.DataFrame:
    """Parse Binance futures metrics."""
    files = find_files(data_dir / "binance" / "futures" / "metrics", "*.zip")
    if not files:
        return pd.DataFrame()

    dfs = [read_csv_zip(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # create_time is UTC datetime string → microseconds
    df["timestamp_us"] = (
        pd.to_datetime(df["create_time"], utc=True)
        .astype(np.int64) // 1_000  # ns → us
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

    return df[keep].sort_values("timestamp_us").reset_index(drop=True)


def parse_binance_book_depth(data_dir: Path) -> pd.DataFrame:
    """Parse Binance futures book depth."""
    files = find_files(data_dir / "binance" / "futures" / "bookDepth", "*.zip")
    if not files:
        return pd.DataFrame()

    dfs = [read_csv_zip(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    df["timestamp_us"] = (
        pd.to_datetime(df["timestamp"], utc=True)
        .astype(np.int64) // 1_000
    )
    df["percentage"] = df["percentage"].astype(np.int8)
    df["depth"] = df["depth"].astype(np.float64)
    df["notional"] = df["notional"].astype(np.float64)

    keep = ["timestamp_us", "percentage", "depth", "notional"]
    return df[keep].sort_values(["timestamp_us", "percentage"]).reset_index(drop=True)


def parse_binance_klines(data_dir: Path, market: str, dtype: str) -> dict[str, pd.DataFrame]:
    """Parse Binance kline-type data. Returns {interval: DataFrame}."""
    base = data_dir / "binance" / market / dtype
    if not base.exists():
        return {}

    # Kline column names (same schema for all kline types)
    kline_cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ]

    result = {}
    for interval_dir in sorted(base.iterdir()):
        if not interval_dir.is_dir():
            continue
        interval = interval_dir.name
        files = find_files(interval_dir, "*.zip")
        if not files:
            continue

        dfs = []
        for f in files:
            if market == "spot":
                df = read_csv_zip(f, header=None, names=kline_cols)
            else:
                df = read_csv_zip(f)
                df.columns = kline_cols

            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        # Normalize open_time to microseconds
        if market == "spot":
            # already microseconds
            df["timestamp_us"] = df["open_time"].astype(np.int64)
        else:
            # milliseconds → microseconds
            df["timestamp_us"] = (df["open_time"] * 1_000).astype(np.int64)

        for c in ["open", "high", "low", "close", "volume", "quote_volume",
                   "taker_buy_volume", "taker_buy_quote_volume"]:
            df[c] = df[c].astype(np.float64)
        df["count"] = df["count"].astype(np.int64)

        keep = ["timestamp_us", "open", "high", "low", "close", "volume",
                "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]
        result[interval] = df[keep].sort_values("timestamp_us").reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_ohlcv_alignment(ohlcv_dict: dict[str, pd.DataFrame], interval: str):
    """Check that OHLCV bars across sources cover the same time range."""
    sources = {k: v for k, v in ohlcv_dict.items() if not v.empty}
    if len(sources) < 2:
        return

    print(f"\n  Alignment check ({interval}):")
    for name, df in sources.items():
        ts_min = pd.Timestamp(df["timestamp_us"].min(), unit="us", tz="UTC")
        ts_max = pd.Timestamp(df["timestamp_us"].max(), unit="us", tz="UTC")
        print(f"    {name:25s}  bars={len(df):>8,}  "
              f"from={ts_min.strftime('%Y-%m-%d %H:%M')}  "
              f"to={ts_max.strftime('%Y-%m-%d %H:%M')}")

    # Check for gaps: find timestamps present in one source but not another
    all_ts = set()
    for df in sources.values():
        all_ts.update(df["timestamp_us"].values)

    for name, df in sources.items():
        source_ts = set(df["timestamp_us"].values)
        missing = len(all_ts - source_ts)
        if missing > 0:
            print(f"    ⚠ {name}: {missing} bars missing vs union of all sources")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


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

    # ------------------------------------------------------------------
    # Step 1: Parse raw trades from all sources
    # ------------------------------------------------------------------
    print("\n[1/5] Parsing raw trades...")

    trade_parsers = {
        "bybit_futures": parse_bybit_futures_trades,
        "bybit_spot": parse_bybit_spot_trades,
        "binance_futures": parse_binance_futures_trades,
        "binance_spot": parse_binance_spot_trades,
    }

    trades = {}
    for source, parser in trade_parsers.items():
        print(f"  Parsing {source}...")
        df = parser(data_dir)
        trades[source] = df
        if not df.empty:
            print(f"    {len(df):,} trades, "
                  f"ts range: {pd.Timestamp(df['timestamp_us'].min(), unit='us', tz='UTC')} → "
                  f"{pd.Timestamp(df['timestamp_us'].max(), unit='us', tz='UTC')}")
        else:
            print(f"    (no data)")

    # Write trades parquet
    print("\n  Writing trades parquet...")
    for source, df in trades.items():
        write_parquet(df, out_dir / "trades" / f"{source}.parquet")

    # ------------------------------------------------------------------
    # Step 2: Build OHLCV bars from trades
    # ------------------------------------------------------------------
    print(f"\n[2/5] Building OHLCV bars...")

    for interval in intervals:
        interval_us = interval_to_us(interval)
        print(f"\n  Interval: {interval} ({interval_us:,} μs)")

        ohlcv_dict = {}
        for source, df in trades.items():
            print(f"    Aggregating {source}...")
            ohlcv = trades_to_ohlcv(df, interval_us)
            ohlcv_dict[source] = ohlcv
            write_parquet(ohlcv, out_dir / "ohlcv" / interval / f"{source}.parquet")

        validate_ohlcv_alignment(ohlcv_dict, interval)

    # ------------------------------------------------------------------
    # Step 3: Binance aggregate trades
    # ------------------------------------------------------------------
    print(f"\n[3/5] Parsing Binance aggregate trades...")

    for market in ["futures", "spot"]:
        print(f"  Parsing {market}...")
        df = parse_binance_agg_trades(data_dir, market)
        write_parquet(df, out_dir / "binance" / f"agg_trades_{market}.parquet")

    # ------------------------------------------------------------------
    # Step 4: Binance extras (metrics, book depth)
    # ------------------------------------------------------------------
    print(f"\n[4/5] Parsing Binance extras...")

    print("  Parsing metrics...")
    metrics = parse_binance_metrics(data_dir)
    write_parquet(metrics, out_dir / "binance" / "metrics.parquet")

    print("  Parsing book depth...")
    book_depth = parse_binance_book_depth(data_dir)
    write_parquet(book_depth, out_dir / "binance" / "book_depth.parquet")

    # ------------------------------------------------------------------
    # Step 5: Binance pre-computed klines (all types)
    # ------------------------------------------------------------------
    print(f"\n[5/5] Parsing Binance klines...")

    kline_types = {
        "futures": ["klines", "indexPriceKlines", "markPriceKlines", "premiumIndexKlines"],
        "spot": ["klines"],
    }

    # Map original dir names to clean parquet dir names
    dir_name_map = {
        "klines": "klines",
        "indexPriceKlines": "index_price_klines",
        "markPriceKlines": "mark_price_klines",
        "premiumIndexKlines": "premium_index_klines",
    }

    for market, dtypes in kline_types.items():
        for dtype in dtypes:
            clean_name = dir_name_map[dtype]
            print(f"  Parsing {market}/{dtype}...")
            klines_by_interval = parse_binance_klines(data_dir, market, dtype)
            for interval, df in klines_by_interval.items():
                write_parquet(
                    df,
                    out_dir / "binance" / f"{clean_name}_{market}" / f"{interval}.parquet",
                )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    total_files = len(list(out_dir.rglob("*.parquet")))
    total_size = sum(f.stat().st_size for f in out_dir.rglob("*.parquet"))
    print(f"Done. {total_files} parquet files, {total_size / (1024*1024):.1f} MB total")
    print(f"Output: {out_dir.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw market data to analysis-ready parquet files.",
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

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
