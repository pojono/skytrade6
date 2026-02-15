#!/usr/bin/env python3
"""
Download historical market data from Bybit, Binance, and OKX public endpoints.

Sources:
  - Bybit Futures:   https://public.bybit.com/trading/{SYMBOL}/
  - Bybit Spot:      https://public.bybit.com/spot/{SYMBOL}/
  - Binance Futures:  https://data.binance.vision/data/futures/um/daily/
  - Binance Spot:     https://data.binance.vision/data/spot/daily/
  - OKX Futures:      https://static.okx.com/cdn/okex/traderecords/trades/daily/
  - OKX Spot:         https://static.okx.com/cdn/okex/traderecords/trades/daily/

Usage:
  python download_market_data.py BTCUSDT 2026-01-01 2026-01-07
  python download_market_data.py ETHUSDT 2025-12-01 2025-12-31 --output ./data
  python download_market_data.py BTCUSDT 2026-01-01 2026-01-07 --sources bybit_futures binance_futures okx_futures
  python download_market_data.py BTCUSDT 2026-01-01 2026-01-07 --sources okx_futures okx_spot
  python download_market_data.py BTCUSDT 2026-01-01 2026-01-07 --binance-data-types trades aggTrades klines metrics
  python download_market_data.py BTCUSDT 2026-01-01 2026-01-07 --kline-intervals 1m 5m 1h

Output structure (archives kept compressed for direct ingestion via pandas/pyarrow):
  {output_dir}/
    {SYMBOL}/
      bybit/
        futures/          # perpetual trades (.csv.gz)
        spot/             # spot trades (.csv.gz)
      binance/
        futures/
          trades/                (.zip)
          aggTrades/             (.zip)
          bookDepth/             (.zip)
          bookTicker/            (.zip)
          metrics/               (.zip)
          klines/{interval}/     (.zip)
          indexPriceKlines/{interval}/
          markPriceKlines/{interval}/
          premiumIndexKlines/{interval}/
        spot/
          trades/         (.zip)
          aggTrades/      (.zip)
          klines/{interval}/
      okx/
        futures/          # perpetual swap trades (.zip)
        spot/             # spot trades (.zip)
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BYBIT_TRADING_URL = "https://public.bybit.com/trading/{symbol}/{symbol}{date}.csv.gz"
BYBIT_SPOT_URL = "https://public.bybit.com/spot/{symbol}/{symbol}_{date}.csv.gz"

BINANCE_FUTURES_URL = (
    "https://data.binance.vision/data/futures/um/daily/{dtype}/{symbol}/"
    "{symbol}-{dtype}-{date}.zip"
)
BINANCE_SPOT_URL = (
    "https://data.binance.vision/data/spot/daily/{dtype}/{symbol}/"
    "{symbol}-{dtype}-{date}.zip"
)

# OKX static download URLs
# SWAP (perpetual futures): instrument = e.g. BTC-USDT-SWAP
# SPOT: instrument = e.g. BTC-USDT
OKX_TRADES_URL = (
    "https://static.okx.com/cdn/okex/traderecords/trades/daily/"
    "{date_compact}/{instrument}-trades-{date}.zip"
)

ALL_SOURCES = [
    "bybit_futures",
    "bybit_spot",
    "binance_futures",
    "binance_spot",
    "okx_futures",
    "okx_spot",
]

# Binance data types that require an interval subpath
INTERVAL_DATA_TYPES = {"klines", "indexPriceKlines", "markPriceKlines", "premiumIndexKlines"}

# Binance data types with flat path (no interval)
FLAT_DATA_TYPES = {"trades", "aggTrades", "bookDepth", "bookTicker", "metrics"}

DEFAULT_BINANCE_FUTURES_DATA_TYPES = [
    "trades", "aggTrades", "bookDepth", "bookTicker", "metrics",
    "klines", "indexPriceKlines", "markPriceKlines", "premiumIndexKlines",
]

DEFAULT_BINANCE_SPOT_DATA_TYPES = ["trades", "aggTrades", "klines"]

DEFAULT_KLINE_INTERVALS = [
    "1m", "5m", "15m", "30m", "1h", "4h", "1d",
]

MAX_CONCURRENT = 5  # per-source concurrency
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2  # seconds

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def symbol_to_okx_pair(symbol: str) -> str:
    """Convert e.g. BTCUSDT -> BTC-USDT for OKX instrument naming."""
    for quote in ("USDT", "USDC", "USD", "BTC", "ETH"):
        if symbol.endswith(quote):
            base = symbol[: -len(quote)]
            return f"{base}-{quote}"
    # Fallback: assume last 4 chars are quote
    return f"{symbol[:-4]}-{symbol[-4:]}"


def date_range(start: str, end: str):
    """Yield date strings YYYY-MM-DD from start to end inclusive."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    while s <= e:
        yield s.strftime("%Y-%m-%d")
        s += timedelta(days=1)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


async def download_file(
    session: aiohttp.ClientSession,
    url: str,
    dest: Path,
    semaphore: asyncio.Semaphore,
):
    """Download a file with retries. Returns (url, success, message).

    Uses atomic write: data is saved to a .tmp file first, then renamed
    to the final path only after the full content is received and verified.
    If dest already exists (without a .tmp), it is known to be complete.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return (url, True, "exists")

    tmp = dest.with_suffix(dest.suffix + ".tmp")

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            async with semaphore:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                    if resp.status == 404:
                        return (url, False, "not found (404)")
                    if resp.status != 200:
                        raise aiohttp.ClientError(f"HTTP {resp.status}")
                    expected = resp.content_length
                    data = await resp.read()

            if expected is not None and len(data) != expected:
                raise aiohttp.ClientError(
                    f"size mismatch: got {len(data)}, expected {expected}")

            ensure_dir(dest.parent)
            tmp.write_bytes(data)
            tmp.rename(dest)

            return (url, True, "ok")

        except asyncio.TimeoutError:
            msg = f"timeout (attempt {attempt}/{RETRY_ATTEMPTS})"
        except (aiohttp.ClientError, OSError) as exc:
            msg = f"{exc} (attempt {attempt}/{RETRY_ATTEMPTS})"

        # Clean up partial temp file
        tmp.unlink(missing_ok=True)

        if attempt < RETRY_ATTEMPTS:
            await asyncio.sleep(RETRY_BACKOFF * attempt)

    return (url, False, msg)


# ---------------------------------------------------------------------------
# Source-specific task builders
# ---------------------------------------------------------------------------


def bybit_futures_tasks(symbol: str, dates, output_dir: Path):
    """Build (url, dest) pairs for Bybit futures."""
    base = output_dir / symbol / "bybit" / "futures"
    for d in dates:
        fname = f"{symbol}{d}.csv.gz"
        url = BYBIT_TRADING_URL.format(symbol=symbol, date=d)
        yield url, base / fname


def bybit_spot_tasks(symbol: str, dates, output_dir: Path):
    """Build (url, dest) pairs for Bybit spot."""
    base = output_dir / symbol / "bybit" / "spot"
    for d in dates:
        fname = f"{symbol}_{d}.csv.gz"
        url = BYBIT_SPOT_URL.format(symbol=symbol, date=d)
        yield url, base / fname


def binance_futures_tasks(symbol: str, dates, output_dir: Path, data_types, kline_intervals):
    """Build (url, dest) pairs for Binance UM futures."""
    for dtype in data_types:
        if dtype in INTERVAL_DATA_TYPES:
            for interval in kline_intervals:
                base = output_dir / symbol / "binance" / "futures" / dtype / interval
                for d in dates:
                    fname = f"{symbol}-{interval}-{d}.zip"
                    url = (
                        f"https://data.binance.vision/data/futures/um/daily/"
                        f"{dtype}/{symbol}/{interval}/{symbol}-{interval}-{d}.zip"
                    )
                    yield url, base / fname
        else:
            base = output_dir / symbol / "binance" / "futures" / dtype
            for d in dates:
                fname = f"{symbol}-{dtype}-{d}.zip"
                url = BINANCE_FUTURES_URL.format(symbol=symbol, dtype=dtype, date=d)
                yield url, base / fname


def binance_spot_tasks(symbol: str, dates, output_dir: Path, data_types, kline_intervals):
    """Build (url, dest) pairs for Binance spot."""
    for dtype in data_types:
        if dtype in INTERVAL_DATA_TYPES:
            for interval in kline_intervals:
                base = output_dir / symbol / "binance" / "spot" / dtype / interval
                for d in dates:
                    fname = f"{symbol}-{interval}-{d}.zip"
                    url = (
                        f"https://data.binance.vision/data/spot/daily/"
                        f"{dtype}/{symbol}/{interval}/{symbol}-{interval}-{d}.zip"
                    )
                    yield url, base / fname
        else:
            base = output_dir / symbol / "binance" / "spot" / dtype
            for d in dates:
                fname = f"{symbol}-{dtype}-{d}.zip"
                url = BINANCE_SPOT_URL.format(symbol=symbol, dtype=dtype, date=d)
                yield url, base / fname


def okx_futures_tasks(symbol: str, dates, output_dir: Path):
    """Build (url, dest) pairs for OKX perpetual swap trades."""
    pair = symbol_to_okx_pair(symbol)
    instrument = f"{pair}-SWAP"
    base = output_dir / symbol / "okx" / "futures"
    for d in dates:
        date_compact = d.replace("-", "")
        fname = f"{instrument}-trades-{d}.zip"
        url = OKX_TRADES_URL.format(instrument=instrument, date=d, date_compact=date_compact)
        yield url, base / fname


def okx_spot_tasks(symbol: str, dates, output_dir: Path):
    """Build (url, dest) pairs for OKX spot trades."""
    instrument = symbol_to_okx_pair(symbol)
    base = output_dir / symbol / "okx" / "spot"
    for d in dates:
        date_compact = d.replace("-", "")
        fname = f"{instrument}-trades-{d}.zip"
        url = OKX_TRADES_URL.format(instrument=instrument, date=d, date_compact=date_compact)
        yield url, base / fname


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


async def run(args):
    symbol = args.symbol.upper()
    dates = list(date_range(args.start_date, args.end_date))
    output_dir = Path(args.output)
    sources = [s.lower() for s in args.sources]
    kline_intervals = args.kline_intervals

    print(f"Symbol:           {symbol}")
    print(f"Date range:       {args.start_date} -> {args.end_date} ({len(dates)} days)")
    print(f"Sources:          {', '.join(sources)}")
    print(f"Binance futures:  {', '.join(args.binance_futures_data_types)}")
    print(f"Binance spot:     {', '.join(args.binance_spot_data_types)}")
    print(f"Kline intervals:  {', '.join(kline_intervals)}")
    print(f"Output directory: {output_dir.resolve()}")
    print()

    # Collect all tasks
    all_tasks = []

    if "bybit_futures" in sources:
        all_tasks.extend(bybit_futures_tasks(symbol, dates, output_dir))
    if "bybit_spot" in sources:
        all_tasks.extend(bybit_spot_tasks(symbol, dates, output_dir))
    if "binance_futures" in sources:
        all_tasks.extend(binance_futures_tasks(
            symbol, dates, output_dir, args.binance_futures_data_types, kline_intervals))
    if "binance_spot" in sources:
        all_tasks.extend(binance_spot_tasks(
            symbol, dates, output_dir, args.binance_spot_data_types, kline_intervals))
    if "okx_futures" in sources:
        all_tasks.extend(okx_futures_tasks(symbol, dates, output_dir))
    if "okx_spot" in sources:
        all_tasks.extend(okx_spot_tasks(symbol, dates, output_dir))

    total = len(all_tasks)
    print(f"Total files to download: {total}")
    print("-" * 60)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    success_count = 0
    skip_count = 0
    fail_count = 0
    not_found_count = 0

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT * 2, force_close=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        coros = [
            download_file(session, url, dest, semaphore)
            for url, dest in all_tasks
        ]

        for i, coro in enumerate(asyncio.as_completed(coros), 1):
            url, ok, msg = await coro
            short = url.split("/")[-1]
            if ok and msg == "exists":
                skip_count += 1
            elif ok:
                success_count += 1
                print(f"  [{i}/{total}] ✓ {short}")
            elif "404" in msg:
                not_found_count += 1
                print(f"  [{i}/{total}] - {short}  (not available)")
            else:
                fail_count += 1
                print(f"  [{i}/{total}] ✗ {short}  ({msg})")

    print()
    print("=" * 60)
    print(f"Done.  downloaded={success_count}  skipped={skip_count}  not_found={not_found_count}  failed={fail_count}")
    print(f"Data saved to: {output_dir.resolve() / symbol}")

    if fail_count > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download historical market data from Bybit, Binance, and OKX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("symbol", help="Trading pair symbol, e.g. BTCUSDT")
    parser.add_argument("start_date", help="Start date inclusive (YYYY-MM-DD)")
    parser.add_argument("end_date", help="End date inclusive (YYYY-MM-DD)")
    parser.add_argument(
        "--output", "-o",
        default="./data",
        help="Output root directory (default: ./data)",
    )
    parser.add_argument(
        "--sources", "-s",
        nargs="+",
        default=ALL_SOURCES,
        choices=ALL_SOURCES,
        help="Which sources to download from (default: all)",
    )
    parser.add_argument(
        "--binance-futures-data-types",
        nargs="+",
        default=DEFAULT_BINANCE_FUTURES_DATA_TYPES,
        help=f"Binance futures data types (default: {DEFAULT_BINANCE_FUTURES_DATA_TYPES})",
    )
    parser.add_argument(
        "--binance-spot-data-types",
        nargs="+",
        default=DEFAULT_BINANCE_SPOT_DATA_TYPES,
        help=f"Binance spot data types (default: {DEFAULT_BINANCE_SPOT_DATA_TYPES})",
    )
    parser.add_argument(
        "--kline-intervals",
        nargs="+",
        default=DEFAULT_KLINE_INTERVALS,
        help=f"Kline intervals when 'klines' is in data types (default: {DEFAULT_KLINE_INTERVALS})",
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=MAX_CONCURRENT,
        help=f"Max concurrent downloads (default: {MAX_CONCURRENT})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate dates
    try:
        s = datetime.strptime(args.start_date, "%Y-%m-%d")
        e = datetime.strptime(args.end_date, "%Y-%m-%d")
        if s > e:
            print("Error: start_date must be <= end_date", file=sys.stderr)
            sys.exit(1)
    except ValueError as exc:
        print(f"Error: invalid date format: {exc}", file=sys.stderr)
        sys.exit(1)

    global MAX_CONCURRENT
    MAX_CONCURRENT = args.concurrency

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
