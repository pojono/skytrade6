#!/usr/bin/env python3
"""
Download Binance market data from data.binance.vision bulk archives (futures or spot).

Sources:
  Futures: https://data.binance.vision/?prefix=data/futures/um/daily/
  Spot:    https://data.binance.vision/?prefix=data/spot/daily/

Available data types — futures (all daily .zip archives):
  - trades:              Individual trades
  - aggTrades:           Aggregated trades
  - klines/1m:           OHLCV 1-minute candles
  - markPriceKlines/1m:  Mark price 1-minute candles
  - premiumIndexKlines/1m: Premium index 1-minute candles
  - indexPriceKlines/1m: Index price 1-minute candles
  - bookDepth:           Order book depth snapshots
  - bookTicker:          Best bid/ask ticker snapshots
  - metrics:             Composite metrics (OI, funding rate, LS ratio, etc.)

Available data types — spot:
  - spotKlines:          OHLCV 1-minute candles
  - spotTrades:          Individual trades
  - spotAggTrades:       Aggregated trades

Usage:
  # Futures (default):
  python download_binance_data.py BTCUSDT 2026-02-01 2026-02-28
  python download_binance_data.py BTCUSDT 2026-02-01 2026-02-07 --types klines,metrics

  # Spot:
  python download_binance_data.py BTCUSDT 2026-02-01 2026-02-28 --market spot
  python download_binance_data.py BTCUSDT 2026-02-01 2026-02-07 --market spot --types all

Output structure (same folder, spot files have _spot postfix):
  binance/{SYMBOL}/
    {YYYY-MM-DD}_kline_1m.csv              (futures)
    {YYYY-MM-DD}_metrics.csv               (futures)
    ...                                    (other futures types)
    {YYYY-MM-DD}_kline_1m_spot.csv         (spot)
    {YYYY-MM-DD}_trades_spot.csv           (spot)
    {YYYY-MM-DD}_aggTrades_spot.csv        (spot)
"""

import argparse
import asyncio
import atexit
import hashlib
import io
import signal
import sys
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BINANCE_FUTURES_BASE = "https://data.binance.vision/data/futures/um/daily"
BINANCE_SPOT_BASE = "https://data.binance.vision/data/spot/daily"

DEFAULT_CONCURRENT = 5
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2  # seconds

# Output directory relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "binance"

# Track .tmp files for cleanup on interrupt
_active_tmp_files: set[Path] = set()


def _cleanup_tmp_files():
    for tmp in list(_active_tmp_files):
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


atexit.register(_cleanup_tmp_files)
for _sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_sig, lambda s, f: sys.exit(1))

# ---------------------------------------------------------------------------
# Data type definitions
# ---------------------------------------------------------------------------

# Each data type maps to:
#   url_template: path under BINANCE_DATA_BASE, with {symbol} and {date} placeholders
#   output_suffix: suffix for the output filename
#   label: human-readable label for progress output
# --- Futures data types ---
DATA_TYPES = {
    "trades": {
        "url_template": "trades/{symbol}/{symbol}-trades-{date}.zip",
        "output_suffix": "trades",
        "label": "trades",
    },
    "aggTrades": {
        "url_template": "aggTrades/{symbol}/{symbol}-aggTrades-{date}.zip",
        "output_suffix": "aggTrades",
        "label": "aggTrades",
    },
    "klines": {
        "url_template": "klines/{symbol}/1m/{symbol}-1m-{date}.zip",
        "output_suffix": "kline_1m",
        "label": "klines (1m)",
    },
    "markPriceKlines": {
        "url_template": "markPriceKlines/{symbol}/1m/{symbol}-1m-{date}.zip",
        "output_suffix": "mark_price_kline_1m",
        "label": "mark price klines (1m)",
    },
    "premiumIndexKlines": {
        "url_template": "premiumIndexKlines/{symbol}/1m/{symbol}-1m-{date}.zip",
        "output_suffix": "premium_index_kline_1m",
        "label": "premium index klines (1m)",
    },
    "indexPriceKlines": {
        "url_template": "indexPriceKlines/{symbol}/1m/{symbol}-1m-{date}.zip",
        "output_suffix": "index_price_kline_1m",
        "label": "index price klines (1m)",
    },
    "bookDepth": {
        "url_template": "bookDepth/{symbol}/{symbol}-bookDepth-{date}.zip",
        "output_suffix": "bookDepth",
        "label": "book depth",
    },
    "bookTicker": {
        "url_template": "bookTicker/{symbol}/{symbol}-bookTicker-{date}.zip",
        "output_suffix": "bookTicker",
        "label": "book ticker",
    },
    "metrics": {
        "url_template": "metrics/{symbol}/{symbol}-metrics-{date}.zip",
        "output_suffix": "metrics",
        "label": "metrics (OI, FR, LS ratio)",
    },
}

DEFAULT_TYPES = [
    "klines", "markPriceKlines", "premiumIndexKlines",
    "indexPriceKlines", "metrics",
]

# --- Spot data types ---
SPOT_DATA_TYPES = {
    "spotKlines": {
        "url_template": "klines/{symbol}/1m/{symbol}-1m-{date}.zip",
        "output_suffix": "kline_1m_spot",
        "label": "spot klines (1m)",
    },
    "spotTrades": {
        "url_template": "trades/{symbol}/{symbol}-trades-{date}.zip",
        "output_suffix": "trades_spot",
        "label": "spot trades",
    },
    "spotAggTrades": {
        "url_template": "aggTrades/{symbol}/{symbol}-aggTrades-{date}.zip",
        "output_suffix": "aggTrades_spot",
        "label": "spot aggTrades",
    },
}

SPOT_DEFAULT_TYPES = ["spotKlines"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def date_range(start: str, end: str):
    """Yield date strings YYYY-MM-DD from start to end inclusive."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    while s <= e:
        yield s.strftime("%Y-%m-%d")
        s += timedelta(days=1)


def _extract_zip_csv(raw: bytes) -> bytes:
    """Extract the first .csv file from a zip archive. Returns raw CSV bytes."""
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        # Find the first CSV file (skip CHECKSUM or other files)
        csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
        if csv_files:
            return zf.read(csv_files[0])
        # Fallback: extract first file
        return zf.read(zf.namelist()[0])


# ---------------------------------------------------------------------------
# First-available-date detection (binary search via HEAD requests)
# ---------------------------------------------------------------------------


async def _head_exists(session: aiohttp.ClientSession, url: str) -> bool:
    """Return True if the URL exists (HTTP 200 on HEAD)."""
    try:
        async with session.head(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            return resp.status == 200
    except Exception:
        return False


async def find_first_available_date(
    session: aiohttp.ClientSession,
    symbol: str,
    start_date: str,
    end_date: str,
    base_url: str = None,
) -> str | None:
    """Binary search for the first date that has data on Binance.

    Probes klines/1m zip files via HEAD requests. Returns the first available
    date string (YYYY-MM-DD), or None if no data exists in the range.
    ~10 requests for a 256-day range.
    """
    if base_url is None:
        base_url = BINANCE_FUTURES_BASE
    fmt = "%Y-%m-%d"
    lo = datetime.strptime(start_date, fmt)
    hi = datetime.strptime(end_date, fmt)

    def _url(d: str) -> str:
        return f"{base_url}/klines/{symbol}/1m/{symbol}-1m-{d}.zip"

    # Quick check: if start_date exists, no need to search
    if await _head_exists(session, _url(start_date)):
        return start_date

    # Find a known-good upper bound (today's zip may not be published yet).
    # Walk backward up to 3 days from end_date.
    upper = None
    d = hi
    for _ in range(4):
        if await _head_exists(session, _url(d.strftime(fmt))):
            upper = d
            break
        d -= timedelta(days=1)
        if d < lo:
            break

    if upper is None:
        return None

    # Binary search: find first date where data exists
    while (upper - lo).days > 1:
        mid = lo + (upper - lo) / 2
        mid_str = mid.strftime(fmt)
        if await _head_exists(session, _url(mid_str)):
            upper = mid
        else:
            lo = mid

    return upper.strftime(fmt)


# ---------------------------------------------------------------------------
# Download engine
# ---------------------------------------------------------------------------


async def download_file(
    session: aiohttp.ClientSession,
    url: str,
    checksum_url: str,
    dest: Path,
    semaphore: asyncio.Semaphore,
):
    """Download a .zip file, extract the CSV, write atomically. Returns (url, success, message).

    Also downloads .CHECKSUM file and verifies SHA256 if available.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return (url, True, "exists")

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    _active_tmp_files.add(tmp)

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            async with semaphore:
                # Download the zip
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=600)
                ) as resp:
                    if resp.status == 404:
                        _active_tmp_files.discard(tmp)
                        return (url, False, "not found (404)")
                    if resp.status == 403:
                        _active_tmp_files.discard(tmp)
                        return (url, False, "forbidden (403)")
                    if resp.status != 200:
                        raise aiohttp.ClientError(f"HTTP {resp.status}")
                    expected = resp.content_length
                    data = await resp.read()

                # Optionally download checksum
                checksum_expected = None
                try:
                    async with session.get(
                        checksum_url,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as cs_resp:
                        if cs_resp.status == 200:
                            cs_text = (await cs_resp.read()).decode().strip()
                            # Format: "<sha256_hex>  <filename>"
                            checksum_expected = cs_text.split()[0] if cs_text else None
                except Exception:
                    pass  # checksum is optional

            if expected is not None and len(data) != expected:
                raise aiohttp.ClientError(
                    f"size mismatch: got {len(data)}, expected {expected}"
                )

            # Verify checksum
            if checksum_expected:
                actual_hash = hashlib.sha256(data).hexdigest()
                if actual_hash != checksum_expected:
                    raise aiohttp.ClientError(
                        f"checksum mismatch: {actual_hash[:16]}... vs {checksum_expected[:16]}..."
                    )

            # Extract CSV from zip
            csv_data = _extract_zip_csv(data)

            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(csv_data)
            tmp.rename(dest)
            _active_tmp_files.discard(tmp)

            size_mb = len(data) / (1024 * 1024)
            return (url, True, f"ok ({size_mb:.1f} MB zip)")

        except asyncio.TimeoutError:
            msg = f"timeout (attempt {attempt}/{RETRY_ATTEMPTS})"
        except (aiohttp.ClientError, OSError, zipfile.BadZipFile) as exc:
            msg = f"{exc} (attempt {attempt}/{RETRY_ATTEMPTS})"

        tmp.unlink(missing_ok=True)
        if attempt < RETRY_ATTEMPTS:
            await asyncio.sleep(RETRY_BACKOFF * attempt)

    _active_tmp_files.discard(tmp)
    return (url, False, msg)


# ---------------------------------------------------------------------------
# Task builder
# ---------------------------------------------------------------------------


def build_tasks(symbol: str, dates, output_dir: Path, data_types: list[str],
                base_url: str = None, type_registry: dict = None):
    """Build (url, checksum_url, dest, label) tuples for all requested data."""
    if base_url is None:
        base_url = BINANCE_FUTURES_BASE
    if type_registry is None:
        type_registry = DATA_TYPES
    base = output_dir / symbol
    for dtype in data_types:
        cfg = type_registry[dtype]
        for d in dates:
            url_path = cfg["url_template"].format(symbol=symbol, date=d)
            url = f"{base_url}/{url_path}"
            checksum_url = url + ".CHECKSUM"
            fname = f"{d}_{cfg['output_suffix']}.csv"
            yield url, checksum_url, base / fname, cfg["label"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_one_symbol(
    symbol: str,
    start_date: str,
    end_date: str,
    concurrency: int,
    data_types: list[str],
    market: str = "futures",
):
    """Download data for a single symbol."""
    output_dir = DATA_DIR
    is_spot = market == "spot"
    base_url = BINANCE_SPOT_BASE if is_spot else BINANCE_FUTURES_BASE
    type_registry = SPOT_DATA_TYPES if is_spot else DATA_TYPES

    # --- Detect first available date (binary search via HEAD) ---
    connector = aiohttp.TCPConnector(limit=5, force_close=False)
    async with aiohttp.ClientSession(connector=connector) as probe_session:
        first_date = await find_first_available_date(
            probe_session, symbol, start_date, end_date, base_url=base_url,
        )

    if first_date is None:
        print(f"\n{symbol}: no data found in {start_date} -> {end_date}, skipping.")
        return 0

    effective_start = first_date if first_date > start_date else start_date
    dates = list(date_range(effective_start, end_date))

    type_labels = [type_registry[t]["label"] for t in data_types]
    print(f"\nSymbol:           {symbol}")
    print(f"Market:           {market}")
    if effective_start != start_date:
        print(f"First available:  {effective_start}  (requested {start_date})")
    print(f"Date range:       {effective_start} -> {end_date} ({len(dates)} days)")
    print(f"Data types:       {', '.join(type_labels)}")
    print(f"Output directory: {output_dir / symbol}")

    all_tasks = list(build_tasks(symbol, dates, output_dir, data_types,
                                 base_url=base_url, type_registry=type_registry))
    total = len(all_tasks)
    print(f"Total files:      {total}")
    print("-" * 60)

    t0 = time.monotonic()
    semaphore = asyncio.Semaphore(concurrency)

    success_count = 0
    skip_count = 0
    fail_count = 0
    not_found_count = 0

    connector = aiohttp.TCPConnector(limit=concurrency * 2, force_close=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        coros = [
            download_file(session, url, cs_url, dest, semaphore)
            for url, cs_url, dest, label in all_tasks
        ]

        for i, coro in enumerate(asyncio.as_completed(coros), 1):
            url, ok, msg = await coro
            short = url.split("/")[-1]
            elapsed = time.monotonic() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            ts = f"[{elapsed:.0f}s elapsed, ETA {eta:.0f}s]"

            if ok and msg == "exists":
                skip_count += 1
                if skip_count <= 5 or skip_count % 20 == 0:
                    print(f"  [{i}/{total}] ~ {short}  (already exists)")
            elif ok:
                success_count += 1
                print(f"  [{i}/{total}] ✓ {short}  {ts}  {msg}")
            elif "404" in msg or "403" in msg:
                not_found_count += 1
                if not_found_count <= 10 or not_found_count % 50 == 0:
                    print(f"  [{i}/{total}] - {short}  ({msg})")
            else:
                fail_count += 1
                print(f"  [{i}/{total}] ✗ {short}  ({msg})")

    total_elapsed = time.monotonic() - t0
    print(
        f"\n{symbol} done in {total_elapsed:.1f}s.  "
        f"downloaded={success_count}  skipped={skip_count}  "
        f"not_found={not_found_count}  failed={fail_count}"
    )
    print(f"Data: {output_dir / symbol}")
    return fail_count


async def run(
    symbols: list[str],
    start_date: str,
    end_date: str,
    concurrency: int,
    data_types: list[str],
    market: str = "futures",
):
    print(f"Symbols:          {', '.join(symbols)}")
    print(f"Market:           {market}")
    print(f"Date range:       {start_date} -> {end_date}")
    print(f"Data types:       {', '.join(data_types)}")
    print(f"Concurrency:      {concurrency}")
    print("=" * 60)

    total_fail = 0
    for i, symbol in enumerate(symbols, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(symbols)}] {symbol}")
        print(f"{'='*60}")
        fails = await run_one_symbol(symbol, start_date, end_date, concurrency, data_types, market)
        total_fail += fails

    print("\n" + "=" * 60)
    print(f"All {len(symbols)} symbols done.")
    if total_fail > 0:
        print(f"Total failures: {total_fail}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download Binance market data from data.binance.vision bulk archives (futures or spot).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Futures data types: {', '.join(DATA_TYPES.keys())}\n"
               f"Spot data types:   {', '.join(SPOT_DATA_TYPES.keys())}\n"
               f"Futures defaults:  {', '.join(DEFAULT_TYPES)}\n"
               f"Spot defaults:     {', '.join(SPOT_DEFAULT_TYPES)}",
    )
    parser.add_argument(
        "symbol",
        help="Trading pair symbol(s), comma-separated. e.g. BTCUSDT or SOLUSDT,DOGEUSDT",
    )
    parser.add_argument("start_date", help="Start date inclusive (YYYY-MM-DD)")
    parser.add_argument("end_date", help="End date inclusive (YYYY-MM-DD)")
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=DEFAULT_CONCURRENT,
        help=f"Max concurrent downloads (default: {DEFAULT_CONCURRENT})",
    )
    parser.add_argument(
        "--market", "-m",
        type=str,
        default="futures",
        choices=["futures", "spot"],
        help="Market type: 'futures' (USDT-M perpetual, default) or 'spot'.",
    )
    parser.add_argument(
        "--types", "-t",
        type=str,
        default=None,
        help="Comma-separated data types to download. "
             "Use 'all' for everything. Defaults depend on --market.",
    )
    parser.add_argument(
        "--no-checksum",
        action="store_true",
        help="Skip checksum verification (faster, less safe)",
    )
    args = parser.parse_args()

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

    # Select type registry based on market
    is_spot = args.market == "spot"
    type_registry = SPOT_DATA_TYPES if is_spot else DATA_TYPES
    default_types = SPOT_DEFAULT_TYPES if is_spot else DEFAULT_TYPES

    # Parse data types
    if args.types is None:
        data_types = default_types
    elif args.types.strip().lower() == "all":
        data_types = list(type_registry.keys())
    else:
        data_types = [t.strip() for t in args.types.split(",") if t.strip()]
        for t in data_types:
            if t not in type_registry:
                print(
                    f"Error: unknown data type '{t}' for market '{args.market}'. "
                    f"Available: {', '.join(type_registry.keys())}",
                    file=sys.stderr,
                )
                sys.exit(1)

    symbols = [sym.strip().upper() for sym in args.symbol.split(",") if sym.strip()]
    asyncio.run(run(symbols, args.start_date, args.end_date, args.concurrency, data_types, args.market))


if __name__ == "__main__":
    main()
