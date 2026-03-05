#!/usr/bin/env python3
"""
Download Binance market data from data.binance.vision bulk archives.

Six canonical data types (same names as Bybit downloader):

  TradesLinear      futures/um/daily/trades        -> {date}_trades.csv.gz
  TradesSpot        spot/daily/trades              -> {date}_trades_spot.csv.gz
  OrderbookLinear   futures/um/daily/bookDepth     -> {date}_bookDepth.csv.gz
  OrderbookSpot     (not available on Binance)
  MetricsLinear     futures klines + markPrice + indexPrice + premiumIndex + metrics
                      -> {date}_kline_1m.csv, {date}_mark_price_kline_1m.csv,
                         {date}_index_price_kline_1m.csv, {date}_premium_index_kline_1m.csv,
                         {date}_metrics.csv
  MetricsSpot       spot/daily/klines/1m           -> {date}_kline_1m_spot.csv

Trades & orderbooks: zip → extract → gzip level 1 (.csv.gz)
Metrics/klines:      zip → extract → save (.csv)

Usage:
  python download_binance_data.py BTCUSDT 2026-02-01 2026-03-01
  python download_binance_data.py BTCUSDT,ETHUSDT 2026-02-01 2026-03-01 -t all
  python download_binance_data.py BTCUSDT 2026-02-01 2026-03-01 -t TradesLinear,MetricsLinear

Output: binance/{SYMBOL}/{date}_{suffix}.csv[.gz]
"""

import argparse
import asyncio
import atexit
import concurrent.futures
import gzip
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

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "binance"

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
# 6 canonical types — sub-download definitions
# ---------------------------------------------------------------------------

# Each sub-download: (url_template, output_suffix, compress_to_gz)
# url_template uses {symbol} and {date} placeholders, relative to base_url.

ALL_TYPES = [
    "TradesLinear", "TradesSpot",
    "OrderbookLinear", "OrderbookSpot",
    "MetricsLinear", "MetricsSpot",
]

DEFAULT_TYPES = ["MetricsLinear", "MetricsSpot"]

# TradesLinear: 1 sub-download per day (large, compress to .csv.gz)
TRADES_LINEAR = [
    ("trades/{symbol}/{symbol}-trades-{date}.zip", "trades", True),
]

# TradesSpot: 1 sub-download per day (large, compress to .csv.gz)
TRADES_SPOT = [
    ("trades/{symbol}/{symbol}-trades-{date}.zip", "trades_spot", True),
]

# OrderbookLinear: 1 sub-download per day (compress to .csv.gz)
ORDERBOOK_LINEAR = [
    ("bookDepth/{symbol}/{symbol}-bookDepth-{date}.zip", "bookDepth", True),
]

# OrderbookSpot: not available on Binance
ORDERBOOK_SPOT = []

# MetricsLinear: 5 sub-downloads per day (small, extract to .csv)
METRICS_LINEAR = [
    ("klines/{symbol}/1m/{symbol}-1m-{date}.zip", "kline_1m", False),
    ("markPriceKlines/{symbol}/1m/{symbol}-1m-{date}.zip", "mark_price_kline_1m", False),
    ("indexPriceKlines/{symbol}/1m/{symbol}-1m-{date}.zip", "index_price_kline_1m", False),
    ("premiumIndexKlines/{symbol}/1m/{symbol}-1m-{date}.zip", "premium_index_kline_1m", False),
    ("metrics/{symbol}/{symbol}-metrics-{date}.zip", "metrics", False),
]

# MetricsSpot: 1 sub-download per day (small, extract to .csv)
METRICS_SPOT = [
    ("klines/{symbol}/1m/{symbol}-1m-{date}.zip", "kline_1m_spot", False),
]

TYPE_SUBS = {
    "TradesLinear":    (BINANCE_FUTURES_BASE, TRADES_LINEAR),
    "TradesSpot":      (BINANCE_SPOT_BASE,    TRADES_SPOT),
    "OrderbookLinear": (BINANCE_FUTURES_BASE, ORDERBOOK_LINEAR),
    "OrderbookSpot":   (BINANCE_SPOT_BASE,    ORDERBOOK_SPOT),
    "MetricsLinear":   (BINANCE_FUTURES_BASE, METRICS_LINEAR),
    "MetricsSpot":     (BINANCE_SPOT_BASE,    METRICS_SPOT),
}

# Probe URLs for first-date detection per type
PROBE_URLS = {
    "TradesLinear":    (BINANCE_FUTURES_BASE, "trades/{symbol}/{symbol}-trades-{date}.zip"),
    "TradesSpot":      (BINANCE_SPOT_BASE,    "trades/{symbol}/{symbol}-trades-{date}.zip"),
    "OrderbookLinear": (BINANCE_FUTURES_BASE, "bookDepth/{symbol}/{symbol}-bookDepth-{date}.zip"),
    "OrderbookSpot":   (BINANCE_SPOT_BASE,    "bookDepth/{symbol}/{symbol}-bookDepth-{date}.zip"),
    "MetricsLinear":   (BINANCE_FUTURES_BASE, "klines/{symbol}/1m/{symbol}-1m-{date}.zip"),
    "MetricsSpot":     (BINANCE_SPOT_BASE,    "klines/{symbol}/1m/{symbol}-1m-{date}.zip"),
}


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
    """Extract the first .csv file from a zip archive."""
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
        if csv_files:
            return zf.read(csv_files[0])
        return zf.read(zf.namelist()[0])


def _process_and_write(data: bytes, compress: bool, dest: Path) -> int:
    """Extract CSV from zip, optionally gzip-compress, write atomically. For ProcessPoolExecutor."""
    csv_data = _extract_zip_csv(data)
    if compress:
        csv_data = gzip.compress(csv_data, compresslevel=1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(csv_data)
    tmp.rename(dest)
    return len(csv_data)


def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024**2:
        return f"{n/1024:.1f}KB"
    if n < 1024**3:
        return f"{n/1024**2:.1f}MB"
    return f"{n/1024**3:.2f}GB"


def _progress(done: int, total: int, t0: float, dl_bytes: int):
    """Print in-place progress bar."""
    if total == 0:
        return
    pct = done / total
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    elapsed = time.monotonic() - t0
    eta = (elapsed / done * (total - done)) if done > 0 else 0
    speed = dl_bytes / elapsed if elapsed > 0 else 0
    size_str = _fmt_size(dl_bytes)
    speed_str = f"{_fmt_size(int(speed))}/s"
    print(
        f"\r  {bar} {pct:4.0%} ({done}/{total})  "
        f"{elapsed:.0f}s ETA {eta:.0f}s  {size_str} @ {speed_str}    ",
        end="", flush=True,
    )


# ---------------------------------------------------------------------------
# First-available-date detection (binary search via HEAD)
# ---------------------------------------------------------------------------


async def _head_exists(session: aiohttp.ClientSession, url: str) -> bool:
    try:
        async with session.head(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            return resp.status == 200
    except Exception:
        return False


async def _binary_search_first_date(
    session: aiohttp.ClientSession,
    symbol: str,
    start_date: str,
    end_date: str,
    base_url: str,
    url_template: str,
) -> str | None:
    """Binary search for the first date that has data."""
    fmt = "%Y-%m-%d"
    lo = datetime.strptime(start_date, fmt)
    hi = datetime.strptime(end_date, fmt)

    def _url(d: str) -> str:
        path = url_template.format(symbol=symbol, date=d)
        return f"{base_url}/{path}"

    if await _head_exists(session, _url(start_date)):
        return start_date

    # Walk backward from end to find a known-good upper bound
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

    while (upper - lo).days > 1:
        mid = lo + (upper - lo) / 2
        mid_str = mid.strftime(fmt)
        if await _head_exists(session, _url(mid_str)):
            upper = mid
        else:
            lo = mid

    return upper.strftime(fmt)


async def find_first_dates(
    session: aiohttp.ClientSession,
    symbol: str,
    start_date: str,
    end_date: str,
    data_types: list[str],
) -> dict[str, str | None]:
    """Find first available date for each type concurrently."""
    results: dict[str, str | None] = {}

    async def _probe(dtype):
        base_url, url_tmpl = PROBE_URLS[dtype]
        first = await _binary_search_first_date(
            session, symbol, start_date, end_date, base_url, url_tmpl,
        )
        return dtype, first

    for coro in asyncio.as_completed([_probe(dt) for dt in data_types]):
        dtype, first = await coro
        results[dtype] = first

    return results


# ---------------------------------------------------------------------------
# Download engine
# ---------------------------------------------------------------------------


async def download_file(
    session: aiohttp.ClientSession,
    url: str,
    dest: Path,
    semaphore: asyncio.Semaphore,
):
    """Download a .zip, return (url, dest, compress_flag, raw_bytes | None, message)."""
    if dest.exists() and dest.stat().st_size > 0:
        return (url, True, "exists", 0)

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    _active_tmp_files.add(tmp)

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            async with semaphore:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=600)
                ) as resp:
                    if resp.status == 404:
                        _active_tmp_files.discard(tmp)
                        return (url, False, "not found (404)", 0)
                    if resp.status == 403:
                        _active_tmp_files.discard(tmp)
                        return (url, False, "forbidden (403)", 0)
                    if resp.status != 200:
                        raise aiohttp.ClientError(f"HTTP {resp.status}")
                    expected = resp.content_length
                    data = await resp.read()

            if expected is not None and len(data) != expected:
                raise aiohttp.ClientError(
                    f"size mismatch: got {len(data)}, expected {expected}"
                )

            _active_tmp_files.discard(tmp)
            return (url, True, "ok", data)

        except asyncio.TimeoutError:
            msg = f"timeout (attempt {attempt}/{RETRY_ATTEMPTS})"
        except (aiohttp.ClientError, OSError) as exc:
            msg = f"{exc} (attempt {attempt}/{RETRY_ATTEMPTS})"

        tmp.unlink(missing_ok=True)
        if attempt < RETRY_ATTEMPTS:
            await asyncio.sleep(RETRY_BACKOFF * attempt)

    _active_tmp_files.discard(tmp)
    return (url, False, msg, 0)


# ---------------------------------------------------------------------------
# Task builders
# ---------------------------------------------------------------------------


def build_tasks(symbol: str, dates: list[str], output_dir: Path, dtype: str):
    """Yield (url, dest, compress) tuples for one data type across all dates."""
    base_url, subs = TYPE_SUBS[dtype]
    base = output_dir / symbol
    for url_tmpl, suffix, compress in subs:
        for d in dates:
            url = f"{base_url}/{url_tmpl.format(symbol=symbol, date=d)}"
            if compress:
                dest = base / f"{d}_{suffix}.csv.gz"
                yield url, dest, True
            else:
                dest = base / f"{d}_{suffix}.csv"
                yield url, dest, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_one_symbol(
    symbol: str,
    start_date: str,
    end_date: str,
    concurrency: int,
    data_types: list[str],
):
    """Download data for a single symbol."""
    output_dir = DATA_DIR

    # --- Detect first available date per type ---
    connector = aiohttp.TCPConnector(limit=10, force_close=False)
    async with aiohttp.ClientSession(connector=connector) as probe_session:
        first_dates = await find_first_dates(
            probe_session, symbol, start_date, end_date, data_types,
        )

    active_types = {dt: fd for dt, fd in first_dates.items() if fd is not None}
    if not active_types:
        print(f"\n{symbol}: no data found in {start_date} -> {end_date}, skipping.")
        return 0

    type_dates: dict[str, list[str]] = {}
    for dt, fd in active_types.items():
        eff = fd if fd > start_date else start_date
        type_dates[dt] = list(date_range(eff, end_date))

    print(f"\nSymbol:           {symbol}")
    for dt in data_types:
        fd = first_dates.get(dt)
        if fd is None:
            print(f"  {dt:20s}  no data")
        elif fd > start_date:
            print(f"  {dt:20s}  {fd} -> {end_date} ({len(type_dates[dt])} days)  (requested {start_date})")
        else:
            print(f"  {dt:20s}  {start_date} -> {end_date} ({len(type_dates[dt])} days)")
    print(f"Output directory: {output_dir / symbol}")

    # Build all tasks, split by compress (CPU-heavy) vs raw (just extract+save)
    all_tasks = []  # (url, dest, compress)
    for dt in active_types:
        all_tasks.extend(build_tasks(symbol, type_dates[dt], output_dir, dt))

    grand_total = len(all_tasks)
    print(f"Tasks:            {grand_total}")
    print("-" * 60)

    t0 = time.monotonic()

    # Shared progress counters
    _done = 0
    _dl_bytes = 0
    _success = 0
    _skip = 0
    _fail = 0
    _not_found = 0

    # Separate: raw-save (compress=False, small) vs CPU-heavy (compress=True, large)
    raw_tasks = []      # extract CSV only
    cpu_tasks = []      # extract CSV + gzip compress
    for url, dest, compress in all_tasks:
        if dest.exists() and dest.stat().st_size > 0:
            _done += 1
            _skip += 1
            _progress(_done, grand_total, t0, _dl_bytes)
        elif compress:
            cpu_tasks.append((url, dest))
        else:
            raw_tasks.append((url, dest))

    # --- Raw tasks: fire all at once, extract CSV in-line (small files) ---
    async def _run_raw():
        nonlocal _done, _dl_bytes, _success, _skip, _fail, _not_found
        if not raw_tasks:
            return
        sem = asyncio.Semaphore(concurrency)
        conn = aiohttp.TCPConnector(limit=concurrency * 2, force_close=False)

        async def _dl_and_extract(session, url, dest):
            u, ok, msg, data = await download_file(session, url, dest, sem)
            return u, ok, msg, data, dest

        async with aiohttp.ClientSession(connector=conn) as session:
            coros = [_dl_and_extract(session, url, dest) for url, dest in raw_tasks]
            for coro in asyncio.as_completed(coros):
                url, ok, msg, data, dest = await coro
                if ok and msg == "exists":
                    _done += 1
                    _skip += 1
                elif ok:
                    try:
                        csv_data = _extract_zip_csv(data)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        tmp = dest.with_suffix(dest.suffix + ".tmp")
                        tmp.write_bytes(csv_data)
                        tmp.rename(dest)
                        _dl_bytes += len(csv_data)
                        _done += 1
                        _success += 1
                    except Exception as exc:
                        _done += 1
                        _fail += 1
                        print(f"\n  ✗ {url.split('/')[-1]}  ({exc})")
                elif "404" in msg or "403" in msg:
                    _done += 1
                    _not_found += 1
                else:
                    _done += 1
                    _fail += 1
                    print(f"\n  ✗ {url.split('/')[-1]}  ({msg})")
                _progress(_done, grand_total, t0, _dl_bytes)

    # --- CPU-heavy tasks: batch download → multiprocess extract+gzip → repeat ---
    async def _run_cpu():
        nonlocal _done, _dl_bytes, _success, _skip, _fail, _not_found
        if not cpu_tasks:
            return
        batch_size = min(concurrency, 8)
        conn = aiohttp.TCPConnector(limit=batch_size * 2, force_close=False)
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=batch_size)
        loop = asyncio.get_event_loop()
        sem = asyncio.Semaphore(batch_size)

        async def _dl_raw(session, url, dest):
            for attempt in range(1, RETRY_ATTEMPTS + 1):
                try:
                    async with sem:
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=600)) as resp:
                            if resp.status == 404:
                                return url, dest, None, "not found (404)"
                            if resp.status == 403:
                                return url, dest, None, "forbidden (403)"
                            if resp.status != 200:
                                raise aiohttp.ClientError(f"HTTP {resp.status}")
                            data = await resp.read()
                            return url, dest, data, "ok"
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    if attempt == RETRY_ATTEMPTS:
                        return url, dest, None, str(exc)
                    await asyncio.sleep(RETRY_BACKOFF * attempt)
            return url, dest, None, "max retries"

        async with aiohttp.ClientSession(connector=conn) as session:
            for batch_start in range(0, len(cpu_tasks), batch_size):
                batch = cpu_tasks[batch_start:batch_start + batch_size]

                raw_results = await asyncio.gather(
                    *[_dl_raw(session, url, dest) for url, dest in batch]
                )

                process_futs = []
                for url, dest, data, msg in raw_results:
                    if data is None:
                        _done += 1
                        if "404" in msg or "403" in msg:
                            _not_found += 1
                        else:
                            _fail += 1
                            print(f"\n  ✗ {url.split('/')[-1]}  ({msg})")
                        _progress(_done, grand_total, t0, _dl_bytes)
                    else:
                        process_futs.append(
                            (url, loop.run_in_executor(
                                pool, _process_and_write, data, True, dest,
                            ))
                        )

                for url, fut in process_futs:
                    try:
                        written = await fut
                        _dl_bytes += written
                        _done += 1
                        _success += 1
                    except Exception as exc:
                        _done += 1
                        _fail += 1
                        print(f"\n  ✗ {url.split('/')[-1]}  ({exc})")
                    _progress(_done, grand_total, t0, _dl_bytes)

        pool.shutdown(wait=False)

    # Run both groups concurrently (same server, but different file sizes)
    await asyncio.gather(_run_raw(), _run_cpu())

    elapsed = time.monotonic() - t0
    print(
        f"\n{symbol} done in {elapsed:.1f}s.  "
        f"downloaded={_success}  skipped={_skip}  "
        f"not_found={_not_found}  failed={_fail}  "
        f"({_fmt_size(_dl_bytes)})"
    )
    return _fail


async def run(
    symbols: list[str],
    start_date: str,
    end_date: str,
    concurrency: int,
    data_types: list[str],
):
    print(f"Symbols:          {', '.join(symbols)}")
    print(f"Date range:       {start_date} -> {end_date}")
    print(f"Data types:       {', '.join(data_types)}")
    print(f"Concurrency:      {concurrency}")
    print("=" * 60)

    total_fail = 0
    for i, symbol in enumerate(symbols, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(symbols)}] {symbol}")
        print(f"{'='*60}")
        fails = await run_one_symbol(symbol, start_date, end_date, concurrency, data_types)
        total_fail += fails

    print("\n" + "=" * 60)
    print(f"All {len(symbols)} symbols done.")
    if total_fail > 0:
        print(f"Total failures: {total_fail}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download Binance market data from data.binance.vision.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available types: {', '.join(ALL_TYPES)}\n"
               f"Default types:   {', '.join(DEFAULT_TYPES)}\n"
               f"Use -t all to download everything.",
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
        "--types", "-t",
        type=str,
        default=None,
        help="Comma-separated data types to download. Use 'all' for everything.",
    )
    args = parser.parse_args()

    try:
        s = datetime.strptime(args.start_date, "%Y-%m-%d")
        e = datetime.strptime(args.end_date, "%Y-%m-%d")
        if s > e:
            print("Error: start_date must be <= end_date", file=sys.stderr)
            sys.exit(1)
    except ValueError as exc:
        print(f"Error: invalid date format: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.types is None:
        data_types = DEFAULT_TYPES
    elif args.types.strip().lower() == "all":
        data_types = list(ALL_TYPES)
    else:
        data_types = [t.strip() for t in args.types.split(",") if t.strip()]
        for t in data_types:
            if t not in ALL_TYPES:
                print(
                    f"Error: unknown type '{t}'. Available: {', '.join(ALL_TYPES)}",
                    file=sys.stderr,
                )
                sys.exit(1)

    symbols = [sym.strip().upper() for sym in args.symbol.split(",") if sym.strip()]
    asyncio.run(run(symbols, args.start_date, args.end_date, args.concurrency, data_types))


if __name__ == "__main__":
    main()
