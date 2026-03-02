#!/usr/bin/env python3
"""
Download Bybit futures trades and ob200 orderbook snapshots from official public archives.

Sources:
  - Trades:    https://public.bybit.com/trading/{SYMBOL}/{SYMBOL}{YYYY-MM-DD}.csv.gz
  - Orderbook: https://quote-saver.bycsi.com/orderbook/linear/{SYMBOL}/{YYYY-MM-DD}_{SYMBOL}_ob200.data.zip

Usage:
  python download_bybit_data.py BTCUSDT 2026-02-01 2026-02-28
  python download_bybit_data.py SOLUSDT 2026-01-15 2026-01-20 --concurrency 10

Output structure:
  flow_research/data/{SYMBOL}/
    {YYYY-MM-DD}_trades.csv       (decompressed from .csv.gz)
    {YYYY-MM-DD}_orderbook.jsonl   (extracted from ob200 .zip, JSONL format)
"""

import argparse
import asyncio
import atexit
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

BYBIT_TRADES_URL = "https://public.bybit.com/trading/{symbol}/{symbol}{date}.csv.gz"
BYBIT_OB200_URL = (
    "https://quote-saver.bycsi.com/orderbook/linear/{symbol}/"
    "{date}_{symbol}_ob200.data.zip"
)

DEFAULT_CONCURRENT = 5
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2  # seconds

# Output directory relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

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
# Helpers
# ---------------------------------------------------------------------------


def date_range(start: str, end: str):
    """Yield date strings YYYY-MM-DD from start to end inclusive."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    while s <= e:
        yield s.strftime("%Y-%m-%d")
        s += timedelta(days=1)


# ---------------------------------------------------------------------------
# Download engine
# ---------------------------------------------------------------------------


def _decompress_gzip(raw: bytes) -> bytes:
    """Decompress gzip data to plain bytes."""
    return gzip.decompress(raw)


def _extract_zip_first(raw: bytes) -> bytes:
    """Extract the first file from a zip archive."""
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        return zf.read(zf.namelist()[0])


async def download_file(
    session: aiohttp.ClientSession,
    url: str,
    dest: Path,
    semaphore: asyncio.Semaphore,
    decompress: str = None,
):
    """Download a file with retries + atomic write. Returns (url, success, message).

    decompress: None (raw), 'gzip' (.csv.gz -> .csv), 'zip' (.zip -> extract first file)
    """
    if dest.exists() and dest.stat().st_size > 0:
        return (url, True, "exists")

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    _active_tmp_files.add(tmp)

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            async with semaphore:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                    if resp.status == 404:
                        _active_tmp_files.discard(tmp)
                        return (url, False, "not found (404)")
                    if resp.status != 200:
                        raise aiohttp.ClientError(f"HTTP {resp.status}")
                    expected = resp.content_length
                    data = await resp.read()

            if expected is not None and len(data) != expected:
                raise aiohttp.ClientError(
                    f"size mismatch: got {len(data)}, expected {expected}"
                )

            if decompress == "gzip":
                data = _decompress_gzip(data)
            elif decompress == "zip":
                data = _extract_zip_first(data)

            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(data)
            tmp.rename(dest)
            _active_tmp_files.discard(tmp)
            return (url, True, "ok")

        except asyncio.TimeoutError:
            msg = f"timeout (attempt {attempt}/{RETRY_ATTEMPTS})"
        except (aiohttp.ClientError, OSError) as exc:
            msg = f"{exc} (attempt {attempt}/{RETRY_ATTEMPTS})"

        tmp.unlink(missing_ok=True)
        if attempt < RETRY_ATTEMPTS:
            await asyncio.sleep(RETRY_BACKOFF * attempt)

    _active_tmp_files.discard(tmp)
    return (url, False, msg)


# ---------------------------------------------------------------------------
# Task builders
# ---------------------------------------------------------------------------


def trades_tasks(symbol: str, dates, output_dir: Path):
    """Build (url, dest, decompress) tuples for Bybit futures trades."""
    base = output_dir / symbol
    for d in dates:
        fname = f"{d}_trades.csv"
        url = BYBIT_TRADES_URL.format(symbol=symbol, date=d)
        yield url, base / fname, "gzip"


def ob200_tasks(symbol: str, dates, output_dir: Path):
    """Build (url, dest, decompress) tuples for Bybit futures orderbook ob200."""
    base = output_dir / symbol
    for d in dates:
        fname = f"{d}_orderbook.jsonl"
        url = BYBIT_OB200_URL.format(symbol=symbol, date=d)
        yield url, base / fname, "zip"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run(symbol: str, start_date: str, end_date: str, concurrency: int):
    dates = list(date_range(start_date, end_date))
    output_dir = DATA_DIR

    all_tasks = []
    all_tasks.extend(trades_tasks(symbol, dates, output_dir))
    all_tasks.extend(ob200_tasks(symbol, dates, output_dir))

    total = len(all_tasks)

    print(f"Symbol:           {symbol}")
    print(f"Date range:       {start_date} -> {end_date} ({len(dates)} days)")
    print(f"Sources:          trades + orderbook (ob200)")
    print(f"Concurrency:      {concurrency}")
    print(f"Output directory: {output_dir / symbol}")
    print(f"Total files:      {total}")
    print("-" * 60)

    semaphore = asyncio.Semaphore(concurrency)
    success_count = 0
    skip_count = 0
    fail_count = 0
    not_found_count = 0
    t0 = time.monotonic()

    connector = aiohttp.TCPConnector(limit=concurrency * 2, force_close=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        coros = [
            download_file(session, url, dest, semaphore, decompress=dec)
            for url, dest, dec in all_tasks
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
                print(f"  [{i}/{total}] ✓ {short}  {ts}")
            elif "404" in msg:
                not_found_count += 1
                print(f"  [{i}/{total}] - {short}  (not available)")
            else:
                fail_count += 1
                print(f"  [{i}/{total}] ✗ {short}  ({msg})")

    total_elapsed = time.monotonic() - t0
    print()
    print("=" * 60)
    print(
        f"Done in {total_elapsed:.1f}s.  "
        f"downloaded={success_count}  skipped={skip_count}  "
        f"not_found={not_found_count}  failed={fail_count}"
    )
    print(f"Data saved to: {output_dir / symbol}")

    if fail_count > 0:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download Bybit futures trades + ob200 orderbook snapshots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("symbol", help="Trading pair symbol, e.g. BTCUSDT")
    parser.add_argument("start_date", help="Start date inclusive (YYYY-MM-DD)")
    parser.add_argument("end_date", help="End date inclusive (YYYY-MM-DD)")
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=DEFAULT_CONCURRENT,
        help=f"Max concurrent downloads (default: {DEFAULT_CONCURRENT})",
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

    asyncio.run(run(args.symbol.upper(), args.start_date, args.end_date, args.concurrency))


if __name__ == "__main__":
    main()
