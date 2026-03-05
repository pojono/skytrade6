#!/usr/bin/env python3
"""
Download Bybit market data.

Six data types that can be downloaded separately or all together:

  TradesLinear     — bulk  https://public.bybit.com/trading/{SYMBOL}/
  TradesSpot       — bulk  https://public.bybit.com/spot/{SYMBOL}/
  OrderbookLinear  — bulk  https://quote-saver.bycsi.com/orderbook/linear/{SYMBOL}/
  OrderbookSpot    — bulk  https://quote-saver.bycsi.com/orderbook/spot/{SYMBOL}/
  MetricsLinear    — REST  kline, markPrice, premiumIndex, fundingRate, OI, LS ratio
  MetricsSpot      — REST  kline

Usage:
  python download_bybit_data.py BTCUSDT 2026-02-01 2026-02-28
  python download_bybit_data.py BTCUSDT 2026-02-01 2026-02-28 -t TradesLinear
  python download_bybit_data.py BTCUSDT 2026-02-01 2026-02-28 -t TradesLinear,OrderbookLinear
  python download_bybit_data.py SOLUSDT,DOGEUSDT 2026-01-15 2026-01-20 -t all -c 10

Default (no -t): MetricsLinear, MetricsSpot

Output structure (spot files have _spot postfix):
  bybit/{SYMBOL}/
    {YYYY-MM-DD}_trades.csv.gz              (TradesLinear)
    {YYYY-MM-DD}_trades_spot.csv.gz         (TradesSpot)
    {YYYY-MM-DD}_orderbook.jsonl.gz         (OrderbookLinear)
    {YYYY-MM-DD}_orderbook_spot.jsonl.gz    (OrderbookSpot)
    {YYYY-MM-DD}_kline_1m.csv               (MetricsLinear)
    {YYYY-MM-DD}_mark_price_kline_1m.csv    (MetricsLinear)
    {YYYY-MM-DD}_premium_index_kline_1m.csv (MetricsLinear)
    {YYYY-MM-DD}_funding_rate.csv           (MetricsLinear)
    {YYYY-MM-DD}_open_interest_5min.csv     (MetricsLinear)
    {YYYY-MM-DD}_long_short_ratio_5min.csv  (MetricsLinear)
    {YYYY-MM-DD}_kline_1m_spot.csv          (MetricsSpot)
"""

import argparse
import asyncio
import atexit
import csv
import gzip
import io
import json
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
BYBIT_SPOT_OB200_URL = (
    "https://quote-saver.bycsi.com/orderbook/spot/{symbol}/"
    "{date}_{symbol}_ob200.data.zip"
)
BYBIT_SPOT_TRADES_URL = "https://public.bybit.com/spot/{symbol}/{symbol}_{date}.csv.gz"

DEFAULT_CONCURRENT = 5
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2  # seconds

# Bybit REST API v5
BYBIT_API_BASE = "https://api.bybit.com"
API_RATE_LIMIT_DELAY = 0.025  # seconds between paginated requests (stay under 120/s IP limit)

# Output directory relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "bybit"

# ---------------------------------------------------------------------------
# Data type definitions — 6 flat types
# ---------------------------------------------------------------------------

ALL_TYPES = [
    "TradesLinear",
    "TradesSpot",
    "OrderbookLinear",
    "OrderbookSpot",
    "MetricsLinear",
    "MetricsSpot",
]

DEFAULT_TYPES = ["MetricsLinear", "MetricsSpot"]

# Lookup: normalised lowercase -> canonical name
_TYPE_LOOKUP = {t.lower(): t for t in ALL_TYPES}

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


def _date_to_ms(date_str: str, end_of_day: bool = False) -> int:
    """Convert YYYY-MM-DD to epoch milliseconds. If end_of_day, use 23:59:59.999."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59, microsecond=999000)
    return int(dt.timestamp() * 1000)


def _write_csv(dest: Path, rows: list[list], header: list[str]):
    """Write rows to a CSV file atomically."""
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    _active_tmp_files.add(tmp)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    tmp.rename(dest)
    _active_tmp_files.discard(tmp)


# ---------------------------------------------------------------------------
# First-available-date detection (binary search via kline API probe)
# ---------------------------------------------------------------------------


async def _probe_date_exists(session: aiohttp.ClientSession, symbol: str, date_str: str, category: str = "linear") -> bool:
    """Return True if kline data exists for this symbol on this date."""
    start_ms = _date_to_ms(date_str)
    end_ms = start_ms + 60_000  # just 1 minute
    try:
        result = await _api_get(session, "/v5/market/kline", {
            "category": category,
            "symbol": symbol,
            "interval": "1",
            "start": str(start_ms),
            "end": str(end_ms),
            "limit": "1",
        })
        return len(result.get("list", [])) > 0
    except Exception:
        return False


async def find_first_available_date(
    session: aiohttp.ClientSession,
    symbol: str,
    start_date: str,
    end_date: str,
    category: str = "linear",
) -> str | None:
    """Binary search for the first date that has data on Bybit.

    Probes the kline API. Returns the first available date string (YYYY-MM-DD),
    or None if no data exists in the range. ~10 requests for a 256-day range.
    """
    fmt = "%Y-%m-%d"
    lo = datetime.strptime(start_date, fmt)
    hi = datetime.strptime(end_date, fmt)

    # Quick check: if start_date exists, no need to search
    if await _probe_date_exists(session, symbol, start_date, category):
        return start_date

    # Find a known-good upper bound (walk backward up to 3 days)
    upper = None
    d = hi
    for _ in range(4):
        if await _probe_date_exists(session, symbol, d.strftime(fmt), category):
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
        if await _probe_date_exists(session, symbol, mid_str, category):
            upper = mid
        else:
            lo = mid

    return upper.strftime(fmt)


# --- Bulk URL probe (HEAD request) for per-type first-date detection ---

def _bulk_probe_url(data_type: str, symbol: str, date_str: str) -> str:
    """Return the bulk archive URL to probe for a given type/symbol/date."""
    if data_type == "TradesLinear":
        return BYBIT_TRADES_URL.format(symbol=symbol, date=date_str)
    elif data_type == "TradesSpot":
        return BYBIT_SPOT_TRADES_URL.format(symbol=symbol, date=date_str)
    elif data_type == "OrderbookLinear":
        return BYBIT_OB200_URL.format(symbol=symbol, date=date_str)
    elif data_type == "OrderbookSpot":
        return BYBIT_SPOT_OB200_URL.format(symbol=symbol, date=date_str)
    raise ValueError(f"No bulk URL for type {data_type}")


async def _probe_bulk_exists(
    session: aiohttp.ClientSession, data_type: str, symbol: str, date_str: str,
) -> bool:
    """HEAD-request probe: does a bulk file exist for this type/symbol/date?"""
    url = _bulk_probe_url(data_type, symbol, date_str)
    try:
        async with session.head(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            return resp.status == 200
    except Exception:
        return False


async def _binary_search_first_date(
    probe_fn, start_date: str, end_date: str,
) -> str | None:
    """Generic binary search using an async probe_fn(date_str) -> bool."""
    fmt = "%Y-%m-%d"
    lo = datetime.strptime(start_date, fmt)
    hi = datetime.strptime(end_date, fmt)

    if await probe_fn(start_date):
        return start_date

    upper = None
    d = hi
    for _ in range(4):
        if await probe_fn(d.strftime(fmt)):
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
        if await probe_fn(mid_str):
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
    """Find the first available date for each requested type independently.

    Returns {type_name: first_date_str or None}.
    Metrics types probe via kline API; bulk types probe via HEAD requests.
    """
    results = {}

    async def _probe_for_type(dtype: str) -> tuple[str, str | None]:
        if dtype == "MetricsLinear":
            first = await _binary_search_first_date(
                lambda d: _probe_date_exists(session, symbol, d, "linear"),
                start_date, end_date,
            )
        elif dtype == "MetricsSpot":
            first = await _binary_search_first_date(
                lambda d: _probe_date_exists(session, symbol, d, "spot"),
                start_date, end_date,
            )
        else:
            # Bulk type: HEAD-request probe
            first = await _binary_search_first_date(
                lambda d: _probe_bulk_exists(session, dtype, symbol, d),
                start_date, end_date,
            )
        return dtype, first

    # Run all probes concurrently
    probes = [_probe_for_type(dt) for dt in data_types]
    for coro in asyncio.as_completed(probes):
        dtype, first = await coro
        results[dtype] = first

    return results


# ---------------------------------------------------------------------------
# REST API v5 — paginated fetchers
# ---------------------------------------------------------------------------


async def _api_get(session: aiohttp.ClientSession, path: str, params: dict) -> dict:
    """GET a Bybit v5 API endpoint with retries. Returns parsed JSON result."""
    url = BYBIT_API_BASE + path
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                # Adaptive rate limiting from response headers
                remaining = resp.headers.get("X-Bapi-Limit-Status")
                if remaining is not None and int(remaining) <= 2:
                    await asyncio.sleep(1.0)  # back off when near limit

                if resp.status == 403:
                    print(f"    403 rate limit hit, waiting 10s...")
                    await asyncio.sleep(10)
                    raise aiohttp.ClientError("403 rate limit")

                body = await resp.json()
                ret_code = body.get("retCode")
                if ret_code == 10006:  # Too many visits
                    print(f"    rate limited (10006), waiting 5s...")
                    await asyncio.sleep(5)
                    raise aiohttp.ClientError("rate limited")
                if ret_code != 0:
                    raise aiohttp.ClientError(
                        f"API error {ret_code}: {body.get('retMsg')}"
                    )
                return body["result"]
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
            if attempt == RETRY_ATTEMPTS:
                raise
            print(f"    retry {attempt}/{RETRY_ATTEMPTS}: {exc}")
            await asyncio.sleep(RETRY_BACKOFF * attempt)
    return {}  # unreachable


async def fetch_kline(
    session: aiohttp.ClientSession,
    symbol: str,
    start_ms: int,
    end_ms: int,
    path: str = "/v5/market/kline",
    interval: str = "1",
    limit: int = 1000,
    category: str = "linear",
) -> list[list]:
    """Paginate kline-style endpoints (kline, mark-price-kline, premium-index-price-kline).

    Returns list of candle arrays sorted ascending by startTime.
    Kline response: [[startTime, open, high, low, close, volume, turnover], ...]
    Mark/Premium response: [[startTime, open, high, low, close], ...]
    API returns newest first; we walk backward from end_ms.
    """
    all_rows = []
    cursor_end = end_ms
    page = 0
    while True:
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "start": str(start_ms),
            "end": str(cursor_end),
            "limit": str(limit),
        }
        result = await _api_get(session, path, params)
        rows = result.get("list", [])
        if not rows:
            break
        all_rows.extend(rows)
        page += 1
        # rows are sorted descending; oldest is last
        oldest_ts = int(rows[-1][0])
        if oldest_ts <= start_ms or len(rows) < limit:
            break
        # next page: end just before oldest
        cursor_end = oldest_ts - 1
        if page % 10 == 0:
            print(f"      ... {len(all_rows)} candles fetched so far")
        await asyncio.sleep(API_RATE_LIMIT_DELAY)

    # deduplicate by startTime and sort ascending
    seen = set()
    unique = []
    for r in all_rows:
        ts = r[0]
        if ts not in seen:
            seen.add(ts)
            unique.append(r)
    unique.sort(key=lambda r: int(r[0]))
    return unique


async def fetch_funding_rate(
    session: aiohttp.ClientSession,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> list[list]:
    """Paginate /v5/market/funding/history. Returns [[timestamp, fundingRate], ...] ascending."""
    all_rows = []
    cursor_end = end_ms
    page = 0
    while True:
        params = {
            "category": "linear",
            "symbol": symbol,
            "startTime": str(start_ms),
            "endTime": str(cursor_end),
            "limit": "200",
        }
        result = await _api_get(session, "/v5/market/funding/history", params)
        items = result.get("list", [])
        if not items:
            break
        for item in items:
            all_rows.append([item["fundingRateTimestamp"], item["fundingRate"]])
        page += 1
        # items are newest-first; walk backward
        oldest_ts = int(items[-1]["fundingRateTimestamp"])
        if oldest_ts <= start_ms or len(items) < 200:
            break
        cursor_end = oldest_ts - 1
        if page % 10 == 0:
            print(f"      ... {len(all_rows)} FR records fetched so far")
        await asyncio.sleep(API_RATE_LIMIT_DELAY)

    # deduplicate + sort ascending
    seen = set()
    unique = []
    for r in all_rows:
        if r[0] not in seen:
            seen.add(r[0])
            unique.append(r)
    unique.sort(key=lambda r: int(r[0]))
    return unique


async def fetch_long_short_ratio(
    session: aiohttp.ClientSession,
    symbol: str,
    start_ms: int,
    end_ms: int,
    period: str = "5min",
) -> list[list]:
    """Paginate /v5/market/account-ratio via time-window walking.

    Returns [[timestamp, buyRatio, sellRatio], ...] ascending.
    API returns newest-first; we walk backward from end_ms.
    """
    all_rows = []
    cursor_end = end_ms
    page = 0
    while True:
        params = {
            "category": "linear",
            "symbol": symbol,
            "period": period,
            "limit": "500",
            "startTime": str(start_ms),
            "endTime": str(cursor_end),
        }
        result = await _api_get(session, "/v5/market/account-ratio", params)
        items = result.get("list", [])
        if not items:
            break
        for item in items:
            all_rows.append([item["timestamp"], item["buyRatio"], item["sellRatio"]])
        page += 1
        oldest_ts = int(items[-1]["timestamp"])
        if oldest_ts <= start_ms or len(items) < 500:
            break
        cursor_end = oldest_ts - 1
        if page % 10 == 0:
            print(f"      ... {len(all_rows)} LS ratio records fetched so far")
        await asyncio.sleep(API_RATE_LIMIT_DELAY)

    # deduplicate + sort ascending
    seen = set()
    unique = []
    for r in all_rows:
        if r[0] not in seen:
            seen.add(r[0])
            unique.append(r)
    unique.sort(key=lambda r: int(r[0]))
    return unique


async def fetch_open_interest(
    session: aiohttp.ClientSession,
    symbol: str,
    start_ms: int,
    end_ms: int,
    interval_time: str = "5min",
) -> list[list]:
    """Paginate /v5/market/open-interest via time-window walking.

    Returns [[timestamp, openInterest], ...] ascending.
    API returns newest-first; we walk backward from end_ms.
    """
    all_rows = []
    cursor_end = end_ms
    page = 0
    while True:
        params = {
            "category": "linear",
            "symbol": symbol,
            "intervalTime": interval_time,
            "limit": "200",
            "startTime": str(start_ms),
            "endTime": str(cursor_end),
        }
        result = await _api_get(session, "/v5/market/open-interest", params)
        items = result.get("list", [])
        if not items:
            break
        for item in items:
            all_rows.append([item["timestamp"], item["openInterest"]])
        page += 1
        oldest_ts = int(items[-1]["timestamp"])
        if oldest_ts <= start_ms or len(items) < 200:
            break
        cursor_end = oldest_ts - 1
        if page % 10 == 0:
            print(f"      ... {len(all_rows)} OI records fetched so far")
        await asyncio.sleep(API_RATE_LIMIT_DELAY)

    # deduplicate + sort ascending
    seen = set()
    unique = []
    for r in all_rows:
        if r[0] not in seen:
            seen.add(r[0])
            unique.append(r)
    unique.sort(key=lambda r: int(r[0]))
    return unique


# ---------------------------------------------------------------------------
# REST API download orchestrator — per-day files
# ---------------------------------------------------------------------------

# Each REST API data type: (suffix, fetcher, header, label)
# Keyed by CLI type name for easy filtering.
REST_API_SOURCES = {
    "klines": {
        "suffix": "kline_1m",
        "fetcher": "kline",
        "path": "/v5/market/kline",
        "category": "linear",
        "header": ["startTime", "open", "high", "low", "close", "volume", "turnover"],
        "label": "kline (1m)",
        "limit": 1000,
    },
    "markPriceKlines": {
        "suffix": "mark_price_kline_1m",
        "fetcher": "kline",
        "path": "/v5/market/mark-price-kline",
        "header": ["startTime", "open", "high", "low", "close"],
        "label": "mark price kline (1m)",
        "limit": 200,
    },
    "premiumIndexKlines": {
        "suffix": "premium_index_kline_1m",
        "fetcher": "kline",
        "path": "/v5/market/premium-index-price-kline",
        "header": ["startTime", "open", "high", "low", "close"],
        "label": "premium index kline (1m)",
        "limit": 200,
    },
    "fundingRate": {
        "suffix": "funding_rate",
        "fetcher": "funding_rate",
        "header": ["timestamp", "fundingRate"],
        "label": "funding rate",
    },
    "openInterest": {
        "suffix": "open_interest_5min",
        "fetcher": "open_interest",
        "header": ["timestamp", "openInterest"],
        "label": "open interest (5min)",
    },
    "longShortRatio": {
        "suffix": "long_short_ratio_5min",
        "fetcher": "long_short_ratio",
        "header": ["timestamp", "buyRatio", "sellRatio"],
        "label": "long/short ratio (5min)",
    },
}

SPOT_REST_API_SOURCES = {
    "spotKlines": {
        "suffix": "kline_1m_spot",
        "fetcher": "kline",
        "path": "/v5/market/kline",
        "category": "spot",
        "header": ["startTime", "open", "high", "low", "close", "volume", "turnover"],
        "label": "spot kline (1m)",
        "limit": 1000,
    },
}


async def _fetch_one_day(session, symbol, day_str, cfg):
    """Fetch data for a single day using the appropriate fetcher."""
    start_ms = _date_to_ms(day_str)
    end_ms = _date_to_ms(day_str, end_of_day=True)
    fetcher = cfg["fetcher"]
    category = cfg.get("category", "linear")

    if fetcher == "kline":
        return await fetch_kline(
            session, symbol, start_ms, end_ms,
            path=cfg["path"], limit=cfg["limit"], category=category,
        )
    elif fetcher == "funding_rate":
        return await fetch_funding_rate(session, symbol, start_ms, end_ms)
    elif fetcher == "open_interest":
        return await fetch_open_interest(session, symbol, start_ms, end_ms)
    elif fetcher == "long_short_ratio":
        return await fetch_long_short_ratio(session, symbol, start_ms, end_ms)
    return []


async def _fetch_and_save(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    symbol: str,
    day_str: str,
    cfg: dict,
    base: Path,
) -> tuple[str, int]:
    """Fetch one (day, data_type) pair and save. Returns (status, 1)."""
    suffix = cfg["suffix"]
    header = cfg["header"]
    fname = f"{day_str}_{suffix}.csv"
    dest = base / fname

    if dest.exists() and dest.stat().st_size > 0:
        return "skip", 1

    async with semaphore:
        try:
            rows = await _fetch_one_day(session, symbol, day_str, cfg)
            if not rows:
                return "nodata", 1
            _write_csv(dest, rows, header)
            print(f"  ✓ {fname}  ({len(rows)} rows)")
            return "ok", 1
        except Exception as exc:
            print(f"  ✗ {fname}  ({exc})")
            return "fail", 1


async def download_rest_api_data(
    session: aiohttp.ClientSession,
    symbol: str,
    dates: list[str],
    output_dir: Path,
    api_concurrency: int = 10,
    rest_types: list[str] | None = None,
    rest_sources: dict | None = None,
) -> tuple[int, int, int]:
    """Download selected REST API data types concurrently.

    rest_types: list of CLI type names to include (e.g. ["klines", "fundingRate"]).
                If None, download all REST types.
    rest_sources: dict of REST source configs to use. Defaults to REST_API_SOURCES.

    Returns (success_count, skip_count, fail_count).
    """
    if rest_sources is None:
        rest_sources = REST_API_SOURCES
    base = output_dir / symbol
    base.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(api_concurrency)
    tasks = []
    for type_name, cfg in rest_sources.items():
        if rest_types is not None and type_name not in rest_types:
            continue
        for day_str in dates:
            tasks.append(_fetch_and_save(session, semaphore, symbol, day_str, cfg, base))

    total = len(tasks)
    success = skip = fail = 0
    done = 0
    t0 = time.monotonic()

    for coro in asyncio.as_completed(tasks):
        status, _ = await coro
        done += 1
        if status == "ok":
            success += 1
        elif status == "skip":
            skip += 1
        else:
            fail += 1
        if done % 50 == 0 or done == total:
            elapsed = time.monotonic() - t0
            print(f"  ... {done}/{total} done ({elapsed:.0f}s)")

    return success, skip, fail


# ---------------------------------------------------------------------------
# Bulk archive download engine
# ---------------------------------------------------------------------------


def _decompress_gzip(raw: bytes) -> bytes:
    """Decompress gzip data to plain bytes."""
    return gzip.decompress(raw)


def _extract_zip_first(raw: bytes) -> bytes:
    """Extract the first file from a zip archive."""
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        return zf.read(zf.namelist()[0])


def _zip_to_gzip(raw: bytes) -> bytes:
    """Extract first file from zip, re-compress as gzip."""
    plain = _extract_zip_first(raw)
    return gzip.compress(plain, compresslevel=6)


async def download_file(
    session: aiohttp.ClientSession,
    url: str,
    dest: Path,
    semaphore: asyncio.Semaphore,
    decompress: str = None,
):
    """Download a file with retries + atomic write. Returns (url, success, message).

    decompress: None (raw), 'gzip' (.csv.gz -> .csv), 'zip' (.zip -> extract first file),
                'zip_to_gz' (.zip -> extract -> gzip compress)
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
                    if resp.status == 403:
                        # CloudFront rate-limit — back off aggressively
                        wait = 30 * attempt
                        print(f"    ⚠ 403 rate-limited, waiting {wait}s ... ({url.split('/')[-1]})")
                        await asyncio.sleep(wait)
                        raise aiohttp.ClientError(f"HTTP 403")
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
            elif decompress == "zip_to_gz":
                data = _zip_to_gzip(data)

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
        fname = f"{d}_trades.csv.gz"
        url = BYBIT_TRADES_URL.format(symbol=symbol, date=d)
        # Treat old uncompressed .csv as existing
        old = base / f"{d}_trades.csv"
        if old.exists() and old.stat().st_size > 0:
            yield url, old, None  # will be skipped as "exists"
        else:
            yield url, base / fname, None  # keep raw .csv.gz


def ob200_tasks(symbol: str, dates, output_dir: Path):
    """Build (url, dest, decompress) tuples for Bybit futures orderbook ob200."""
    base = output_dir / symbol
    for d in dates:
        fname = f"{d}_orderbook.jsonl.gz"
        url = BYBIT_OB200_URL.format(symbol=symbol, date=d)
        # Also treat old uncompressed .jsonl as existing
        old = base / f"{d}_orderbook.jsonl"
        if old.exists() and old.stat().st_size > 0:
            yield url, old, None  # will be skipped as "exists"
        else:
            yield url, base / fname, "zip_to_gz"


def spot_ob200_tasks(symbol: str, dates, output_dir: Path):
    """Build (url, dest, decompress) tuples for Bybit spot orderbook ob200."""
    base = output_dir / symbol
    for d in dates:
        fname = f"{d}_orderbook_spot.jsonl.gz"
        url = BYBIT_SPOT_OB200_URL.format(symbol=symbol, date=d)
        old = base / f"{d}_orderbook_spot.jsonl"
        if old.exists() and old.stat().st_size > 0:
            yield url, old, None
        else:
            yield url, base / fname, "zip_to_gz"


def spot_trades_tasks(symbol: str, dates, output_dir: Path):
    """Build (url, dest, decompress) tuples for Bybit spot trades."""
    base = output_dir / symbol
    for d in dates:
        fname = f"{d}_trades_spot.csv.gz"
        url = BYBIT_SPOT_TRADES_URL.format(symbol=symbol, date=d)
        # Treat old uncompressed .csv as existing
        old = base / f"{d}_trades_spot.csv"
        if old.exists() and old.stat().st_size > 0:
            yield url, old, None
        else:
            yield url, base / fname, None  # keep raw .csv.gz


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
    """Download data for a single symbol.

    data_types: list of canonical type names from ALL_TYPES.
    Each type gets its own first-available-date via independent binary search.
    """
    output_dir = DATA_DIR

    # --- Detect first available date per type (concurrent probes) ---
    api_connector = aiohttp.TCPConnector(limit=10, force_close=False)
    async with aiohttp.ClientSession(connector=api_connector) as probe_session:
        first_dates = await find_first_dates(
            probe_session, symbol, start_date, end_date, data_types,
        )

    # Filter out types with no data at all
    active_types = {dt: fd for dt, fd in first_dates.items() if fd is not None}
    if not active_types:
        print(f"\n{symbol}: no data found in {start_date} -> {end_date}, skipping.")
        return 0

    # Build per-type date lists
    type_dates: dict[str, list[str]] = {}
    for dt, fd in active_types.items():
        eff = fd if fd > start_date else start_date
        type_dates[dt] = list(date_range(eff, end_date))

    # Classify
    bulk_active = [t for t in active_types if t.startswith("Trades") or t.startswith("Orderbook")]
    rest_active = [t for t in active_types if t.startswith("Metrics")]

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

    total_fail = 0
    t0 = time.monotonic()

    # --- Phase 1: Bulk archive downloads ---
    if bulk_active:
        all_tasks = []
        task_builders = {
            "TradesLinear":    trades_tasks,
            "TradesSpot":      spot_trades_tasks,
            "OrderbookLinear": ob200_tasks,
            "OrderbookSpot":   spot_ob200_tasks,
        }
        for bt in bulk_active:
            builder = task_builders[bt]
            all_tasks.extend(builder(symbol, type_dates[bt], output_dir))

        total = len(all_tasks)
        print(f"Bulk files:       {total}")
        print("-" * 60)

        phase_label = "Phase 1/2" if rest_active else "Bulk archives"
        print(f"\n[{phase_label}] Bulk archive downloads ({', '.join(bulk_active)})")
        semaphore = asyncio.Semaphore(concurrency)
        success_count = 0
        skip_count = 0
        fail_count = 0
        not_found_count = 0

        connector = aiohttp.TCPConnector(limit=concurrency * 2, force_close=False)
        async with aiohttp.ClientSession(connector=connector) as session:

            async def _throttled(url, dest, dec, idx):
                # Stagger requests to avoid CloudFront burst detection
                await asyncio.sleep(idx * 0.05)
                return await download_file(session, url, dest, semaphore, decompress=dec)

            coros = [
                _throttled(url, dest, dec, idx)
                for idx, (url, dest, dec) in enumerate(all_tasks)
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

        phase1_elapsed = time.monotonic() - t0
        print(
            f"\nBulk downloads done in {phase1_elapsed:.1f}s.  "
            f"downloaded={success_count}  skipped={skip_count}  "
            f"not_found={not_found_count}  failed={fail_count}"
        )
        total_fail += fail_count

    # --- Phase 2: REST API paginated downloads (per-day files) ---
    if rest_active:
        phase_label = "Phase 2/2" if bulk_active else "REST API"

        for metrics_type in rest_active:
            if metrics_type == "MetricsLinear":
                rest_sources = REST_API_SOURCES
            else:
                rest_sources = SPOT_REST_API_SOURCES
            dates_for_metrics = type_dates[metrics_type]

            rest_labels = [cfg["label"] for cfg in rest_sources.values()]
            print(f"\n[{phase_label}] {metrics_type} ({', '.join(rest_labels)})")
            print(f"  Date range: {dates_for_metrics[0]} -> {dates_for_metrics[-1]} ({len(dates_for_metrics)} days)")
            print("-" * 60)
            t1 = time.monotonic()

            api_connector = aiohttp.TCPConnector(limit=50, force_close=False)
            async with aiohttp.ClientSession(connector=api_connector) as api_session:
                api_success, api_skip, api_fail = await download_rest_api_data(
                    api_session, symbol, dates_for_metrics, output_dir,
                    rest_types=list(rest_sources.keys()),
                    rest_sources=rest_sources,
                )

            phase2_elapsed = time.monotonic() - t1
            print(
                f"\n{metrics_type} done in {phase2_elapsed:.1f}s.  "
                f"downloaded={api_success}  skipped={api_skip}  failed={api_fail}"
            )
            total_fail += api_fail

    # --- Summary ---
    total_elapsed = time.monotonic() - t0
    print(f"\n{symbol} done in {total_elapsed:.1f}s.  Data: {output_dir / symbol}")
    return total_fail


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


def _parse_types(raw: str | None) -> list[str]:
    """Parse -t argument into canonical type names. Case-insensitive."""
    if raw is None:
        return list(DEFAULT_TYPES)
    if raw.strip().lower() == "all":
        return list(ALL_TYPES)
    result = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        canonical = _TYPE_LOOKUP.get(token.lower())
        if canonical is None:
            print(
                f"Error: unknown type '{token}'. "
                f"Available: {', '.join(ALL_TYPES)}",
                file=sys.stderr,
            )
            sys.exit(1)
        result.append(canonical)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Download Bybit market data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available types:  {', '.join(ALL_TYPES)}\n"
               f"Default:          {', '.join(DEFAULT_TYPES)}\n"
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
        help=f"Comma-separated data types to download (case-insensitive). "
             f"Use 'all' for everything. Default: {', '.join(DEFAULT_TYPES)}.",
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

    data_types = _parse_types(args.types)
    symbols = [sym.strip().upper() for sym in args.symbol.split(",") if sym.strip()]
    asyncio.run(run(symbols, args.start_date, args.end_date, args.concurrency, data_types))


if __name__ == "__main__":
    main()
