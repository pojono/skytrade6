#!/usr/bin/env python3
"""
Download OKX market data (perpetual swap or spot).

Sources — swap (REST API v5, paginated):
  - Kline (1m):              GET /api/v5/market/history-candles
  - Mark Price Kline (1m):   GET /api/v5/market/history-mark-price-candles
  - Funding Rate History:    GET /api/v5/public/funding-rate-history
  - Open Interest History:   GET /api/v5/rubik/stat/contracts/open-interest-history
  - Long/Short Ratio:        GET /api/v5/rubik/stat/contracts/long-short-account-ratio-contract
  - Taker Volume:            GET /api/v5/rubik/stat/taker-volume-contract
  - Premium History:         GET /api/v5/public/premium-history

Sources — swap (bulk archives):
  - Trades: https://static.okx.com/cdn/okex/traderecords/trades/daily/{YYYYMMDD}/{INST}-trades-{YYYY-MM-DD}.zip

Sources — spot (REST API v5):
  - Kline (1m):              GET /api/v5/market/history-candles (instId=BTC-USDT)

Sources — spot (bulk archives):
  - Trades: https://static.okx.com/cdn/okex/traderecords/trades/daily/{YYYYMMDD}/{INST}-trades-{YYYY-MM-DD}.zip

Usage:
  # Swap (default):
  python download_okx_data.py BTCUSDT 2026-02-01 2026-02-28
  python download_okx_data.py BTCUSDT 2026-02-01 2026-02-07 --types klines,fundingRate

  # Spot:
  python download_okx_data.py BTCUSDT 2026-02-01 2026-02-28 --market spot
  python download_okx_data.py BTCUSDT 2026-02-01 2026-02-07 --market spot --types all

Output structure (same folder, spot files have _spot postfix):
  okx/{SYMBOL}/
    {YYYY-MM-DD}_trades.csv               (swap trades)
    {YYYY-MM-DD}_kline_1m.csv             (swap OHLCV 1m)
    {YYYY-MM-DD}_mark_price_kline_1m.csv  (swap mark price 1m)
    {YYYY-MM-DD}_funding_rate.csv         (swap funding rates)
    {YYYY-MM-DD}_open_interest_5min.csv   (swap open interest 5min)
    {YYYY-MM-DD}_long_short_ratio_5min.csv (swap long/short ratio 5min)
    {YYYY-MM-DD}_taker_volume_5min.csv    (swap taker volume 5min)
    {YYYY-MM-DD}_premium_history.csv      (swap premium index)
    {YYYY-MM-DD}_kline_1m_spot.csv        (spot OHLCV 1m)
    {YYYY-MM-DD}_trades_spot.csv          (spot trades)
"""

import argparse
import asyncio
import atexit
import csv
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

OKX_API_BASE = "https://www.okx.com"
OKX_TRADES_URL = (
    "https://static.okx.com/cdn/okex/traderecords/trades/daily/"
    "{date_compact}/{instrument}-trades-{date}.zip"
)

DEFAULT_CONCURRENT = 5
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2  # seconds
API_RATE_LIMIT_DELAY = 0.25  # seconds between paginated requests within a single task

# Output directory relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "okx"

# ---------------------------------------------------------------------------
# Data type definitions (unified interface)
# ---------------------------------------------------------------------------

# --- Swap (perpetual futures) data types ---
DATA_TYPES = {
    "trades":           {"source": "bulk",  "label": "trades"},
    "klines":           {"source": "rest",  "label": "klines (1m)"},
    "markPriceKlines":  {"source": "rest",  "label": "mark price klines (1m)"},
    "fundingRate":      {"source": "rest",  "label": "funding rate"},
    "openInterest":     {"source": "rest",  "label": "open interest (5min)"},
    "longShortRatio":   {"source": "rest",  "label": "long/short ratio (5min)"},
    "takerVolume":      {"source": "rest",  "label": "taker volume (5min)"},
    "premiumHistory":   {"source": "rest",  "label": "premium history"},
}

DEFAULT_TYPES = [
    "klines", "markPriceKlines", "fundingRate", "openInterest",
    "longShortRatio", "takerVolume", "premiumHistory",
]

# --- Spot data types ---
SPOT_DATA_TYPES = {
    "spotKlines":  {"source": "rest",  "label": "spot klines (1m)"},
    "spotTrades":  {"source": "bulk",  "label": "spot trades"},
}

SPOT_DEFAULT_TYPES = ["spotKlines"]

# Global rate limiter — enforces min interval between ANY two API requests
# OKX rubik/stat endpoints: 5 req/2s; market endpoints: 20-40 req/2s
# We target ~4 req/s globally which is safe for all endpoint types.
_throttle_lock: asyncio.Lock | None = None
_throttle_last: float = 0.0
GLOBAL_MIN_INTERVAL = 0.28  # seconds between any two API calls (~3.5 req/s)


async def _throttle():
    """Global rate limiter. Ensures min interval between consecutive API calls."""
    global _throttle_lock, _throttle_last
    if _throttle_lock is None:
        _throttle_lock = asyncio.Lock()
    async with _throttle_lock:
        now = time.monotonic()
        wait = GLOBAL_MIN_INTERVAL - (now - _throttle_last)
        if wait > 0:
            await asyncio.sleep(wait)
        _throttle_last = time.monotonic()


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


def symbol_to_okx_inst(symbol: str, market: str = "swap") -> str:
    """Convert e.g. BTCUSDT -> BTC-USDT-SWAP (swap) or BTC-USDT (spot)."""
    for quote in ("USDT", "USDC", "USD"):
        if symbol.endswith(quote):
            base_ccy = symbol[: -len(quote)]
            if market == "spot":
                return f"{base_ccy}-{quote}"
            return f"{base_ccy}-{quote}-SWAP"
    # Fallback: assume last 4 chars are quote
    if market == "spot":
        return f"{symbol[:-4]}-{symbol[-4:]}"
    return f"{symbol[:-4]}-{symbol[-4:]}-SWAP"


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


def _extract_zip_csv(raw: bytes) -> bytes:
    """Extract the first .csv file from a zip archive. Returns raw CSV bytes."""
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
        if csv_files:
            return zf.read(csv_files[0])
        # Fallback: extract first file
        return zf.read(zf.namelist()[0])


# ---------------------------------------------------------------------------
# First-available-date detection (binary search via kline API probe)
# ---------------------------------------------------------------------------


async def _probe_date_exists(session: aiohttp.ClientSession, inst_id: str, date_str: str) -> bool:
    """Return True if kline data exists for this instrument on this date."""
    start_ms = _date_to_ms(date_str)
    end_ms = start_ms + 60_000  # just 1 minute
    try:
        result = await _api_get(session, "/api/v5/market/history-candles", {
            "instId": inst_id,
            "bar": "1m",
            "after": str(end_ms),
            "before": str(start_ms - 1),
            "limit": "1",
        })
        return len(result) > 0
    except Exception:
        return False


async def find_first_available_date(
    session: aiohttp.ClientSession,
    inst_id: str,
    start_date: str,
    end_date: str,
) -> str | None:
    """Binary search for the first date that has data on OKX.

    Probes the history-candles API. Returns the first available date string,
    or None if no data exists in the range. ~10 requests for a 256-day range.
    """
    fmt = "%Y-%m-%d"
    lo = datetime.strptime(start_date, fmt)
    hi = datetime.strptime(end_date, fmt)

    # Quick check: if start_date exists, no need to search
    if await _probe_date_exists(session, inst_id, start_date):
        return start_date

    # Find a known-good upper bound (walk backward up to 3 days)
    upper = None
    d = hi
    for _ in range(4):
        if await _probe_date_exists(session, inst_id, d.strftime(fmt)):
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
        if await _probe_date_exists(session, inst_id, mid_str):
            upper = mid
        else:
            lo = mid

    return upper.strftime(fmt)


# ---------------------------------------------------------------------------
# REST API v5 — generic GET with retries
# ---------------------------------------------------------------------------


async def _api_get(session: aiohttp.ClientSession, path: str, params: dict) -> list:
    """GET an OKX v5 API endpoint with retries. Returns parsed JSON data list."""
    url = OKX_API_BASE + path
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        await _throttle()
        try:
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 429:
                    print(f"    429 rate limit hit, waiting 5s...")
                    await asyncio.sleep(5)
                    raise aiohttp.ClientError("429 rate limit")

                body = await resp.json()
                code = body.get("code")
                if code != "0":
                    raise aiohttp.ClientError(
                        f"API error {code}: {body.get('msg')}"
                    )
                return body.get("data", [])
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
            if attempt == RETRY_ATTEMPTS:
                raise
            print(f"    retry {attempt}/{RETRY_ATTEMPTS}: {exc}")
            await asyncio.sleep(RETRY_BACKOFF * attempt)
    return []  # unreachable


# ---------------------------------------------------------------------------
# REST API v5 — paginated fetchers
# ---------------------------------------------------------------------------


async def fetch_kline(
    session: aiohttp.ClientSession,
    inst_id: str,
    start_ms: int,
    end_ms: int,
    path: str = "/api/v5/market/history-candles",
    bar: str = "1m",
    limit: int = 100,
) -> list[list]:
    """Paginate kline-style endpoints (candles, mark-price-candles).

    OKX candles: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    Mark price:  [ts, o, h, l, c, confirm]
    API returns newest first; we walk backward from end_ms using 'after' param.

    OKX pagination: 'after' = return data older than this ts,
                    'before' = return data newer than this ts.
    """
    all_rows = []
    cursor_after = str(end_ms + 1)  # start from end, walk backward
    page = 0
    while True:
        params = {
            "instId": inst_id,
            "bar": bar,
            "after": cursor_after,
            "before": str(start_ms - 1),
            "limit": str(limit),
        }
        data = await _api_get(session, path, params)
        if not data:
            break
        all_rows.extend(data)
        page += 1
        # data is sorted descending; oldest is last
        oldest_ts = data[-1][0]
        if int(oldest_ts) <= start_ms or len(data) < limit:
            break
        cursor_after = oldest_ts
        if page % 10 == 0:
            print(f"      ... {len(all_rows)} candles fetched so far")
        await asyncio.sleep(API_RATE_LIMIT_DELAY)

    # deduplicate by ts and sort ascending
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
    inst_id: str,
    start_ms: int,
    end_ms: int,
) -> list[list]:
    """Paginate /api/v5/public/funding-rate-history.

    Returns [[fundingTime, fundingRate, realizedRate], ...] ascending.
    API returns newest first; paginate via 'after' param.
    """
    all_rows = []
    cursor_after = str(end_ms + 1)
    page = 0
    while True:
        params = {
            "instId": inst_id,
            "after": cursor_after,
            "before": str(start_ms - 1),
            "limit": "100",
        }
        data = await _api_get(session, "/api/v5/public/funding-rate-history", params)
        if not data:
            break
        for item in data:
            all_rows.append([
                item["fundingTime"],
                item["fundingRate"],
                item.get("realizedRate", ""),
            ])
        page += 1
        oldest_ts = data[-1]["fundingTime"]
        if int(oldest_ts) <= start_ms or len(data) < 100:
            break
        cursor_after = oldest_ts
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


async def fetch_open_interest(
    session: aiohttp.ClientSession,
    inst_id: str,
    start_ms: int,
    end_ms: int,
    period: str = "5m",
) -> list[list]:
    """Paginate /api/v5/rubik/stat/contracts/open-interest-history.

    Returns [[ts, oi, oiCcy, oiUsd], ...] ascending.
    Max 100 per page, newest first. Paginate via 'end' timestamp.
    """
    all_rows = []
    cursor_end = end_ms
    page = 0
    while True:
        params = {
            "instId": inst_id,
            "period": period,
            "end": str(cursor_end),
            "begin": str(start_ms),
            "limit": "100",
        }
        data = await _api_get(
            session,
            "/api/v5/rubik/stat/contracts/open-interest-history",
            params,
        )
        if not data:
            break
        all_rows.extend(data)
        page += 1
        oldest_ts = int(data[-1][0])
        if oldest_ts <= start_ms or len(data) < 100:
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


async def fetch_long_short_ratio(
    session: aiohttp.ClientSession,
    inst_id: str,
    start_ms: int,
    end_ms: int,
    period: str = "5m",
) -> list[list]:
    """Paginate /api/v5/rubik/stat/contracts/long-short-account-ratio-contract.

    Returns [[ts, longShortAcctRatio], ...] ascending.
    """
    all_rows = []
    cursor_end = end_ms
    page = 0
    while True:
        params = {
            "instId": inst_id,
            "period": period,
            "end": str(cursor_end),
            "begin": str(start_ms),
            "limit": "100",
        }
        data = await _api_get(
            session,
            "/api/v5/rubik/stat/contracts/long-short-account-ratio-contract",
            params,
        )
        if not data:
            break
        all_rows.extend(data)
        page += 1
        oldest_ts = int(data[-1][0])
        if oldest_ts <= start_ms or len(data) < 100:
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


async def fetch_taker_volume(
    session: aiohttp.ClientSession,
    inst_id: str,
    start_ms: int,
    end_ms: int,
    period: str = "5m",
) -> list[list]:
    """Paginate /api/v5/rubik/stat/taker-volume-contract.

    Returns [[ts, sellVol, buyVol], ...] ascending.
    """
    all_rows = []
    cursor_end = end_ms
    page = 0
    while True:
        params = {
            "instId": inst_id,
            "period": period,
            "end": str(cursor_end),
            "begin": str(start_ms),
            "limit": "100",
        }
        data = await _api_get(
            session,
            "/api/v5/rubik/stat/taker-volume-contract",
            params,
        )
        if not data:
            break
        all_rows.extend(data)
        page += 1
        oldest_ts = int(data[-1][0])
        if oldest_ts <= start_ms or len(data) < 100:
            break
        cursor_end = oldest_ts - 1
        if page % 10 == 0:
            print(f"      ... {len(all_rows)} taker vol records fetched so far")
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


async def fetch_premium_history(
    session: aiohttp.ClientSession,
    inst_id: str,
    start_ms: int,
    end_ms: int,
) -> list[list]:
    """Paginate /api/v5/public/premium-history.

    Returns [[ts, premium], ...] ascending.
    """
    all_rows = []
    cursor_after = str(end_ms + 1)
    page = 0
    while True:
        params = {
            "instId": inst_id,
            "after": cursor_after,
            "before": str(start_ms - 1),
            "limit": "100",
        }
        data = await _api_get(session, "/api/v5/public/premium-history", params)
        if not data:
            break
        for item in data:
            all_rows.append([item["ts"], item["premium"]])
        page += 1
        oldest_ts = data[-1]["ts"]
        if int(oldest_ts) <= start_ms or len(data) < 100:
            break
        cursor_after = oldest_ts
        if page % 10 == 0:
            print(f"      ... {len(all_rows)} premium records fetched so far")
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

REST_API_SOURCES = {
    "klines": {
        "suffix": "kline_1m",
        "fetcher": "kline",
        "path": "/api/v5/market/history-candles",
        "header": ["startTime", "open", "high", "low", "close", "volume", "volCcy", "volCcyQuote", "confirm"],
        "label": "kline (1m)",
        "limit": 100,
    },
    "markPriceKlines": {
        "suffix": "mark_price_kline_1m",
        "fetcher": "kline",
        "path": "/api/v5/market/history-mark-price-candles",
        "header": ["startTime", "open", "high", "low", "close", "confirm"],
        "label": "mark price kline (1m)",
        "limit": 100,
    },
    "fundingRate": {
        "suffix": "funding_rate",
        "fetcher": "funding_rate",
        "header": ["fundingTime", "fundingRate", "realizedRate"],
        "label": "funding rate",
    },
    "openInterest": {
        "suffix": "open_interest_5min",
        "fetcher": "open_interest",
        "header": ["timestamp", "oi", "oiCcy", "oiUsd"],
        "label": "open interest (5min)",
    },
    "longShortRatio": {
        "suffix": "long_short_ratio_5min",
        "fetcher": "long_short_ratio",
        "header": ["timestamp", "longShortAcctRatio"],
        "label": "long/short ratio (5min)",
    },
    "takerVolume": {
        "suffix": "taker_volume_5min",
        "fetcher": "taker_volume",
        "header": ["timestamp", "sellVol", "buyVol"],
        "label": "taker volume (5min)",
    },
    "premiumHistory": {
        "suffix": "premium_history",
        "fetcher": "premium_history",
        "header": ["timestamp", "premium"],
        "label": "premium history",
    },
}

SPOT_REST_API_SOURCES = {
    "spotKlines": {
        "suffix": "kline_1m_spot",
        "fetcher": "kline",
        "path": "/api/v5/market/history-candles",
        "header": ["startTime", "open", "high", "low", "close", "volume", "volCcy", "volCcyQuote", "confirm"],
        "label": "spot kline (1m)",
        "limit": 100,
    },
}


async def _fetch_one_day(session, inst_id, day_str, cfg):
    """Fetch data for a single day using the appropriate fetcher."""
    start_ms = _date_to_ms(day_str)
    end_ms = _date_to_ms(day_str, end_of_day=True)
    fetcher = cfg["fetcher"]

    if fetcher == "kline":
        return await fetch_kline(
            session, inst_id, start_ms, end_ms,
            path=cfg["path"], limit=cfg["limit"],
        )
    elif fetcher == "funding_rate":
        return await fetch_funding_rate(session, inst_id, start_ms, end_ms)
    elif fetcher == "open_interest":
        return await fetch_open_interest(session, inst_id, start_ms, end_ms)
    elif fetcher == "long_short_ratio":
        return await fetch_long_short_ratio(session, inst_id, start_ms, end_ms)
    elif fetcher == "taker_volume":
        return await fetch_taker_volume(session, inst_id, start_ms, end_ms)
    elif fetcher == "premium_history":
        return await fetch_premium_history(session, inst_id, start_ms, end_ms)
    return []


async def _fetch_and_save(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    symbol: str,
    inst_id: str,
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
            rows = await _fetch_one_day(session, inst_id, day_str, cfg)
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
    inst_id: str,
    dates: list[str],
    output_dir: Path,
    api_concurrency: int = 10,
    rest_types: list[str] | None = None,
    rest_sources: dict | None = None,
) -> tuple[int, int, int]:
    """Download selected REST API data types concurrently.

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
            tasks.append(
                _fetch_and_save(session, semaphore, symbol, inst_id, day_str, cfg, base)
            )

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


async def download_file(
    session: aiohttp.ClientSession,
    url: str,
    dest: Path,
    semaphore: asyncio.Semaphore,
):
    """Download a .zip file, extract CSV, write atomically. Returns (url, success, message)."""
    if dest.exists() and dest.stat().st_size > 0:
        return (url, True, "exists")

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
                        return (url, False, "not found (404)")
                    if resp.status == 403:
                        _active_tmp_files.discard(tmp)
                        return (url, False, "forbidden (403)")
                    if resp.status != 200:
                        raise aiohttp.ClientError(f"HTTP {resp.status}")
                    expected = resp.content_length
                    data = await resp.read()

            if expected is not None and len(data) != expected:
                raise aiohttp.ClientError(
                    f"size mismatch: got {len(data)}, expected {expected}"
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
# Task builders
# ---------------------------------------------------------------------------


def trades_tasks(symbol: str, dates, output_dir: Path):
    """Build (url, dest) tuples for OKX perpetual swap trades."""
    inst_id = symbol_to_okx_inst(symbol, market="swap")
    base = output_dir / symbol
    for d in dates:
        date_compact = d.replace("-", "")
        fname = f"{d}_trades.csv"
        url = OKX_TRADES_URL.format(instrument=inst_id, date=d, date_compact=date_compact)
        yield url, base / fname


def spot_trades_tasks(symbol: str, dates, output_dir: Path):
    """Build (url, dest) tuples for OKX spot trades."""
    inst_id = symbol_to_okx_inst(symbol, market="spot")
    base = output_dir / symbol
    for d in dates:
        date_compact = d.replace("-", "")
        fname = f"{d}_trades_spot.csv"
        url = OKX_TRADES_URL.format(instrument=inst_id, date=d, date_compact=date_compact)
        yield url, base / fname


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_one_symbol(
    symbol: str,
    start_date: str,
    end_date: str,
    concurrency: int,
    data_types: list[str],
    market: str = "swap",
):
    """Download data for a single symbol."""
    output_dir = DATA_DIR
    is_spot = market == "spot"
    inst_id = symbol_to_okx_inst(symbol, market=market)

    # Pick the right type registry
    type_registry = SPOT_DATA_TYPES if is_spot else DATA_TYPES
    rest_source_registry = SPOT_REST_API_SOURCES if is_spot else REST_API_SOURCES

    # --- Detect first available date (binary search via kline API probe) ---
    api_connector = aiohttp.TCPConnector(limit=5, force_close=False)
    async with aiohttp.ClientSession(connector=api_connector) as probe_session:
        first_date = await find_first_available_date(
            probe_session, inst_id, start_date, end_date,
        )

    if first_date is None:
        print(f"\n{symbol}: no data found in {start_date} -> {end_date}, skipping.")
        return 0

    effective_start = first_date if first_date > start_date else start_date
    dates = list(date_range(effective_start, end_date))

    bulk_types = [t for t in data_types if type_registry[t]["source"] == "bulk"]
    rest_types = [t for t in data_types if type_registry[t]["source"] == "rest"]

    type_labels = [type_registry[t]["label"] for t in data_types]
    print(f"\nSymbol:           {symbol}")
    print(f"Market:           {market}")
    print(f"OKX instrument:   {inst_id}")
    if effective_start != start_date:
        print(f"First available:  {effective_start}  (requested {start_date})")
    print(f"Date range:       {effective_start} -> {end_date} ({len(dates)} days)")
    print(f"Data types:       {', '.join(type_labels)}")
    print(f"Output directory: {output_dir / symbol}")

    total_fail = 0
    t0 = time.monotonic()

    # --- Phase 1: Bulk archive downloads ---
    if bulk_types:
        all_tasks = []
        if "trades" in bulk_types:
            all_tasks.extend(trades_tasks(symbol, dates, output_dir))
        if "spotTrades" in bulk_types:
            all_tasks.extend(spot_trades_tasks(symbol, dates, output_dir))
        total = len(all_tasks)
        print(f"Bulk files:       {total}")
        print("-" * 60)

        phase_label = "Phase 1/2" if rest_types else "Bulk archives"
        print(f"\n[{phase_label}] Bulk archive downloads ({', '.join(bulk_types)})")
        semaphore = asyncio.Semaphore(concurrency)
        success_count = 0
        skip_count = 0
        fail_count = 0
        not_found_count = 0

        connector = aiohttp.TCPConnector(limit=concurrency * 2, force_close=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            coros = [
                download_file(session, url, dest, semaphore)
                for url, dest in all_tasks
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

        phase1_elapsed = time.monotonic() - t0
        print(
            f"\nBulk downloads done in {phase1_elapsed:.1f}s.  "
            f"downloaded={success_count}  skipped={skip_count}  "
            f"not_found={not_found_count}  failed={fail_count}"
        )
        total_fail += fail_count

    # --- Phase 2: REST API paginated downloads (per-day files) ---
    if rest_types:
        rest_labels = [type_registry[t]["label"] for t in rest_types]
        phase_label = "Phase 2/2" if bulk_types else "REST API"
        print(f"\n[{phase_label}] REST API downloads ({', '.join(rest_labels)})")
        print("-" * 60)
        t1 = time.monotonic()

        api_connector = aiohttp.TCPConnector(limit=50, force_close=False)
        async with aiohttp.ClientSession(connector=api_connector) as api_session:
            api_success, api_skip, api_fail = await download_rest_api_data(
                api_session, symbol, inst_id, dates, output_dir,
                rest_types=rest_types,
                rest_sources=rest_source_registry,
            )

        phase2_elapsed = time.monotonic() - t1
        print(
            f"\nAPI downloads done in {phase2_elapsed:.1f}s.  "
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
    market: str = "swap",
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
        description="Download OKX market data (perpetual swap or spot).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Swap data types:  {', '.join(DATA_TYPES.keys())}\n"
               f"Spot data types:  {', '.join(SPOT_DATA_TYPES.keys())}\n"
               f"Swap defaults:    {', '.join(DEFAULT_TYPES)}\n"
               f"Spot defaults:    {', '.join(SPOT_DEFAULT_TYPES)}",
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
        default="swap",
        choices=["swap", "spot"],
        help="Market type: 'swap' (perpetual futures, default) or 'spot'.",
    )
    parser.add_argument(
        "--types", "-t",
        type=str,
        default=None,
        help="Comma-separated data types to download. "
             "Use 'all' for everything including bulk trades. "
             "Defaults depend on --market.",
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
