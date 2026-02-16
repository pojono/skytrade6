#!/usr/bin/env python3
"""
Validate downloaded raw data from Bybit/Binance/OKX.

Checks:
  1. All expected daily files exist for each source/symbol/date
  2. No gaps in date coverage (2024-01-01 to 2026-01-31)
  3. File sizes are reasonable (not zero, not suspiciously small)
  4. Archives are not corrupted (gzip/zip can be opened)
"""

import gzip
import io
import os
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("./data")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
START_DATE = "2024-01-01"
END_DATE = "2026-01-31"

# Minimum file size thresholds (bytes) — anything below is suspicious
MIN_SIZE_GZ = 100       # .csv.gz
MIN_SIZE_ZIP = 100      # .zip

# ── Helpers ─────────────────────────────────────────────────────────────────

def date_range(start: str, end: str):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    dates = []
    while s <= e:
        dates.append(s.strftime("%Y-%m-%d"))
        s += timedelta(days=1)
    return dates


def check_gz(path: Path) -> str:
    """Try to decompress a .csv.gz file. Returns error string or empty."""
    try:
        with gzip.open(path, "rb") as f:
            data = f.read(4096)  # read first 4KB to verify
            if len(data) == 0:
                return "empty after decompression"
        return ""
    except Exception as e:
        return f"gzip error: {e}"


def check_zip(path: Path) -> str:
    """Try to open a .zip file and read its first entry. Returns error string or empty."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                return f"corrupt entry: {bad}"
            names = zf.namelist()
            if len(names) == 0:
                return "zip has no entries"
            # Try reading first entry
            with zf.open(names[0]) as f:
                data = f.read(4096)
                if len(data) == 0:
                    return "first entry empty"
        return ""
    except Exception as e:
        return f"zip error: {e}"


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024*1024):.1f} MB"
    else:
        return f"{size_bytes / (1024*1024*1024):.2f} GB"


# ── Source definitions ──────────────────────────────────────────────────────

def get_bybit_futures_files(symbol, dates):
    """Expected Bybit futures files."""
    base = DATA_DIR / symbol / "bybit" / "futures"
    for d in dates:
        yield d, base / f"{symbol}{d}.csv.gz"


def get_bybit_spot_files(symbol, dates):
    base = DATA_DIR / symbol / "bybit" / "spot"
    for d in dates:
        yield d, base / f"{symbol}_{d}.csv.gz"


def get_binance_futures_trades(symbol, dates):
    base = DATA_DIR / symbol / "binance" / "futures" / "trades"
    for d in dates:
        yield d, base / f"{symbol}-trades-{d}.zip"


def get_binance_futures_aggTrades(symbol, dates):
    base = DATA_DIR / symbol / "binance" / "futures" / "aggTrades"
    for d in dates:
        yield d, base / f"{symbol}-aggTrades-{d}.zip"


def get_binance_futures_bookDepth(symbol, dates):
    base = DATA_DIR / symbol / "binance" / "futures" / "bookDepth"
    for d in dates:
        yield d, base / f"{symbol}-bookDepth-{d}.zip"


def get_binance_futures_bookTicker(symbol, dates):
    base = DATA_DIR / symbol / "binance" / "futures" / "bookTicker"
    for d in dates:
        yield d, base / f"{symbol}-bookTicker-{d}.zip"


def get_binance_futures_metrics(symbol, dates):
    base = DATA_DIR / symbol / "binance" / "futures" / "metrics"
    for d in dates:
        yield d, base / f"{symbol}-metrics-{d}.zip"


def get_binance_futures_klines(symbol, dates, interval):
    base = DATA_DIR / symbol / "binance" / "futures" / "klines" / interval
    for d in dates:
        yield d, base / f"{symbol}-{interval}-{d}.zip"


def get_binance_futures_indexPriceKlines(symbol, dates, interval):
    base = DATA_DIR / symbol / "binance" / "futures" / "indexPriceKlines" / interval
    for d in dates:
        yield d, base / f"{symbol}-{interval}-{d}.zip"


def get_binance_futures_markPriceKlines(symbol, dates, interval):
    base = DATA_DIR / symbol / "binance" / "futures" / "markPriceKlines" / interval
    for d in dates:
        yield d, base / f"{symbol}-{interval}-{d}.zip"


def get_binance_futures_premiumIndexKlines(symbol, dates, interval):
    base = DATA_DIR / symbol / "binance" / "futures" / "premiumIndexKlines" / interval
    for d in dates:
        yield d, base / f"{symbol}-{interval}-{d}.zip"


def get_binance_spot_trades(symbol, dates):
    base = DATA_DIR / symbol / "binance" / "spot" / "trades"
    for d in dates:
        yield d, base / f"{symbol}-trades-{d}.zip"


def get_binance_spot_aggTrades(symbol, dates):
    base = DATA_DIR / symbol / "binance" / "spot" / "aggTrades"
    for d in dates:
        yield d, base / f"{symbol}-aggTrades-{d}.zip"


def get_binance_spot_klines(symbol, dates, interval):
    base = DATA_DIR / symbol / "binance" / "spot" / "klines" / interval
    for d in dates:
        yield d, base / f"{symbol}-{interval}-{d}.zip"


def get_okx_futures_files(symbol, dates):
    pair_map = {
        "BTCUSDT": "BTC-USDT", "ETHUSDT": "ETH-USDT",
        "SOLUSDT": "SOL-USDT", "XRPUSDT": "XRP-USDT",
        "DOGEUSDT": "DOGE-USDT",
    }
    pair = pair_map.get(symbol, symbol)
    instrument = f"{pair}-SWAP"
    base = DATA_DIR / symbol / "okx" / "futures"
    for d in dates:
        yield d, base / f"{instrument}-trades-{d}.zip"


def get_okx_spot_files(symbol, dates):
    pair_map = {
        "BTCUSDT": "BTC-USDT", "ETHUSDT": "ETH-USDT",
        "SOLUSDT": "SOL-USDT", "XRPUSDT": "XRP-USDT",
        "DOGEUSDT": "DOGE-USDT",
    }
    pair = pair_map.get(symbol, symbol)
    base = DATA_DIR / symbol / "okx" / "spot"
    for d in dates:
        yield d, base / f"{pair}-trades-{d}.zip"


KLINE_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]


def build_all_sources(symbol, dates):
    """Return list of (source_name, generator_of_(date, path)) for a symbol."""
    sources = []

    # Bybit
    sources.append(("bybit/futures", get_bybit_futures_files(symbol, dates)))
    sources.append(("bybit/spot", get_bybit_spot_files(symbol, dates)))

    # Binance futures
    sources.append(("binance/futures/trades", get_binance_futures_trades(symbol, dates)))
    sources.append(("binance/futures/aggTrades", get_binance_futures_aggTrades(symbol, dates)))
    sources.append(("binance/futures/bookDepth", get_binance_futures_bookDepth(symbol, dates)))
    sources.append(("binance/futures/bookTicker", get_binance_futures_bookTicker(symbol, dates)))
    sources.append(("binance/futures/metrics", get_binance_futures_metrics(symbol, dates)))

    for interval in KLINE_INTERVALS:
        sources.append((f"binance/futures/klines/{interval}",
                        get_binance_futures_klines(symbol, dates, interval)))
        sources.append((f"binance/futures/indexPriceKlines/{interval}",
                        get_binance_futures_indexPriceKlines(symbol, dates, interval)))
        sources.append((f"binance/futures/markPriceKlines/{interval}",
                        get_binance_futures_markPriceKlines(symbol, dates, interval)))
        sources.append((f"binance/futures/premiumIndexKlines/{interval}",
                        get_binance_futures_premiumIndexKlines(symbol, dates, interval)))

    # Binance spot
    sources.append(("binance/spot/trades", get_binance_spot_trades(symbol, dates)))
    sources.append(("binance/spot/aggTrades", get_binance_spot_aggTrades(symbol, dates)))
    for interval in KLINE_INTERVALS:
        sources.append((f"binance/spot/klines/{interval}",
                        get_binance_spot_klines(symbol, dates, interval)))

    # OKX
    sources.append(("okx/futures", get_okx_futures_files(symbol, dates)))
    sources.append(("okx/spot", get_okx_spot_files(symbol, dates)))

    return sources


# ── Validation ──────────────────────────────────────────────────────────────

def find_gap_ranges(missing_dates):
    """Convert list of date strings to gap ranges for compact display."""
    if not missing_dates:
        return []
    sorted_dates = sorted(missing_dates)
    ranges = []
    start = sorted_dates[0]
    prev = sorted_dates[0]
    for d in sorted_dates[1:]:
        prev_dt = datetime.strptime(prev, "%Y-%m-%d")
        curr_dt = datetime.strptime(d, "%Y-%m-%d")
        if (curr_dt - prev_dt).days == 1:
            prev = d
        else:
            ranges.append((start, prev))
            start = d
            prev = d
    ranges.append((start, prev))
    return ranges


def validate_source(symbol, source_name, file_iter, all_dates_set):
    """Validate a single source. Returns dict with results."""
    missing = []
    corrupt = []
    tiny = []
    sizes = []
    present_count = 0
    tmp_files = []

    for date_str, path in file_iter:
        if not path.exists():
            missing.append(date_str)
            continue

        present_count += 1
        fsize = path.stat().st_size
        sizes.append(fsize)

        # Check for .tmp files (incomplete downloads)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_files.append(date_str)

        # Size check
        min_size = MIN_SIZE_GZ if path.suffix == ".gz" else MIN_SIZE_ZIP
        if fsize < min_size:
            tiny.append((date_str, fsize))

    return {
        "present": present_count,
        "missing": missing,
        "corrupt": corrupt,
        "tiny": tiny,
        "tmp_files": tmp_files,
        "sizes": sizes,
    }


def validate_source_with_integrity(symbol, source_name, file_iter, sample_rate=0.05):
    """Validate with archive integrity checks on a sample of files."""
    missing = []
    corrupt = []
    tiny = []
    sizes = []
    present_count = 0

    files_to_check = []

    for date_str, path in file_iter:
        if not path.exists():
            missing.append(date_str)
            continue

        present_count += 1
        fsize = path.stat().st_size
        sizes.append(fsize)

        min_size = MIN_SIZE_GZ if path.suffix == ".gz" else MIN_SIZE_ZIP
        if fsize < min_size:
            tiny.append((date_str, fsize))

        files_to_check.append((date_str, path))

    # Sample integrity check
    import random
    random.seed(42)
    sample_size = max(3, int(len(files_to_check) * sample_rate))
    sample = random.sample(files_to_check, min(sample_size, len(files_to_check)))

    for date_str, path in sample:
        if path.suffix == ".gz":
            err = check_gz(path)
        else:
            err = check_zip(path)
        if err:
            corrupt.append((date_str, err))

    return {
        "present": present_count,
        "missing": missing,
        "corrupt": corrupt,
        "tiny": tiny,
        "sizes": sizes,
        "integrity_checked": len(sample),
    }


def main():
    t0 = time.time()
    dates = date_range(START_DATE, END_DATE)
    total_days = len(dates)
    all_dates_set = set(dates)

    print("=" * 80)
    print(f"  RAW DATA VALIDATION")
    print(f"  Date range: {START_DATE} to {END_DATE} ({total_days} days)")
    print(f"  Symbols:    {', '.join(SYMBOLS)}")
    print(f"  Data dir:   {DATA_DIR.resolve()}")
    print("=" * 80)
    print()

    # First pass: quick overview of what directories exist
    print("─" * 80)
    print("PHASE 1: Directory structure check")
    print("─" * 80)
    for symbol in SYMBOLS:
        sym_dir = DATA_DIR / symbol
        if not sym_dir.exists():
            print(f"  ✗ {symbol}: directory missing!")
            continue
        exchanges = sorted([d.name for d in sym_dir.iterdir() if d.is_dir()])
        print(f"  {symbol}: {', '.join(exchanges)}")
    print()

    # Second pass: count files per source
    print("─" * 80)
    print("PHASE 2: File count & gap analysis")
    print("─" * 80)

    grand_total_files = 0
    grand_total_missing = 0
    grand_total_size = 0
    all_issues = []

    for si, symbol in enumerate(SYMBOLS):
        print(f"\n{'━' * 80}")
        print(f"  [{si+1}/{len(SYMBOLS)}] {symbol}")
        print(f"{'━' * 80}")

        sources = build_all_sources(symbol, dates)
        sym_files = 0
        sym_missing = 0
        sym_size = 0

        for source_name, file_iter in sources:
            result = validate_source(symbol, source_name, file_iter, all_dates_set)

            present = result["present"]
            n_missing = len(result["missing"])
            total_size = sum(result["sizes"]) if result["sizes"] else 0

            sym_files += present
            sym_missing += n_missing
            sym_size += total_size

            # Determine status
            if present == 0 and n_missing == total_days:
                # Entire source missing — likely not downloaded for this symbol
                status = "NOT DOWNLOADED"
                icon = "⊘"
            elif n_missing == 0:
                status = "COMPLETE"
                icon = "✓"
            else:
                status = f"GAPS: {n_missing} missing"
                icon = "✗"

            # Only print non-trivial sources (skip "NOT DOWNLOADED" for brevity unless it's a primary source)
            is_primary = source_name in ("bybit/futures", "bybit/spot", "binance/futures/trades",
                                          "binance/spot/trades", "okx/futures", "okx/spot")

            if present > 0 or is_primary:
                avg_size = format_size(total_size // present) if present > 0 else "N/A"
                min_size = format_size(min(result["sizes"])) if result["sizes"] else "N/A"
                max_size = format_size(max(result["sizes"])) if result["sizes"] else "N/A"

                print(f"\n  {icon} {source_name}")
                print(f"    Files: {present}/{total_days}  |  Status: {status}")
                if present > 0:
                    print(f"    Size:  total={format_size(total_size)}  avg={avg_size}  min={min_size}  max={max_size}")

                # Show gap ranges if any
                if 0 < n_missing < total_days:
                    gaps = find_gap_ranges(result["missing"])
                    if len(gaps) <= 10:
                        gap_strs = [f"{s} to {e}" if s != e else s for s, e in gaps]
                        print(f"    Gaps:  {'; '.join(gap_strs)}")
                    else:
                        print(f"    Gaps:  {len(gaps)} gap ranges (first: {gaps[0][0]} to {gaps[0][1]}, last: {gaps[-1][0]} to {gaps[-1][1]})")

                    all_issues.append((symbol, source_name, n_missing, gaps))

                # Show tiny files
                if result["tiny"]:
                    print(f"    ⚠ Tiny files ({len(result['tiny'])}): ", end="")
                    for d, sz in result["tiny"][:5]:
                        print(f"{d}({sz}B) ", end="")
                    if len(result["tiny"]) > 5:
                        print(f"... +{len(result['tiny'])-5} more", end="")
                    print()
                    all_issues.append((symbol, source_name + " [tiny]", len(result["tiny"]), []))

        grand_total_files += sym_files
        grand_total_missing += sym_missing
        grand_total_size += sym_size

        elapsed = time.time() - t0
        print(f"\n  Summary: {sym_files} files, {format_size(sym_size)} total, "
              f"{sym_missing} missing  [{elapsed:.1f}s elapsed]")

    # Phase 3: Integrity spot-check
    print(f"\n\n{'─' * 80}")
    print("PHASE 3: Archive integrity spot-check (5% sample per source)")
    print("─" * 80)

    integrity_issues = []
    checked_total = 0

    for si, symbol in enumerate(SYMBOLS):
        sources = build_all_sources(symbol, dates)
        sym_checked = 0
        sym_corrupt = 0

        for source_name, file_iter in sources:
            result = validate_source_with_integrity(symbol, source_name, file_iter)
            n_checked = result.get("integrity_checked", 0)
            sym_checked += n_checked
            checked_total += n_checked

            if result["corrupt"]:
                sym_corrupt += len(result["corrupt"])
                for d, err in result["corrupt"]:
                    integrity_issues.append((symbol, source_name, d, err))

        elapsed = time.time() - t0
        status = "✓ OK" if sym_corrupt == 0 else f"✗ {sym_corrupt} corrupt"
        print(f"  {symbol}: checked {sym_checked} files — {status}  [{elapsed:.1f}s]")

    # ── Grand Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n\n{'=' * 80}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Total files present:    {grand_total_files:,}")
    print(f"  Total size:             {format_size(grand_total_size)}")
    print(f"  Total missing:          {grand_total_missing:,}")
    print(f"  Integrity checked:      {checked_total:,}")
    print(f"  Corrupt files found:    {len(integrity_issues)}")
    print(f"  Elapsed:                {elapsed:.1f}s")
    print()

    if all_issues:
        print("  ISSUES FOUND:")
        for symbol, source, count, gaps in all_issues:
            print(f"    - {symbol}/{source}: {count} issues")
    else:
        print("  ✓ No issues found — all data complete and valid!")

    if integrity_issues:
        print("\n  CORRUPT FILES:")
        for symbol, source, date, err in integrity_issues:
            print(f"    - {symbol}/{source}/{date}: {err}")

    print()


if __name__ == "__main__":
    main()
