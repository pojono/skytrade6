#!/usr/bin/env python3
"""
Validate downloaded Bybit raw data files (futures + spot).

Checks for all 5 symbols over 2024-01-01 to 2026-01-31:
  1. All expected daily .csv.gz files exist
  2. No gaps in date coverage
  3. File sizes are reasonable (not zero or suspiciously small)
  4. Archives are not corrupted (gzip can be opened)

Usage:
  python3 validate_bybit_files.py
  python3 validate_bybit_files.py --data-dir ./data
  python3 validate_bybit_files.py --symbols BTCUSDT ETHUSDT
"""

import argparse
import gzip
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("./data")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
START_DATE = "2024-01-01"
END_DATE = "2026-01-31"
MIN_SIZE = 100  # bytes — anything below is suspicious


def date_range(start: str, end: str) -> list[str]:
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    dates = []
    while s <= e:
        dates.append(s.strftime("%Y-%m-%d"))
        s += timedelta(days=1)
    return dates


def format_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    if n < 1024 ** 3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


def gap_ranges(missing: list[str]) -> list[tuple[str, str]]:
    if not missing:
        return []
    s = sorted(missing)
    ranges, start, prev = [], s[0], s[0]
    for d in s[1:]:
        if (datetime.strptime(d, "%Y-%m-%d") - datetime.strptime(prev, "%Y-%m-%d")).days == 1:
            prev = d
        else:
            ranges.append((start, prev))
            start = prev = d
    ranges.append((start, prev))
    return ranges


def check_gz(path: Path) -> str:
    try:
        with gzip.open(path, "rb") as f:
            data = f.read(4096)
            if len(data) == 0:
                return "empty after decompression"
        return ""
    except Exception as e:
        return str(e)


def validate_source(symbol: str, label: str, base: Path, dates: list[str],
                    filename_fn) -> dict:
    missing, tiny, sizes = [], [], []
    for d in dates:
        p = base / filename_fn(d)
        if not p.exists():
            missing.append(d)
            continue
        sz = p.stat().st_size
        sizes.append(sz)
        if sz < MIN_SIZE:
            tiny.append((d, sz))
    return {"missing": missing, "tiny": tiny, "sizes": sizes,
            "present": len(dates) - len(missing)}


def integrity_sample(symbol: str, base: Path, dates: list[str],
                     filename_fn, rate: float = 0.05) -> list[tuple[str, str]]:
    existing = [(d, base / filename_fn(d)) for d in dates if (base / filename_fn(d)).exists()]
    random.seed(42)
    n = max(3, int(len(existing) * rate))
    sample = random.sample(existing, min(n, len(existing)))
    errors = []
    for d, p in sample:
        err = check_gz(p)
        if err:
            errors.append((d, err))
    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate Bybit raw data files")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    symbols = [s.upper() for s in args.symbols]
    dates = date_range(args.start, args.end)
    total_days = len(dates)
    t0 = time.time()

    print("=" * 70)
    print(f"  BYBIT DATA VALIDATION")
    print(f"  Range: {args.start} → {args.end} ({total_days} days)")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Data dir: {data_dir.resolve()}")
    print("=" * 70)

    grand_files = 0
    grand_size = 0
    grand_missing = 0
    all_ok = True

    for si, symbol in enumerate(symbols):
        elapsed = time.time() - t0
        print(f"\n{'━' * 70}")
        print(f"  [{si+1}/{len(symbols)}] {symbol}  ({elapsed:.1f}s)")
        print(f"{'━' * 70}")

        for label, subdir, fn in [
            ("futures", "bybit/futures", lambda d: f"{symbol}{d}.csv.gz"),
            ("spot",    "bybit/spot",    lambda d: f"{symbol}_{d}.csv.gz"),
        ]:
            base = data_dir / symbol / subdir
            r = validate_source(symbol, label, base, dates, fn)
            n_miss = len(r["missing"])
            total_sz = sum(r["sizes"]) if r["sizes"] else 0
            grand_files += r["present"]
            grand_size += total_sz
            grand_missing += n_miss

            icon = "✓" if n_miss == 0 else "✗"
            status = "COMPLETE" if n_miss == 0 else f"{n_miss} MISSING"
            print(f"\n  {icon} bybit/{label}")
            print(f"    Files: {r['present']}/{total_days}  |  {status}")

            if r["present"] > 0:
                avg = format_size(total_sz // r["present"])
                mn = format_size(min(r["sizes"]))
                mx = format_size(max(r["sizes"]))
                print(f"    Size:  total={format_size(total_sz)}  avg={avg}  min={mn}  max={mx}")

            if 0 < n_miss < total_days:
                all_ok = False
                gaps = gap_ranges(r["missing"])
                if len(gaps) <= 10:
                    for s, e in gaps:
                        print(f"    Gap: {s} → {e}" if s != e else f"    Gap: {s}")
                else:
                    print(f"    {len(gaps)} gap ranges (first: {gaps[0][0]}→{gaps[0][1]}, last: {gaps[-1][0]}→{gaps[-1][1]})")
            elif n_miss == total_days:
                all_ok = False

            if r["tiny"]:
                all_ok = False
                print(f"    ⚠ {len(r['tiny'])} tiny files (<{MIN_SIZE}B):")
                for d, sz in r["tiny"][:5]:
                    print(f"      {d} — {sz} B")
                if len(r["tiny"]) > 5:
                    print(f"      ... +{len(r['tiny'])-5} more")

            # Integrity spot-check
            errors = integrity_sample(symbol, base, dates, fn)
            if errors:
                all_ok = False
                print(f"    ⚠ {len(errors)} corrupt files:")
                for d, err in errors:
                    print(f"      {d}: {err}")
            else:
                n_checked = max(3, int(r["present"] * 0.05))
                n_checked = min(n_checked, r["present"])
                print(f"    Integrity: {n_checked} sampled — OK")

    elapsed = time.time() - t0
    print(f"\n\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Files present:  {grand_files:,}")
    print(f"  Total size:     {format_size(grand_size)}")
    print(f"  Missing:        {grand_missing:,}")
    print(f"  Elapsed:        {elapsed:.1f}s")

    if all_ok and grand_missing == 0:
        print(f"\n  ✓ All Bybit data complete and valid!")
    elif grand_missing > 0:
        print(f"\n  ✗ {grand_missing} files missing — see details above")
    print()


if __name__ == "__main__":
    main()
