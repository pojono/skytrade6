#!/usr/bin/env python3
"""
Validate downloaded Bybit raw data AND derived parquet files.

Phase 1 — Raw .csv.gz files (futures + spot):
  1. All expected daily files exist for 2024-01-01 to 2026-01-31
  2. No gaps in date coverage
  3. File sizes are reasonable
  4. Archives are not corrupted (gzip spot-check)

Phase 2 — Parquet files (trades + OHLCV):
  1. All expected daily parquet files exist (matching raw file dates)
  2. No gaps in date coverage
  3. Schema consistency — same columns and dtypes across all files
  4. Files are not corrupted (pyarrow spot-check)
  5. Reasonable file sizes

Usage:
  python3 validate_bybit_files.py
  python3 validate_bybit_files.py --data-dir ./data --parquet-dir ./parquet
  python3 validate_bybit_files.py --symbols BTCUSDT ETHUSDT
  python3 validate_bybit_files.py --skip-parquet
"""

import argparse
import gzip
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("./data")
PARQUET_DIR = Path("./parquet")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
START_DATE = "2023-01-01"
END_DATE = "2026-01-31"
MIN_SIZE_RAW = 100      # bytes — raw .csv.gz
MIN_SIZE_PARQUET = 200  # bytes — parquet files

OHLCV_INTERVALS = ["1m", "5m", "15m", "1h"]
BYBIT_SOURCES = ["bybit_futures", "bybit_spot"]

# Expected schemas (column_name -> pyarrow type string)
EXPECTED_TRADES_SCHEMA = {
    "timestamp_us": "int64",
    "price": "double",
    "quantity": "double",
    "quote_quantity": "double",
    "side": "int8",
    "trade_id": "string",
}

EXPECTED_OHLCV_SCHEMA = {
    "timestamp_us": "int64",
    "open": "double",
    "high": "double",
    "low": "double",
    "close": "double",
    "volume": "double",
    "quote_volume": "double",
    "trade_count": "int64",
    "buy_volume": "double",
    "sell_volume": "double",
    "vwap": "double",
}


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
                    filename_fn, min_size: int = 100) -> dict:
    missing, tiny, sizes = [], [], []
    for d in dates:
        p = base / filename_fn(d)
        if not p.exists():
            missing.append(d)
            continue
        sz = p.stat().st_size
        sizes.append(sz)
        if sz < min_size:
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


# ── Parquet helpers ─────────────────────────────────────────────────────────

def check_parquet(path: Path) -> str:
    """Try to open a parquet file and read metadata + first row group. Returns error or empty."""
    try:
        pf = pq.ParquetFile(path)
        meta = pf.metadata
        if meta.num_rows == 0:
            return "0 rows"
        # Read first row group to verify data is readable
        pf.read_row_group(0, columns=[pf.schema.names[0]])
        return ""
    except Exception as e:
        return str(e)


def get_parquet_schema(path: Path) -> dict[str, str] | None:
    """Read parquet schema as {col_name: type_string}. Returns None on error."""
    try:
        schema = pq.read_schema(path)
        return {field.name: str(field.type) for field in schema}
    except Exception:
        return None


def schema_str(schema: dict[str, str]) -> str:
    return ", ".join(f"{k}:{v}" for k, v in schema.items())


def parquet_integrity_sample(files: list[Path], rate: float = 0.05) -> list[tuple[str, str]]:
    """Spot-check a sample of parquet files for corruption."""
    random.seed(42)
    n = max(3, int(len(files) * rate))
    sample = random.sample(files, min(n, len(files)))
    errors = []
    for p in sample:
        err = check_parquet(p)
        if err:
            errors.append((p.stem, err))
    return errors


def validate_parquet_source(base_dir: Path, dates: list[str]) -> dict:
    """Validate a directory of {YYYY-MM-DD}.parquet files against expected dates."""
    if not base_dir.exists():
        return {"present": 0, "missing": dates[:], "tiny": [], "sizes": [],
                "schema_issues": [], "corrupt": [], "exists": False}

    all_files = sorted(base_dir.glob("*.parquet"))
    existing_dates = {f.stem for f in all_files}

    missing = [d for d in dates if d not in existing_dates]
    extra = existing_dates - set(dates)

    tiny, sizes = [], []
    for f in all_files:
        sz = f.stat().st_size
        sizes.append(sz)
        if sz < MIN_SIZE_PARQUET:
            tiny.append((f.stem, sz))

    # Schema consistency: check all files have the same schema
    schema_issues = []
    ref_schema = None
    for f in all_files:
        s = get_parquet_schema(f)
        if s is None:
            schema_issues.append((f.stem, "unreadable schema"))
            continue
        if ref_schema is None:
            ref_schema = s
        elif s != ref_schema:
            diffs = []
            for k in set(list(s.keys()) + list(ref_schema.keys())):
                if k not in ref_schema:
                    diffs.append(f"+{k}:{s[k]}")
                elif k not in s:
                    diffs.append(f"-{k}:{ref_schema[k]}")
                elif s[k] != ref_schema[k]:
                    diffs.append(f"{k}:{ref_schema[k]}→{s[k]}")
            schema_issues.append((f.stem, "; ".join(diffs)))

    # Integrity spot-check
    corrupt = parquet_integrity_sample(all_files) if all_files else []

    return {
        "present": len(all_files),
        "missing": missing,
        "extra": sorted(extra),
        "tiny": tiny,
        "sizes": sizes,
        "schema_issues": schema_issues,
        "corrupt": corrupt,
        "ref_schema": ref_schema,
        "integrity_checked": min(max(3, int(len(all_files) * 0.05)), len(all_files)),
        "exists": True,
    }


def print_parquet_result(label: str, r: dict, expected_schema: dict | None,
                         total_expected: int) -> bool:
    """Print validation result for one parquet source. Returns True if all OK."""
    ok = True

    if not r["exists"]:
        print(f"\n  ⊘ {label}")
        print(f"    Directory not found")
        return False

    n_miss = len(r["missing"])
    icon = "✓" if n_miss == 0 else "✗"
    status = "COMPLETE" if n_miss == 0 else f"{n_miss} MISSING"
    print(f"\n  {icon} {label}")
    print(f"    Files: {r['present']}/{total_expected}  |  {status}")

    if r["present"] > 0:
        total_sz = sum(r["sizes"])
        avg = format_size(total_sz // r["present"])
        mn = format_size(min(r["sizes"]))
        mx = format_size(max(r["sizes"]))
        print(f"    Size:  total={format_size(total_sz)}  avg={avg}  min={mn}  max={mx}")

    if 0 < n_miss < total_expected:
        ok = False
        gaps = gap_ranges(r["missing"])
        if len(gaps) <= 5:
            for s, e in gaps:
                print(f"    Gap: {s} → {e}" if s != e else f"    Gap: {s}")
        else:
            print(f"    {len(gaps)} gap ranges (first: {gaps[0][0]}→{gaps[0][1]}, last: {gaps[-1][0]}→{gaps[-1][1]})")
    elif n_miss == total_expected:
        ok = False

    if r["tiny"]:
        ok = False
        print(f"    ⚠ {len(r['tiny'])} tiny files (<{MIN_SIZE_PARQUET}B):")
        for d, sz in r["tiny"][:5]:
            print(f"      {d} — {sz} B")
        if len(r["tiny"]) > 5:
            print(f"      ... +{len(r['tiny'])-5} more")

    # Schema check
    if r["ref_schema"] and expected_schema:
        if r["ref_schema"] != expected_schema:
            ok = False
            print(f"    ⚠ Schema mismatch vs expected!")
            print(f"      Got:      {schema_str(r['ref_schema'])}")
            print(f"      Expected: {schema_str(expected_schema)}")
        else:
            print(f"    Schema: OK ({len(expected_schema)} cols)")

    if r["schema_issues"]:
        ok = False
        print(f"    ⚠ {len(r['schema_issues'])} files with inconsistent schema:")
        for d, diff in r["schema_issues"][:5]:
            print(f"      {d}: {diff}")
        if len(r["schema_issues"]) > 5:
            print(f"      ... +{len(r['schema_issues'])-5} more")

    if r["corrupt"]:
        ok = False
        print(f"    ⚠ {len(r['corrupt'])} corrupt files:")
        for d, err in r["corrupt"]:
            print(f"      {d}: {err}")
    else:
        n_chk = r.get("integrity_checked", 0)
        if n_chk > 0:
            print(f"    Integrity: {n_chk} sampled — OK")

    return ok


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate Bybit raw + parquet data")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--parquet-dir", default="./parquet")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--skip-parquet", action="store_true",
                        help="Skip parquet validation")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    parquet_dir = Path(args.parquet_dir)
    symbols = [s.upper() for s in args.symbols]
    dates = date_range(args.start, args.end)
    total_days = len(dates)
    do_parquet = not args.skip_parquet
    t0 = time.time()

    if do_parquet and not HAS_PYARROW:
        print("⚠ pyarrow not installed — skipping parquet validation")
        do_parquet = False

    print("=" * 70)
    print(f"  BYBIT DATA VALIDATION")
    print(f"  Range: {args.start} → {args.end} ({total_days} days)")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Raw dir:     {data_dir.resolve()}")
    if do_parquet:
        print(f"  Parquet dir: {parquet_dir.resolve()}")
    print("=" * 70)

    # ── Phase 1: Raw .csv.gz ────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  PHASE 1: Raw .csv.gz files")
    print(f"{'─' * 70}")

    grand_raw_files = 0
    grand_raw_size = 0
    grand_raw_missing = 0
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
            r = validate_source(symbol, label, base, dates, fn, MIN_SIZE_RAW)
            n_miss = len(r["missing"])
            total_sz = sum(r["sizes"]) if r["sizes"] else 0
            grand_raw_files += r["present"]
            grand_raw_size += total_sz
            grand_raw_missing += n_miss

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
                print(f"    ⚠ {len(r['tiny'])} tiny files (<{MIN_SIZE_RAW}B):")
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

    # ── Phase 2: Parquet files ──────────────────────────────────────────
    grand_pq_files = 0
    grand_pq_size = 0
    grand_pq_missing = 0

    if do_parquet:
        print(f"\n\n{'─' * 70}")
        print(f"  PHASE 2: Parquet files (Bybit only)")
        print(f"{'─' * 70}")

        for si, symbol in enumerate(symbols):
            elapsed = time.time() - t0
            print(f"\n{'━' * 70}")
            print(f"  [{si+1}/{len(symbols)}] {symbol}  ({elapsed:.1f}s)")
            print(f"{'━' * 70}")

            sym_pq_dir = parquet_dir / symbol

            # Determine expected parquet dates from raw files that actually exist
            # (parquet may cover a subset of the raw date range)
            for source in BYBIT_SOURCES:
                # Figure out which raw dates exist for this source
                if source == "bybit_futures":
                    raw_base = data_dir / symbol / "bybit" / "futures"
                    raw_files = sorted(raw_base.glob("*.csv.gz")) if raw_base.exists() else []
                else:
                    raw_base = data_dir / symbol / "bybit" / "spot"
                    raw_files = sorted(raw_base.glob("*.csv.gz")) if raw_base.exists() else []

                raw_dates = []
                for f in raw_files:
                    m = re.search(r"(\d{4}-\d{2}-\d{2})", f.name)
                    if m:
                        raw_dates.append(m.group(1))

                # Determine actual parquet date range from what exists on disk
                trades_dir = sym_pq_dir / "trades" / source
                if trades_dir.exists():
                    pq_files = sorted(trades_dir.glob("*.parquet"))
                    if pq_files:
                        pq_first = pq_files[0].stem
                        pq_last = pq_files[-1].stem
                        # Expected dates = raw dates within the parquet range
                        expected_dates = [d for d in raw_dates if pq_first <= d <= pq_last]
                    else:
                        expected_dates = raw_dates
                else:
                    expected_dates = raw_dates

                n_expected = len(expected_dates)
                source_label = source.replace("_", "/")

                # Trades
                r = validate_parquet_source(trades_dir, expected_dates)
                ok = print_parquet_result(
                    f"trades/{source_label}", r, EXPECTED_TRADES_SCHEMA, n_expected)
                if not ok:
                    all_ok = False
                grand_pq_files += r["present"]
                grand_pq_size += sum(r["sizes"]) if r["sizes"] else 0
                grand_pq_missing += len(r["missing"])

                # OHLCV per interval
                for iv in OHLCV_INTERVALS:
                    ohlcv_dir = sym_pq_dir / "ohlcv" / iv / source
                    r = validate_parquet_source(ohlcv_dir, expected_dates)
                    ok = print_parquet_result(
                        f"ohlcv/{iv}/{source_label}", r, EXPECTED_OHLCV_SCHEMA, n_expected)
                    if not ok:
                        all_ok = False
                    grand_pq_files += r["present"]
                    grand_pq_size += sum(r["sizes"]) if r["sizes"] else 0
                    grand_pq_missing += len(r["missing"])

    # ── Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Raw .csv.gz:")
    print(f"    Files present:  {grand_raw_files:,}")
    print(f"    Total size:     {format_size(grand_raw_size)}")
    print(f"    Missing:        {grand_raw_missing:,}")
    if do_parquet:
        print(f"  Parquet (Bybit):")
        print(f"    Files present:  {grand_pq_files:,}")
        print(f"    Total size:     {format_size(grand_pq_size)}")
        print(f"    Missing:        {grand_pq_missing:,}")
    print(f"  Elapsed:          {elapsed:.1f}s")

    if all_ok and grand_raw_missing == 0 and grand_pq_missing == 0:
        print(f"\n  ✓ All Bybit data (raw + parquet) complete and valid!")
    else:
        issues = grand_raw_missing + grand_pq_missing
        print(f"\n  ✗ {issues} total issues — see details above")
    print()


if __name__ == "__main__":
    main()
