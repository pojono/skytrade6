#!/usr/bin/env python3
"""
Compress existing uncompressed data files to gzip and remove originals.

Targets:
  - Bybit orderbook .jsonl  -> .jsonl.gz
  - Trades .csv (all exchanges: bybit/, binance/, okx/)  -> .csv.gz
    Matches: *_trades.csv, *_trades_spot.csv, *_aggTrades.csv, *_aggTrades_spot.csv,
             *_bookDepth.csv, *_bookTicker.csv

Usage:
  python3 compress_orderbooks.py                # dry-run (show what would be done)
  python3 compress_orderbooks.py --run          # actually compress and delete
  python3 compress_orderbooks.py --run -j 4     # use 4 parallel workers
"""

import argparse
import functools
import gzip
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

print = functools.partial(print, flush=True)

DATALAKE_DIR = Path(__file__).resolve().parent
BYBIT_DIR = DATALAKE_DIR / "bybit"

# Patterns for uncompressed files that should be gzipped
# Trades and large bulk data across all exchanges
COMPRESS_PATTERNS = [
    "*_trades.csv",
    "*_trades_spot.csv",
    "*_aggTrades.csv",
    "*_aggTrades_spot.csv",
    "*_bookDepth.csv",
    "*_bookTicker.csv",
]


def compress_one(file_path: Path, base_dir: Path) -> tuple[str, int, int, str]:
    """Compress a single file to .gz. Returns (path, old_size, new_size, status)."""
    if file_path.suffix == ".jsonl":
        gz_path = file_path.with_suffix(".jsonl.gz")
    else:
        gz_path = Path(str(file_path) + ".gz")
    rel = str(file_path.relative_to(base_dir))

    if gz_path.exists() and gz_path.stat().st_size > 0:
        return (rel, 0, 0, "gz already exists, skipped")

    old_size = file_path.stat().st_size
    try:
        tmp = gz_path.with_suffix(".gz.tmp")
        with open(file_path, "rb") as f_in, gzip.open(tmp, "wb", compresslevel=6) as f_out:
            while True:
                chunk = f_in.read(8 * 1024 * 1024)  # 8 MB chunks
                if not chunk:
                    break
                f_out.write(chunk)
        tmp.rename(gz_path)
        new_size = gz_path.stat().st_size
        file_path.unlink()
        return (rel, old_size, new_size, "ok")
    except Exception as e:
        # Clean up on failure
        tmp = gz_path.with_suffix(".gz.tmp")
        tmp.unlink(missing_ok=True)
        return (rel, old_size, 0, f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Compress uncompressed orderbook and trades files to gzip"
    )
    parser.add_argument("--run", action="store_true", help="Actually compress and delete (default: dry-run)")
    parser.add_argument("-j", "--workers", type=int, default=2, help="Parallel workers (default: 2)")
    args = parser.parse_args()

    # Collect all files to compress
    files = []
    base_dirs = {}  # map file -> base_dir for relative path display

    # 1. Bybit orderbook .jsonl files
    if BYBIT_DIR.exists():
        for f in sorted(BYBIT_DIR.rglob("*_orderbook.jsonl")):
            files.append(f)
            base_dirs[f] = BYBIT_DIR

    # 2. Trades/bulk CSV files across all exchanges
    for exchange in ["bybit", "binance", "okx"]:
        exchange_dir = DATALAKE_DIR / exchange
        if not exchange_dir.exists():
            continue
        for pattern in COMPRESS_PATTERNS:
            for f in sorted(exchange_dir.rglob(pattern)):
                files.append(f)
                base_dirs[f] = exchange_dir

    if not files:
        print("No uncompressed files found.")
        return

    total_old = sum(f.stat().st_size for f in files)
    print(f"Found {len(files)} uncompressed files ({total_old / (1024**3):.1f} GB)")

    # Count by type
    orderbook_count = sum(1 for f in files if f.suffix == ".jsonl")
    csv_count = len(files) - orderbook_count
    if orderbook_count:
        print(f"  Orderbook .jsonl: {orderbook_count}")
    if csv_count:
        print(f"  Trades/bulk .csv: {csv_count}")

    if not args.run:
        print(f"\nDry-run mode. Use --run to compress and delete originals.")
        for f in files[:15]:
            sz = f.stat().st_size / (1024**2)
            base = base_dirs[f]
            print(f"  {base.name}/{f.relative_to(base)}  ({sz:.0f} MB)")
        if len(files) > 15:
            print(f"  ... and {len(files) - 15} more")
        return

    print(f"Compressing with {args.workers} workers...\n")

    done = 0
    saved_total = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(compress_one, f, base_dirs[f]): f for f in files}
        for future in as_completed(futures):
            rel, old_sz, new_sz, status = future.result()
            done += 1
            if status == "ok":
                saved = old_sz - new_sz
                saved_total += saved
                ratio = old_sz / new_sz if new_sz else 0
                print(f"  [{done}/{len(files)}] {rel}  {old_sz/(1024**2):.0f}MB -> {new_sz/(1024**2):.0f}MB  ({ratio:.1f}x)")
            elif "ERROR" in status:
                errors += 1
                print(f"  [{done}/{len(files)}] {rel}  {status}")
            else:
                print(f"  [{done}/{len(files)}] {rel}  {status}")

    print(f"\nDone. {done - errors} compressed, {errors} errors.")
    print(f"Space saved: {saved_total / (1024**3):.1f} GB")


if __name__ == "__main__":
    main()
