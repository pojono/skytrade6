#!/usr/bin/env python3
"""
Convert symbol=ALL JSONL ticker data to Parquet.

Input:  data_all/{exchange}/{stream}/{stream}_{date}_hr{HH}.jsonl.gz
Output: data_all/{exchange}/{stream}.parquet

Streams:
  - binance/ticker:      Binance 24hr ticker (rest, linear, ~679 symbols/snapshot)
  - binance/fundingRate:  Binance funding rate (rest, linear, ~688 symbols/snapshot)
  - bybit/ticker:        Bybit ticker (rest, linear, ~649 symbols/snapshot)

Each JSONL line is one API snapshot containing ALL symbols.
We flatten to one row per (ts, symbol) with typed columns.

Incremental mode (default):
  - Tracks processed files in data_all/{exchange}/{stream}.manifest.json
  - Only processes new JSONL files not in the manifest
  - Appends new row groups to existing parquet
  - Skips files with .partial marker (incomplete hours)

Full rebuild: python build_ticker_all_parquet.py --full
"""
import argparse
import gzip
import json
import time
import sys
from pathlib import Path
from glob import glob

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path("data_all")

# ── Column type definitions ──────────────────────────────────────────────────

BINANCE_TICKER_FLOAT_COLS = [
    "priceChange", "priceChangePercent", "weightedAvgPrice",
    "lastPrice", "lastQty", "openPrice", "highPrice", "lowPrice",
    "volume", "quoteVolume",
]
BINANCE_TICKER_INT_COLS = [
    "openTime", "closeTime", "firstId", "lastId", "count",
]

BINANCE_FUNDING_FLOAT_COLS = [
    "markPrice", "indexPrice", "estimatedSettlePrice",
    "lastFundingRate", "interestRate",
]
BINANCE_FUNDING_INT_COLS = [
    "nextFundingTime", "time",
]

BYBIT_TICKER_FLOAT_COLS = [
    "lastPrice", "indexPrice", "markPrice",
    "prevPrice24h", "price24hPcnt", "highPrice24h", "lowPrice24h",
    "prevPrice1h", "openInterest", "openInterestValue",
    "turnover24h", "volume24h", "fundingRate",
    "ask1Size", "bid1Price", "ask1Price", "bid1Size",
]
BYBIT_TICKER_INT_COLS = [
    "nextFundingTime", "deliveryTime",
]
BYBIT_TICKER_STR_COLS = [
    "fundingIntervalHour", "fundingCap",
]


def extract_binance_ticker(rec):
    """Extract rows from a Binance ticker snapshot."""
    ts = rec["ts"]
    rows = []
    for item in rec["result"]:
        row = {"ts": ts, "symbol": item["symbol"]}
        for col in BINANCE_TICKER_FLOAT_COLS:
            row[col] = item.get(col)
        for col in BINANCE_TICKER_INT_COLS:
            row[col] = item.get(col)
        rows.append(row)
    return rows


def extract_binance_funding(rec):
    """Extract rows from a Binance fundingRate snapshot."""
    ts = rec["ts"]
    rows = []
    for item in rec["result"]:
        row = {"ts": ts, "symbol": item["symbol"]}
        for col in BINANCE_FUNDING_FLOAT_COLS:
            row[col] = item.get(col)
        for col in BINANCE_FUNDING_INT_COLS:
            row[col] = item.get(col)
        rows.append(row)
    return rows


def extract_bybit_ticker(rec):
    """Extract rows from a Bybit ticker snapshot."""
    ts = rec["ts"]
    inner = rec["result"]["result"]["list"]
    rows = []
    for item in inner:
        row = {"ts": ts, "symbol": item["symbol"]}
        for col in BYBIT_TICKER_FLOAT_COLS:
            row[col] = item.get(col)
        for col in BYBIT_TICKER_INT_COLS:
            row[col] = item.get(col)
        for col in BYBIT_TICKER_STR_COLS:
            row[col] = item.get(col)
        rows.append(row)
    return rows


def cast_columns(df, float_cols, int_cols):
    """Cast string columns to numeric types."""
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def load_manifest(manifest_path):
    """Load the set of already-processed files from manifest."""
    if manifest_path.exists():
        with open(manifest_path) as f:
            data = json.load(f)
        return set(data.get("processed_files", []))
    return set()


def save_manifest(manifest_path, processed_files):
    """Save the set of processed files to manifest."""
    with open(manifest_path, "w") as f:
        json.dump({"processed_files": sorted(processed_files), "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f, indent=2)


def process_stream(exchange, stream, extractor, float_cols, int_cols, str_cols=None, full_rebuild=False):
    """Process JSONL files for one stream into Parquet. Supports incremental append."""
    input_dir = DATA_DIR / exchange / stream
    output_file = DATA_DIR / exchange / f"{stream}.parquet"
    manifest_file = DATA_DIR / exchange / f"{stream}.manifest.json"

    all_files = sorted(glob(str(input_dir / f"{stream}_*.jsonl.gz")))
    if not all_files:
        print(f"  No files found in {input_dir}")
        return 0

    # Filter out files with .partial marker
    files_with_partial = []
    for fp in all_files:
        if Path(f"{fp}.partial").exists():
            files_with_partial.append(Path(fp).name)
    clean_files = [fp for fp in all_files if not Path(f"{fp}.partial").exists()]

    # Incremental: only process files not in manifest
    if full_rebuild:
        processed = set()
        # Remove existing parquet for full rebuild
        if output_file.exists():
            output_file.unlink()
    else:
        processed = load_manifest(manifest_file)

    new_files = [fp for fp in clean_files if Path(fp).name not in processed]

    print(f"\n{'='*70}")
    print(f"  {exchange}/{stream}")
    print(f"  Total JSONL files:  {len(all_files)}")
    print(f"  Partial (skipped):  {len(files_with_partial)}")
    if files_with_partial:
        for pf in files_with_partial:
            print(f"    ⏳ {pf}")
    print(f"  Already processed:  {len(processed)}")
    print(f"  New to process:     {len(new_files)}")
    print(f"  Output: {output_file}")
    mode = "FULL REBUILD" if full_rebuild else "INCREMENTAL"
    print(f"  Mode: {mode}")
    print(f"{'='*70}")

    if not new_files:
        print(f"  ✓ Nothing new to process")
        if output_file.exists():
            out_size = output_file.stat().st_size / (1024 * 1024)
            meta = pq.read_metadata(str(output_file))
            print(f"    Existing parquet: {meta.num_rows:,} rows, {out_size:,.1f} MB")
        return 0

    t_start = time.time()
    total_snapshots = 0
    total_rows = 0
    all_rows = []
    newly_processed = set()

    # For incremental: read existing schema so we append with matching schema
    existing_schema = None
    if not full_rebuild and output_file.exists():
        existing_schema = pq.read_schema(str(output_file))

    # Process file by file with chunked flushing to keep memory bounded
    FLUSH_EVERY = 6  # flush every 6 files (~6 hours)
    writer = None
    schema = existing_schema

    # For incremental, we append to a temp file then merge
    if not full_rebuild and output_file.exists():
        append_file = DATA_DIR / exchange / f"{stream}.parquet.new"
    else:
        append_file = output_file

    for fi, fpath in enumerate(new_files):
        fname = Path(fpath).name
        file_rows = []
        file_snapshots = 0
        file_errors = 0

        with gzip.open(fpath, "rt") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    rows = extractor(rec)
                    file_rows.extend(rows)
                    file_snapshots += 1
                except Exception as e:
                    file_errors += 1
                    if file_errors <= 3:
                        print(f"    WARN: {fname} line {line_no}: {e}")

        all_rows.extend(file_rows)
        total_snapshots += file_snapshots
        total_rows += len(file_rows)
        newly_processed.add(fname)

        elapsed = time.time() - t_start
        rate = total_rows / elapsed if elapsed > 0 else 0
        print(
            f"  [{fi+1:3d}/{len(new_files)}] {fname}: "
            f"{file_snapshots:,} snaps, {len(file_rows):,} rows "
            f"(total: {total_rows:,.0f} rows, {elapsed:.0f}s, {rate:,.0f} rows/s)"
            + (f" [{file_errors} errors]" if file_errors else "")
        )

        # Flush chunk to parquet periodically
        if (fi + 1) % FLUSH_EVERY == 0 or (fi + 1) == len(new_files):
            if all_rows:
                chunk_df = pd.DataFrame(all_rows)
                chunk_df = cast_columns(chunk_df, float_cols, int_cols)
                # Convert ts to datetime
                chunk_df["ts"] = pd.to_datetime(chunk_df["ts"], unit="ms", utc=True)
                chunk_df.sort_values(["ts", "symbol"], inplace=True)

                table = pa.Table.from_pandas(chunk_df, preserve_index=False)

                if writer is None:
                    if schema is None:
                        schema = table.schema
                    writer = pq.ParquetWriter(
                        str(append_file),
                        schema,
                        compression="zstd",
                        compression_level=3,
                    )

                # Ensure schema matches (handle evolving symbol sets)
                table = table.cast(schema)
                writer.write_table(table)
                all_rows = []

                print(
                    f"    → Flushed chunk to parquet "
                    f"({total_rows:,.0f} rows total so far)"
                )

    if writer is not None:
        writer.close()

    # For incremental: merge old + new parquet files
    if append_file != output_file and append_file.exists():
        print(f"  Merging old + new parquet...")
        old_table = pq.read_table(str(output_file))
        new_table = pq.read_table(str(append_file))
        merged = pa.concat_tables([old_table, new_table])
        pq.write_table(merged, str(output_file), compression="zstd", compression_level=3)
        append_file.unlink()
        del old_table, new_table, merged
        print(f"    → Merged successfully")

    # Update manifest
    all_processed = processed | newly_processed
    save_manifest(manifest_file, all_processed)

    elapsed = time.time() - t_start
    if output_file.exists():
        out_size = output_file.stat().st_size / (1024 * 1024)
        meta = pq.read_metadata(str(output_file))
        total_in_pq = meta.num_rows
    else:
        out_size = 0
        total_in_pq = 0

    print(f"\n  ✓ {exchange}/{stream} complete")
    print(f"    New snapshots:  {total_snapshots:,}")
    print(f"    New rows:       {total_rows:,}")
    print(f"    Total in pq:    {total_in_pq:,}")
    print(f"    Tracked files:  {len(all_processed)}")
    print(f"    Output:         {output_file} ({out_size:,.1f} MB)")
    print(f"    Time:           {elapsed:.1f}s")
    if elapsed > 0:
        print(f"    Rate:           {total_rows / elapsed:,.0f} rows/s")

    return total_rows


def main():
    parser = argparse.ArgumentParser(description="Convert symbol=ALL JSONL to Parquet")
    parser.add_argument("--full", action="store_true", help="Full rebuild (ignore manifest, overwrite parquet)")
    parser.add_argument("--stream", type=str, default=None, help="Only process specific stream, e.g. 'binance/ticker'")
    args = parser.parse_args()

    mode = "FULL REBUILD" if args.full else "INCREMENTAL"
    print("=" * 70)
    print(f"CONVERT symbol=ALL JSONL → PARQUET  [{mode}]")
    print("=" * 70)
    t0 = time.time()

    results = {}

    streams = [
        ("binance", "ticker", extract_binance_ticker, BINANCE_TICKER_FLOAT_COLS, BINANCE_TICKER_INT_COLS, None),
        ("binance", "fundingRate", extract_binance_funding, BINANCE_FUNDING_FLOAT_COLS, BINANCE_FUNDING_INT_COLS, None),
        ("bybit", "ticker", extract_bybit_ticker, BYBIT_TICKER_FLOAT_COLS, BYBIT_TICKER_INT_COLS, BYBIT_TICKER_STR_COLS),
    ]

    for exchange, stream, extractor, fcols, icols, scols in streams:
        key = f"{exchange}/{stream}"
        if args.stream and args.stream != key:
            continue
        n = process_stream(exchange, stream, extractor, fcols, icols, scols, full_rebuild=args.full)
        results[key] = n

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"ALL CONVERSIONS COMPLETE  [{elapsed:.0f}s]")
    print(f"{'='*70}")
    for stream, rows in results.items():
        pq_path = DATA_DIR / stream.replace("/", "/") 
        # find the parquet
        exchange, sname = stream.split("/")
        pq_file = DATA_DIR / exchange / f"{sname}.parquet"
        if pq_file.exists():
            sz = pq_file.stat().st_size / (1024 * 1024)
            print(f"  {stream}: {rows:,} rows, {sz:,.1f} MB")
        else:
            print(f"  {stream}: {rows} rows")
    print(f"{'='*70}")

    # Quick schema summary
    print(f"\nParquet schema summary:")
    for exchange in ["binance", "bybit"]:
        for sname in ["ticker", "fundingRate"]:
            pq_file = DATA_DIR / exchange / f"{sname}.parquet"
            if pq_file.exists():
                pf = pq.read_metadata(str(pq_file))
                schema = pq.read_schema(str(pq_file))
                print(f"\n  {exchange}/{sname}: {pf.num_rows:,} rows, {pf.num_row_groups} row groups")
                for i in range(schema.__len__()):
                    print(f"    {schema.field(i).name}: {schema.field(i).type}")


if __name__ == "__main__":
    main()
