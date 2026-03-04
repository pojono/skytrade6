#!/usr/bin/env python3
"""Rank symbols under one dense filter configuration using cached entry records."""

from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dense_filter_sweep import DEFAULT_SYMBOLS, build_symbol_records, load_symbols


OUT_DIR = Path(__file__).resolve().parent / "out"


def avg_net(rows):
    return sum(r.net_pnl_bps for r in rows) / len(rows) if rows else math.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols-file", type=Path, default=DEFAULT_SYMBOLS)
    parser.add_argument("--min-overlap-days", type=int, default=90)
    parser.add_argument("--min-signal-bps", type=float, default=10.0)
    parser.add_argument("--fee-bps-roundtrip", type=float, default=6.0)
    parser.add_argument("--recent-days", type=int, default=45)
    parser.add_argument("--test-days", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ls-threshold", type=float, default=0.15)
    parser.add_argument("--oi-threshold-bps", type=float, default=5.0)
    parser.add_argument("--carry-threshold-bps", type=float, default=0.0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = load_symbols(args.symbols_file)

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                build_symbol_records,
                symbol,
                args.min_overlap_days,
                args.min_signal_bps,
                args.fee_bps_roundtrip,
                args.recent_days,
                args.test_days,
            ): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            records = future.result()
            if not records:
                continue
            raw_train = [r for r in records if r.split == "train"]
            raw_test = [r for r in records if r.split == "test"]
            filt_train = [
                r for r in raw_train
                if r.ls_diff_signed >= args.ls_threshold
                and r.oi_diff_signed_bps >= args.oi_threshold_bps
                and r.carry_diff_signed_bps >= args.carry_threshold_bps
            ]
            filt_test = [
                r for r in raw_test
                if r.ls_diff_signed >= args.ls_threshold
                and r.oi_diff_signed_bps >= args.oi_threshold_bps
                and r.carry_diff_signed_bps >= args.carry_threshold_bps
            ]
            rows.append(
                (
                    symbol,
                    len(raw_train),
                    avg_net(raw_train),
                    len(raw_test),
                    avg_net(raw_test),
                    len(filt_train),
                    avg_net(filt_train),
                    len(filt_test),
                    avg_net(filt_test),
                )
            )

    rows.sort(
        key=lambda row: (
            -(row[8] if not math.isnan(row[8]) else float("-inf")),
            -row[7],
            row[0],
        )
    )

    out_path = OUT_DIR / "dense_filter_rank.csv"
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "symbol",
                "raw_train_signals",
                "raw_train_avg_net_bps",
                "raw_test_signals",
                "raw_test_avg_net_bps",
                "filt_train_signals",
                "filt_train_avg_net_bps",
                "filt_test_signals",
                "filt_test_avg_net_bps",
            ]
        )
        writer.writerows(rows)

    print(f"Ranked symbols: {len(rows)}")
    print(f"Wrote {out_path}")
    print("Top 20 symbols:")
    for row in rows[:20]:
        print(
            f"  {row[0]}: filt_test={row[8]:.4f}bps ({row[7]}) "
            f"filt_train={row[6]:.4f}bps ({row[5]}) raw_test={row[4]:.4f}bps ({row[3]})"
        )


if __name__ == "__main__":
    main()
