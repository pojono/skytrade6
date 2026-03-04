#!/usr/bin/env python3
"""Run a fast pilot sweep across a small symbol basket before scaling up."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from cross_exchange_edge_scan import analyze_symbol, collect_dates, BINANCE, BYBIT, OUT_DIR


def eligible_symbols(min_overlap_days: int) -> list[str]:
    binance_symbols = {p.name for p in BINANCE.iterdir() if p.is_dir()}
    bybit_symbols = {p.name for p in BYBIT.iterdir() if p.is_dir()}
    common = sorted(binance_symbols & bybit_symbols)
    picked: list[tuple[int, str]] = []
    for symbol in common:
        overlap = len(
            collect_dates(BINANCE / symbol, "kline_1m.csv")
            & collect_dates(BYBIT / symbol, "kline_1m.csv")
        )
        if overlap >= min_overlap_days:
            picked.append((overlap, symbol))
    picked.sort(key=lambda item: (-item[0], item[1]))
    return [symbol for _, symbol in picked]


def run_config(
    symbols: list[str],
    min_overlap_days: int,
    min_signal_bps: float,
    fee_bps_roundtrip: float,
    test_days: int,
    workers: int,
    executor_kind: str,
    recent_days: int,
) -> tuple[float, float, int, float, int]:
    results = []
    executor_cls = ThreadPoolExecutor if executor_kind == "thread" else ProcessPoolExecutor
    with executor_cls(max_workers=workers) as pool:
        futures = {
            pool.submit(
                analyze_symbol,
                symbol,
                min_overlap_days,
                min_signal_bps,
                fee_bps_roundtrip,
                test_days,
                recent_days,
            ): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    total_test_signals = sum(row.test.signal_count for row in results)
    total_test_net = sum(row.test.net_sum for row in results)
    avg_test_net = total_test_net / total_test_signals if total_test_signals else float("nan")
    profitable = sum(1 for row in results if row.test.signal_count and row.test.avg_net_pnl_bps > 0)
    return min_signal_bps, fee_bps_roundtrip, len(results), avg_test_net, profitable


def parse_csv_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-overlap-days", type=int, default=90)
    parser.add_argument("--pilot-symbols", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--executor", choices=["thread", "process"], default="thread")
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--signal-grid", type=str, default="4,6,8,10,12")
    parser.add_argument("--fee-grid", type=str, default="0,2,4,6")
    parser.add_argument("--recent-days", type=int, default=60)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = eligible_symbols(args.min_overlap_days)[: args.pilot_symbols]
    signal_grid = parse_csv_floats(args.signal_grid)
    fee_grid = parse_csv_floats(args.fee_grid)

    rows = []
    for min_signal_bps in signal_grid:
        for fee_bps_roundtrip in fee_grid:
            rows.append(
                run_config(
                    symbols=symbols,
                    min_overlap_days=args.min_overlap_days,
                    min_signal_bps=min_signal_bps,
                    fee_bps_roundtrip=fee_bps_roundtrip,
                    test_days=args.test_days,
                    workers=max(1, args.workers),
                    executor_kind=args.executor,
                    recent_days=args.recent_days,
                )
            )

    rows.sort(key=lambda row: (-row[3], row[1], row[0]))

    out_path = OUT_DIR / "pilot_iterate_summary.csv"
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "min_signal_bps",
                "fee_bps_roundtrip",
                "symbols_analyzed",
                "avg_test_net_pnl_bps",
                "test_profitable_symbols",
            ]
        )
        writer.writerows(rows)

    print(f"Pilot symbols: {', '.join(symbols)}")
    print(f"Wrote {out_path}")
    print("Top 10 pilot configs:")
    for row in rows[:10]:
        print(
            f"  signal={row[0]:.1f} fee={row[1]:.1f} avg_test_net={row[3]:.4f}bps "
            f"profitable={row[4]}/{row[2]}"
        )


if __name__ == "__main__":
    main()
