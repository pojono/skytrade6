#!/usr/bin/env python3
"""Validate the fixed candidate basket across time, symbols, and extra slippage assumptions."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from dense_filter_sweep import build_day_records, collect_dates, BINANCE, BYBIT, load_symbols


OUT_DIR = Path(__file__).resolve().parent / "out"
DEFAULT_SYMBOLS = OUT_DIR / "candidate_basket.txt"


@dataclass
class RichRecord:
    symbol: str
    day: str
    split: str
    net_pnl_bps: float
    ls_diff_signed: float
    oi_diff_signed_bps: float
    carry_diff_signed_bps: float
    entry_spread_abs_bps: float
    entry_spread_velocity_bps: float


def passes_filter(
    row: RichRecord,
    ls_threshold: float,
    oi_threshold_bps: float,
    carry_threshold_bps: float,
) -> bool:
    return (
        row.ls_diff_signed >= ls_threshold
        and row.oi_diff_signed_bps >= oi_threshold_bps
        and row.carry_diff_signed_bps >= carry_threshold_bps
    )


def build_symbol_rich_records(
    symbol: str,
    min_overlap_days: int,
    min_signal_bps: float,
    fee_bps_roundtrip: float,
    recent_days: int,
    test_days: int,
) -> list[RichRecord]:
    overlap = sorted(
        collect_dates(BINANCE / symbol, "kline_1m.csv")
        & collect_dates(BYBIT / symbol, "kline_1m.csv")
    )
    if len(overlap) < min_overlap_days:
        return []
    dates = overlap[-recent_days:] if recent_days > 0 else overlap
    if test_days >= len(dates):
        test_days = max(1, len(dates) // 3)
    split_idx = len(dates) - test_days
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]

    out: list[RichRecord] = []
    for split, use_dates in (("train", train_dates), ("test", test_dates)):
        for day in use_dates:
            rows = build_day_records(symbol, day, split, min_signal_bps, fee_bps_roundtrip)
            out.extend(
                RichRecord(
                    symbol=symbol,
                    day=day,
                    split=split,
                    net_pnl_bps=row.net_pnl_bps,
                    ls_diff_signed=row.ls_diff_signed,
                    oi_diff_signed_bps=row.oi_diff_signed_bps,
                    carry_diff_signed_bps=row.carry_diff_signed_bps,
                    entry_spread_abs_bps=row.entry_spread_abs_bps,
                    entry_spread_velocity_bps=row.entry_spread_velocity_bps,
                )
                for row in rows
            )
    return out


def mean_net(rows: list[RichRecord], extra_slippage_bps: float = 0.0) -> float:
    if not rows:
        return math.nan
    total = sum(row.net_pnl_bps - extra_slippage_bps for row in rows)
    return total / len(rows)


def aggregate_monthly(rows: list[RichRecord]) -> list[tuple[str, int, float]]:
    buckets: dict[str, list[RichRecord]] = defaultdict(list)
    for row in rows:
        buckets[row.day[:7]].append(row)
    out = []
    for month in sorted(buckets):
        month_rows = buckets[month]
        out.append((month, len(month_rows), mean_net(month_rows)))
    return out


def split_stats(rows: list[RichRecord]) -> tuple[int, float, int, float]:
    train_rows = [row for row in rows if row.split == "train"]
    test_rows = [row for row in rows if row.split == "test"]
    return len(train_rows), mean_net(train_rows), len(test_rows), mean_net(test_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols-file", type=Path, default=DEFAULT_SYMBOLS)
    parser.add_argument("--min-overlap-days", type=int, default=90)
    parser.add_argument("--min-signal-bps", type=float, default=10.0)
    parser.add_argument("--fee-bps-roundtrip", type=float, default=6.0)
    parser.add_argument("--recent-days", type=int, default=180)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ls-threshold", type=float, default=0.15)
    parser.add_argument("--oi-threshold-bps", type=float, default=5.0)
    parser.add_argument("--carry-threshold-bps", type=float, default=2.0)
    parser.add_argument("--extra-slippage-grid", type=str, default="0,1,2,3")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = load_symbols(args.symbols_file)

    all_rows: list[RichRecord] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                build_symbol_rich_records,
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
            all_rows.extend(future.result())

    filt_rows = [
        row
        for row in all_rows
        if passes_filter(
            row,
            args.ls_threshold,
            args.oi_threshold_bps,
            args.carry_threshold_bps,
        )
    ]
    train_rows = [row for row in filt_rows if row.split == "train"]
    test_rows = [row for row in filt_rows if row.split == "test"]

    monthly_rows = aggregate_monthly(filt_rows)
    monthly_path = OUT_DIR / "candidate_monthly.csv"
    with monthly_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["month", "signals", "avg_net_bps"])
        writer.writerows(monthly_rows)

    symbol_stats = []
    for symbol in symbols:
        rows = [row for row in filt_rows if row.symbol == symbol]
        tr = [row for row in rows if row.split == "train"]
        te = [row for row in rows if row.split == "test"]
        total_net = sum(row.net_pnl_bps for row in rows)
        symbol_stats.append(
            (
                symbol,
                len(tr),
                mean_net(tr),
                len(te),
                mean_net(te),
                len(rows),
                total_net,
            )
        )
    symbol_stats.sort(key=lambda row: (-row[6], row[0]))

    symbol_path = OUT_DIR / "candidate_symbol_contrib.csv"
    with symbol_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "symbol",
                "train_signals",
                "train_avg_net_bps",
                "test_signals",
                "test_avg_net_bps",
                "total_signals",
                "total_net_bps_sum",
            ]
        )
        writer.writerows(symbol_stats)

    slip_grid = [float(x) for x in args.extra_slippage_grid.split(",") if x.strip()]
    slip_rows = []
    for extra_slippage in slip_grid:
        slip_rows.append(
            (
                extra_slippage,
                len(train_rows),
                mean_net(train_rows, extra_slippage),
                len(test_rows),
                mean_net(test_rows, extra_slippage),
            )
        )
    slip_path = OUT_DIR / "candidate_slippage_sweep.csv"
    with slip_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "extra_slippage_bps",
                "train_signals",
                "train_avg_net_bps",
                "test_signals",
                "test_avg_net_bps",
            ]
        )
        writer.writerows(slip_rows)

    leave_one_out_rows = []
    for symbol in symbols:
        subset_rows = [row for row in filt_rows if row.symbol != symbol]
        train_n, train_avg, test_n, test_avg = split_stats(subset_rows)
        leave_one_out_rows.append((symbol, train_n, train_avg, test_n, test_avg))

    leave_path = OUT_DIR / "candidate_leave_one_out.csv"
    with leave_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "excluded_symbol",
                "train_signals",
                "train_avg_net_bps",
                "test_signals",
                "test_avg_net_bps",
            ]
        )
        writer.writerows(leave_one_out_rows)

    top_two = [row[0] for row in symbol_stats[:2]]
    drop_top_two_rows = [row for row in filt_rows if row.symbol not in set(top_two)]
    drop_top_two_stats = split_stats(drop_top_two_rows)

    total_net_sum = sum(row[6] for row in symbol_stats)
    top_symbol_share = (symbol_stats[0][6] / total_net_sum) if total_net_sum else math.nan
    positive_months = sum(1 for _, _, avg in monthly_rows if not math.isnan(avg) and avg > 0)

    report_path = OUT_DIR / "candidate_robustness_report.md"
    lines = [
        "# Candidate Robustness Report",
        "",
        "## Configuration",
        "",
        f"- Symbols: {', '.join(symbols)}",
        f"- Recent days: {args.recent_days}",
        f"- Test days: {args.test_days}",
        f"- Spread threshold: {args.min_signal_bps:.2f} bps",
        f"- Fee: {args.fee_bps_roundtrip:.2f} bps",
        f"- Filter: ls>={args.ls_threshold:.2f}, oi>={args.oi_threshold_bps:.2f}, carry>={args.carry_threshold_bps:.2f}",
        "",
        "## Aggregate",
        "",
        f"- Train avg net: {mean_net(train_rows):.4f} bps on {len(train_rows)} signals",
        f"- Test avg net: {mean_net(test_rows):.4f} bps on {len(test_rows)} signals",
        f"- Positive months: {positive_months}/{len(monthly_rows)}",
        f"- Top symbol share of total net PnL: {top_symbol_share:.2%}",
        "",
        "## Monthly",
        "",
        "| Month | Signals | Avg Net bps |",
        "|---|---:|---:|",
    ]
    for month, count, avg in monthly_rows:
        lines.append(f"| {month} | {count} | {avg:.4f} |")
    lines.extend(
        [
            "",
            "## Symbol Contribution",
            "",
            "| Symbol | Test Signals | Test Avg Net bps | Total Net Sum |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in symbol_stats:
        lines.append(f"| {row[0]} | {row[3]} | {row[4]:.4f} | {row[6]:.2f} |")
    lines.extend(
        [
            "",
            "## Extra Slippage Sweep",
            "",
            "| Extra Slippage bps | Train Avg Net | Test Avg Net |",
            "|---|---:|---:|",
        ]
    )
    for row in slip_rows:
        lines.append(f"| {row[0]:.2f} | {row[2]:.4f} | {row[4]:.4f} |")
    lines.extend(
        [
            "",
            "## Leave-One-Out",
            "",
            "| Excluded Symbol | Train Avg Net | Test Avg Net |",
            "|---|---:|---:|",
        ]
    )
    for row in leave_one_out_rows:
        lines.append(f"| {row[0]} | {row[2]:.4f} | {row[4]:.4f} |")
    lines.extend(
        [
            "",
            "## Drop Top Two Contributors",
            "",
            f"- Excluded: {', '.join(top_two)}",
            f"- Train avg net: {drop_top_two_stats[1]:.4f} bps on {drop_top_two_stats[0]} signals",
            f"- Test avg net: {drop_top_two_stats[3]:.4f} bps on {drop_top_two_stats[2]} signals",
            "",
        ]
    )
    report_path.write_text("\n".join(lines))

    print(f"Loaded filtered signals: train={len(train_rows)} test={len(test_rows)}")
    print(f"Positive months: {positive_months}/{len(monthly_rows)}")
    print(f"Top symbol share: {top_symbol_share:.2%}")
    print(f"Wrote {monthly_path}")
    print(f"Wrote {symbol_path}")
    print(f"Wrote {slip_path}")
    print(f"Wrote {leave_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
