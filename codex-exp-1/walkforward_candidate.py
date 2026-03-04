#!/usr/bin/env python3
"""Walk-forward style validation for the fixed basket with optional daily trade caps."""

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
DEFAULT_SYMBOLS = OUT_DIR / "candidate_basket_v2.txt"


@dataclass
class WFRecord:
    symbol: str
    day: str
    month: str
    net_pnl_bps: float
    ls_diff_signed: float
    oi_diff_signed_bps: float
    carry_diff_signed_bps: float
    entry_spread_abs_bps: float
    entry_spread_velocity_bps: float

    @property
    def score(self) -> float:
        # Higher score means stronger agreement with the filter direction.
        return self.ls_diff_signed + (self.oi_diff_signed_bps / 5.0) + (self.carry_diff_signed_bps / 2.0)


def passes_filter(
    row: WFRecord,
    ls_threshold: float,
    oi_threshold_bps: float,
    carry_threshold_bps: float,
) -> bool:
    return (
        row.ls_diff_signed >= ls_threshold
        and row.oi_diff_signed_bps >= oi_threshold_bps
        and row.carry_diff_signed_bps >= carry_threshold_bps
    )


def build_symbol_records(
    symbol: str,
    min_overlap_days: int,
    min_signal_bps: float,
    fee_bps_roundtrip: float,
    recent_days: int,
) -> list[WFRecord]:
    overlap = sorted(
        collect_dates(BINANCE / symbol, "kline_1m.csv")
        & collect_dates(BYBIT / symbol, "kline_1m.csv")
    )
    if len(overlap) < min_overlap_days:
        return []
    dates = overlap[-recent_days:] if recent_days > 0 else overlap
    out: list[WFRecord] = []
    for day in dates:
        rows = build_day_records(symbol, day, "all", min_signal_bps, fee_bps_roundtrip)
        out.extend(
            WFRecord(
                symbol=symbol,
                day=day,
                month=day[:7],
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


def trade_slippage(
    row: WFRecord,
    fixed_extra_slippage_bps: float,
    spread_slip_coeff: float,
    velocity_slip_coeff: float,
    min_signal_bps: float,
) -> float:
    stretch = max(0.0, row.entry_spread_abs_bps - min_signal_bps)
    return (
        fixed_extra_slippage_bps
        + spread_slip_coeff * stretch
        + velocity_slip_coeff * row.entry_spread_velocity_bps
    )


def avg_net(
    rows: list[WFRecord],
    fixed_extra_slippage_bps: float = 0.0,
    spread_slip_coeff: float = 0.0,
    velocity_slip_coeff: float = 0.0,
    min_signal_bps: float = 0.0,
) -> float:
    if not rows:
        return math.nan
    return (
        sum(
            row.net_pnl_bps
            - trade_slippage(
                row,
                fixed_extra_slippage_bps,
                spread_slip_coeff,
                velocity_slip_coeff,
                min_signal_bps,
            )
            for row in rows
        )
        / len(rows)
    )


def apply_daily_cap(rows: list[WFRecord], max_trades_per_symbol_day: int) -> list[WFRecord]:
    if max_trades_per_symbol_day <= 0:
        return rows
    buckets: dict[tuple[str, str], list[WFRecord]] = defaultdict(list)
    for row in rows:
        buckets[(row.symbol, row.day)].append(row)
    capped: list[WFRecord] = []
    for key in sorted(buckets):
        day_rows = buckets[key]
        day_rows.sort(key=lambda row: (-row.score, -row.net_pnl_bps))
        capped.extend(day_rows[:max_trades_per_symbol_day])
    return capped


def split_train_test_months(months: list[str], test_months: int) -> tuple[list[str], list[str]]:
    if test_months >= len(months):
        test_months = max(1, len(months) // 3)
    return months[:-test_months], months[-test_months:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols-file", type=Path, default=DEFAULT_SYMBOLS)
    parser.add_argument("--min-overlap-days", type=int, default=90)
    parser.add_argument("--min-signal-bps", type=float, default=10.0)
    parser.add_argument("--fee-bps-roundtrip", type=float, default=6.0)
    parser.add_argument("--recent-days", type=int, default=210)
    parser.add_argument("--test-months", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ls-threshold", type=float, default=0.15)
    parser.add_argument("--oi-threshold-bps", type=float, default=5.0)
    parser.add_argument("--carry-threshold-bps", type=float, default=2.0)
    parser.add_argument("--daily-cap", type=int, default=0)
    parser.add_argument("--extra-slippage-bps", type=float, default=0.0)
    parser.add_argument("--spread-slip-coeff", type=float, default=0.0)
    parser.add_argument("--velocity-slip-coeff", type=float, default=0.0)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = load_symbols(args.symbols_file)

    all_rows: list[WFRecord] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                build_symbol_records,
                symbol,
                args.min_overlap_days,
                args.min_signal_bps,
                args.fee_bps_roundtrip,
                args.recent_days,
            ): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            all_rows.extend(future.result())

    filt_rows = [
        row
        for row in all_rows
        if passes_filter(row, args.ls_threshold, args.oi_threshold_bps, args.carry_threshold_bps)
    ]
    filt_rows = apply_daily_cap(filt_rows, args.daily_cap)

    months = sorted({row.month for row in filt_rows})
    train_months, test_months = split_train_test_months(months, args.test_months)

    month_rows = []
    for month in months:
        rows = [row for row in filt_rows if row.month == month]
        month_rows.append(
            (
                month,
                len(rows),
                avg_net(
                    rows,
                    args.extra_slippage_bps,
                    args.spread_slip_coeff,
                    args.velocity_slip_coeff,
                    args.min_signal_bps,
                ),
            )
        )

    symbol_rows = []
    for symbol in symbols:
        rows = [row for row in filt_rows if row.symbol == symbol]
        symbol_rows.append(
            (
                symbol,
                len(rows),
                avg_net(
                    rows,
                    args.extra_slippage_bps,
                    args.spread_slip_coeff,
                    args.velocity_slip_coeff,
                    args.min_signal_bps,
                ),
            )
        )
    symbol_rows.sort(key=lambda row: (-row[2], -row[1], row[0]))

    train_rows = [row for row in filt_rows if row.month in set(train_months)]
    test_rows = [row for row in filt_rows if row.month in set(test_months)]

    suffix = f"_{args.tag}" if args.tag else ""
    monthly_path = OUT_DIR / f"walkforward_monthly{suffix}.csv"
    with monthly_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["month", "signals", "avg_net_bps"])
        writer.writerows(month_rows)

    symbol_path = OUT_DIR / f"walkforward_symbol{suffix}.csv"
    with symbol_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["symbol", "signals", "avg_net_bps"])
        writer.writerows(symbol_rows)

    report_path = OUT_DIR / f"walkforward_report{suffix}.md"
    positive_months = sum(1 for _, _, avg in month_rows if not math.isnan(avg) and avg > 0)
    lines = [
        "# Walk-Forward Candidate Report",
        "",
        "## Configuration",
        "",
        f"- Symbols: {', '.join(symbols)}",
        f"- Recent days: {args.recent_days}",
        f"- Test months: {args.test_months}",
        f"- Spread threshold: {args.min_signal_bps:.2f} bps",
        f"- Base fee: {args.fee_bps_roundtrip:.2f} bps",
        f"- Extra slippage: {args.extra_slippage_bps:.2f} bps",
        f"- Spread slippage coeff: {args.spread_slip_coeff:.4f}",
        f"- Velocity slippage coeff: {args.velocity_slip_coeff:.4f}",
        f"- Filter: ls>={args.ls_threshold:.2f}, oi>={args.oi_threshold_bps:.2f}, carry>={args.carry_threshold_bps:.2f}",
        f"- Daily cap per symbol: {args.daily_cap if args.daily_cap > 0 else 'none'}",
        "",
        "## Aggregate",
        "",
        f"- Train months: {', '.join(train_months)}",
        f"- Test months: {', '.join(test_months)}",
        f"- Train avg net: {avg_net(train_rows, args.extra_slippage_bps, args.spread_slip_coeff, args.velocity_slip_coeff, args.min_signal_bps):.4f} bps on {len(train_rows)} signals",
        f"- Test avg net: {avg_net(test_rows, args.extra_slippage_bps, args.spread_slip_coeff, args.velocity_slip_coeff, args.min_signal_bps):.4f} bps on {len(test_rows)} signals",
        f"- Positive months: {positive_months}/{len(month_rows)}",
        "",
        "## Monthly",
        "",
        "| Month | Signals | Avg Net bps |",
        "|---|---:|---:|",
    ]
    for month, count, avg in month_rows:
        lines.append(f"| {month} | {count} | {avg:.4f} |")
    lines.extend(
        [
            "",
            "## Symbol",
            "",
            "| Symbol | Signals | Avg Net bps |",
            "|---|---:|---:|",
        ]
    )
    for symbol, count, avg in symbol_rows:
        lines.append(f"| {symbol} | {count} | {avg:.4f} |")
    lines.append("")
    report_path.write_text("\n".join(lines))

    print(f"Loaded filtered records: {len(filt_rows)}")
    print(
        f"Train avg net={avg_net(train_rows, args.extra_slippage_bps, args.spread_slip_coeff, args.velocity_slip_coeff, args.min_signal_bps):.4f}bps on {len(train_rows)} signals"
    )
    print(
        f"Test avg net={avg_net(test_rows, args.extra_slippage_bps, args.spread_slip_coeff, args.velocity_slip_coeff, args.min_signal_bps):.4f}bps on {len(test_rows)} signals"
    )
    print(f"Positive months: {positive_months}/{len(month_rows)}")
    print(f"Wrote {monthly_path}")
    print(f"Wrote {symbol_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
