#!/usr/bin/env python3
"""Analyze downside metrics for a paper-trading fill stream."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / "out"
DEFAULT_INPUT = OUT_DIR / "paper_fills_v3_replayopt_sized25.csv"


@dataclass(frozen=True)
class Fill:
    symbol: str
    day: str
    month: str
    entry_ts_ms: int
    exit_ts_ms: int
    net_pnl_bps: float
    pnl_dollars: float


def load_fills(path: Path) -> list[Fill]:
    fills: list[Fill] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fills.append(
                Fill(
                    symbol=row["symbol"],
                    day=row["day"],
                    month=row["month"],
                    entry_ts_ms=int(row["entry_ts_ms"]),
                    exit_ts_ms=int(row["exit_ts_ms"]),
                    net_pnl_bps=float(row["net_pnl_bps"]),
                    pnl_dollars=float(row["pnl_dollars"]),
                )
            )
    fills.sort(key=lambda row: (row.exit_ts_ms, row.entry_ts_ms, row.symbol))
    return fills


def iso_week_label(day: str) -> str:
    d = datetime.strptime(day, "%Y-%m-%d").date()
    year, week, _ = d.isocalendar()
    return f"{year}-W{week:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--starting-capital", type=float, default=100000.0)
    parser.add_argument("--report-output", type=Path, default=OUT_DIR / "downside_report_v3_replayopt_sized25.md")
    parser.add_argument("--weekly-output", type=Path, default=OUT_DIR / "downside_weekly_v3_replayopt_sized25.csv")
    parser.add_argument("--monthly-output", type=Path, default=OUT_DIR / "downside_monthly_v3_replayopt_sized25.csv")
    args = parser.parse_args()

    fills = load_fills(args.input)
    balance = args.starting_capital
    peak = balance
    max_drawdown_dollars = 0.0
    max_drawdown_pct = 0.0
    drawdown_start_peak = peak
    drawdown_trough = balance
    drawdown_start_ts = fills[0].exit_ts_ms if fills else 0
    drawdown_end_ts = drawdown_start_ts
    peak_ts = drawdown_start_ts

    weekly_pnl: dict[str, float] = defaultdict(float)
    weekly_fills: dict[str, int] = defaultdict(int)
    monthly_pnl: dict[str, float] = defaultdict(float)
    monthly_fills: dict[str, int] = defaultdict(int)

    for fill in fills:
        balance += fill.pnl_dollars
        week = iso_week_label(fill.day)
        weekly_pnl[week] += fill.pnl_dollars
        weekly_fills[week] += 1
        monthly_pnl[fill.month] += fill.pnl_dollars
        monthly_fills[fill.month] += 1

        if balance > peak:
            peak = balance
            peak_ts = fill.exit_ts_ms
        drawdown_dollars = peak - balance
        drawdown_pct = drawdown_dollars / peak if peak else 0.0
        if drawdown_dollars > max_drawdown_dollars:
            max_drawdown_dollars = drawdown_dollars
            max_drawdown_pct = drawdown_pct
            drawdown_start_peak = peak
            drawdown_trough = balance
            drawdown_start_ts = peak_ts
            drawdown_end_ts = fill.exit_ts_ms

    weekly_rows = []
    for week in sorted(weekly_pnl):
        pnl = weekly_pnl[week]
        weekly_rows.append([week, weekly_fills[week], f"{pnl:.2f}"])

    monthly_rows = []
    positive_months = 0
    negative_months = 0
    for month in sorted(monthly_pnl):
        pnl = monthly_pnl[month]
        if pnl > 0:
            positive_months += 1
        elif pnl < 0:
            negative_months += 1
        monthly_rows.append([month, monthly_fills[month], f"{pnl:.2f}"])

    with args.weekly_output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iso_week", "fills", "pnl_dollars"])
        writer.writerows(weekly_rows)

    with args.monthly_output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["month", "fills", "pnl_dollars"])
        writer.writerows(monthly_rows)

    worst_week = min(weekly_pnl.items(), key=lambda row: row[1]) if weekly_pnl else ("", 0.0)
    best_week = max(weekly_pnl.items(), key=lambda row: row[1]) if weekly_pnl else ("", 0.0)

    def fmt_ts(ts_ms: int) -> str:
        if not ts_ms:
            return ""
        return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# Downside Report",
        "",
        f"- Input: {args.input}",
        f"- Filled trades: {len(fills)}",
        f"- Starting capital: ${args.starting_capital:,.2f}",
        f"- Ending capital: ${balance:,.2f}",
        "",
        "## Max Drawdown",
        "",
        f"- Max drawdown dollars: ${max_drawdown_dollars:,.2f}",
        f"- Max drawdown percent: {max_drawdown_pct:.2%}",
        f"- Peak before drawdown: ${drawdown_start_peak:,.2f}",
        f"- Trough at max drawdown: ${drawdown_trough:,.2f}",
        f"- Drawdown start: {fmt_ts(drawdown_start_ts)}",
        f"- Drawdown end: {fmt_ts(drawdown_end_ts)}",
        "",
        "## Weekly Stability",
        "",
        f"- Weeks observed: {len(weekly_pnl)}",
        f"- Positive weeks: {sum(1 for v in weekly_pnl.values() if v > 0)}",
        f"- Negative weeks: {sum(1 for v in weekly_pnl.values() if v < 0)}",
        f"- Worst week: {worst_week[0]} (${worst_week[1]:,.2f})",
        f"- Best week: {best_week[0]} (${best_week[1]:,.2f})",
        "",
        "## Monthly Stability",
        "",
        f"- Months observed: {len(monthly_pnl)}",
        f"- Positive months: {positive_months}",
        f"- Negative months: {negative_months}",
        "",
    ]
    args.report_output.write_text("\n".join(lines))
    print(f"Wrote {args.weekly_output}")
    print(f"Wrote {args.monthly_output}")
    print(f"Wrote {args.report_output}")


if __name__ == "__main__":
    main()
