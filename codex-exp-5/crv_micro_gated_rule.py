#!/usr/bin/env python3
"""Backtest a tighter CRVUSDT rule using recent microstructure gates."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import crv_microstructure_audit as micro


OUT_DIR = Path(__file__).resolve().parent / "out"


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def load_rows(symbol: str, taker_fee_bps_roundtrip: float) -> tuple[list[micro.MicroTrade], list[str]]:
    bn_dir = micro.base.BINANCE / symbol
    bb_dir = micro.base.BYBIT / symbol
    audited_days = sorted(
        micro.base.collect_dates(bn_dir, "trades.csv")
        & micro.base.collect_dates(bb_dir, "trades.csv")
        & micro.base.collect_dates(bb_dir, "orderbook.jsonl")
        & micro.base.collect_dates(bn_dir, "kline_1m.csv")
        & micro.base.collect_dates(bb_dir, "kline_1m.csv")
    )
    rows: list[micro.MicroTrade] = []
    for day in audited_days:
        rows.extend(micro.audit_day(symbol, day, taker_fee_bps_roundtrip))
    return rows, audited_days


def build_report(
    symbol: str,
    baseline_rows: list[micro.MicroTrade],
    filtered_rows: list[micro.MicroTrade],
    audited_days: list[str],
    args: argparse.Namespace,
) -> str:
    baseline_avg = avg([row.net_taker_bps for row in baseline_rows])
    filtered_avg = avg([row.net_taker_bps for row in filtered_rows])
    baseline_win = avg([1.0 if row.net_taker_bps > 0 else 0.0 for row in baseline_rows])
    filtered_win = avg([1.0 if row.net_taker_bps > 0 else 0.0 for row in filtered_rows])

    month_rows = []
    month_buckets: dict[str, list[micro.MicroTrade]] = defaultdict(list)
    for row in filtered_rows:
        month_buckets[row.month].append(row)
    for month in sorted(month_buckets):
        vals = [row.net_taker_bps for row in month_buckets[month]]
        month_rows.append(f"| {month} | {len(vals)} | {avg(vals):.4f} |")

    lines = [
        "# CRV Micro-Gated Rule",
        "",
        "## Scope",
        "",
        f"- Symbol: {symbol}",
        f"- Audited days: {', '.join(audited_days)}",
        f"- Taker fee round trip: {args.taker_fee_bps_roundtrip:.2f} bps",
        "",
        "## Gates",
        "",
        f"- Min signal score: {args.min_score:.2f}",
        f"- Max Bybit trigger book spread: {args.max_book_spread_bps:.2f} bps",
        f"- Require flow fading: {args.require_nonpositive_combined_flow}",
        "",
        "## Comparison",
        "",
        f"- Baseline recent triggers: {len(baseline_rows)}",
        f"- Baseline avg net after taker fee: {baseline_avg:.4f} bps",
        f"- Baseline win rate: {100.0 * baseline_win:.2f}%",
        f"- Filtered triggers: {len(filtered_rows)}",
        f"- Filtered avg net after taker fee: {filtered_avg:.4f} bps",
        f"- Filtered win rate: {100.0 * filtered_win:.2f}%",
        "",
        "## Monthly",
        "",
        "| Month | Trades | Avg Net Taker bps |",
        "|---|---:|---:|",
        *month_rows,
        "",
        "## Interpretation",
        "",
        "- This is a recent microstructure-covered sample, not the full historical backtest window.",
        "- The point of this rule is to test whether adding execution-aware gates improves the surviving CRV edge materially.",
    ]
    return "\n".join(lines) + "\n"


def write_trade_log(path: Path, rows: list[micro.MicroTrade]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "day",
                "month",
                "entry_ts_ms",
                "exit_ts_ms",
                "net_taker_bps",
                "score",
                "spread_abs_bps",
                "combined_flow_usd",
                "bybit_book_spread_bps",
                "bybit_top_depth_usd",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.day,
                    row.month,
                    row.entry_ts_ms,
                    row.exit_ts_ms,
                    f"{row.net_taker_bps:.6f}",
                    f"{row.score:.6f}",
                    f"{row.spread_abs_bps:.6f}",
                    f"{row.combined_flow_usd:.6f}",
                    f"{row.bybit_book_spread_bps:.6f}",
                    f"{row.bybit_top_depth_usd:.6f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="CRVUSDT")
    parser.add_argument("--taker-fee-bps-roundtrip", type=float, default=20.0)
    parser.add_argument("--min-score", type=float, default=18.0)
    parser.add_argument("--max-book-spread-bps", type=float, default=8.0)
    parser.add_argument("--require-nonpositive-combined-flow", action="store_true")
    args = parser.parse_args()

    baseline_rows, audited_days = load_rows(args.symbol, args.taker_fee_bps_roundtrip)
    if not baseline_rows:
        raise SystemExit("No baseline microstructure trades found.")

    filtered_rows = [
        row
        for row in baseline_rows
        if row.score >= args.min_score
        and row.bybit_book_spread_bps <= args.max_book_spread_bps
        and (
            (not args.require_nonpositive_combined_flow)
            or row.combined_flow_usd <= 0.0
        )
    ]
    if not filtered_rows:
        raise SystemExit("No trades passed the selected microstructure gates.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "crv_micro_gated_trades.csv"
    md_path = OUT_DIR / "crv_micro_gated_report.md"
    write_trade_log(csv_path, filtered_rows)
    md_path.write_text(build_report(args.symbol, baseline_rows, filtered_rows, audited_days, args))
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
