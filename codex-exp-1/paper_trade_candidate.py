#!/usr/bin/env python3
"""Simple paper simulator for the frozen basket using exported trade records."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / "out"
DEFAULT_INPUT = OUT_DIR / "candidate_trades_v3.csv"


@dataclass
class Trade:
    symbol: str
    day: str
    month: str
    entry_ts_ms: int
    exit_ts_ms: int
    gross_pnl_bps: float
    entry_spread_abs_bps: float
    entry_spread_velocity_bps: float
    score: float


@dataclass
class Fill:
    symbol: str
    day: str
    month: str
    entry_ts_ms: int
    exit_ts_ms: int
    score: float
    gross_pnl_bps: float
    net_pnl_bps: float
    alloc_dollars: float
    pnl_dollars: float


def trade_priority(trade: Trade, selector_mode: str) -> tuple:
    if selector_mode == "spread":
        return (-trade.entry_spread_abs_bps, -trade.score, trade.symbol)
    if selector_mode == "velocity":
        return (-trade.entry_spread_velocity_bps, -trade.score, trade.symbol)
    return (-trade.score, -trade.entry_spread_abs_bps, trade.symbol)


def load_trades(path: Path) -> list[Trade]:
    rows: list[Trade] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                Trade(
                    symbol=row["symbol"],
                    day=row["day"],
                    month=row["month"],
                    entry_ts_ms=int(row["entry_ts_ms"]),
                    exit_ts_ms=int(row["exit_ts_ms"]),
                    gross_pnl_bps=float(row["gross_pnl_bps"]),
                    entry_spread_abs_bps=float(row["entry_spread_abs_bps"]),
                    entry_spread_velocity_bps=float(row["entry_spread_velocity_bps"]),
                    score=float(row["score"]),
                )
            )
    rows.sort(key=lambda row: (row.entry_ts_ms, -row.score, row.symbol))
    return rows


def trade_slippage(
    trade: Trade,
    min_signal_bps: float,
    fixed_extra_slippage_bps: float,
    spread_slip_coeff: float,
    velocity_slip_coeff: float,
    size_slip_coeff: float,
    alloc_fraction: float,
    base_alloc_ref: float,
) -> float:
    stretch = max(0.0, trade.entry_spread_abs_bps - min_signal_bps)
    size_multiple_above_base = 0.0
    if base_alloc_ref > 0:
        size_multiple_above_base = max(0.0, alloc_fraction / base_alloc_ref - 1.0)
    return (
        fixed_extra_slippage_bps
        + spread_slip_coeff * stretch
        + velocity_slip_coeff * trade.entry_spread_velocity_bps
        + size_slip_coeff * size_multiple_above_base
    )


def should_block_new_trade(
    trade: Trade,
    day_pnl: dict[str, float],
    month_pnl: dict[str, float],
    day_start_equity: dict[str, float],
    month_start_equity: dict[str, float],
    daily_loss_stop_pct: float,
    monthly_loss_stop_pct: float,
) -> bool:
    if daily_loss_stop_pct > 0:
        start_equity = day_start_equity.get(trade.day)
        if start_equity is not None:
            if day_pnl.get(trade.day, 0.0) <= -(start_equity * daily_loss_stop_pct):
                return True
    if monthly_loss_stop_pct > 0:
        start_equity = month_start_equity.get(trade.month)
        if start_equity is not None:
            if month_pnl.get(trade.month, 0.0) <= -(start_equity * monthly_loss_stop_pct):
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--starting-capital", type=float, default=100000.0)
    parser.add_argument("--per-trade-allocation", type=float, default=0.10)
    parser.add_argument("--max-open-positions", type=int, default=3)
    parser.add_argument("--max-open-per-symbol", type=int, default=1)
    parser.add_argument("--max-symbol-allocation", type=float, default=0.10)
    parser.add_argument("--daily-cap-per-symbol", type=int, default=3)
    parser.add_argument("--selector-mode", choices=["score", "spread", "velocity"], default="score")
    parser.add_argument("--daily-loss-stop-pct", type=float, default=0.0)
    parser.add_argument("--monthly-loss-stop-pct", type=float, default=0.0)
    parser.add_argument("--min-signal-bps", type=float, default=10.0)
    parser.add_argument("--fee-bps-roundtrip", type=float, default=6.0)
    parser.add_argument("--extra-slippage-bps", type=float, default=1.0)
    parser.add_argument("--spread-slip-coeff", type=float, default=0.10)
    parser.add_argument("--velocity-slip-coeff", type=float, default=0.05)
    parser.add_argument("--size-slip-coeff", type=float, default=0.0)
    parser.add_argument("--base-allocation-ref", type=float, default=0.10)
    parser.add_argument("--output-fills", type=Path, default=OUT_DIR / "paper_fills_v3.csv")
    parser.add_argument("--output-monthly", type=Path, default=OUT_DIR / "paper_monthly_v3.csv")
    parser.add_argument("--output-report", type=Path, default=OUT_DIR / "paper_report_v3.md")
    args = parser.parse_args()

    trades = load_trades(args.input)
    balance = args.starting_capital
    open_positions: list[Fill] = []
    daily_counts: dict[tuple[str, str], int] = {}
    day_pnl: dict[str, float] = {}
    month_pnl: dict[str, float] = {}
    day_start_equity: dict[str, float] = {}
    month_start_equity: dict[str, float] = {}
    fills: list[Fill] = []

    idx = 0
    while idx < len(trades):
        entry_ts = trades[idx].entry_ts_ms

        still_open: list[Fill] = []
        for fill in open_positions:
            if fill.exit_ts_ms <= entry_ts:
                balance += fill.pnl_dollars
                day_pnl[fill.day] = day_pnl.get(fill.day, 0.0) + fill.pnl_dollars
                month_pnl[fill.month] = month_pnl.get(fill.month, 0.0) + fill.pnl_dollars
            else:
                still_open.append(fill)
        open_positions = still_open

        batch: list[Trade] = []
        while idx < len(trades) and trades[idx].entry_ts_ms == entry_ts:
            batch.append(trades[idx])
            idx += 1

        available_slots = max(0, args.max_open_positions - len(open_positions))
        if available_slots == 0:
            continue

        batch.sort(key=lambda trade: trade_priority(trade, args.selector_mode))
        accepted = 0
        for trade in batch:
            if accepted >= available_slots:
                break
            if trade.day not in day_start_equity:
                day_start_equity[trade.day] = balance
            if trade.month not in month_start_equity:
                month_start_equity[trade.month] = balance
            if should_block_new_trade(
                trade,
                day_pnl,
                month_pnl,
                day_start_equity,
                month_start_equity,
                args.daily_loss_stop_pct,
                args.monthly_loss_stop_pct,
            ):
                continue
            key = (trade.symbol, trade.day)
            if daily_counts.get(key, 0) >= args.daily_cap_per_symbol:
                continue
            open_same_symbol = [fill for fill in open_positions if fill.symbol == trade.symbol]
            if len(open_same_symbol) >= args.max_open_per_symbol:
                continue

            current_symbol_alloc = sum(fill.alloc_dollars for fill in open_same_symbol)
            max_symbol_dollars = balance * args.max_symbol_allocation
            alloc = balance * args.per_trade_allocation
            if current_symbol_alloc + alloc > max_symbol_dollars + 1e-9:
                continue
            slip = trade_slippage(
                trade,
                args.min_signal_bps,
                args.extra_slippage_bps,
                args.spread_slip_coeff,
                args.velocity_slip_coeff,
                args.size_slip_coeff,
                args.per_trade_allocation,
                args.base_allocation_ref,
            )
            net_bps = trade.gross_pnl_bps - args.fee_bps_roundtrip - slip
            pnl_dollars = alloc * (net_bps / 10000.0)
            fill = Fill(
                symbol=trade.symbol,
                day=trade.day,
                month=trade.month,
                entry_ts_ms=trade.entry_ts_ms,
                exit_ts_ms=trade.exit_ts_ms,
                score=trade.score,
                gross_pnl_bps=trade.gross_pnl_bps,
                net_pnl_bps=net_bps,
                alloc_dollars=alloc,
                pnl_dollars=pnl_dollars,
            )
            fills.append(fill)
            open_positions.append(fill)
            daily_counts[key] = daily_counts.get(key, 0) + 1
            accepted += 1

    for fill in open_positions:
        balance += fill.pnl_dollars
        day_pnl[fill.day] = day_pnl.get(fill.day, 0.0) + fill.pnl_dollars
        month_pnl[fill.month] = month_pnl.get(fill.month, 0.0) + fill.pnl_dollars

    with args.output_fills.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "symbol",
                "day",
                "month",
                "entry_ts_ms",
                "exit_ts_ms",
                "score",
                "gross_pnl_bps",
                "net_pnl_bps",
                "alloc_dollars",
                "pnl_dollars",
            ]
        )
        for fill in fills:
            writer.writerow(
                [
                    fill.symbol,
                    fill.day,
                    fill.month,
                    fill.entry_ts_ms,
                    fill.exit_ts_ms,
                    f"{fill.score:.6f}",
                    f"{fill.gross_pnl_bps:.6f}",
                    f"{fill.net_pnl_bps:.6f}",
                    f"{fill.alloc_dollars:.2f}",
                    f"{fill.pnl_dollars:.2f}",
                ]
            )

    total_pnl = balance - args.starting_capital
    avg_net_bps = sum(fill.net_pnl_bps for fill in fills) / len(fills) if fills else float("nan")
    win_rate = sum(1 for fill in fills if fill.pnl_dollars > 0) / len(fills) if fills else float("nan")

    by_symbol: dict[str, tuple[int, float]] = {}
    for fill in fills:
        count, pnl = by_symbol.get(fill.symbol, (0, 0.0))
        by_symbol[fill.symbol] = (count + 1, pnl + fill.pnl_dollars)
    symbol_rows = sorted(by_symbol.items(), key=lambda item: (-item[1][1], item[0]))

    by_month: dict[str, list[Fill]] = defaultdict(list)
    running_pnl = 0.0
    monthly_rows = []
    for fill in fills:
        by_month[fill.month].append(fill)
    for month in sorted(by_month):
        month_fills = by_month[month]
        month_pnl = sum(fill.pnl_dollars for fill in month_fills)
        month_avg = sum(fill.net_pnl_bps for fill in month_fills) / len(month_fills)
        running_pnl += month_pnl
        monthly_rows.append(
            (
                month,
                len(month_fills),
                month_avg,
                month_pnl,
                args.starting_capital + running_pnl,
            )
        )

    with args.output_monthly.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "month",
                "filled_trades",
                "avg_net_bps",
                "pnl_dollars",
                "equity_end_dollars",
            ]
        )
        writer.writerows(monthly_rows)

    lines = [
        "# Paper Trading Report",
        "",
        "## Configuration",
        "",
        f"- Input: {args.input}",
        f"- Starting capital: {args.starting_capital:.2f}",
        f"- Per-trade allocation: {args.per_trade_allocation:.2%}",
        f"- Max open positions: {args.max_open_positions}",
        f"- Max open per symbol: {args.max_open_per_symbol}",
        f"- Max symbol allocation: {args.max_symbol_allocation:.2%}",
        f"- Daily cap per symbol: {args.daily_cap_per_symbol}",
        f"- Selector mode: {args.selector_mode}",
        f"- Daily loss stop: {args.daily_loss_stop_pct:.2%}",
        f"- Monthly loss stop: {args.monthly_loss_stop_pct:.2%}",
        f"- Base fee: {args.fee_bps_roundtrip:.2f} bps",
        f"- Extra slippage: {args.extra_slippage_bps:.2f} bps",
        f"- Spread slippage coeff: {args.spread_slip_coeff:.4f}",
        f"- Velocity slippage coeff: {args.velocity_slip_coeff:.4f}",
        f"- Size slippage coeff: {args.size_slip_coeff:.4f}",
        f"- Base allocation ref: {args.base_allocation_ref:.2%}",
        "",
        "## Results",
        "",
        f"- Filled trades: {len(fills)}",
        f"- Final capital: {balance:.2f}",
        f"- Total PnL: {total_pnl:.2f}",
        f"- Average net edge per fill: {avg_net_bps:.4f} bps",
        f"- Win rate: {win_rate:.2%}",
        "",
        "## Symbol Contribution",
        "",
        "| Symbol | Filled Trades | PnL Dollars |",
        "|---|---:|---:|",
    ]
    for symbol, (count, pnl) in symbol_rows:
        lines.append(f"| {symbol} | {count} | {pnl:.2f} |")
    lines.extend(
        [
            "",
            "## Monthly",
            "",
            "| Month | Filled Trades | Avg Net bps | PnL Dollars | Equity End |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in monthly_rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]:.4f} | {row[3]:.2f} | {row[4]:.2f} |")
    lines.append("")
    args.output_report.write_text("\n".join(lines))

    print(f"Filled trades: {len(fills)}")
    print(f"Final capital: {balance:.2f}")
    print(f"Total PnL: {total_pnl:.2f}")
    print(f"Average net edge: {avg_net_bps:.4f}bps")
    print(f"Wrote {args.output_fills}")
    print(f"Wrote {args.output_monthly}")
    print(f"Wrote {args.output_report}")


if __name__ == "__main__":
    main()
