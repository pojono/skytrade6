#!/usr/bin/env python3
"""Simple paper simulator for the frozen basket using exported trade records."""

from __future__ import annotations

import argparse
import csv
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
) -> float:
    stretch = max(0.0, trade.entry_spread_abs_bps - min_signal_bps)
    return (
        fixed_extra_slippage_bps
        + spread_slip_coeff * stretch
        + velocity_slip_coeff * trade.entry_spread_velocity_bps
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--starting-capital", type=float, default=100000.0)
    parser.add_argument("--per-trade-allocation", type=float, default=0.10)
    parser.add_argument("--max-open-positions", type=int, default=3)
    parser.add_argument("--daily-cap-per-symbol", type=int, default=3)
    parser.add_argument("--min-signal-bps", type=float, default=10.0)
    parser.add_argument("--fee-bps-roundtrip", type=float, default=6.0)
    parser.add_argument("--extra-slippage-bps", type=float, default=1.0)
    parser.add_argument("--spread-slip-coeff", type=float, default=0.10)
    parser.add_argument("--velocity-slip-coeff", type=float, default=0.05)
    parser.add_argument("--output-fills", type=Path, default=OUT_DIR / "paper_fills_v3.csv")
    parser.add_argument("--output-report", type=Path, default=OUT_DIR / "paper_report_v3.md")
    args = parser.parse_args()

    trades = load_trades(args.input)
    balance = args.starting_capital
    open_positions: list[Fill] = []
    daily_counts: dict[tuple[str, str], int] = {}
    fills: list[Fill] = []

    idx = 0
    while idx < len(trades):
        entry_ts = trades[idx].entry_ts_ms

        still_open: list[Fill] = []
        for fill in open_positions:
            if fill.exit_ts_ms <= entry_ts:
                balance += fill.pnl_dollars
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

        accepted = 0
        for trade in batch:
            if accepted >= available_slots:
                break
            key = (trade.symbol, trade.day)
            if daily_counts.get(key, 0) >= args.daily_cap_per_symbol:
                continue

            alloc = balance * args.per_trade_allocation
            slip = trade_slippage(
                trade,
                args.min_signal_bps,
                args.extra_slippage_bps,
                args.spread_slip_coeff,
                args.velocity_slip_coeff,
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

    lines = [
        "# Paper Trading Report",
        "",
        "## Configuration",
        "",
        f"- Input: {args.input}",
        f"- Starting capital: {args.starting_capital:.2f}",
        f"- Per-trade allocation: {args.per_trade_allocation:.2%}",
        f"- Max open positions: {args.max_open_positions}",
        f"- Daily cap per symbol: {args.daily_cap_per_symbol}",
        f"- Base fee: {args.fee_bps_roundtrip:.2f} bps",
        f"- Extra slippage: {args.extra_slippage_bps:.2f} bps",
        f"- Spread slippage coeff: {args.spread_slip_coeff:.4f}",
        f"- Velocity slippage coeff: {args.velocity_slip_coeff:.4f}",
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
    lines.append("")
    args.output_report.write_text("\n".join(lines))

    print(f"Filled trades: {len(fills)}")
    print(f"Final capital: {balance:.2f}")
    print(f"Total PnL: {total_pnl:.2f}")
    print(f"Average net edge: {avg_net_bps:.4f}bps")
    print(f"Wrote {args.output_fills}")
    print(f"Wrote {args.output_report}")


if __name__ == "__main__":
    main()
