#!/usr/bin/env python3
"""Stress test the frozen replay-optimized 25% strategy under harsher execution assumptions."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from paper_trade_candidate import load_trades, should_block_new_trade, trade_priority, trade_slippage


OUT_DIR = Path(__file__).resolve().parent / "out"
DEFAULT_INPUT = OUT_DIR / "candidate_trades_v3_replayopt.csv"


@dataclass(frozen=True)
class Scenario:
    name: str
    fee_bps_roundtrip: float
    extra_slippage_bps: float
    spread_slip_coeff: float
    velocity_slip_coeff: float
    size_slip_coeff: float


def iso_week(day: str) -> str:
    d = datetime.strptime(day, "%Y-%m-%d").date()
    year, week, _ = d.isocalendar()
    return f"{year}-W{week:02d}"


def run_scenario(input_path: Path, scenario: Scenario) -> dict[str, float | int | str]:
    trades = load_trades(input_path)

    starting_capital = 100000.0
    per_trade_allocation = 0.25
    max_open_positions = 1
    max_open_per_symbol = 1
    max_symbol_allocation = 0.25
    daily_cap_per_symbol = 3
    selector_mode = "spread"
    daily_loss_stop_pct = 0.01
    monthly_loss_stop_pct = 0.03
    min_signal_bps = 10.0
    base_allocation_ref = 0.10

    balance = starting_capital
    open_positions = []
    daily_counts: dict[tuple[str, str], int] = {}
    day_pnl: dict[str, float] = {}
    month_pnl: dict[str, float] = {}
    day_start_equity: dict[str, float] = {}
    month_start_equity: dict[str, float] = {}

    fills = 0
    win_count = 0
    net_bps_sum = 0.0
    weekly_pnl: dict[str, float] = defaultdict(float)
    monthly_realized: dict[str, float] = defaultdict(float)

    peak = balance
    max_drawdown_dollars = 0.0
    max_drawdown_pct = 0.0

    idx = 0
    while idx < len(trades):
        entry_ts = trades[idx].entry_ts_ms

        still_open = []
        for fill in open_positions:
            if fill["exit_ts_ms"] <= entry_ts:
                balance += fill["pnl_dollars"]
                day = fill["day"]
                month = fill["month"]
                day_pnl[day] = day_pnl.get(day, 0.0) + fill["pnl_dollars"]
                month_pnl[month] = month_pnl.get(month, 0.0) + fill["pnl_dollars"]
                weekly_pnl[iso_week(day)] += fill["pnl_dollars"]
                monthly_realized[month] += fill["pnl_dollars"]
                if balance > peak:
                    peak = balance
                drawdown_dollars = peak - balance
                drawdown_pct = drawdown_dollars / peak if peak else 0.0
                if drawdown_dollars > max_drawdown_dollars:
                    max_drawdown_dollars = drawdown_dollars
                    max_drawdown_pct = drawdown_pct
            else:
                still_open.append(fill)
        open_positions = still_open

        batch = []
        while idx < len(trades) and trades[idx].entry_ts_ms == entry_ts:
            batch.append(trades[idx])
            idx += 1

        available_slots = max(0, max_open_positions - len(open_positions))
        if available_slots == 0:
            continue

        batch.sort(key=lambda trade: trade_priority(trade, selector_mode))
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
                daily_loss_stop_pct,
                monthly_loss_stop_pct,
            ):
                continue
            key = (trade.symbol, trade.day)
            if daily_counts.get(key, 0) >= daily_cap_per_symbol:
                continue
            open_same_symbol = [fill for fill in open_positions if fill["symbol"] == trade.symbol]
            if len(open_same_symbol) >= max_open_per_symbol:
                continue

            current_symbol_alloc = sum(fill["alloc_dollars"] for fill in open_same_symbol)
            alloc = balance * per_trade_allocation
            max_symbol_dollars = balance * max_symbol_allocation
            if current_symbol_alloc + alloc > max_symbol_dollars + 1e-9:
                continue

            slip = trade_slippage(
                trade,
                min_signal_bps,
                scenario.extra_slippage_bps,
                scenario.spread_slip_coeff,
                scenario.velocity_slip_coeff,
                scenario.size_slip_coeff,
                per_trade_allocation,
                base_allocation_ref,
            )
            net_bps = trade.gross_pnl_bps - scenario.fee_bps_roundtrip - slip
            pnl_dollars = alloc * (net_bps / 10000.0)
            open_positions.append(
                {
                    "symbol": trade.symbol,
                    "day": trade.day,
                    "month": trade.month,
                    "exit_ts_ms": trade.exit_ts_ms,
                    "alloc_dollars": alloc,
                    "pnl_dollars": pnl_dollars,
                }
            )
            fills += 1
            net_bps_sum += net_bps
            if net_bps > 0:
                win_count += 1
            daily_counts[key] = daily_counts.get(key, 0) + 1
            accepted += 1

    for fill in open_positions:
        balance += fill["pnl_dollars"]
        day = fill["day"]
        month = fill["month"]
        day_pnl[day] = day_pnl.get(day, 0.0) + fill["pnl_dollars"]
        month_pnl[month] = month_pnl.get(month, 0.0) + fill["pnl_dollars"]
        weekly_pnl[iso_week(day)] += fill["pnl_dollars"]
        monthly_realized[month] += fill["pnl_dollars"]
        if balance > peak:
            peak = balance
        drawdown_dollars = peak - balance
        drawdown_pct = drawdown_dollars / peak if peak else 0.0
        if drawdown_dollars > max_drawdown_dollars:
            max_drawdown_dollars = drawdown_dollars
            max_drawdown_pct = drawdown_pct

    positive_weeks = sum(1 for value in weekly_pnl.values() if value > 0)
    negative_weeks = sum(1 for value in weekly_pnl.values() if value < 0)
    positive_months = sum(1 for value in monthly_realized.values() if value > 0)
    negative_months = sum(1 for value in monthly_realized.values() if value < 0)
    worst_week = min(weekly_pnl.values()) if weekly_pnl else 0.0

    return {
        "scenario": scenario.name,
        "fills": fills,
        "final_capital": balance,
        "total_pnl_dollars": balance - starting_capital,
        "avg_net_bps": net_bps_sum / fills if fills else 0.0,
        "win_rate": win_count / fills if fills else 0.0,
        "max_drawdown_dollars": max_drawdown_dollars,
        "max_drawdown_pct": max_drawdown_pct,
        "positive_weeks": positive_weeks,
        "negative_weeks": negative_weeks,
        "positive_months": positive_months,
        "negative_months": negative_months,
        "worst_week_dollars": worst_week,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=OUT_DIR / "stress_matrix_replayopt_25.csv")
    parser.add_argument("--output-md", type=Path, default=OUT_DIR / "stress_matrix_replayopt_25.md")
    args = parser.parse_args()

    scenarios = [
        Scenario("base", 6.0, 1.0, 0.10, 0.05, 1.5),
        Scenario("higher_fees", 8.0, 1.0, 0.10, 0.05, 1.5),
        Scenario("higher_fixed_slip", 6.0, 2.0, 0.10, 0.05, 1.5),
        Scenario("higher_variable_slip", 6.0, 1.0, 0.15, 0.08, 1.5),
        Scenario("higher_size_slip", 6.0, 1.0, 0.10, 0.05, 2.0),
        Scenario("harsh_combo", 8.0, 2.0, 0.15, 0.08, 2.0),
        Scenario("very_harsh_combo", 10.0, 3.0, 0.20, 0.10, 2.5),
    ]

    rows = [run_scenario(args.input, scenario) for scenario in scenarios]

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scenario",
                "fills",
                "final_capital",
                "total_pnl_dollars",
                "avg_net_bps",
                "win_rate",
                "max_drawdown_dollars",
                "max_drawdown_pct",
                "positive_weeks",
                "negative_weeks",
                "positive_months",
                "negative_months",
                "worst_week_dollars",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["scenario"],
                    row["fills"],
                    f"{row['final_capital']:.2f}",
                    f"{row['total_pnl_dollars']:.2f}",
                    f"{row['avg_net_bps']:.6f}",
                    f"{row['win_rate']:.6f}",
                    f"{row['max_drawdown_dollars']:.2f}",
                    f"{row['max_drawdown_pct']:.6f}",
                    row["positive_weeks"],
                    row["negative_weeks"],
                    row["positive_months"],
                    row["negative_months"],
                    f"{row['worst_week_dollars']:.2f}",
                ]
            )

    base_row = rows[0]
    positive_rows = [row for row in rows if row["total_pnl_dollars"] > 0]
    lines = [
        "# Stress Matrix: Replay-Optimized 25% Sleeve",
        "",
        f"- Input: {args.input}",
        "- Strategy is frozen; only execution costs change across scenarios.",
        "",
        "## Results",
        "",
        "| Scenario | PnL Dollars | Avg Net bps | Win Rate | Max DD % | Neg Weeks | Neg Months |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['total_pnl_dollars']:.2f} | {row['avg_net_bps']:.4f} | {row['win_rate']:.2%} | {row['max_drawdown_pct']:.2%} | {row['negative_weeks']} | {row['negative_months']} |"
        )
    lines.extend(
        [
            "",
            "## Survival Summary",
            "",
            f"- Positive scenarios: {len(positive_rows)} / {len(rows)}",
            f"- Base case PnL: ${base_row['total_pnl_dollars']:.2f}",
            "",
        ]
    )
    harsh = next((row for row in rows if row["scenario"] == "harsh_combo"), None)
    very_harsh = next((row for row in rows if row["scenario"] == "very_harsh_combo"), None)
    if harsh is not None:
        lines.append(f"- Harsh combo remains {'positive' if harsh['total_pnl_dollars'] > 0 else 'negative'} at ${harsh['total_pnl_dollars']:.2f}")
    if very_harsh is not None:
        lines.append(f"- Very harsh combo remains {'positive' if very_harsh['total_pnl_dollars'] > 0 else 'negative'} at ${very_harsh['total_pnl_dollars']:.2f}")
    lines.append("")
    args.output_md.write_text("\n".join(lines))

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
