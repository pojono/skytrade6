#!/usr/bin/env python3
"""Forward-only rolling paper runner for the frozen replay-optimized 25% strategy."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from paper_trade_candidate import load_trades, should_block_new_trade, trade_priority, trade_slippage


OUT_DIR = Path(__file__).resolve().parent / "out"
DEFAULT_INPUT = OUT_DIR / "candidate_trades_v3_replayopt.csv"
DEFAULT_LOG = OUT_DIR / "rolling_paper_log_replayopt_25.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--starting-capital", type=float, default=100000.0)
    args = parser.parse_args()

    # Frozen strategy configuration.
    per_trade_allocation = 0.25
    max_open_positions = 1
    max_open_per_symbol = 1
    max_symbol_allocation = 0.25
    daily_cap_per_symbol = 3
    selector_mode = "spread"
    daily_loss_stop_pct = 0.01
    monthly_loss_stop_pct = 0.03
    min_signal_bps = 10.0
    fee_bps_roundtrip = 6.0
    extra_slippage_bps = 1.0
    spread_slip_coeff = 0.10
    velocity_slip_coeff = 0.05
    size_slip_coeff = 1.5
    base_allocation_ref = 0.10

    trades = load_trades(args.input)
    existing_days: set[str] = set()
    balance = args.starting_capital
    historical_month_pnl: dict[str, float] = {}
    if args.log.exists():
        with args.log.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                day = row["day"]
                month = row["month"]
                existing_days.add(day)
                balance = float(row["equity_end_dollars"])
                historical_month_pnl[month] = float(row["month_realized_pnl_dollars"])

    filtered = [trade for trade in trades if trade.day not in existing_days]
    filtered.sort(key=lambda trade: (trade.entry_ts_ms, trade.symbol))

    open_positions = []
    daily_counts: dict[tuple[str, str], int] = {}
    day_pnl: dict[str, float] = {}
    month_pnl: dict[str, float] = dict(historical_month_pnl)
    day_start_equity: dict[str, float] = {}
    month_start_equity: dict[str, float] = {}
    if historical_month_pnl:
        last_month = max(historical_month_pnl)
        month_start_equity[last_month] = balance - historical_month_pnl[last_month]

    idx = 0
    while idx < len(filtered):
        entry_ts = filtered[idx].entry_ts_ms

        still_open = []
        for fill in open_positions:
            if fill["exit_ts_ms"] <= entry_ts:
                balance += fill["pnl_dollars"]
                day_pnl[fill["day"]] = day_pnl.get(fill["day"], 0.0) + fill["pnl_dollars"]
                month_pnl[fill["month"]] = month_pnl.get(fill["month"], 0.0) + fill["pnl_dollars"]
            else:
                still_open.append(fill)
        open_positions = still_open

        batch = []
        while idx < len(filtered) and filtered[idx].entry_ts_ms == entry_ts:
            batch.append(filtered[idx])
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

            alloc = balance * per_trade_allocation
            max_symbol_dollars = balance * max_symbol_allocation
            current_symbol_alloc = sum(fill["alloc_dollars"] for fill in open_same_symbol)
            if current_symbol_alloc + alloc > max_symbol_dollars + 1e-9:
                continue

            slip = trade_slippage(
                trade,
                min_signal_bps,
                extra_slippage_bps,
                spread_slip_coeff,
                velocity_slip_coeff,
                size_slip_coeff,
                per_trade_allocation,
                base_allocation_ref,
            )
            net_bps = trade.gross_pnl_bps - fee_bps_roundtrip - slip
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
            daily_counts[key] = daily_counts.get(key, 0) + 1
            accepted += 1

    for fill in open_positions:
        balance += fill["pnl_dollars"]
        day_pnl[fill["day"]] = day_pnl.get(fill["day"], 0.0) + fill["pnl_dollars"]
        month_pnl[fill["month"]] = month_pnl.get(fill["month"], 0.0) + fill["pnl_dollars"]

    rolling_rows = []
    for day in sorted(day_pnl):
        month = day[:7]
        start_eq = day_start_equity.get(day, balance - day_pnl[day])
        rolling_rows.append(
            (
                day,
                month,
                f"{start_eq:.2f}",
                f"{day_pnl[day]:.2f}",
                f"{(day_pnl[day] / start_eq * 100.0) if start_eq else 0.0:.4f}",
                f"{month_pnl.get(month, 0.0):.2f}",
                f"{(start_eq + day_pnl[day]):.2f}",
            )
        )

    write_header = not args.log.exists()
    with args.log.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(
                [
                    "day",
                    "month",
                    "equity_start_dollars",
                    "day_pnl_dollars",
                    "day_return_pct",
                    "month_realized_pnl_dollars",
                    "equity_end_dollars",
                ]
            )
        writer.writerows(rolling_rows)

    print(f"Appended days: {len(rolling_rows)}")
    print(f"Ending equity: {balance:.2f}")
    print(f"Wrote {args.log}")


if __name__ == "__main__":
    main()
