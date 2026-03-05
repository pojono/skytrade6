#!/usr/bin/env python3
"""Scan symbols for the strict extreme-spread cross-exchange rule."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import extreme_spread_crv as base


OUT_DIR = Path(__file__).resolve().parent / "out"


def format_seconds(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def overlap_dates(symbol: str) -> list[str]:
    bn_dir = base.BINANCE / symbol
    bb_dir = base.BYBIT / symbol
    return sorted(
        base.collect_dates(bn_dir, "kline_1m.csv")
        & base.collect_dates(bb_dir, "kline_1m.csv")
        & base.collect_dates(bn_dir, "metrics.csv")
        & base.collect_dates(bb_dir, "long_short_ratio_5min.csv")
        & base.collect_dates(bb_dir, "open_interest_5min.csv")
        & base.collect_dates(bn_dir, "mark_price_kline_1m.csv")
        & base.collect_dates(bn_dir, "index_price_kline_1m.csv")
        & base.collect_dates(bb_dir, "premium_index_kline_1m.csv")
    )


def scan_symbol(symbol: str, args: argparse.Namespace) -> tuple | None:
    dates = overlap_dates(symbol)
    if len(dates) < args.min_overlap_days:
        return None
    dates = dates[-args.recent_days :] if args.recent_days > 0 else dates
    trades = []
    for day in dates:
        trades.extend(
            base.build_day_trades(
                symbol,
                day,
                args.min_spread_bps,
                args.min_ls,
                args.min_oi_bps,
                args.min_carry_bps,
                args.min_score,
            )
        )
    trades = base.apply_daily_cap(trades, args.daily_cap)
    months = sorted({trade.month for trade in trades})
    if len(trades) < args.min_trades or len(months) < (args.test_months + 1):
        return None

    test_months = months[-args.test_months :]
    train_months = months[: -args.test_months]
    train = [trade for trade in trades if trade.month in set(train_months)]
    test = [trade for trade in trades if trade.month in set(test_months)]
    if len(train) < args.min_split_trades or len(test) < args.min_split_trades:
        return None

    train_taker = base.summarize_fee(train, args.taker_fee_bps_roundtrip)
    test_taker = base.summarize_fee(test, args.taker_fee_bps_roundtrip)
    train_maker = base.summarize_fee(train, args.maker_fee_bps_roundtrip)
    test_maker = base.summarize_fee(test, args.maker_fee_bps_roundtrip)

    return (
        symbol,
        len(dates),
        len(months),
        len(train),
        train_taker["avg_net_bps"],
        len(test),
        test_taker["avg_net_bps"],
        test_taker["win_rate"],
        train_maker["avg_net_bps"],
        test_maker["avg_net_bps"],
    )


def load_default_symbols() -> list[str]:
    return sorted(
        set(path.name for path in base.BINANCE.iterdir() if path.is_dir())
        & set(path.name for path in base.BYBIT.iterdir() if path.is_dir())
    )


def write_outputs(rows: list[tuple], limit: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "scan_leaderboard.csv"
    md_path = OUT_DIR / "scan_report.md"

    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "symbol",
                "days_scanned",
                "months",
                "train_trades",
                "train_avg_net_taker_bps",
                "test_trades",
                "test_avg_net_taker_bps",
                "test_win_rate",
                "train_avg_net_maker_bps",
                "test_avg_net_maker_bps",
            ]
        )
        writer.writerows(rows)

    lines = [
        "# Extreme Spread Scan Report",
        "",
        f"- Symbols passing filters: {len(rows)}",
        "",
        "| Symbol | Train Trades | Train Taker bps | Test Trades | Test Taker bps | Test Win Rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:limit]:
        lines.append(
            f"| {row[0]} | {row[3]} | {row[4]:.4f} | {row[5]} | {row[6]:.4f} | {100.0 * row[7]:.2f}% |"
        )
    md_path.write_text("\n".join(lines) + "\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--recent-days", type=int, default=210)
    parser.add_argument("--test-months", type=int, default=2)
    parser.add_argument("--min-overlap-days", type=int, default=90)
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--min-split-trades", type=int, default=10)
    parser.add_argument("--min-spread-bps", type=float, default=32.0)
    parser.add_argument("--min-ls", type=float, default=0.15)
    parser.add_argument("--min-oi-bps", type=float, default=5.0)
    parser.add_argument("--min-carry-bps", type=float, default=2.0)
    parser.add_argument("--min-score", type=float, default=14.0)
    parser.add_argument("--daily-cap", type=int, default=3)
    parser.add_argument("--maker-fee-bps-roundtrip", type=float, default=8.0)
    parser.add_argument("--taker-fee-bps-roundtrip", type=float, default=20.0)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.symbols.strip():
        symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
    else:
        symbols = load_default_symbols()

    print(
        f"Starting scan: {len(symbols)} symbols, workers={max(1, args.workers)}, "
        f"recent_days={args.recent_days}, spread>={args.min_spread_bps:.2f} bps"
    )

    rows = []
    started_at = perf_counter()
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(scan_symbol, symbol, args): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            completed += 1
            row = future.result()
            if row is not None:
                rows.append(row)
            elapsed = perf_counter() - started_at
            avg_per_symbol = elapsed / completed if completed else 0.0
            remaining = len(symbols) - completed
            eta = avg_per_symbol * remaining
            status = "PASS" if row is not None else "skip"
            extra = ""
            if row is not None:
                extra = (
                    f", test_trades={row[5]}, test_taker={row[6]:.4f} bps, "
                    f"win_rate={100.0 * row[7]:.2f}%"
                )
            print(
                f"[{completed}/{len(symbols)}] {symbol}: {status}"
                f"{extra} | survivors={len(rows)} | elapsed={format_seconds(elapsed)}"
                f" | eta={format_seconds(eta)}"
            )

    rows.sort(key=lambda row: (row[6], row[5], row[4], row[0]), reverse=True)
    write_outputs(rows, args.top)


if __name__ == "__main__":
    main()
