#!/usr/bin/env python3
"""Scan microstructure-gated multi-symbol variants on locally covered symbols."""

from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import crv_microstructure_audit as micro


OUT_DIR = Path(__file__).resolve().parent / "out"


def format_seconds(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def covered_symbols() -> list[str]:
    symbols = []
    common = sorted(
        set(path.name for path in micro.base.BINANCE.iterdir() if path.is_dir())
        & set(path.name for path in micro.base.BYBIT.iterdir() if path.is_dir())
    )
    for symbol in common:
        bn_dir = micro.base.BINANCE / symbol
        bb_dir = micro.base.BYBIT / symbol
        overlap = (
            micro.base.collect_dates(bn_dir, "trades.csv")
            & micro.base.collect_dates(bn_dir, "bookDepth.csv")
            & micro.base.collect_dates(bb_dir, "trades.csv")
            & micro.base.collect_dates(bb_dir, "orderbook.jsonl")
            & micro.base.collect_dates(bn_dir, "kline_1m.csv")
            & micro.base.collect_dates(bb_dir, "kline_1m.csv")
        )
        if overlap:
            symbols.append(symbol)
    return symbols


def load_symbol(symbol: str, taker_fee_bps_roundtrip: float) -> tuple[str, list[micro.MicroTrade]]:
    bn_dir = micro.base.BINANCE / symbol
    bb_dir = micro.base.BYBIT / symbol
    days = sorted(
        micro.base.collect_dates(bn_dir, "trades.csv")
        & micro.base.collect_dates(bn_dir, "bookDepth.csv")
        & micro.base.collect_dates(bb_dir, "trades.csv")
        & micro.base.collect_dates(bb_dir, "orderbook.jsonl")
        & micro.base.collect_dates(bn_dir, "kline_1m.csv")
        & micro.base.collect_dates(bb_dir, "kline_1m.csv")
    )
    rows: list[micro.MicroTrade] = []
    for day in days:
        rows.extend(micro.audit_day(symbol, day, taker_fee_bps_roundtrip))
    return symbol, rows


def passes_gate(
    row: micro.MicroTrade,
    min_score: float,
    max_book_spread_bps: float,
    flow_mode: str,
) -> bool:
    if row.score < min_score:
        return False
    if row.bybit_book_spread_bps > max_book_spread_bps:
        return False
    if flow_mode == "nonpos" and row.combined_flow_usd > 0.0:
        return False
    if flow_mode == "pos" and row.combined_flow_usd <= 0.0:
        return False
    if flow_mode == "smallabs" and abs(row.combined_flow_usd) > 10_000.0:
        return False
    return True


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def write_outputs(results: list[tuple], args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "multi_symbol_micro_scan.csv"
    md_path = OUT_DIR / "multi_symbol_micro_scan.md"

    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "min_score",
                "max_book_spread_bps",
                "flow_mode",
                "symbols_used",
                "positive_symbols",
                "total_trades",
                "portfolio_avg_net_taker_bps",
                "portfolio_win_rate",
                "top_symbols",
            ]
        )
        writer.writerows(results)

    lines = [
        "# Multi-Symbol Microstructure Scan",
        "",
        f"- Covered symbols scanned: {args.symbol_count}",
        f"- Minimum trades per symbol kept: {args.min_symbol_trades}",
        f"- Minimum positive symbols required: {args.min_positive_symbols}",
        "",
        "| Score | Max Book bps | Flow | Symbols Used | Positive Symbols | Trades | Portfolio Avg bps | Win Rate | Top Symbols |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in results[: args.top]:
        lines.append(
            f"| {row[0]:.2f} | {row[1]:.2f} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]:.4f} | {100.0 * row[7]:.2f}% | {row[8]} |"
        )
    md_path.write_text("\n".join(lines) + "\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--taker-fee-bps-roundtrip", type=float, default=20.0)
    parser.add_argument("--min-symbol-trades", type=int, default=5)
    parser.add_argument("--min-positive-symbols", type=int, default=2)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    symbols = covered_symbols()
    args.symbol_count = len(symbols)
    print(
        f"Loading microstructure data: {len(symbols)} symbols, workers={max(1, args.workers)}"
    )

    rows_by_symbol: dict[str, list[micro.MicroTrade]] = {}
    started_at = perf_counter()
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(load_symbol, symbol, args.taker_fee_bps_roundtrip): symbol for symbol in symbols
        }
        for future in as_completed(futures):
            symbol, rows = future.result()
            rows_by_symbol[symbol] = rows
            completed += 1
            elapsed = perf_counter() - started_at
            avg_per_symbol = elapsed / completed
            eta = avg_per_symbol * (len(symbols) - completed)
            print(
                f"[{completed}/{len(symbols)}] {symbol}: triggers={len(rows)}"
                f" | elapsed={format_seconds(elapsed)} | eta={format_seconds(eta)}"
            )

    results: list[tuple] = []
    score_grid = [14.0, 16.0, 18.0, 20.0]
    book_grid = [12.0, 10.0, 8.0, 6.0]
    flow_grid = ["any", "nonpos", "pos", "smallabs"]

    for min_score in score_grid:
        for max_book in book_grid:
            for flow_mode in flow_grid:
                symbol_stats = []
                portfolio_values: list[float] = []
                for symbol, rows in rows_by_symbol.items():
                    subset = [
                        row
                        for row in rows
                        if passes_gate(row, min_score, max_book, flow_mode)
                    ]
                    if len(subset) < args.min_symbol_trades:
                        continue
                    vals = [row.net_taker_bps for row in subset]
                    avg_net = avg(vals)
                    symbol_stats.append((symbol, len(subset), avg_net))
                    portfolio_values.extend(vals)
                if not symbol_stats:
                    continue
                positive_symbols = [row for row in symbol_stats if row[2] > 0]
                if len(positive_symbols) < args.min_positive_symbols:
                    continue
                if not portfolio_values:
                    continue
                portfolio_avg = avg(portfolio_values)
                win_rate = sum(1 for value in portfolio_values if value > 0) / len(portfolio_values)
                top_symbols = ", ".join(
                    symbol for symbol, _, _ in sorted(symbol_stats, key=lambda row: row[2], reverse=True)[:3]
                )
                results.append(
                    (
                        min_score,
                        max_book,
                        flow_mode,
                        len(symbol_stats),
                        len(positive_symbols),
                        len(portfolio_values),
                        portfolio_avg,
                        win_rate,
                        top_symbols,
                    )
                )

    results.sort(key=lambda row: (row[6], row[4], row[5]), reverse=True)
    write_outputs(results, args)


if __name__ == "__main__":
    main()
