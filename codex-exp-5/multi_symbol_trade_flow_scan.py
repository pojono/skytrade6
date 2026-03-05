#!/usr/bin/env python3
"""Scan broader multi-symbol variants using two-exchange trade flow only."""

from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import crv_microstructure_audit as micro
import extreme_spread_crv as base


OUT_DIR = Path(__file__).resolve().parent / "out"


def format_seconds(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def trade_covered_symbols(min_overlap_days: int) -> list[str]:
    common = sorted(
        set(path.name for path in base.BINANCE.iterdir() if path.is_dir())
        & set(path.name for path in base.BYBIT.iterdir() if path.is_dir())
    )
    out = []
    for symbol in common:
        overlap = (
            base.collect_dates(base.BINANCE / symbol, "trades.csv")
            & base.collect_dates(base.BYBIT / symbol, "trades.csv")
            & base.collect_dates(base.BINANCE / symbol, "kline_1m.csv")
            & base.collect_dates(base.BYBIT / symbol, "kline_1m.csv")
            & base.collect_dates(base.BINANCE / symbol, "metrics.csv")
            & base.collect_dates(base.BYBIT / symbol, "long_short_ratio_5min.csv")
            & base.collect_dates(base.BYBIT / symbol, "open_interest_5min.csv")
            & base.collect_dates(base.BINANCE / symbol, "mark_price_kline_1m.csv")
            & base.collect_dates(base.BINANCE / symbol, "index_price_kline_1m.csv")
            & base.collect_dates(base.BYBIT / symbol, "premium_index_kline_1m.csv")
        )
        if len(overlap) >= min_overlap_days:
            out.append(symbol)
    return out


def load_symbol(symbol: str, taker_fee_bps_roundtrip: float) -> tuple[str, list[dict]]:
    bn_dir = base.BINANCE / symbol
    bb_dir = base.BYBIT / symbol
    overlap = sorted(
        base.collect_dates(bn_dir, "trades.csv")
        & base.collect_dates(bb_dir, "trades.csv")
        & base.collect_dates(bn_dir, "kline_1m.csv")
        & base.collect_dates(bb_dir, "kline_1m.csv")
        & base.collect_dates(bn_dir, "metrics.csv")
        & base.collect_dates(bb_dir, "long_short_ratio_5min.csv")
        & base.collect_dates(bb_dir, "open_interest_5min.csv")
        & base.collect_dates(bn_dir, "mark_price_kline_1m.csv")
        & base.collect_dates(bn_dir, "index_price_kline_1m.csv")
        & base.collect_dates(bb_dir, "premium_index_kline_1m.csv")
    )
    bn_flow_cache: dict[str, list[tuple[int, float]]] = {}
    bb_flow_cache: dict[str, list[tuple[int, float]]] = {}
    rows: list[dict] = []
    for day in overlap:
        trades = base.build_day_trades(symbol, day, 32.0, 0.15, 5.0, 2.0, 14.0)
        if not trades:
            continue
        day_trades = base.apply_daily_cap(trades, 3)
        if day not in bn_flow_cache:
            bn_flow_cache[day] = micro.load_binance_flow(bn_dir / f"{day}_trades.csv")
            bb_flow_cache[day] = micro.load_bybit_flow(bb_dir / f"{day}_trades.csv")
        bn_flow = bn_flow_cache[day]
        bb_flow = bb_flow_cache[day]
        if not (bn_flow and bb_flow):
            continue
        for trade in day_trades:
            start_ts = trade.entry_ts_ms - 60_000
            end_ts = trade.entry_ts_ms
            bn_signed = micro.sum_window(bn_flow, start_ts, end_ts)
            bb_signed = micro.sum_window(bb_flow, start_ts, end_ts)
            signal_dir = 1.0 if trade.spread_bps > 0 else -1.0
            combined_flow = signal_dir * (bn_signed - bb_signed)
            rows.append(
                {
                    "symbol": symbol,
                    "day": day,
                    "month": day[:7],
                    "net_taker_bps": trade.gross_pnl_bps - taker_fee_bps_roundtrip,
                    "score": trade.score,
                    "spread_abs_bps": trade.spread_abs_bps,
                    "combined_flow_usd": combined_flow,
                }
            )
    return symbol, rows


def passes_gate(row: dict, min_score: float, flow_mode: str, max_abs_flow: float) -> bool:
    if row["score"] < min_score:
        return False
    flow = row["combined_flow_usd"]
    if flow_mode == "nonpos" and flow > 0.0:
        return False
    if flow_mode == "pos" and flow <= 0.0:
        return False
    if flow_mode == "smallabs" and abs(flow) > max_abs_flow:
        return False
    return True


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def write_outputs(results: list[tuple], args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "multi_symbol_trade_flow_scan.csv"
    md_path = OUT_DIR / "multi_symbol_trade_flow_scan.md"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "min_score",
                "flow_mode",
                "max_abs_flow",
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
        "# Multi-Symbol Trade-Flow Scan",
        "",
        f"- Covered symbols scanned: {args.symbol_count}",
        f"- Minimum trades per symbol kept: {args.min_symbol_trades}",
        f"- Minimum positive symbols required: {args.min_positive_symbols}",
        "",
        "| Score | Flow | Max Abs Flow | Symbols Used | Positive Symbols | Trades | Portfolio Avg bps | Win Rate | Top Symbols |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results[: args.top]:
        lines.append(
            f"| {row[0]:.2f} | {row[1]} | {row[2]:.0f} | {row[3]} | {row[4]} | {row[5]} | {row[6]:.4f} | {100.0 * row[7]:.2f}% | {row[8]} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--taker-fee-bps-roundtrip", type=float, default=20.0)
    parser.add_argument("--min-overlap-days", type=int, default=5)
    parser.add_argument("--min-symbol-trades", type=int, default=5)
    parser.add_argument("--min-positive-symbols", type=int, default=3)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    symbols = trade_covered_symbols(args.min_overlap_days)
    args.symbol_count = len(symbols)
    print(f"Loading trade-flow data: {len(symbols)} symbols, workers={max(1, args.workers)}")

    rows_by_symbol: dict[str, list[dict]] = {}
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
            eta = (elapsed / completed) * (len(symbols) - completed) if completed else 0.0
            print(
                f"[{completed}/{len(symbols)}] {symbol}: triggers={len(rows)}"
                f" | elapsed={format_seconds(elapsed)} | eta={format_seconds(eta)}"
            )

    results: list[tuple] = []
    score_grid = [14.0, 16.0, 18.0, 20.0]
    flow_grid = ["any", "nonpos", "pos", "smallabs"]
    flow_caps = [5_000.0, 10_000.0, 20_000.0]
    for min_score in score_grid:
        for flow_mode in flow_grid:
            for flow_cap in flow_caps:
                symbol_stats = []
                portfolio_values: list[float] = []
                for symbol, rows in rows_by_symbol.items():
                    subset = [
                        row
                        for row in rows
                        if passes_gate(row, min_score, flow_mode, flow_cap)
                    ]
                    if len(subset) < args.min_symbol_trades:
                        continue
                    vals = [row["net_taker_bps"] for row in subset]
                    avg_net = avg(vals)
                    symbol_stats.append((symbol, len(subset), avg_net))
                    portfolio_values.extend(vals)
                if not symbol_stats:
                    continue
                positive_symbols = [row for row in symbol_stats if row[2] > 0]
                if len(positive_symbols) < args.min_positive_symbols:
                    continue
                portfolio_avg = avg(portfolio_values)
                win_rate = sum(1 for value in portfolio_values if value > 0) / len(portfolio_values)
                top_symbols = ", ".join(
                    symbol for symbol, _, _ in sorted(symbol_stats, key=lambda row: row[2], reverse=True)[:4]
                )
                results.append(
                    (
                        min_score,
                        flow_mode,
                        flow_cap,
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
