#!/usr/bin/env python3
"""Anti-overfit validation for multi-symbol trade-flow gates."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import multi_symbol_trade_flow_scan as scan


OUT_DIR = Path(__file__).resolve().parent / "out"


@dataclass
class SplitMetrics:
    trades: int
    avg_bps: float
    win_rate: float


@dataclass
class SymbolSplit:
    train: list[dict]
    valid: list[dict]
    holdout: list[dict]


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def win_rate(rows: list[dict]) -> float:
    if not rows:
        return math.nan
    return sum(1 for row in rows if row["net_taker_bps"] > 0) / len(rows)


def split_rows(rows: list[dict]) -> SymbolSplit | None:
    if not rows:
        return None
    by_day: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_day[row["day"]].append(row)
    days = sorted(by_day)
    if len(days) < 6:
        return None

    n_days = len(days)
    train_days = max(3, int(n_days * 0.6))
    valid_days = max(1, int(n_days * 0.2))
    holdout_days = n_days - train_days - valid_days
    if holdout_days < 1:
        holdout_days = 1
        if train_days > 3:
            train_days -= 1
        else:
            valid_days = max(1, valid_days - 1)

    train_set = set(days[:train_days])
    valid_set = set(days[train_days : train_days + valid_days])
    holdout_set = set(days[train_days + valid_days :])

    return SymbolSplit(
        train=[row for row in rows if row["day"] in train_set],
        valid=[row for row in rows if row["day"] in valid_set],
        holdout=[row for row in rows if row["day"] in holdout_set],
    )


def passes_gate(row: dict, min_score: float, flow_mode: str, max_abs_flow: float) -> bool:
    return scan.passes_gate(row, min_score, flow_mode, max_abs_flow)


def evaluate_split(rows: list[dict]) -> SplitMetrics:
    values = [row["net_taker_bps"] for row in rows]
    return SplitMetrics(
        trades=len(rows),
        avg_bps=avg(values),
        win_rate=win_rate(rows),
    )


def load_symbol(symbol: str, fee_bps: float) -> tuple[str, list[dict]]:
    return scan.load_symbol(symbol, fee_bps)


def top_share(symbol_holdout_nets: dict[str, float]) -> float:
    positives = {k: v for k, v in symbol_holdout_nets.items() if v > 0}
    total = sum(positives.values())
    if total <= 0:
        return math.nan
    return max(positives.values()) / total


def run_validation(args: argparse.Namespace) -> list[tuple]:
    symbols = scan.trade_covered_symbols(args.min_overlap_days)
    print(f"Loading symbols: {len(symbols)} (workers={args.workers})")

    rows_by_symbol: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(load_symbol, symbol, args.taker_fee_bps_roundtrip): symbol for symbol in symbols}
        done = 0
        for future in as_completed(futures):
            symbol, rows = future.result()
            rows_by_symbol[symbol] = rows
            done += 1
            print(f"[{done}/{len(symbols)}] {symbol}: triggers={len(rows)}")

    split_by_symbol: dict[str, SymbolSplit] = {}
    for symbol, rows in rows_by_symbol.items():
        split = split_rows(rows)
        if split is not None:
            split_by_symbol[symbol] = split

    print(f"Symbols with usable temporal splits: {len(split_by_symbol)}")

    score_grid = [14.0, 16.0, 18.0, 20.0]
    flow_grid = ["any", "nonpos", "pos", "smallabs"]
    flow_caps = [5_000.0, 10_000.0, 20_000.0]

    results = []
    for min_score in score_grid:
        for flow_mode in flow_grid:
            for flow_cap in flow_caps:
                symbol_metrics = {}
                symbol_holdout_nets = {}
                total_holdout_rows = []
                for symbol, split in split_by_symbol.items():
                    train = [row for row in split.train if passes_gate(row, min_score, flow_mode, flow_cap)]
                    valid = [row for row in split.valid if passes_gate(row, min_score, flow_mode, flow_cap)]
                    holdout = [row for row in split.holdout if passes_gate(row, min_score, flow_mode, flow_cap)]
                    if (
                        len(train) < args.min_symbol_trades_split
                        or len(valid) < args.min_symbol_trades_split
                        or len(holdout) < args.min_symbol_trades_split
                    ):
                        continue
                    train_m = evaluate_split(train)
                    valid_m = evaluate_split(valid)
                    holdout_m = evaluate_split(holdout)
                    symbol_metrics[symbol] = (train_m, valid_m, holdout_m)
                    symbol_holdout_nets[symbol] = sum(row["net_taker_bps"] for row in holdout)
                    total_holdout_rows.extend(holdout)

                if not symbol_metrics:
                    continue
                positive_symbols = [
                    symbol for symbol, (_, _, holdout_m) in symbol_metrics.items() if holdout_m.avg_bps > 0
                ]
                if len(positive_symbols) < args.min_positive_symbols:
                    continue

                portfolio_train = avg([m[0].avg_bps for m in symbol_metrics.values()])
                portfolio_valid = avg([m[1].avg_bps for m in symbol_metrics.values()])
                portfolio_holdout = avg([m[2].avg_bps for m in symbol_metrics.values()])
                if any(math.isnan(value) for value in [portfolio_train, portfolio_valid, portfolio_holdout]):
                    continue
                if (
                    portfolio_train <= 0
                    or portfolio_valid <= 0
                    or portfolio_holdout <= 0
                ):
                    continue

                holdout_top_share = top_share(symbol_holdout_nets)
                if not math.isnan(holdout_top_share) and holdout_top_share > args.max_top_symbol_share:
                    continue

                holdout_trades = len(total_holdout_rows)
                if holdout_trades < args.min_holdout_trades:
                    continue

                holdout_win = win_rate(total_holdout_rows)
                top_symbols = ", ".join(
                    symbol
                    for symbol, _ in sorted(
                        symbol_holdout_nets.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:4]
                )
                results.append(
                    (
                        min_score,
                        flow_mode,
                        flow_cap,
                        len(symbol_metrics),
                        len(positive_symbols),
                        holdout_trades,
                        portfolio_train,
                        portfolio_valid,
                        portfolio_holdout,
                        holdout_win,
                        holdout_top_share,
                        top_symbols,
                    )
                )

    results.sort(key=lambda row: (row[8], row[4], row[5]), reverse=True)
    return results


def write_outputs(results: list[tuple], args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "anti_overfit_validation.csv"
    md_path = OUT_DIR / "anti_overfit_validation.md"

    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "min_score",
                "flow_mode",
                "max_abs_flow",
                "symbols_kept",
                "positive_symbols",
                "holdout_trades",
                "portfolio_train_avg_bps",
                "portfolio_valid_avg_bps",
                "portfolio_holdout_avg_bps",
                "holdout_win_rate",
                "holdout_top_symbol_share",
                "top_holdout_symbols",
            ]
        )
        writer.writerows(results)

    lines = [
        "# Anti-Overfit Validation",
        "",
        "A config is accepted only if it passes all of:",
        f"- minimum positive symbols: {args.min_positive_symbols}",
        f"- minimum trades per symbol per split: {args.min_symbol_trades_split}",
        f"- minimum holdout trades total: {args.min_holdout_trades}",
        f"- positive portfolio avg in train, validation, and holdout",
        f"- holdout top-symbol contribution <= {args.max_top_symbol_share:.2f}",
        "",
    ]
    if not results:
        lines.extend(
            [
                "## Result",
                "",
                "No configuration passed the anti-overfit bar on current local data.",
            ]
        )
    else:
        lines.extend(
            [
                "## Passing Configurations",
                "",
                "| Score | Flow | Max Abs Flow | Symbols | Positive Symbols | Holdout Trades | Train bps | Valid bps | Holdout bps | Holdout Win | Top Share | Top Symbols |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in results[: args.top]:
            lines.append(
                f"| {row[0]:.2f} | {row[1]} | {row[2]:.0f} | {row[3]} | {row[4]} | {row[5]} | {row[6]:.4f} | {row[7]:.4f} | {row[8]:.4f} | {100.0 * row[9]:.2f}% | {row[10]:.2f} | {row[11]} |"
            )

    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--taker-fee-bps-roundtrip", type=float, default=20.0)
    parser.add_argument("--min-overlap-days", type=int, default=5)
    parser.add_argument("--min-positive-symbols", type=int, default=5)
    parser.add_argument("--min-symbol-trades-split", type=int, default=3)
    parser.add_argument("--min-holdout-trades", type=int, default=40)
    parser.add_argument("--max-top-symbol-share", type=float, default=0.45)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    results = run_validation(args)
    write_outputs(results, args)


if __name__ == "__main__":
    main()
