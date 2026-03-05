#!/usr/bin/env python3
"""Walk-forward robust strategy with anti-overfit constraints."""

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
class GateConfig:
    min_score: float
    flow_mode: str
    max_abs_flow: float


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def pass_gate(row: dict, cfg: GateConfig) -> bool:
    return scan.passes_gate(row, cfg.min_score, cfg.flow_mode, cfg.max_abs_flow)


def select_config(
    rows_by_symbol: dict[str, list[dict]],
    train_days: set[str],
    valid_days: set[str],
    args: argparse.Namespace,
) -> GateConfig | None:
    candidates = [
        GateConfig(min_score, flow_mode, cap)
        for min_score in [14.0, 16.0, 18.0, 20.0]
        for flow_mode in ["any", "nonpos", "pos", "smallabs"]
        for cap in [5_000.0, 10_000.0, 20_000.0]
    ]
    best_cfg: GateConfig | None = None
    best_score = float("-inf")

    for cfg in candidates:
        per_symbol_train = {}
        per_symbol_valid = {}
        for symbol, rows in rows_by_symbol.items():
            tr = [r for r in rows if r["day"] in train_days and pass_gate(r, cfg)]
            va = [r for r in rows if r["day"] in valid_days and pass_gate(r, cfg)]
            if len(tr) < args.min_symbol_trades_train or len(va) < args.min_symbol_trades_valid:
                continue
            per_symbol_train[symbol] = tr
            per_symbol_valid[symbol] = va
        if len(per_symbol_train) < args.min_symbols:
            continue

        train_symbol_avg = {s: avg([r["net_taker_bps"] for r in rs]) for s, rs in per_symbol_train.items()}
        valid_symbol_avg = {s: avg([r["net_taker_bps"] for r in rs]) for s, rs in per_symbol_valid.items()}
        positive_symbols = [s for s, val in valid_symbol_avg.items() if val > 0]
        if len(positive_symbols) < args.min_positive_symbols:
            continue

        train_port = avg(list(train_symbol_avg.values()))
        valid_port = avg(list(valid_symbol_avg.values()))
        if train_port <= 0 or valid_port <= 0:
            continue

        valid_pos_sum = sum(v for v in valid_symbol_avg.values() if v > 0)
        if valid_pos_sum > 0:
            top_share = max(v for v in valid_symbol_avg.values() if v > 0) / valid_pos_sum
            if top_share > args.max_top_symbol_share:
                continue

        # Conservative objective: prioritize validation edge and breadth.
        score = valid_port + 0.2 * train_port + 0.1 * len(positive_symbols)
        if score > best_score:
            best_score = score
            best_cfg = cfg
    return best_cfg


def execute_day(
    rows_by_symbol: dict[str, list[dict]],
    test_day: str,
    cfg: GateConfig,
    args: argparse.Namespace,
) -> list[dict]:
    selected = []
    for symbol, rows in rows_by_symbol.items():
        day_rows = [r for r in rows if r["day"] == test_day and pass_gate(r, cfg)]
        if not day_rows:
            continue
        day_rows.sort(key=lambda r: r["score"], reverse=True)
        selected.extend(day_rows[: args.max_trades_per_symbol_day])
    return selected


def build_report(
    trades: list[dict],
    day_configs: list[tuple[str, GateConfig]],
    args: argparse.Namespace,
) -> str:
    vals = [r["net_taker_bps"] for r in trades]
    by_symbol = defaultdict(list)
    by_month = defaultdict(list)
    for r in trades:
        by_symbol[r["symbol"]].append(r["net_taker_bps"])
        by_month[r["month"]].append(r["net_taker_bps"])

    lines = [
        "# Robust Walk-Forward Strategy",
        "",
        "## Setup",
        "",
        f"- Universe source: trade-flow covered symbols (min overlap days={args.min_overlap_days})",
        f"- Walk-forward warmup days: {args.warmup_days}",
        f"- Validation tail days per step: {args.validation_days}",
        f"- Min symbols per config: {args.min_symbols}",
        f"- Min positive symbols in validation: {args.min_positive_symbols}",
        f"- Max top-symbol share in validation: {args.max_top_symbol_share:.2f}",
        f"- Max trades per symbol per day: {args.max_trades_per_symbol_day}",
        "",
        "## Outcome",
        "",
        f"- Days traded: {len({d for d, _ in day_configs})}",
        f"- Total trades: {len(trades)}",
        f"- Avg net after taker fees: {avg(vals):.4f} bps",
        f"- Win rate: {100.0 * sum(1 for v in vals if v > 0) / len(vals):.2f}%" if vals else "- Win rate: nan",
        "",
        "## Symbol",
        "",
        "| Symbol | Trades | Avg Net bps |",
        "|---|---:|---:|",
    ]
    for symbol, svals in sorted(by_symbol.items(), key=lambda kv: avg(kv[1]), reverse=True):
        lines.append(f"| {symbol} | {len(svals)} | {avg(svals):.4f} |")

    lines.extend(
        [
            "",
            "## Monthly",
            "",
            "| Month | Trades | Avg Net bps |",
            "|---|---:|---:|",
        ]
    )
    for month in sorted(by_month):
        mvals = by_month[month]
        lines.append(f"| {month} | {len(mvals)} | {avg(mvals):.4f} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Gate selection is strictly walk-forward: only prior days are used to choose the next-day config.",
            "- This is still bounded by local data coverage; if almost all symbols have zero triggers, diversification remains limited.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(trades: list[dict], day_configs: list[tuple[str, GateConfig]], args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trade_path = OUT_DIR / "robust_walkforward_trades.csv"
    cfg_path = OUT_DIR / "robust_walkforward_configs.csv"
    report_path = OUT_DIR / "robust_walkforward_report.md"

    with trade_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["day", "month", "symbol", "net_taker_bps", "score", "combined_flow_usd", "spread_abs_bps"])
        for r in trades:
            writer.writerow(
                [
                    r["day"],
                    r["month"],
                    r["symbol"],
                    f"{r['net_taker_bps']:.6f}",
                    f"{r['score']:.6f}",
                    f"{r['combined_flow_usd']:.6f}",
                    f"{r['spread_abs_bps']:.6f}",
                ]
            )

    with cfg_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["day", "min_score", "flow_mode", "max_abs_flow"])
        for day, cfg in day_configs:
            writer.writerow([day, f"{cfg.min_score:.2f}", cfg.flow_mode, f"{cfg.max_abs_flow:.0f}"])

    report_path.write_text(build_report(trades, day_configs, args))
    print(f"Wrote {trade_path}")
    print(f"Wrote {cfg_path}")
    print(f"Wrote {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--taker-fee-bps-roundtrip", type=float, default=20.0)
    parser.add_argument("--min-overlap-days", type=int, default=5)
    parser.add_argument("--warmup-days", type=int, default=10)
    parser.add_argument("--validation-days", type=int, default=4)
    parser.add_argument("--min-symbols", type=int, default=2)
    parser.add_argument("--min-positive-symbols", type=int, default=2)
    parser.add_argument("--min-symbol-trades-train", type=int, default=2)
    parser.add_argument("--min-symbol-trades-valid", type=int, default=1)
    parser.add_argument("--max-top-symbol-share", type=float, default=0.80)
    parser.add_argument("--max-trades-per-symbol-day", type=int, default=1)
    args = parser.parse_args()

    symbols = scan.trade_covered_symbols(args.min_overlap_days)
    rows_by_symbol: dict[str, list[dict]] = {}
    print(f"Loading rows for {len(symbols)} symbols")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(scan.load_symbol, symbol, args.taker_fee_bps_roundtrip): symbol for symbol in symbols}
        done = 0
        for future in as_completed(futures):
            symbol, rows = future.result()
            rows_by_symbol[symbol] = rows
            done += 1
            print(f"[{done}/{len(symbols)}] {symbol}: {len(rows)} rows")

    all_days = sorted({row["day"] for rows in rows_by_symbol.values() for row in rows})
    trades = []
    day_configs: list[tuple[str, GateConfig]] = []
    for idx in range(args.warmup_days, len(all_days)):
        test_day = all_days[idx]
        train_window = all_days[:idx]
        if len(train_window) <= args.validation_days:
            continue
        train_days = set(train_window[: -args.validation_days])
        valid_days = set(train_window[-args.validation_days :])
        cfg = select_config(rows_by_symbol, train_days, valid_days, args)
        if cfg is None:
            continue
        day_configs.append((test_day, cfg))
        trades.extend(execute_day(rows_by_symbol, test_day, cfg, args))

    if not trades:
        raise SystemExit("No trades generated by walk-forward robust strategy.")

    write_outputs(trades, day_configs, args)


if __name__ == "__main__":
    main()
