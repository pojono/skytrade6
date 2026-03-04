#!/usr/bin/env python3
"""Constrained holdout test for simple microstructure filters on the recent 7-day window."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / "out"
DEFAULT_INPUT = OUT_DIR / "microstructure_window_analysis.csv"


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else math.nan


def summarize(rows: list[dict[str, str]]) -> tuple[int, float, float]:
    if not rows:
        return 0, math.nan, math.nan
    nets = [float(row["net_pnl_bps_25pct_model"]) for row in rows]
    wins = sum(1 for row in rows if row["is_winner_25pct_model"] == "1")
    return len(rows), sum(nets) / len(nets), wins / len(rows)


def passes(row: dict[str, str], cfg: dict[str, float]) -> bool:
    if float(row["bybit_trade_count_5s"]) < cfg["min_bybit_trade_count_5s"]:
        return False
    if float(row["binance_trade_count_5s"]) < cfg["min_binance_trade_count_5s"]:
        return False
    if float(row["bybit_book_spread_bps"]) > cfg["max_bybit_book_spread_bps"]:
        return False
    if float(row["bybit_trade_imbalance_5s"]) > cfg["max_bybit_trade_imbalance_5s"]:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--test-days", type=int, default=2)
    parser.add_argument("--output-csv", type=Path, default=OUT_DIR / "microstructure_filter_search.csv")
    parser.add_argument("--output-md", type=Path, default=OUT_DIR / "microstructure_filter_report.md")
    parser.add_argument("--output-filtered", type=Path, default=OUT_DIR / "microstructure_window_filtered.csv")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.input.open()))
    days = sorted({row["day"] for row in rows})
    test_days = days[-args.test_days :]
    train_days = days[: -args.test_days]
    train_day_set = set(train_days)
    test_day_set = set(test_days)
    train_rows = [row for row in rows if row["day"] in train_day_set]
    test_rows = [row for row in rows if row["day"] in test_day_set]

    search_space = {
        "min_bybit_trade_count_5s": [0.0, 2.0, 4.0, 6.0, 8.0],
        "min_binance_trade_count_5s": [0.0, 2.0, 4.0, 6.0],
        "max_bybit_book_spread_bps": [10.0, 6.0, 4.5, 3.8, 3.5],
        "max_bybit_trade_imbalance_5s": [1.0, 0.5, 0.2, 0.0],
    }
    baseline = {
        "min_bybit_trade_count_5s": 0.0,
        "min_binance_trade_count_5s": 0.0,
        "max_bybit_book_spread_bps": 10.0,
        "max_bybit_trade_imbalance_5s": 1.0,
    }

    best_cfg = dict(baseline)
    best_score = float("-inf")
    results: list[tuple[dict[str, float], tuple[int, float, float], tuple[int, float, float], float]] = []
    keys = [
        "min_bybit_trade_count_5s",
        "min_binance_trade_count_5s",
        "max_bybit_book_spread_bps",
        "max_bybit_trade_imbalance_5s",
    ]

    for _ in range(2):
        for key in keys:
            local_best_val = best_cfg[key]
            local_best_score = float("-inf")
            for value in search_space[key]:
                cfg = dict(best_cfg)
                cfg[key] = value
                keep_train = [row for row in train_rows if passes(row, cfg)]
                train_n, train_avg, train_wr = summarize(keep_train)
                if train_n < 40 or math.isnan(train_avg):
                    continue
                score = train_avg * math.log(train_n)
                keep_test = [row for row in test_rows if passes(row, cfg)]
                test_n, test_avg, test_wr = summarize(keep_test)
                results.append((dict(cfg), (train_n, train_avg, train_wr), (test_n, test_avg, test_wr), score))
                if score > local_best_score:
                    local_best_score = score
                    local_best_val = value
                if score > best_score:
                    best_score = score
                    best_cfg = dict(cfg)
            best_cfg[key] = local_best_val

    baseline_train = summarize([row for row in train_rows if passes(row, baseline)])
    baseline_test = summarize([row for row in test_rows if passes(row, baseline)])
    best_train_rows = [row for row in train_rows if passes(row, best_cfg)]
    best_test_rows = [row for row in test_rows if passes(row, best_cfg)]
    best_train = summarize(best_train_rows)
    best_test = summarize(best_test_rows)

    hypothesis_cfg = {
        "min_bybit_trade_count_5s": 4.0,
        "min_binance_trade_count_5s": 0.0,
        "max_bybit_book_spread_bps": 4.5,
        "max_bybit_trade_imbalance_5s": 1.0,
    }
    hypothesis_train = summarize([row for row in train_rows if passes(row, hypothesis_cfg)])
    hypothesis_test = summarize([row for row in test_rows if passes(row, hypothesis_cfg)])

    all_keep = [row for row in rows if passes(row, best_cfg)]
    with args.output_filtered.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(all_keep)

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "min_bybit_trade_count_5s",
                "min_binance_trade_count_5s",
                "max_bybit_book_spread_bps",
                "max_bybit_trade_imbalance_5s",
                "train_n",
                "train_avg_net_bps",
                "train_win_rate",
                "test_n",
                "test_avg_net_bps",
                "test_win_rate",
            ]
        )
        for cfg, train, test, _ in sorted(results, key=lambda row: (-row[1][1], -row[1][0]))[:100]:
            writer.writerow(
                [
                    cfg["min_bybit_trade_count_5s"],
                    cfg["min_binance_trade_count_5s"],
                    cfg["max_bybit_book_spread_bps"],
                    cfg["max_bybit_trade_imbalance_5s"],
                    train[0],
                    f"{train[1]:.6f}",
                    f"{train[2]:.6f}",
                    test[0],
                    f"{test[1]:.6f}",
                    f"{test[2]:.6f}",
                ]
            )

    lines = [
        "# Microstructure Filter Report",
        "",
        f"- Train days: {', '.join(train_days)}",
        f"- Test days: {', '.join(test_days)}",
        "",
        "## Baseline (No Extra Microstructure Gate)",
        "",
        f"- Train: {baseline_train[0]} trades, {baseline_train[1]:.4f} bps, {baseline_train[2]:.2%} win rate",
        f"- Test: {baseline_test[0]} trades, {baseline_test[1]:.4f} bps, {baseline_test[2]:.2%} win rate",
        "",
        "## Selected Filter (Train Only)",
        "",
        "```json",
        "{",
        f'  "min_bybit_trade_count_5s": {best_cfg["min_bybit_trade_count_5s"]},',
        f'  "min_binance_trade_count_5s": {best_cfg["min_binance_trade_count_5s"]},',
        f'  "max_bybit_book_spread_bps": {best_cfg["max_bybit_book_spread_bps"]},',
        f'  "max_bybit_trade_imbalance_5s": {best_cfg["max_bybit_trade_imbalance_5s"]}',
        "}",
        "```",
        "",
        f"- Train: {best_train[0]} trades, {best_train[1]:.4f} bps, {best_train[2]:.2%} win rate",
        f"- Test: {best_test[0]} trades, {best_test[1]:.4f} bps, {best_test[2]:.2%} win rate",
        "",
        "## Hypothesis-Driven Gate (Activity + Tight Bybit Book)",
        "",
        "This is not chosen on the holdout. It is a hand-picked follow-through from the descriptive microstructure findings:",
        "",
        "```json",
        "{",
        '  "min_bybit_trade_count_5s": 4.0,',
        '  "min_binance_trade_count_5s": 0.0,',
        '  "max_bybit_book_spread_bps": 4.5,',
        '  "max_bybit_trade_imbalance_5s": 1.0',
        "}",
        "```",
        "",
        f"- Train: {hypothesis_train[0]} trades, {hypothesis_train[1]:.4f} bps, {hypothesis_train[2]:.2%} win rate",
        f"- Test: {hypothesis_test[0]} trades, {hypothesis_test[1]:.4f} bps, {hypothesis_test[2]:.2%} win rate",
        "",
        f"- Kept trades in full 7-day window: {len(all_keep)} / {len(rows)}",
        "",
    ]
    args.output_md.write_text("\n".join(lines))

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_filtered}")


if __name__ == "__main__":
    main()
