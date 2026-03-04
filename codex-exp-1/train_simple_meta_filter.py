#!/usr/bin/env python3
"""Search a simple threshold-based pre-trade filter on train months and evaluate on test months."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / "out"
DEFAULT_INPUT = OUT_DIR / "candidate_trades_v3.csv"


@dataclass(frozen=True)
class Trade:
    month: str
    symbol: str
    score: float
    velocity: float
    spread_abs: float
    ls: float
    oi: float
    carry: float
    net_bps: float


def calc_net(
    row: dict[str, str],
    fee_bps_roundtrip: float,
    extra_slippage_bps: float,
    spread_slip_coeff: float,
    velocity_slip_coeff: float,
    min_signal_bps: float,
) -> float:
    gross = float(row["gross_pnl_bps"])
    spread_abs = float(row["entry_spread_abs_bps"])
    velocity = float(row["entry_spread_velocity_bps"])
    stretch = max(0.0, spread_abs - min_signal_bps)
    slip = extra_slippage_bps + spread_slip_coeff * stretch + velocity_slip_coeff * velocity
    return gross - fee_bps_roundtrip - slip


def passes(trade: Trade, cfg: dict[str, float]) -> bool:
    need_score = cfg["min_score"] + (cfg["sei_score_extra"] if trade.symbol == "SEIUSDT" else 0.0)
    return (
        trade.score >= need_score
        and trade.velocity <= cfg["max_velocity"]
        and trade.spread_abs >= cfg["min_spread_abs"]
        and trade.ls >= cfg["min_ls"]
        and trade.oi >= cfg["min_oi"]
        and trade.carry >= cfg["min_carry"]
    )


def summarize(count: int, net_sum: float, win_count: int) -> tuple[int, float, float]:
    if not count:
        return 0, math.nan, math.nan
    return count, net_sum / count, win_count / count


def evaluate_bucket(
    bucket: list[Trade],
    cfg: dict[str, float],
) -> tuple[int, float, float]:
    count = 0
    net_sum = 0.0
    win_count = 0
    for row in bucket:
        if not passes(row, cfg):
            continue
        count += 1
        net_sum += row.net_bps
        if row.net_bps > 0:
            win_count += 1
    return summarize(count, net_sum, win_count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--test-months", type=int, default=2)
    parser.add_argument("--fee-bps-roundtrip", type=float, default=6.0)
    parser.add_argument("--extra-slippage-bps", type=float, default=1.0)
    parser.add_argument("--spread-slip-coeff", type=float, default=0.10)
    parser.add_argument("--velocity-slip-coeff", type=float, default=0.05)
    parser.add_argument("--min-signal-bps", type=float, default=10.0)
    parser.add_argument("--min-train-trades", type=int, default=500)
    parser.add_argument("--output-csv", type=Path, default=OUT_DIR / "meta_filter_search.csv")
    parser.add_argument("--output-json", type=Path, default=OUT_DIR / "meta_filter_config.json")
    parser.add_argument("--output-md", type=Path, default=OUT_DIR / "meta_filter_report.md")
    args = parser.parse_args()

    raw_rows = list(csv.DictReader(args.input.open()))
    rows = [
        Trade(
            month=row["month"],
            symbol=row["symbol"],
            score=float(row["score"]),
            velocity=float(row["entry_spread_velocity_bps"]),
            spread_abs=float(row["entry_spread_abs_bps"]),
            ls=float(row["ls_diff_signed"]),
            oi=float(row["oi_diff_signed_bps"]),
            carry=float(row["carry_diff_signed_bps"]),
            net_bps=calc_net(
                row,
                args.fee_bps_roundtrip,
                args.extra_slippage_bps,
                args.spread_slip_coeff,
                args.velocity_slip_coeff,
                args.min_signal_bps,
            ),
        )
        for row in raw_rows
    ]
    months = sorted({row.month for row in rows})
    test_months = months[-args.test_months :]
    train_months = months[: -args.test_months]
    train_months_set = set(train_months)
    test_months_set = set(test_months)
    train = [row for row in rows if row.month in train_months_set]
    test = [row for row in rows if row.month in test_months_set]

    search_space = {
        "min_score": [4.0, 5.0, 6.0, 7.0, 8.0],
        "sei_score_extra": [6.0, 8.0, 10.0],
        "max_velocity": [6.0, 8.0, 10.0, 12.0],
        "min_spread_abs": [10.0, 11.0, 12.0],
        "min_ls": [0.15, 0.25, 0.35],
        "min_oi": [5.0, 10.0, 15.0],
        "min_carry": [2.0, 4.0, 6.0],
    }
    selected = {
        "min_score": 4.0,
        "sei_score_extra": 6.0,
        "max_velocity": 10.0,
        "min_spread_abs": 10.0,
        "min_ls": 0.15,
        "min_oi": 5.0,
        "min_carry": 2.0,
    }

    result_rows = []
    best_cfg = dict(selected)
    best_score = float("-inf")
    search_order = [
        "min_score",
        "sei_score_extra",
        "max_velocity",
        "min_spread_abs",
        "min_ls",
        "min_oi",
        "min_carry",
    ]
    for _ in range(2):
        for key in search_order:
            local_best_value = selected[key]
            local_best_score = float("-inf")
            for value in search_space[key]:
                cfg = dict(selected)
                cfg[key] = value
                train_n, train_avg, train_wr = evaluate_bucket(train, cfg)
                if train_n < args.min_train_trades or math.isnan(train_avg):
                    continue
                score = train_avg * math.log(train_n)
                test_n, test_avg, test_wr = evaluate_bucket(test, cfg)
                result_rows.append(
                    (
                        dict(cfg),
                        train_n,
                        train_avg,
                        train_wr,
                        test_n,
                        test_avg,
                        test_wr,
                        score,
                    )
                )
                if score > local_best_score:
                    local_best_score = score
                    local_best_value = value
                if score > best_score:
                    best_score = score
                    best_cfg = dict(cfg)
            selected[key] = local_best_value

    result_rows.sort(
        key=lambda row: (
            -(row[2] if not math.isnan(row[2]) else float("-inf")),
            -row[1],
        )
    )

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "min_score",
                "sei_score_extra",
                "max_velocity",
                "min_spread_abs",
                "min_ls",
                "min_oi",
                "min_carry",
                "train_trades",
                "train_avg_net_bps",
                "train_win_rate",
                "test_trades",
                "test_avg_net_bps",
                "test_win_rate",
            ]
        )
        for cfg, train_n, train_avg, train_wr, test_n, test_avg, test_wr, _ in result_rows[:200]:
            writer.writerow(
                [
                    cfg["min_score"],
                    cfg["sei_score_extra"],
                    cfg["max_velocity"],
                    cfg["min_spread_abs"],
                    cfg["min_ls"],
                    cfg["min_oi"],
                    cfg["min_carry"],
                    train_n,
                    f"{train_avg:.6f}",
                    f"{train_wr:.6f}",
                    test_n,
                    f"{test_avg:.6f}",
                    f"{test_wr:.6f}",
                ]
            )

    selected = best_cfg
    train_n, train_avg, train_wr = evaluate_bucket(train, selected)
    test_n, test_avg, test_wr = evaluate_bucket(test, selected)

    args.output_json.write_text(json.dumps(selected, indent=2, sort_keys=True))
    lines = [
        "# Meta Filter Report",
        "",
        f"- Train months: {', '.join(train_months)}",
        f"- Test months: {', '.join(test_months)}",
        "",
        "## Selected Rule",
        "",
        "```json",
        json.dumps(selected, indent=2, sort_keys=True),
        "```",
        "",
        "## Performance",
        "",
        f"- Train trades: {train_n}",
        f"- Train avg net: {train_avg:.4f} bps",
        f"- Train win rate: {train_wr:.2%}",
        f"- Test trades: {test_n}",
        f"- Test avg net: {test_avg:.4f} bps",
        f"- Test win rate: {test_wr:.2%}",
        "",
    ]
    args.output_md.write_text("\n".join(lines))
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(f"Selected test avg={test_avg:.4f}bps test winrate={test_wr:.2%} trades={test_n}")


if __name__ == "__main__":
    main()
