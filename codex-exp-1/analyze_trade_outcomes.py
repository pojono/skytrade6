#!/usr/bin/env python3
"""Analyze winner vs loser patterns on pre-trade candidate trades."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / "out"
DEFAULT_INPUT = OUT_DIR / "candidate_trades_v3.csv"


def net_bps(
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


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else math.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--fee-bps-roundtrip", type=float, default=6.0)
    parser.add_argument("--extra-slippage-bps", type=float, default=1.0)
    parser.add_argument("--spread-slip-coeff", type=float, default=0.10)
    parser.add_argument("--velocity-slip-coeff", type=float, default=0.05)
    parser.add_argument("--min-signal-bps", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "trade_outcome_analysis.md")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.input.open()))
    for row in rows:
        row["_net_bps"] = net_bps(
            row,
            args.fee_bps_roundtrip,
            args.extra_slippage_bps,
            args.spread_slip_coeff,
            args.velocity_slip_coeff,
            args.min_signal_bps,
        )

    winners = [row for row in rows if row["_net_bps"] > 0]
    losers = [row for row in rows if row["_net_bps"] <= 0]
    features = [
        ("score", "score"),
        ("spread_abs", "entry_spread_abs_bps"),
        ("velocity", "entry_spread_velocity_bps"),
        ("ls_diff", "ls_diff_signed"),
        ("oi_diff", "oi_diff_signed_bps"),
        ("carry_diff", "carry_diff_signed_bps"),
    ]

    lines = [
        "# Trade Outcome Analysis",
        "",
        f"- Trades analyzed: {len(rows)}",
        f"- Winners: {len(winners)}",
        f"- Losers: {len(losers)}",
        f"- Win rate: {(len(winners) / len(rows)):.2%}" if rows else "- Win rate: n/a",
        f"- Mean net bps: {mean([row['_net_bps'] for row in rows]):.4f}",
        "",
        "## Feature Means",
        "",
        "| Feature | Winners | Losers | Delta (W-L) |",
        "|---|---:|---:|---:|",
    ]
    for label, key in features:
        w = mean([float(row[key]) for row in winners])
        l = mean([float(row[key]) for row in losers])
        lines.append(f"| {label} | {w:.4f} | {l:.4f} | {(w-l):.4f} |")

    by_symbol: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_symbol.setdefault(row["symbol"], []).append(row)
    lines.extend(
        [
            "",
            "## By Symbol",
            "",
            "| Symbol | Trades | Win Rate | Mean Net bps |",
            "|---|---:|---:|---:|",
        ]
    )
    for symbol in sorted(by_symbol):
        group = by_symbol[symbol]
        wins = sum(1 for row in group if row["_net_bps"] > 0)
        lines.append(
            f"| {symbol} | {len(group)} | {(wins/len(group)):.2%} | {mean([row['_net_bps'] for row in group]):.4f} |"
        )

    args.output.write_text("\n".join(lines))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
