#!/usr/bin/env python3
"""Filter cross-exchange results down to statistically useful symbol candidates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / "out"


def parse_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def parse_int(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError):
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=OUT_DIR / "cross_exchange_edge_summary.csv",
        help="Input CSV from cross_exchange_edge_scan.py",
    )
    parser.add_argument("--min-test-signals", type=int, default=500)
    parser.add_argument("--min-total-signals", type=int, default=2000)
    parser.add_argument("--min-test-net-bps", type=float, default=0.0)
    parser.add_argument("--require-total-net-positive", action="store_true")
    args = parser.parse_args()

    survivors = []
    with args.input.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            test_signals = parse_int(row, "signal_count_test")
            total_signals = parse_int(row, "signal_count_total")
            test_net = parse_float(row, "avg_net_pnl_bps_test")
            total_net = parse_float(row, "avg_net_pnl_bps_total")
            if test_signals < args.min_test_signals:
                continue
            if total_signals < args.min_total_signals:
                continue
            if test_net < args.min_test_net_bps:
                continue
            if args.require_total_net_positive and total_net <= 0:
                continue
            survivors.append(
                (
                    row["symbol"],
                    test_signals,
                    total_signals,
                    test_net,
                    total_net,
                )
            )

    survivors.sort(key=lambda item: (-item[3], -item[1], item[0]))

    out_path = OUT_DIR / "survivor_symbols.txt"
    out_path.write_text(
        "\n".join(item[0] for item in survivors) + ("\n" if survivors else "")
    )

    print(f"Survivors: {len(survivors)}")
    print(f"Wrote {out_path}")
    for symbol, test_signals, total_signals, test_net, total_net in survivors[:20]:
        print(
            f"  {symbol}: test_net={test_net:.4f}bps test_signals={test_signals} "
            f"total_net={total_net:.4f}bps total_signals={total_signals}"
        )


if __name__ == "__main__":
    main()
