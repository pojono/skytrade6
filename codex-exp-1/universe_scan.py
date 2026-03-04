#!/usr/bin/env python3
"""Scan the local datalake and build an eligible cross-exchange symbol universe."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATALAKE = ROOT / "datalake"
BINANCE = DATALAKE / "binance"
BYBIT = DATALAKE / "bybit"
OUT_DIR = Path(__file__).resolve().parent / "out"


@dataclass
class SymbolSummary:
    symbol: str
    binance_days: int
    bybit_days: int
    overlap_days: int
    binance_metrics_days: int
    bybit_funding_days: int
    bybit_open_interest_days: int
    bybit_long_short_days: int
    eligible: bool


def collect_dates(symbol_dir: Path, suffix: str) -> set[str]:
    dates: set[str] = set()
    if not symbol_dir.exists():
        return dates
    pattern = f"*_{suffix}"
    for path in symbol_dir.glob(pattern):
        name = path.name
        if "_" not in name:
            continue
        dates.add(name.split("_", 1)[0])
    return dates


def scan_symbol(symbol: str, min_overlap_days: int) -> SymbolSummary:
    binance_dir = BINANCE / symbol
    bybit_dir = BYBIT / symbol

    binance_kline = collect_dates(binance_dir, "kline_1m.csv")
    bybit_kline = collect_dates(bybit_dir, "kline_1m.csv")
    overlap = binance_kline & bybit_kline

    binance_metrics = collect_dates(binance_dir, "metrics.csv")
    bybit_funding = collect_dates(bybit_dir, "funding_rate.csv")
    bybit_open_interest = collect_dates(bybit_dir, "open_interest_5min.csv")
    bybit_long_short = collect_dates(bybit_dir, "long_short_ratio_5min.csv")

    eligible = len(overlap) >= min_overlap_days

    return SymbolSummary(
        symbol=symbol,
        binance_days=len(binance_kline),
        bybit_days=len(bybit_kline),
        overlap_days=len(overlap),
        binance_metrics_days=len(binance_metrics),
        bybit_funding_days=len(bybit_funding),
        bybit_open_interest_days=len(bybit_open_interest),
        bybit_long_short_days=len(bybit_long_short),
        eligible=eligible,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-overlap-days",
        type=int,
        default=90,
        help="Minimum overlapping daily kline files required to mark a symbol eligible.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    binance_symbols = {p.name for p in BINANCE.iterdir() if p.is_dir()}
    bybit_symbols = {p.name for p in BYBIT.iterdir() if p.is_dir()}
    common_symbols = sorted(binance_symbols & bybit_symbols)

    rows = [scan_symbol(symbol, args.min_overlap_days) for symbol in common_symbols]
    rows.sort(key=lambda row: (-row.overlap_days, row.symbol))

    csv_path = OUT_DIR / "universe_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "symbol",
                "binance_days",
                "bybit_days",
                "overlap_days",
                "binance_metrics_days",
                "bybit_funding_days",
                "bybit_open_interest_days",
                "bybit_long_short_days",
                "eligible",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.symbol,
                    row.binance_days,
                    row.bybit_days,
                    row.overlap_days,
                    row.binance_metrics_days,
                    row.bybit_funding_days,
                    row.bybit_open_interest_days,
                    row.bybit_long_short_days,
                    int(row.eligible),
                ]
            )

    eligible_symbols = [row.symbol for row in rows if row.eligible]
    eligible_path = OUT_DIR / "eligible_symbols.txt"
    eligible_path.write_text("\n".join(eligible_symbols) + ("\n" if eligible_symbols else ""))

    print(f"Common symbols: {len(common_symbols)}")
    print(f"Eligible symbols (>= {args.min_overlap_days} overlap days): {len(eligible_symbols)}")
    if eligible_symbols:
        print(f"Top 10 by overlap: {', '.join(eligible_symbols[:10])}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {eligible_path}")


if __name__ == "__main__":
    main()
