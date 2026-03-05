#!/usr/bin/env python3
"""Scan shared Binance/Bybit symbols for short-horizon lead-lag structure."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BINANCE = ROOT / "datalake" / "binance"
BYBIT = ROOT / "datalake" / "bybit"
OUT_DIR = Path(__file__).resolve().parent / "out"


@dataclass
class SymbolLeadLag:
    symbol: str
    overlap_days: int
    rows: int
    best_leader: str
    best_lag_min: int
    best_corr: float
    binance_leads_corr: float
    bybit_leads_corr: float


def collect_dates(symbol_dir: Path, suffix: str) -> set[str]:
    dates: set[str] = set()
    if not symbol_dir.exists():
        return dates
    for path in symbol_dir.glob(f"*_{suffix}"):
        if "_" in path.name:
            dates.add(path.name.split("_", 1)[0])
    return dates


def load_close_series(symbol_dir: Path, days: list[str], exchange: str) -> pd.Series | None:
    rows: list[pd.DataFrame] = []
    for day in days:
        path = symbol_dir / f"{day}_kline_1m.csv"
        if not path.exists():
            continue
        usecols = ["open_time", "close"] if exchange == "binance" else ["startTime", "close"]
        try:
            frame = pd.read_csv(path, usecols=usecols)
        except ValueError:
            continue
        if frame.empty:
            continue
        if exchange == "bybit":
            frame = frame.rename(columns={"startTime": "open_time"})
        rows.append(frame)
    if not rows:
        return None
    frame = pd.concat(rows, ignore_index=True)
    frame = frame.drop_duplicates("open_time").sort_values("open_time")
    idx = pd.to_datetime(pd.to_numeric(frame["open_time"], errors="coerce"), unit="ms", utc=True)
    values = pd.to_numeric(frame["close"], errors="coerce")
    series = pd.Series(values.to_numpy(dtype=float), index=idx).replace([np.inf, -np.inf], np.nan).dropna()
    return series if not series.empty else None


def analyze_symbol(symbol: str, days: list[str], max_lag: int) -> SymbolLeadLag | None:
    bn = load_close_series(BINANCE / symbol, days, "binance")
    bb = load_close_series(BYBIT / symbol, days, "bybit")
    if bn is None or bb is None:
        return None
    aligned = pd.concat([bn.rename("binance"), bb.rename("bybit")], axis=1).dropna()
    if len(aligned) < 500:
        return None
    returns = np.log(aligned).diff().dropna()
    if returns.empty:
        return None

    best_leader = "none"
    best_lag = 0
    best_corr = 0.0
    best_bn = 0.0
    best_bb = 0.0

    for lag in range(1, max_lag + 1):
        bn_leads = returns["binance"].shift(lag).corr(returns["bybit"])
        bb_leads = returns["bybit"].shift(lag).corr(returns["binance"])
        bn_score = abs(float(bn_leads)) if pd.notna(bn_leads) else -1.0
        bb_score = abs(float(bb_leads)) if pd.notna(bb_leads) else -1.0
        if bn_score > abs(best_corr):
            best_leader = "binance"
            best_lag = lag
            best_corr = float(bn_leads)
        if bb_score > abs(best_corr):
            best_leader = "bybit"
            best_lag = lag
            best_corr = float(bb_leads)
        if lag == 1:
            best_bn = float(bn_leads) if pd.notna(bn_leads) else float("nan")
            best_bb = float(bb_leads) if pd.notna(bb_leads) else float("nan")

    return SymbolLeadLag(
        symbol=symbol,
        overlap_days=len(days),
        rows=len(returns),
        best_leader=best_leader,
        best_lag_min=best_lag,
        best_corr=best_corr,
        binance_leads_corr=best_bn,
        bybit_leads_corr=best_bb,
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-overlap-days", type=int, default=90)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--max-symbols", type=int, default=40)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--output-tag", type=str, default="")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    binance_symbols = {path.name for path in BINANCE.iterdir() if path.is_dir()}
    bybit_symbols = {path.name for path in BYBIT.iterdir() if path.is_dir()}
    common = sorted(binance_symbols & bybit_symbols)

    candidates: list[tuple[int, str, list[str]]] = []
    for symbol in common:
        overlap = sorted(
            collect_dates(BINANCE / symbol, "kline_1m.csv")
            & collect_dates(BYBIT / symbol, "kline_1m.csv")
        )
        if len(overlap) < args.min_overlap_days:
            continue
        days = overlap[-args.lookback_days :] if args.lookback_days > 0 else overlap
        candidates.append((len(overlap), symbol, days))
    candidates.sort(key=lambda item: (-item[0], item[1]))

    rows: list[SymbolLeadLag] = []
    for _, symbol, days in candidates[: max(1, args.max_symbols)]:
        result = analyze_symbol(symbol, days, args.max_lag)
        if result is not None:
            rows.append(result)

    rows.sort(key=lambda row: (-abs(row.best_corr), row.symbol))
    suffix = f"_{args.output_tag}" if args.output_tag else ""
    csv_path = OUT_DIR / f"lead_lag_summary{suffix}.csv"
    report_path = OUT_DIR / f"lead_lag_report{suffix}.md"

    write_csv(
        csv_path,
        [row.__dict__ for row in rows],
        [
            "symbol",
            "overlap_days",
            "rows",
            "best_leader",
            "best_lag_min",
            "best_corr",
            "binance_leads_corr",
            "bybit_leads_corr",
        ],
    )

    binance_wins = sum(1 for row in rows if row.best_leader == "binance")
    bybit_wins = sum(1 for row in rows if row.best_leader == "bybit")
    avg_abs = float(np.mean([abs(row.best_corr) for row in rows])) if rows else float("nan")

    lines = [
        "# Lead-Lag Report",
        "",
        f"- Symbols analyzed: `{len(rows)}`",
        f"- Lookback days: `{args.lookback_days}`",
        f"- Max lag: `{args.max_lag}` minutes",
        f"- Avg absolute best correlation: `{avg_abs:.4f}`",
        f"- Best-leader count: Binance `{binance_wins}`, Bybit `{bybit_wins}`",
        "",
        "## Top Symbols",
        "",
    ]
    for row in rows[:10]:
        lines.append(
            f"- {row.symbol}: leader `{row.best_leader}` at `{row.best_lag_min}m`, "
            f"corr `{row.best_corr:.4f}`"
        )
    report_path.write_text("\n".join(lines) + "\n")

    print(f"Analyzed {len(rows)} symbols")
    if rows:
        top = rows[0]
        print(
            f"Top lead-lag: {top.symbol} leader={top.best_leader} "
            f"lag={top.best_lag_min}m corr={top.best_corr:.4f}"
        )
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
