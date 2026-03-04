#!/usr/bin/env python3
"""Cache entry records once, then sweep dense cross-exchange filters in memory."""

from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATALAKE = ROOT / "datalake"
BINANCE = DATALAKE / "binance"
BYBIT = DATALAKE / "bybit"
OUT_DIR = Path(__file__).resolve().parent / "out"
DEFAULT_SYMBOLS = OUT_DIR / "survivor_symbols.txt"


@dataclass
class EntryRecord:
    split: str
    net_pnl_bps: float
    ls_diff_signed: float
    oi_diff_signed_bps: float
    carry_diff_signed_bps: float
    entry_spread_abs_bps: float
    entry_spread_velocity_bps: float


def collect_dates(symbol_dir: Path, suffix: str) -> set[str]:
    dates: set[str] = set()
    if not symbol_dir.exists():
        return dates
    for path in symbol_dir.glob(f"*_{suffix}"):
        if "_" in path.name:
            dates.add(path.name.split("_", 1)[0])
    return dates


def load_close_rows(path: Path, time_field: str, close_field: str) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    if not path.exists():
        return rows
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                rows.append((int(float(row[time_field])), float(row[close_field])))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def load_binance_metrics(path: Path) -> list[tuple[int, float, float]]:
    rows: list[tuple[int, float, float]] = []
    if not path.exists():
        return rows
    prev_oi: float | None = None
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                dt = datetime.strptime(row["create_time"], "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
                ts = int(dt.timestamp() * 1000)
                ls_ratio = float(row["count_long_short_ratio"])
                oi = float(row["sum_open_interest"])
            except (KeyError, TypeError, ValueError):
                continue
            if ls_ratio <= 0:
                continue
            oi_chg = 0.0
            if prev_oi and prev_oi != 0:
                oi_chg = 10000.0 * (oi / prev_oi - 1.0)
            prev_oi = oi
            rows.append((ts, math.log(ls_ratio), oi_chg))
    return rows


def load_bybit_long_short(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    if not path.exists():
        return rows
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                ts = int(float(row["timestamp"]))
                buy = float(row["buyRatio"])
                sell = float(row["sellRatio"])
            except (KeyError, TypeError, ValueError):
                continue
            if buy <= 0 or sell <= 0:
                continue
            rows.append((ts, math.log(buy / sell)))
    return rows


def load_bybit_oi(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    if not path.exists():
        return rows
    prev_oi: float | None = None
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                ts = int(float(row["timestamp"]))
                oi = float(row["openInterest"])
            except (KeyError, TypeError, ValueError):
                continue
            oi_chg = 0.0
            if prev_oi and prev_oi != 0:
                oi_chg = 10000.0 * (oi / prev_oi - 1.0)
            prev_oi = oi
            rows.append((ts, oi_chg))
    return rows


def load_binance_basis(mark_path: Path, index_path: Path) -> list[tuple[int, float]]:
    mark_rows = load_close_rows(mark_path, "open_time", "close")
    index_rows = load_close_rows(index_path, "open_time", "close")
    rows: list[tuple[int, float]] = []
    i = 0
    j = 0
    while i < len(mark_rows) and j < len(index_rows):
        mts, mclose = mark_rows[i]
        its, iclose = index_rows[j]
        if mts == its:
            if iclose != 0:
                rows.append((mts, 10000.0 * (mclose / iclose - 1.0)))
            i += 1
            j += 1
        elif mts < its:
            i += 1
        else:
            j += 1
    return rows


def load_bybit_premium(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    if not path.exists():
        return rows
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                rows.append((int(float(row["startTime"])), float(row["close"]) * 10000.0))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def advance_idx(rows, idx: int, ts: int) -> int:
    while idx + 1 < len(rows) and rows[idx + 1][0] <= ts:
        idx += 1
    return idx


def build_day_records(
    symbol: str,
    day: str,
    split: str,
    min_signal_bps: float,
    fee_bps_roundtrip: float,
) -> list[EntryRecord]:
    bn_dir = BINANCE / symbol
    bb_dir = BYBIT / symbol
    bn_close = load_close_rows(bn_dir / f"{day}_kline_1m.csv", "open_time", "close")
    bb_close = load_close_rows(bb_dir / f"{day}_kline_1m.csv", "startTime", "close")
    bn_metrics = load_binance_metrics(bn_dir / f"{day}_metrics.csv")
    bb_ls = load_bybit_long_short(bb_dir / f"{day}_long_short_ratio_5min.csv")
    bb_oi = load_bybit_oi(bb_dir / f"{day}_open_interest_5min.csv")
    bn_basis = load_binance_basis(
        bn_dir / f"{day}_mark_price_kline_1m.csv",
        bn_dir / f"{day}_index_price_kline_1m.csv",
    )
    bb_premium = load_bybit_premium(bb_dir / f"{day}_premium_index_kline_1m.csv")

    if not (bn_close and bb_close and bn_metrics and bb_ls and bb_oi and bn_basis and bb_premium):
        return []

    out: list[EntryRecord] = []
    i = j = 0
    m_idx = ls_idx = oi_idx = basis_idx = prem_idx = 0
    prev_spread: float | None = None
    prev_prev_spread: float | None = None
    prev_ls_diff = 0.0
    prev_oi_diff = 0.0
    prev_carry_diff = 0.0

    while i < len(bn_close) and j < len(bb_close):
        bn_ts, bn_px = bn_close[i]
        bb_ts, bb_px = bb_close[j]
        if bn_ts == bb_ts:
            if bb_px != 0:
                m_idx = advance_idx(bn_metrics, m_idx, bn_ts)
                ls_idx = advance_idx(bb_ls, ls_idx, bn_ts)
                oi_idx = advance_idx(bb_oi, oi_idx, bn_ts)
                basis_idx = advance_idx(bn_basis, basis_idx, bn_ts)
                prem_idx = advance_idx(bb_premium, prem_idx, bn_ts)

                spread = 10000.0 * (bn_px / bb_px - 1.0)
                sign = 1.0 if spread > 0 else -1.0 if spread < 0 else 0.0
                ls_diff = sign * (bn_metrics[m_idx][1] - bb_ls[ls_idx][1])
                oi_diff = sign * (bn_metrics[m_idx][2] - bb_oi[oi_idx][1])
                carry_diff = sign * (bn_basis[basis_idx][1] - bb_premium[prem_idx][1])

                if prev_spread is not None and abs(prev_spread) >= min_signal_bps:
                    direction = 1.0 if prev_spread > 0 else -1.0
                    net = direction * (prev_spread - spread) - fee_bps_roundtrip
                    spread_velocity = (
                        abs(prev_spread - prev_prev_spread) if prev_prev_spread is not None else 0.0
                    )
                    out.append(
                        EntryRecord(
                            split=split,
                            net_pnl_bps=net,
                            ls_diff_signed=prev_ls_diff,
                            oi_diff_signed_bps=prev_oi_diff,
                            carry_diff_signed_bps=prev_carry_diff,
                            entry_spread_abs_bps=abs(prev_spread),
                            entry_spread_velocity_bps=spread_velocity,
                        )
                    )

                prev_prev_spread = prev_spread
                prev_spread = spread
                prev_ls_diff = ls_diff
                prev_oi_diff = oi_diff
                prev_carry_diff = carry_diff
            i += 1
            j += 1
        elif bn_ts < bb_ts:
            i += 1
        else:
            j += 1
    return out


def build_symbol_records(
    symbol: str,
    min_overlap_days: int,
    min_signal_bps: float,
    fee_bps_roundtrip: float,
    recent_days: int,
    test_days: int,
) -> list[EntryRecord]:
    overlap = sorted(
        collect_dates(BINANCE / symbol, "kline_1m.csv")
        & collect_dates(BYBIT / symbol, "kline_1m.csv")
    )
    if len(overlap) < min_overlap_days:
        return []
    dates = overlap[-recent_days:] if recent_days > 0 else overlap
    if test_days >= len(dates):
        test_days = max(1, len(dates) // 3)
    split_idx = len(dates) - test_days
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]
    records: list[EntryRecord] = []
    for day in train_dates:
        records.extend(build_day_records(symbol, day, "train", min_signal_bps, fee_bps_roundtrip))
    for day in test_dates:
        records.extend(build_day_records(symbol, day, "test", min_signal_bps, fee_bps_roundtrip))
    return records


def load_symbols(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def mean_net(records: list[EntryRecord]) -> float:
    return sum(r.net_pnl_bps for r in records) / len(records) if records else math.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols-file", type=Path, default=DEFAULT_SYMBOLS)
    parser.add_argument("--min-overlap-days", type=int, default=90)
    parser.add_argument("--min-signal-bps", type=float, default=10.0)
    parser.add_argument("--fee-bps-roundtrip", type=float, default=6.0)
    parser.add_argument("--recent-days", type=int, default=45)
    parser.add_argument("--test-days", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ls-grid", type=str, default="0,0.05,0.1,0.15")
    parser.add_argument("--oi-grid", type=str, default="0,2,5")
    parser.add_argument("--carry-grid", type=str, default="0,0.5,1.0,2.0")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = load_symbols(args.symbols_file)

    all_records: list[EntryRecord] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                build_symbol_records,
                symbol,
                args.min_overlap_days,
                args.min_signal_bps,
                args.fee_bps_roundtrip,
                args.recent_days,
                args.test_days,
            ): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            all_records.extend(future.result())

    ls_grid = [float(x) for x in args.ls_grid.split(",") if x.strip()]
    oi_grid = [float(x) for x in args.oi_grid.split(",") if x.strip()]
    carry_grid = [float(x) for x in args.carry_grid.split(",") if x.strip()]

    rows = []
    raw_train = [r for r in all_records if r.split == "train"]
    raw_test = [r for r in all_records if r.split == "test"]
    for ls_thr, oi_thr, carry_thr in product(ls_grid, oi_grid, carry_grid):
        filt_train = [
            r for r in raw_train
            if r.ls_diff_signed >= ls_thr
            and r.oi_diff_signed_bps >= oi_thr
            and r.carry_diff_signed_bps >= carry_thr
        ]
        filt_test = [
            r for r in raw_test
            if r.ls_diff_signed >= ls_thr
            and r.oi_diff_signed_bps >= oi_thr
            and r.carry_diff_signed_bps >= carry_thr
        ]
        rows.append(
            (
                ls_thr,
                oi_thr,
                carry_thr,
                len(filt_train),
                mean_net(filt_train),
                len(filt_test),
                mean_net(filt_test),
            )
        )

    rows.sort(
        key=lambda row: (
            -(row[6] if not math.isnan(row[6]) else float("-inf")),
            -row[5],
            row[0],
            row[1],
            row[2],
        )
    )

    out_path = OUT_DIR / "dense_filter_sweep.csv"
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "ls_threshold",
                "oi_threshold_bps",
                "carry_threshold_bps",
                "train_signals",
                "train_avg_net_bps",
                "test_signals",
                "test_avg_net_bps",
            ]
        )
        writer.writerows(rows)

    print(f"Loaded records: train={len(raw_train)} test={len(raw_test)}")
    print(f"Wrote {out_path}")
    print("Top 15 configs:")
    for row in rows[:15]:
        print(
            f"  ls>={row[0]:.2f} oi>={row[1]:.1f} carry>={row[2]:.1f}: "
            f"test_avg={row[6]:.4f}bps test_n={row[5]} train_avg={row[4]:.4f}bps train_n={row[3]}"
        )


if __name__ == "__main__":
    main()
