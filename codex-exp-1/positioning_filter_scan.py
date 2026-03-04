#!/usr/bin/env python3
"""Test spread-reversion entries only when cross-exchange positioning confirms the stretch."""

from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATALAKE = ROOT / "datalake"
BINANCE = DATALAKE / "binance"
BYBIT = DATALAKE / "bybit"
OUT_DIR = Path(__file__).resolve().parent / "out"
DEFAULT_SYMBOLS = OUT_DIR / "survivor_symbols.txt"


@dataclass
class BucketStats:
    raw_signals: int = 0
    filt_signals: int = 0
    raw_net_sum: float = 0.0
    filt_net_sum: float = 0.0
    raw_wins: int = 0
    filt_wins: int = 0

    def absorb(self, other: "BucketStats") -> None:
        self.raw_signals += other.raw_signals
        self.filt_signals += other.filt_signals
        self.raw_net_sum += other.raw_net_sum
        self.filt_net_sum += other.filt_net_sum
        self.raw_wins += other.raw_wins
        self.filt_wins += other.filt_wins

    @property
    def raw_avg_net(self) -> float:
        return self.raw_net_sum / self.raw_signals if self.raw_signals else math.nan

    @property
    def filt_avg_net(self) -> float:
        return self.filt_net_sum / self.filt_signals if self.filt_signals else math.nan

    @property
    def raw_hit_rate(self) -> float:
        return self.raw_wins / self.raw_signals if self.raw_signals else math.nan

    @property
    def filt_hit_rate(self) -> float:
        return self.filt_wins / self.filt_signals if self.filt_signals else math.nan


@dataclass
class SymbolResult:
    symbol: str
    overlap_days: int
    train: BucketStats
    test: BucketStats


def collect_dates(symbol_dir: Path, suffix: str) -> set[str]:
    dates: set[str] = set()
    if not symbol_dir.exists():
        return dates
    for path in symbol_dir.glob(f"*_{suffix}"):
        if "_" in path.name:
            dates.add(path.name.split("_", 1)[0])
    return dates


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
                long_short = float(row["count_long_short_ratio"])
                oi = float(row["sum_open_interest"])
            except (KeyError, TypeError, ValueError):
                continue
            oi_chg_bps = 0.0
            if prev_oi and prev_oi != 0:
                oi_chg_bps = 10000.0 * (oi / prev_oi - 1.0)
            prev_oi = oi
            if long_short <= 0:
                continue
            rows.append((ts, math.log(long_short), oi_chg_bps))
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
                buy_ratio = float(row["buyRatio"])
                sell_ratio = float(row["sellRatio"])
            except (KeyError, TypeError, ValueError):
                continue
            if buy_ratio <= 0 or sell_ratio <= 0:
                continue
            rows.append((ts, math.log(buy_ratio / sell_ratio)))
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
            oi_chg_bps = 0.0
            if prev_oi and prev_oi != 0:
                oi_chg_bps = 10000.0 * (oi / prev_oi - 1.0)
            prev_oi = oi
            rows.append((ts, oi_chg_bps))
    return rows


def load_binance_basis(mark_path: Path, index_path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    if not mark_path.exists() or not index_path.exists():
        return rows
    mark_rows = load_close_rows(mark_path, "open_time", "close")
    index_rows = load_close_rows(index_path, "open_time", "close")
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
                ts = int(float(row["startTime"]))
                close = float(row["close"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append((ts, close * 10000.0))
    return rows


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


def latest_value(rows, idx: int, ts: int):
    while idx + 1 < len(rows) and rows[idx + 1][0] <= ts:
        idx += 1
    return idx


def analyze_day(
    bn_kline_path: Path,
    bb_kline_path: Path,
    bn_metrics_path: Path,
    bb_ls_path: Path,
    bb_oi_path: Path,
    bn_mark_path: Path,
    bn_index_path: Path,
    bb_premium_path: Path,
    min_signal_bps: float,
    fee_bps_roundtrip: float,
    min_ls_diff: float,
    min_oi_diff_bps: float,
    min_carry_diff_bps: float,
    require_oi_confirm: bool,
    require_carry_confirm: bool,
) -> BucketStats:
    stats = BucketStats()

    bn_rows = load_close_rows(bn_kline_path, "open_time", "close")
    bb_rows = load_close_rows(bb_kline_path, "startTime", "close")
    if not bn_rows or not bb_rows:
        return stats

    bn_metrics = load_binance_metrics(bn_metrics_path)
    bb_ls = load_bybit_long_short(bb_ls_path)
    bb_oi = load_bybit_oi(bb_oi_path)
    bn_basis = load_binance_basis(bn_mark_path, bn_index_path)
    bb_premium = load_bybit_premium(bb_premium_path)
    if not bn_metrics or not bb_ls or not bb_oi or not bn_basis or not bb_premium:
        return stats

    bn_idx = 0
    bb_idx = 0
    metric_idx = 0
    bb_ls_idx = 0
    bb_oi_idx = 0
    bn_basis_idx = 0
    bb_premium_idx = 0
    prev_spread: float | None = None
    prev_confirmed = False

    while bn_idx < len(bn_rows) and bb_idx < len(bb_rows):
        bn_ts, bn_close = bn_rows[bn_idx]
        bb_ts, bb_close = bb_rows[bb_idx]

        if bn_ts == bb_ts:
            if bb_close != 0:
                metric_idx = latest_value(bn_metrics, metric_idx, bn_ts)
                bb_ls_idx = latest_value(bb_ls, bb_ls_idx, bn_ts)
                bb_oi_idx = latest_value(bb_oi, bb_oi_idx, bn_ts)
                bn_basis_idx = latest_value(bn_basis, bn_basis_idx, bn_ts)
                bb_premium_idx = latest_value(bb_premium, bb_premium_idx, bn_ts)

                spread_bps = 10000.0 * (bn_close / bb_close - 1.0)
                sign = 1.0 if spread_bps > 0 else -1.0 if spread_bps < 0 else 0.0
                bn_ls_log = bn_metrics[metric_idx][1]
                bn_oi_chg_bps = bn_metrics[metric_idx][2]
                bb_ls_log = bb_ls[bb_ls_idx][1]
                bb_oi_chg_bps = bb_oi[bb_oi_idx][1]
                bn_basis_bps = bn_basis[bn_basis_idx][1]
                bb_premium_bps = bb_premium[bb_premium_idx][1]

                ls_confirm = sign * (bn_ls_log - bb_ls_log) >= min_ls_diff
                oi_confirm = sign * (bn_oi_chg_bps - bb_oi_chg_bps) >= min_oi_diff_bps
                carry_confirm = sign * (bn_basis_bps - bb_premium_bps) >= min_carry_diff_bps
                confirmed = (
                    ls_confirm
                    and (oi_confirm if require_oi_confirm else True)
                    and (carry_confirm if require_carry_confirm else True)
                )

                if prev_spread is not None and abs(prev_spread) >= min_signal_bps:
                    direction = 1.0 if prev_spread > 0 else -1.0
                    net_pnl = direction * (prev_spread - spread_bps) - fee_bps_roundtrip
                    stats.raw_signals += 1
                    stats.raw_net_sum += net_pnl
                    if net_pnl > 0:
                        stats.raw_wins += 1

                    if prev_confirmed:
                        stats.filt_signals += 1
                        stats.filt_net_sum += net_pnl
                        if net_pnl > 0:
                            stats.filt_wins += 1

                prev_spread = spread_bps
                prev_confirmed = confirmed

            bn_idx += 1
            bb_idx += 1
            continue

        if bn_ts < bb_ts:
            bn_idx += 1
        else:
            bb_idx += 1

    return stats


def analyze_symbol(
    symbol: str,
    min_overlap_days: int,
    min_signal_bps: float,
    fee_bps_roundtrip: float,
    test_days: int,
    recent_days: int,
    min_ls_diff: float,
    min_oi_diff_bps: float,
    min_carry_diff_bps: float,
    require_oi_confirm: bool,
    require_carry_confirm: bool,
) -> SymbolResult | None:
    overlap = sorted(
        collect_dates(BINANCE / symbol, "kline_1m.csv") & collect_dates(BYBIT / symbol, "kline_1m.csv")
    )
    if len(overlap) < min_overlap_days:
        return None
    dates = overlap[-recent_days:] if recent_days > 0 else overlap
    if test_days >= len(dates):
        test_days = max(1, len(dates) // 3)
    split_idx = len(dates) - max(0, test_days)
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]

    train = BucketStats()
    test = BucketStats()
    for bucket, use_dates in ((train, train_dates), (test, test_dates)):
        for day in use_dates:
            bucket.absorb(
                analyze_day(
                    bn_kline_path=BINANCE / symbol / f"{day}_kline_1m.csv",
                    bb_kline_path=BYBIT / symbol / f"{day}_kline_1m.csv",
                    bn_metrics_path=BINANCE / symbol / f"{day}_metrics.csv",
                    bb_ls_path=BYBIT / symbol / f"{day}_long_short_ratio_5min.csv",
                    bb_oi_path=BYBIT / symbol / f"{day}_open_interest_5min.csv",
                    bn_mark_path=BINANCE / symbol / f"{day}_mark_price_kline_1m.csv",
                    bn_index_path=BINANCE / symbol / f"{day}_index_price_kline_1m.csv",
                    bb_premium_path=BYBIT / symbol / f"{day}_premium_index_kline_1m.csv",
                    min_signal_bps=min_signal_bps,
                    fee_bps_roundtrip=fee_bps_roundtrip,
                    min_ls_diff=min_ls_diff,
                    min_oi_diff_bps=min_oi_diff_bps,
                    min_carry_diff_bps=min_carry_diff_bps,
                    require_oi_confirm=require_oi_confirm,
                    require_carry_confirm=require_carry_confirm,
                )
            )

    if train.raw_signals + test.raw_signals == 0:
        return None

    return SymbolResult(symbol=symbol, overlap_days=len(overlap), train=train, test=test)


def load_symbol_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def aggregate(results: list[SymbolResult], attr: str) -> BucketStats:
    out = BucketStats()
    for row in results:
        out.absorb(getattr(row, attr))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols-file", type=Path, default=DEFAULT_SYMBOLS)
    parser.add_argument("--min-overlap-days", type=int, default=90)
    parser.add_argument("--min-signal-bps", type=float, default=10.0)
    parser.add_argument("--fee-bps-roundtrip", type=float, default=6.0)
    parser.add_argument("--test-days", type=int, default=15)
    parser.add_argument("--recent-days", type=int, default=45)
    parser.add_argument("--min-ls-diff", type=float, default=0.0)
    parser.add_argument("--min-oi-diff-bps", type=float, default=0.0)
    parser.add_argument("--min-carry-diff-bps", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--require-oi-confirm", action="store_true")
    parser.add_argument("--require-carry-confirm", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = load_symbol_list(args.symbols_file)
    results: list[SymbolResult] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                analyze_symbol,
                symbol,
                args.min_overlap_days,
                args.min_signal_bps,
                args.fee_bps_roundtrip,
                args.test_days,
                args.recent_days,
                args.min_ls_diff,
                args.min_oi_diff_bps,
                args.min_carry_diff_bps,
                args.require_oi_confirm,
                args.require_carry_confirm,
            ): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            row = future.result()
            if row is not None:
                results.append(row)

    results.sort(
        key=lambda row: (
            -(row.test.filt_avg_net if row.test.filt_signals else float("-inf")),
            -row.test.filt_signals,
            row.symbol,
        )
    )

    summary_path = OUT_DIR / "positioning_filter_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "symbol",
                "overlap_days",
                "raw_test_signals",
                "raw_test_avg_net_bps",
                "filt_test_signals",
                "filt_test_avg_net_bps",
                "raw_total_signals",
                "filt_total_signals",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row.symbol,
                    row.overlap_days,
                    row.test.raw_signals,
                    f"{row.test.raw_avg_net:.6f}",
                    row.test.filt_signals,
                    f"{row.test.filt_avg_net:.6f}",
                    row.train.raw_signals + row.test.raw_signals,
                    row.train.filt_signals + row.test.filt_signals,
                ]
            )

    total_train = aggregate(results, "train")
    total_test = aggregate(results, "test")
    report_path = OUT_DIR / "positioning_filter_report.md"
    lines = [
        "# Positioning Filter Report",
        "",
        "## Configuration",
        "",
        f"- Symbols: {len(results)}",
        f"- Min signal spread: {args.min_signal_bps:.2f} bps",
        f"- Fee: {args.fee_bps_roundtrip:.2f} bps",
        f"- Recent days: {args.recent_days}",
        f"- Test days: {args.test_days}",
        f"- Min long-short diff: {args.min_ls_diff:.4f}",
        f"- Min OI diff: {args.min_oi_diff_bps:.4f} bps",
        f"- Min carry diff: {args.min_carry_diff_bps:.4f} bps",
        f"- Require OI confirm: {args.require_oi_confirm}",
        f"- Require carry confirm: {args.require_carry_confirm}",
        "",
        "## Aggregate Results",
        "",
        f"- Raw train avg net: {total_train.raw_avg_net:.4f} bps on {total_train.raw_signals} signals",
        f"- Filtered train avg net: {total_train.filt_avg_net:.4f} bps on {total_train.filt_signals} signals",
        f"- Raw test avg net: {total_test.raw_avg_net:.4f} bps on {total_test.raw_signals} signals",
        f"- Filtered test avg net: {total_test.filt_avg_net:.4f} bps on {total_test.filt_signals} signals",
        "",
        "## Top Symbols By Filtered Test Avg Net",
        "",
        "| Symbol | Raw Test Avg | Filtered Test Avg | Filtered Test Signals |",
        "|---|---:|---:|---:|",
    ]
    for row in results[:15]:
        lines.append(
            f"| {row.symbol} | {row.test.raw_avg_net:.4f} | {row.test.filt_avg_net:.4f} | {row.test.filt_signals} |"
        )
    lines.append("")
    report_path.write_text("\n".join(lines))

    print(f"Analyzed symbols: {len(results)}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")
    print(
        f"Raw test avg net={total_test.raw_avg_net:.4f}bps on {total_test.raw_signals} signals"
    )
    print(
        f"Filtered test avg net={total_test.filt_avg_net:.4f}bps on {total_test.filt_signals} signals"
    )
    if results:
        print("Top filtered symbols:")
        for row in results[:10]:
            print(
                f"  {row.symbol}: filt_test_avg={row.test.filt_avg_net:.4f}bps "
                f"filt_test_signals={row.test.filt_signals} raw_test_avg={row.test.raw_avg_net:.4f}bps"
            )


if __name__ == "__main__":
    main()
