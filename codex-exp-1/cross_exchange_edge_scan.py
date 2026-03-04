#!/usr/bin/env python3
"""Test a cross-exchange spread-reversion hypothesis on 1-minute bars."""

from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATALAKE = ROOT / "datalake"
BINANCE = DATALAKE / "binance"
BYBIT = DATALAKE / "bybit"
OUT_DIR = Path(__file__).resolve().parent / "out"


@dataclass
class BucketStats:
    aligned_bars: int = 0
    signal_count: int = 0
    spread_abs_sum: float = 0.0
    signal_spread_sum: float = 0.0
    gross_sum: float = 0.0
    net_sum: float = 0.0
    wins: int = 0

    def absorb(self, other: "BucketStats") -> None:
        self.aligned_bars += other.aligned_bars
        self.signal_count += other.signal_count
        self.spread_abs_sum += other.spread_abs_sum
        self.signal_spread_sum += other.signal_spread_sum
        self.gross_sum += other.gross_sum
        self.net_sum += other.net_sum
        self.wins += other.wins

    @property
    def mean_abs_spread_bps(self) -> float:
        return self.spread_abs_sum / self.aligned_bars if self.aligned_bars else math.nan

    @property
    def mean_signal_spread_bps(self) -> float:
        return self.signal_spread_sum / self.signal_count if self.signal_count else math.nan

    @property
    def avg_gross_pnl_bps(self) -> float:
        return self.gross_sum / self.signal_count if self.signal_count else math.nan

    @property
    def avg_net_pnl_bps(self) -> float:
        return self.net_sum / self.signal_count if self.signal_count else math.nan

    @property
    def hit_rate(self) -> float:
        return self.wins / self.signal_count if self.signal_count else math.nan


@dataclass
class SymbolResult:
    symbol: str
    overlap_days: int
    train_days: int
    test_days: int
    total: BucketStats
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


def iter_close_rows(path: Path, time_field: str, close_field: str):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                yield int(float(row[time_field])), float(row[close_field])
            except (KeyError, TypeError, ValueError):
                continue


def analyze_day(
    bn_path: Path,
    bb_path: Path,
    min_signal_bps: float,
    fee_bps_roundtrip: float,
) -> BucketStats:
    stats = BucketStats()
    bn_iter = iter_close_rows(bn_path, "open_time", "close")
    bb_iter = iter_close_rows(bb_path, "startTime", "close")

    try:
        bn_row = next(bn_iter)
        bb_row = next(bb_iter)
    except StopIteration:
        return stats

    prev_spread: float | None = None

    while True:
        bn_ts, bn_close = bn_row
        bb_ts, bb_close = bb_row

        if bn_ts == bb_ts:
            if bb_close != 0:
                spread_bps = 10000.0 * (bn_close / bb_close - 1.0)
                stats.aligned_bars += 1
                stats.spread_abs_sum += abs(spread_bps)

                if prev_spread is not None and abs(prev_spread) >= min_signal_bps:
                    direction = 1.0 if prev_spread > 0 else -1.0
                    gross_pnl = direction * (prev_spread - spread_bps)
                    net_pnl = gross_pnl - fee_bps_roundtrip
                    stats.signal_count += 1
                    stats.signal_spread_sum += abs(prev_spread)
                    stats.gross_sum += gross_pnl
                    stats.net_sum += net_pnl
                    if net_pnl > 0:
                        stats.wins += 1

                prev_spread = spread_bps

            try:
                bn_row = next(bn_iter)
                bb_row = next(bb_iter)
            except StopIteration:
                break
            continue

        if bn_ts < bb_ts:
            try:
                bn_row = next(bn_iter)
            except StopIteration:
                break
        else:
            try:
                bb_row = next(bb_iter)
            except StopIteration:
                break

    return stats


def analyze_symbol(
    symbol: str,
    min_overlap_days: int,
    min_signal_bps: float,
    fee_bps_roundtrip: float,
    test_days: int,
    recent_days: int,
) -> SymbolResult | None:
    binance_dir = BINANCE / symbol
    bybit_dir = BYBIT / symbol
    full_overlap_days = sorted(
        collect_dates(binance_dir, "kline_1m.csv") & collect_dates(bybit_dir, "kline_1m.csv")
    )
    if len(full_overlap_days) < min_overlap_days:
        return None
    overlap_days = full_overlap_days[-recent_days:] if recent_days > 0 else full_overlap_days

    if test_days < 0:
        test_days = 0
    if test_days >= len(overlap_days):
        test_days = max(1, len(overlap_days) // 4)

    split_idx = len(overlap_days) - test_days
    train_dates = overlap_days[:split_idx]
    test_dates = overlap_days[split_idx:]

    total = BucketStats()
    train = BucketStats()
    test = BucketStats()

    for bucket_name, dates in (("train", train_dates), ("test", test_dates)):
        for day in dates:
            bn_path = binance_dir / f"{day}_kline_1m.csv"
            bb_path = bybit_dir / f"{day}_kline_1m.csv"
            if not bn_path.exists() or not bb_path.exists():
                continue
            day_stats = analyze_day(
                bn_path=bn_path,
                bb_path=bb_path,
                min_signal_bps=min_signal_bps,
                fee_bps_roundtrip=fee_bps_roundtrip,
            )
            total.absorb(day_stats)
            if bucket_name == "train":
                train.absorb(day_stats)
            else:
                test.absorb(day_stats)

    if total.signal_count == 0:
        return None

    return SymbolResult(
        symbol=symbol,
        overlap_days=len(full_overlap_days),
        train_days=len(train_dates),
        test_days=len(test_dates),
        total=total,
        train=train,
        test=test,
    )


def analyze_symbols_parallel(
    symbols: list[str],
    min_overlap_days: int,
    min_signal_bps: float,
    fee_bps_roundtrip: float,
    test_days: int,
    workers: int,
    executor_kind: str,
    recent_days: int,
) -> list[SymbolResult]:
    results: list[SymbolResult] = []
    executor_cls = ThreadPoolExecutor if executor_kind == "thread" else ProcessPoolExecutor
    with executor_cls(max_workers=workers) as pool:
        futures = {
            pool.submit(
                analyze_symbol,
                symbol,
                min_overlap_days,
                min_signal_bps,
                fee_bps_roundtrip,
                test_days,
                recent_days,
            ): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
    return results


def aggregate_bucket(results: list[SymbolResult], attr: str) -> BucketStats:
    agg = BucketStats()
    for row in results:
        agg.absorb(getattr(row, attr))
    return agg


def write_report(
    results: list[SymbolResult],
    summary_csv: Path,
    report_md: Path,
    min_overlap_days: int,
    min_signal_bps: float,
    fee_bps_roundtrip: float,
    test_days: int,
    workers: int,
    recent_days: int,
) -> None:
    with summary_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "symbol",
                "overlap_days",
                "train_days",
                "test_days",
                "aligned_bars_total",
                "signal_count_total",
                "avg_net_pnl_bps_total",
                "hit_rate_total",
                "signal_count_train",
                "avg_net_pnl_bps_train",
                "hit_rate_train",
                "signal_count_test",
                "avg_net_pnl_bps_test",
                "hit_rate_test",
                "mean_signal_spread_bps_total",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row.symbol,
                    row.overlap_days,
                    row.train_days,
                    row.test_days,
                    row.total.aligned_bars,
                    row.total.signal_count,
                    f"{row.total.avg_net_pnl_bps:.6f}",
                    f"{row.total.hit_rate:.6f}",
                    row.train.signal_count,
                    f"{row.train.avg_net_pnl_bps:.6f}",
                    f"{row.train.hit_rate:.6f}",
                    row.test.signal_count,
                    f"{row.test.avg_net_pnl_bps:.6f}",
                    f"{row.test.hit_rate:.6f}",
                    f"{row.total.mean_signal_spread_bps:.6f}",
                ]
            )

    total = aggregate_bucket(results, "total")
    train = aggregate_bucket(results, "train")
    test = aggregate_bucket(results, "test")
    profitable_test_symbols = sum(
        1 for row in results if row.test.signal_count and row.test.avg_net_pnl_bps > 0
    )

    lines = [
        "# Cross-Exchange Edge Report",
        "",
        "## Configuration",
        "",
        f"- Minimum overlap days: {min_overlap_days}",
        f"- Minimum entry spread: {min_signal_bps:.2f} bps",
        f"- Fee assumption: {fee_bps_roundtrip:.2f} bps round trip",
        f"- Out-of-sample window: last {test_days} days",
        f"- Worker slots: {workers}",
        f"- Recent-day cap: {recent_days if recent_days > 0 else 'full history'}",
        "",
        "## Portfolio-Level Summary",
        "",
        f"- Symbols analyzed: {len(results)}",
        f"- Total signals: {total.signal_count}",
        f"- Total avg net PnL per signal: {total.avg_net_pnl_bps:.4f} bps",
        f"- Train signals: {train.signal_count}",
        f"- Train avg net PnL per signal: {train.avg_net_pnl_bps:.4f} bps",
        f"- Test signals: {test.signal_count}",
        f"- Test avg net PnL per signal: {test.avg_net_pnl_bps:.4f} bps",
        f"- Test-profitable symbols: {profitable_test_symbols}/{len(results)}",
        "",
        "## Top 15 Symbols By Test Avg Net PnL",
        "",
        "| Symbol | Test Signals | Test Avg Net bps | Test Hit Rate | Total Avg Net bps |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in results[:15]:
        lines.append(
            f"| {row.symbol} | {row.test.signal_count} | {row.test.avg_net_pnl_bps:.4f} | "
            f"{row.test.hit_rate:.2%} | {row.total.avg_net_pnl_bps:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This uses synchronized 1-minute closes and a one-bar holding period.",
            "- Streaming row-merge keeps memory bounded per file pair.",
            "- Strong train results with weak test results should be treated as noise, not edge.",
            "",
        ]
    )

    report_md.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-overlap-days",
        type=int,
        default=90,
        help="Minimum overlapping daily kline files required to analyze a symbol.",
    )
    parser.add_argument(
        "--min-signal-bps",
        type=float,
        default=8.0,
        help="Only count entry signals when absolute spread is at least this large.",
    )
    parser.add_argument(
        "--fee-bps-roundtrip",
        type=float,
        default=0.0,
        help="Per-trade round-trip cost deducted from each signal in basis points.",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=120,
        help="Maximum number of common symbols to process.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Worker slots for per-symbol parallelism.",
    )
    parser.add_argument(
        "--executor",
        choices=["thread", "process"],
        default="thread",
        help="Parallel executor type. 'thread' is sandbox-safe and works well for file I/O.",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=30,
        help="Reserve the last N overlapping dates per symbol as out-of-sample.",
    )
    parser.add_argument(
        "--recent-days",
        type=int,
        default=0,
        help="Only analyze the most recent N overlapping dates per symbol. 0 means full history.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    binance_symbols = {p.name for p in BINANCE.iterdir() if p.is_dir()}
    bybit_symbols = {p.name for p in BYBIT.iterdir() if p.is_dir()}
    common_symbols = sorted(binance_symbols & bybit_symbols)[: args.max_symbols]

    results = analyze_symbols_parallel(
        symbols=common_symbols,
        min_overlap_days=args.min_overlap_days,
        min_signal_bps=args.min_signal_bps,
        fee_bps_roundtrip=args.fee_bps_roundtrip,
        test_days=args.test_days,
        workers=max(1, args.workers),
        executor_kind=args.executor,
        recent_days=args.recent_days,
    )

    results.sort(
        key=lambda row: (
            -(row.test.avg_net_pnl_bps if row.test.signal_count else float("-inf")),
            -row.test.signal_count,
            row.symbol,
        )
    )

    summary_csv = OUT_DIR / "cross_exchange_edge_summary.csv"
    report_md = OUT_DIR / "cross_exchange_edge_report.md"
    write_report(
        results=results,
        summary_csv=summary_csv,
        report_md=report_md,
        min_overlap_days=args.min_overlap_days,
        min_signal_bps=args.min_signal_bps,
        fee_bps_roundtrip=args.fee_bps_roundtrip,
        test_days=args.test_days,
        workers=max(1, args.workers),
        recent_days=args.recent_days,
    )

    print(f"Analyzed symbols: {len(results)}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {report_md}")
    if results:
        print("Top 10 symbols by test avg net pnl:")
        for row in results[:10]:
            print(
                f"  {row.symbol}: test_avg_net={row.test.avg_net_pnl_bps:.4f}bps "
                f"test_signals={row.test.signal_count} total_avg_net={row.total.avg_net_pnl_bps:.4f}bps"
            )


if __name__ == "__main__":
    main()
