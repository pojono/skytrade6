#!/usr/bin/env python3
"""Backtest a strict CRVUSDT cross-exchange spread-reversion rule."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATALAKE = ROOT / "datalake"
BINANCE = DATALAKE / "binance"
BYBIT = DATALAKE / "bybit"
OUT_DIR = Path(__file__).resolve().parent / "out"


@dataclass
class Trade:
    symbol: str
    day: str
    month: str
    entry_ts_ms: int
    exit_ts_ms: int
    spread_bps: float
    spread_abs_bps: float
    spread_velocity_bps: float
    ls_diff_signed: float
    oi_diff_signed_bps: float
    carry_diff_signed_bps: float
    score: float
    gross_pnl_bps: float


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
            oi_chg_bps = 0.0
            if prev_oi and prev_oi != 0:
                oi_chg_bps = 10000.0 * (oi / prev_oi - 1.0)
            prev_oi = oi
            rows.append((ts, math.log(ls_ratio), oi_chg_bps))
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
            oi_chg_bps = 0.0
            if prev_oi and prev_oi != 0:
                oi_chg_bps = 10000.0 * (oi / prev_oi - 1.0)
            prev_oi = oi
            rows.append((ts, oi_chg_bps))
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


def advance_idx(rows: list[tuple[int, float] | tuple[int, float, float]], idx: int, ts: int) -> int:
    while idx + 1 < len(rows) and rows[idx + 1][0] <= ts:
        idx += 1
    return idx


def score_trade(ls_diff_signed: float, oi_diff_signed_bps: float, carry_diff_signed_bps: float) -> float:
    return ls_diff_signed + (oi_diff_signed_bps / 5.0) + (carry_diff_signed_bps / 2.0)


def build_day_trades(
    symbol: str,
    day: str,
    min_spread_bps: float,
    min_ls: float,
    min_oi_bps: float,
    min_carry_bps: float,
    min_score: float,
) -> list[Trade]:
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

    trades: list[Trade] = []
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

                if prev_spread is not None:
                    spread_abs = abs(prev_spread)
                    spread_velocity = (
                        abs(prev_spread - prev_prev_spread) if prev_prev_spread is not None else 0.0
                    )
                    trade_score = score_trade(prev_ls_diff, prev_oi_diff, prev_carry_diff)
                    if (
                        spread_abs >= min_spread_bps
                        and prev_ls_diff >= min_ls
                        and prev_oi_diff >= min_oi_bps
                        and prev_carry_diff >= min_carry_bps
                        and trade_score >= min_score
                    ):
                        direction = 1.0 if prev_spread > 0 else -1.0
                        gross_pnl_bps = direction * (prev_spread - spread)
                        trades.append(
                            Trade(
                                symbol=symbol,
                                day=day,
                                month=day[:7],
                                entry_ts_ms=bn_ts - 60_000,
                                exit_ts_ms=bn_ts,
                                spread_bps=prev_spread,
                                spread_abs_bps=spread_abs,
                                spread_velocity_bps=spread_velocity,
                                ls_diff_signed=prev_ls_diff,
                                oi_diff_signed_bps=prev_oi_diff,
                                carry_diff_signed_bps=prev_carry_diff,
                                score=trade_score,
                                gross_pnl_bps=gross_pnl_bps,
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
    return trades


def apply_daily_cap(trades: list[Trade], daily_cap: int) -> list[Trade]:
    if daily_cap <= 0:
        return trades
    kept: list[Trade] = []
    count_by_day: dict[str, int] = defaultdict(int)
    for trade in sorted(trades, key=lambda row: row.entry_ts_ms):
        if count_by_day[trade.day] >= daily_cap:
            continue
        kept.append(trade)
        count_by_day[trade.day] += 1
    return kept


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def summarize_fee(trades: list[Trade], fee_bps_roundtrip: float) -> dict[str, float]:
    net_values = [trade.gross_pnl_bps - fee_bps_roundtrip for trade in trades]
    wins = sum(1 for value in net_values if value > 0)
    return {
        "trades": float(len(trades)),
        "avg_net_bps": mean(net_values),
        "net_sum_bps": sum(net_values),
        "win_rate": (wins / len(trades)) if trades else math.nan,
    }


def write_trade_log(path: Path, trades: list[Trade], maker_fee_bps: float, taker_fee_bps: float) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "symbol",
                "day",
                "month",
                "entry_ts_ms",
                "exit_ts_ms",
                "spread_bps",
                "spread_abs_bps",
                "spread_velocity_bps",
                "ls_diff_signed",
                "oi_diff_signed_bps",
                "carry_diff_signed_bps",
                "score",
                "gross_pnl_bps",
                "net_maker_bps",
                "net_taker_bps",
            ]
        )
        for trade in trades:
            writer.writerow(
                [
                    trade.symbol,
                    trade.day,
                    trade.month,
                    trade.entry_ts_ms,
                    trade.exit_ts_ms,
                    f"{trade.spread_bps:.6f}",
                    f"{trade.spread_abs_bps:.6f}",
                    f"{trade.spread_velocity_bps:.6f}",
                    f"{trade.ls_diff_signed:.6f}",
                    f"{trade.oi_diff_signed_bps:.6f}",
                    f"{trade.carry_diff_signed_bps:.6f}",
                    f"{trade.score:.6f}",
                    f"{trade.gross_pnl_bps:.6f}",
                    f"{trade.gross_pnl_bps - maker_fee_bps:.6f}",
                    f"{trade.gross_pnl_bps - taker_fee_bps:.6f}",
                ]
            )


def build_report(
    symbol: str,
    trades: list[Trade],
    train_months: list[str],
    test_months: list[str],
    maker_fee_bps: float,
    taker_fee_bps: float,
    args: argparse.Namespace,
) -> str:
    train_set = set(train_months)
    test_set = set(test_months)
    train = [trade for trade in trades if trade.month in train_set]
    test = [trade for trade in trades if trade.month in test_set]
    maker_train = summarize_fee(train, maker_fee_bps)
    maker_test = summarize_fee(test, maker_fee_bps)
    taker_train = summarize_fee(train, taker_fee_bps)
    taker_test = summarize_fee(test, taker_fee_bps)

    month_rows: list[str] = []
    month_buckets: dict[str, list[Trade]] = defaultdict(list)
    for trade in trades:
        month_buckets[trade.month].append(trade)
    for month in sorted(month_buckets):
        month_trades = month_buckets[month]
        maker_avg = summarize_fee(month_trades, maker_fee_bps)["avg_net_bps"]
        taker_avg = summarize_fee(month_trades, taker_fee_bps)["avg_net_bps"]
        month_rows.append(
            f"| {month} | {len(month_trades)} | {maker_avg:.4f} | {taker_avg:.4f} |"
        )

    lines = [
        "# Extreme Spread CRV Report",
        "",
        "## Strategy",
        "",
        f"- Symbol: {symbol}",
        "- Entry idea: fade the 1-minute Binance vs Bybit close spread when it is extremely stretched and the supporting positioning/basis signal points in the same direction.",
        "- Exit: hold for exactly one aligned minute bar, then close on the next synchronized close.",
        "- Direction: short the rich exchange, long the cheap exchange.",
        "",
        "## Parameters",
        "",
        f"- Recent common days scanned: {args.recent_days}",
        f"- Spread threshold: {args.min_spread_bps:.2f} bps",
        f"- Min long/short diff: {args.min_ls:.2f}",
        f"- Min OI diff: {args.min_oi_bps:.2f} bps",
        f"- Min carry diff: {args.min_carry_bps:.2f} bps",
        f"- Min score: {args.min_score:.2f}",
        f"- Daily cap: {args.daily_cap}",
        f"- Maker round-trip fee: {maker_fee_bps:.2f} bps",
        f"- Taker round-trip fee: {taker_fee_bps:.2f} bps",
        f"- Train months: {', '.join(train_months) if train_months else '(none)'}",
        f"- Test months: {', '.join(test_months) if test_months else '(none)'}",
        "",
        "## Aggregate",
        "",
        f"- Total trades: {len(trades)}",
        f"- Train trades: {len(train)}",
        f"- Test trades: {len(test)}",
        f"- Train avg net after maker fees: {maker_train['avg_net_bps']:.4f} bps",
        f"- Test avg net after maker fees: {maker_test['avg_net_bps']:.4f} bps",
        f"- Train avg net after taker fees: {taker_train['avg_net_bps']:.4f} bps",
        f"- Test avg net after taker fees: {taker_test['avg_net_bps']:.4f} bps",
        f"- Test win rate after taker fees: {100.0 * taker_test['win_rate']:.2f}%",
        "",
        "## Monthly",
        "",
        "| Month | Trades | Avg Net Maker bps | Avg Net Taker bps |",
        "|---|---:|---:|---:|",
        *month_rows,
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="CRVUSDT")
    parser.add_argument("--recent-days", type=int, default=210)
    parser.add_argument("--test-months", type=int, default=2)
    parser.add_argument("--min-spread-bps", type=float, default=32.0)
    parser.add_argument("--min-ls", type=float, default=0.15)
    parser.add_argument("--min-oi-bps", type=float, default=5.0)
    parser.add_argument("--min-carry-bps", type=float, default=2.0)
    parser.add_argument("--min-score", type=float, default=14.0)
    parser.add_argument("--daily-cap", type=int, default=3)
    parser.add_argument("--maker-fee-bps-roundtrip", type=float, default=8.0)
    parser.add_argument("--taker-fee-bps-roundtrip", type=float, default=20.0)
    args = parser.parse_args()

    bn_dir = BINANCE / args.symbol
    bb_dir = BYBIT / args.symbol
    overlap = sorted(
        collect_dates(bn_dir, "kline_1m.csv")
        & collect_dates(bb_dir, "kline_1m.csv")
        & collect_dates(bn_dir, "metrics.csv")
        & collect_dates(bb_dir, "long_short_ratio_5min.csv")
        & collect_dates(bb_dir, "open_interest_5min.csv")
        & collect_dates(bn_dir, "mark_price_kline_1m.csv")
        & collect_dates(bn_dir, "index_price_kline_1m.csv")
        & collect_dates(bb_dir, "premium_index_kline_1m.csv")
    )
    if not overlap:
        raise SystemExit(f"No overlapping datalake coverage found for {args.symbol}.")

    dates = overlap[-args.recent_days :] if args.recent_days > 0 else overlap
    raw_trades: list[Trade] = []
    for day in dates:
        raw_trades.extend(
            build_day_trades(
                args.symbol,
                day,
                args.min_spread_bps,
                args.min_ls,
                args.min_oi_bps,
                args.min_carry_bps,
                args.min_score,
            )
        )
    trades = apply_daily_cap(raw_trades, args.daily_cap)

    months = sorted({trade.month for trade in trades})
    if not months:
        raise SystemExit("The configured filter produced no trades.")
    if args.test_months >= len(months):
        args.test_months = max(1, len(months) // 2)
    train_months = months[:-args.test_months]
    test_months = months[-args.test_months :]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trade_path = OUT_DIR / "trade_log.csv"
    report_path = OUT_DIR / "report.md"
    write_trade_log(
        trade_path,
        trades,
        args.maker_fee_bps_roundtrip,
        args.taker_fee_bps_roundtrip,
    )
    report_path.write_text(
        build_report(
            args.symbol,
            trades,
            train_months,
            test_months,
            args.maker_fee_bps_roundtrip,
            args.taker_fee_bps_roundtrip,
            args,
        )
    )

    print(f"Wrote {trade_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
