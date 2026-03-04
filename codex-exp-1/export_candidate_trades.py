#!/usr/bin/env python3
"""Export timestamped trade records for the frozen cross-exchange basket."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from dense_filter_sweep import (
    BINANCE,
    BYBIT,
    advance_idx,
    collect_dates,
    load_binance_basis,
    load_binance_metrics,
    load_bybit_long_short,
    load_bybit_oi,
    load_bybit_premium,
    load_close_rows,
    load_symbols,
)


OUT_DIR = Path(__file__).resolve().parent / "out"
DEFAULT_SYMBOLS = OUT_DIR / "candidate_basket_v3.txt"


@dataclass
class TradeRecord:
    symbol: str
    day: str
    month: str
    entry_ts_ms: int
    exit_ts_ms: int
    gross_pnl_bps: float
    entry_spread_bps: float
    entry_spread_abs_bps: float
    entry_spread_velocity_bps: float
    ls_diff_signed: float
    oi_diff_signed_bps: float
    carry_diff_signed_bps: float

    @property
    def score(self) -> float:
        return self.ls_diff_signed + (self.oi_diff_signed_bps / 5.0) + (self.carry_diff_signed_bps / 2.0)


def passes_filter(
    row: TradeRecord,
    ls_threshold: float,
    oi_threshold_bps: float,
    carry_threshold_bps: float,
) -> bool:
    return (
        row.ls_diff_signed >= ls_threshold
        and row.oi_diff_signed_bps >= oi_threshold_bps
        and row.carry_diff_signed_bps >= carry_threshold_bps
    )


def build_day_trade_records(symbol: str, day: str, min_signal_bps: float) -> list[TradeRecord]:
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

    out: list[TradeRecord] = []
    i = j = 0
    m_idx = ls_idx = oi_idx = basis_idx = prem_idx = 0
    prev_ts: int | None = None
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

                if prev_spread is not None and prev_ts is not None and abs(prev_spread) >= min_signal_bps:
                    direction = 1.0 if prev_spread > 0 else -1.0
                    gross_pnl = direction * (prev_spread - spread)
                    spread_velocity = (
                        abs(prev_spread - prev_prev_spread) if prev_prev_spread is not None else 0.0
                    )
                    out.append(
                        TradeRecord(
                            symbol=symbol,
                            day=day,
                            month=day[:7],
                            entry_ts_ms=prev_ts,
                            exit_ts_ms=bn_ts,
                            gross_pnl_bps=gross_pnl,
                            entry_spread_bps=prev_spread,
                            entry_spread_abs_bps=abs(prev_spread),
                            entry_spread_velocity_bps=spread_velocity,
                            ls_diff_signed=prev_ls_diff,
                            oi_diff_signed_bps=prev_oi_diff,
                            carry_diff_signed_bps=prev_carry_diff,
                        )
                    )

                prev_prev_spread = prev_spread
                prev_ts = bn_ts
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


def build_symbol_trade_records(
    symbol: str,
    min_overlap_days: int,
    min_signal_bps: float,
    recent_days: int,
    ls_threshold: float,
    oi_threshold_bps: float,
    carry_threshold_bps: float,
) -> list[TradeRecord]:
    overlap = sorted(
        collect_dates(BINANCE / symbol, "kline_1m.csv")
        & collect_dates(BYBIT / symbol, "kline_1m.csv")
    )
    if len(overlap) < min_overlap_days:
        return []
    days = overlap[-recent_days:] if recent_days > 0 else overlap
    out: list[TradeRecord] = []
    for day in days:
        day_rows = build_day_trade_records(symbol, day, min_signal_bps)
        out.extend(
            row
            for row in day_rows
            if passes_filter(row, ls_threshold, oi_threshold_bps, carry_threshold_bps)
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols-file", type=Path, default=DEFAULT_SYMBOLS)
    parser.add_argument("--min-overlap-days", type=int, default=90)
    parser.add_argument("--min-signal-bps", type=float, default=10.0)
    parser.add_argument("--recent-days", type=int, default=210)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ls-threshold", type=float, default=0.15)
    parser.add_argument("--oi-threshold-bps", type=float, default=5.0)
    parser.add_argument("--carry-threshold-bps", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "candidate_trades_v3.csv")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = load_symbols(args.symbols_file)

    rows: list[TradeRecord] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                build_symbol_trade_records,
                symbol,
                args.min_overlap_days,
                args.min_signal_bps,
                args.recent_days,
                args.ls_threshold,
                args.oi_threshold_bps,
                args.carry_threshold_bps,
            ): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            rows.extend(future.result())

    rows.sort(key=lambda row: (row.entry_ts_ms, -row.score, row.symbol))

    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "symbol",
                "day",
                "month",
                "entry_ts_ms",
                "exit_ts_ms",
                "gross_pnl_bps",
                "entry_spread_bps",
                "entry_spread_abs_bps",
                "entry_spread_velocity_bps",
                "ls_diff_signed",
                "oi_diff_signed_bps",
                "carry_diff_signed_bps",
                "score",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.symbol,
                    row.day,
                    row.month,
                    row.entry_ts_ms,
                    row.exit_ts_ms,
                    f"{row.gross_pnl_bps:.6f}",
                    f"{row.entry_spread_bps:.6f}",
                    f"{row.entry_spread_abs_bps:.6f}",
                    f"{row.entry_spread_velocity_bps:.6f}",
                    f"{row.ls_diff_signed:.6f}",
                    f"{row.oi_diff_signed_bps:.6f}",
                    f"{row.carry_diff_signed_bps:.6f}",
                    f"{row.score:.6f}",
                ]
            )

    print(f"Exported trades: {len(rows)}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
