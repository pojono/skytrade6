#!/usr/bin/env python3
"""Audit recent CRVUSDT strategy triggers using local trade and order book data."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import extreme_spread_crv as base


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "out"


@dataclass
class MicroTrade:
    day: str
    month: str
    entry_ts_ms: int
    exit_ts_ms: int
    gross_pnl_bps: float
    net_taker_bps: float
    spread_abs_bps: float
    spread_velocity_bps: float
    score: float
    binance_flow_usd: float
    bybit_flow_usd: float
    combined_flow_usd: float
    bybit_book_spread_bps: float
    bybit_top_depth_usd: float
    flow_aligned: int


def load_binance_flow(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    if not path.exists():
        return rows
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                ts = int(float(row["time"]))
                price = float(row["price"])
                qty = float(row["qty"])
                is_buyer_maker = str(row["is_buyer_maker"]).lower() == "true"
            except (KeyError, TypeError, ValueError):
                continue
            signed_notional = price * qty * (-1.0 if is_buyer_maker else 1.0)
            rows.append((ts, signed_notional))
    return rows


def load_bybit_flow(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    if not path.exists():
        return rows
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                ts = int(float(row["timestamp"]) * 1000)
                side = row["side"]
                size = float(row["size"])
                price = float(row["price"])
            except (KeyError, TypeError, ValueError):
                continue
            signed_notional = price * size * (1.0 if side == "Buy" else -1.0)
            rows.append((ts, signed_notional))
    return rows


def sum_window(rows: list[tuple[int, float]], start_ts: int, end_ts: int) -> float:
    total = 0.0
    for ts, value in rows:
        if ts < start_ts:
            continue
        if ts >= end_ts:
            break
        total += value
    return total


def load_orderbook_snapshots(path: Path) -> list[tuple[int, float, float]]:
    rows: list[tuple[int, float, float]] = []
    if not path.exists():
        return rows
    best_bid = 0.0
    best_ask = 0.0
    bid_qty = 0.0
    ask_qty = 0.0
    with path.open() as handle:
        for line in handle:
            try:
                payload = json.loads(line)
                ts = int(payload["ts"])
                data = payload["data"]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            bids = data.get("b") or []
            asks = data.get("a") or []
            if bids:
                try:
                    best_bid = float(bids[0][0])
                    bid_qty = float(bids[0][1])
                except (TypeError, ValueError, IndexError):
                    pass
            if asks:
                try:
                    best_ask = float(asks[0][0])
                    ask_qty = float(asks[0][1])
                except (TypeError, ValueError, IndexError):
                    pass
            if best_bid > 0 and best_ask > 0 and best_ask >= best_bid:
                mid = (best_bid + best_ask) / 2.0
                spread_bps = 10000.0 * (best_ask / best_bid - 1.0) if best_bid else math.nan
                top_depth_usd = mid * (bid_qty + ask_qty)
                rows.append((ts, spread_bps, top_depth_usd))
    return rows


def nearest_snapshot(rows: list[tuple[int, float, float]], ts: int) -> tuple[float, float]:
    if not rows:
        return math.nan, math.nan
    best = rows[0]
    best_dist = abs(rows[0][0] - ts)
    for row in rows[1:]:
        dist = abs(row[0] - ts)
        if dist < best_dist:
            best = row
            best_dist = dist
        elif row[0] > ts and dist > best_dist:
            break
    return best[1], best[2]


def audit_day(symbol: str, day: str, taker_fee_bps: float) -> list[MicroTrade]:
    trades = base.build_day_trades(symbol, day, 32.0, 0.15, 5.0, 2.0, 14.0)
    trades = base.apply_daily_cap(trades, 3)
    if not trades:
        return []

    bn_dir = base.BINANCE / symbol
    bb_dir = base.BYBIT / symbol
    bn_flow = load_binance_flow(bn_dir / f"{day}_trades.csv")
    bb_flow = load_bybit_flow(bb_dir / f"{day}_trades.csv")
    ob_rows = load_orderbook_snapshots(bb_dir / f"{day}_orderbook.jsonl")
    if not (bn_flow and bb_flow and ob_rows):
        return []

    out: list[MicroTrade] = []
    for trade in trades:
        start_ts = trade.entry_ts_ms - 60_000
        end_ts = trade.entry_ts_ms
        bn_signed = sum_window(bn_flow, start_ts, end_ts)
        bb_signed = sum_window(bb_flow, start_ts, end_ts)
        spread_bps, top_depth_usd = nearest_snapshot(ob_rows, trade.entry_ts_ms)
        signal_dir = 1.0 if trade.spread_bps > 0 else -1.0
        combined_flow = signal_dir * (bn_signed - bb_signed)
        out.append(
            MicroTrade(
                day=trade.day,
                month=trade.month,
                entry_ts_ms=trade.entry_ts_ms,
                exit_ts_ms=trade.exit_ts_ms,
                gross_pnl_bps=trade.gross_pnl_bps,
                net_taker_bps=trade.gross_pnl_bps - taker_fee_bps,
                spread_abs_bps=trade.spread_abs_bps,
                spread_velocity_bps=trade.spread_velocity_bps,
                score=trade.score,
                binance_flow_usd=bn_signed,
                bybit_flow_usd=bb_signed,
                combined_flow_usd=combined_flow,
                bybit_book_spread_bps=spread_bps,
                bybit_top_depth_usd=top_depth_usd,
                flow_aligned=1 if combined_flow > 0 else 0,
            )
        )
    return out


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def build_report(rows: list[MicroTrade], audited_days: list[str]) -> str:
    total = len(rows)
    positive = [row for row in rows if row.net_taker_bps > 0]
    aligned = [row for row in rows if row.flow_aligned]
    misaligned = [row for row in rows if not row.flow_aligned]
    high_score = [row for row in rows if row.score >= 20.0]
    high_depth = [row for row in rows if row.bybit_top_depth_usd >= 25_000.0]
    tight_book = [row for row in rows if row.bybit_book_spread_bps <= 12.0]
    aligned_avg = avg([row.net_taker_bps for row in aligned])
    misaligned_avg = avg([row.net_taker_bps for row in misaligned])
    high_score_avg = avg([row.net_taker_bps for row in high_score])
    high_depth_avg = avg([row.net_taker_bps for row in high_depth])
    tight_book_avg = avg([row.net_taker_bps for row in tight_book])

    observations: list[str] = []
    if not math.isnan(high_score_avg) and high_score_avg > avg([row.net_taker_bps for row in rows]):
        observations.append("- Higher-score signals remain the cleanest positive subset and should stay central to the filter.")
    if not math.isnan(tight_book_avg) and tight_book_avg > avg([row.net_taker_bps for row in rows]):
        observations.append("- Tighter Bybit top-of-book conditions improve average outcome and look like a useful execution gate.")
    if not math.isnan(high_depth_avg) and high_depth_avg < avg([row.net_taker_bps for row in rows]):
        observations.append("- Larger displayed top depth does not help here; the better trades are not simply the deepest-book moments.")
    if not math.isnan(misaligned_avg) and misaligned_avg > aligned_avg:
        observations.append("- The best trades often arrive after the most obvious aggressive flow burst has already started fading, not while it is still strongly aligned.")
    if not observations:
        observations.append("- No single microstructure bucket dominates enough yet; the next pass should test combined conditions directly.")

    lines = [
        "# CRV Microstructure Audit",
        "",
        "## Scope",
        "",
        "- Symbol: CRVUSDT",
        "- Purpose: test whether the surviving CRV edge is concentrated in recent trade-flow / liquidity conditions that are observable with local microstructure data.",
        f"- Audited days: {', '.join(audited_days)}",
        f"- Trigger count audited: {total}",
        "",
        "## Aggregate",
        "",
        f"- Avg net after 20 bps taker fee: {avg([row.net_taker_bps for row in rows]):.4f} bps",
        f"- Positive trigger share: {100.0 * len(positive) / total:.2f}%",
        f"- Avg combined signed flow before entry: {avg([row.combined_flow_usd for row in rows]):.2f} USD",
        f"- Avg Bybit top-of-book spread: {avg([row.bybit_book_spread_bps for row in rows]):.4f} bps",
        f"- Avg Bybit top depth: {avg([row.bybit_top_depth_usd for row in rows]):.2f} USD",
        "",
        "## Buckets",
        "",
        f"- Flow aligned with signal: {len(aligned)} trades, avg net {aligned_avg:.4f} bps",
        f"- Flow not aligned: {len(misaligned)} trades, avg net {misaligned_avg:.4f} bps",
        f"- High score (>=20): {len(high_score)} trades, avg net {high_score_avg:.4f} bps",
        f"- Higher top depth (>=25k USD): {len(high_depth)} trades, avg net {high_depth_avg:.4f} bps",
        f"- Tight Bybit book (<=12 bps): {len(tight_book)} trades, avg net {tight_book_avg:.4f} bps",
        "",
        "## Interpretation",
        "",
        "- This is a recent-days execution/regime audit, not a replacement for the full historical signal test.",
        *observations,
    ]
    return "\n".join(lines) + "\n"


def write_trade_log(path: Path, rows: list[MicroTrade]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "day",
                "month",
                "entry_ts_ms",
                "exit_ts_ms",
                "gross_pnl_bps",
                "net_taker_bps",
                "spread_abs_bps",
                "spread_velocity_bps",
                "score",
                "binance_flow_usd",
                "bybit_flow_usd",
                "combined_flow_usd",
                "flow_aligned",
                "bybit_book_spread_bps",
                "bybit_top_depth_usd",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.day,
                    row.month,
                    row.entry_ts_ms,
                    row.exit_ts_ms,
                    f"{row.gross_pnl_bps:.6f}",
                    f"{row.net_taker_bps:.6f}",
                    f"{row.spread_abs_bps:.6f}",
                    f"{row.spread_velocity_bps:.6f}",
                    f"{row.score:.6f}",
                    f"{row.binance_flow_usd:.6f}",
                    f"{row.bybit_flow_usd:.6f}",
                    f"{row.combined_flow_usd:.6f}",
                    row.flow_aligned,
                    f"{row.bybit_book_spread_bps:.6f}",
                    f"{row.bybit_top_depth_usd:.6f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="CRVUSDT")
    parser.add_argument("--taker-fee-bps-roundtrip", type=float, default=20.0)
    args = parser.parse_args()

    bn_dir = base.BINANCE / args.symbol
    bb_dir = base.BYBIT / args.symbol
    audited_days = sorted(
        base.collect_dates(bn_dir, "trades.csv")
        & base.collect_dates(bb_dir, "trades.csv")
        & base.collect_dates(bb_dir, "orderbook.jsonl")
        & base.collect_dates(bn_dir, "kline_1m.csv")
        & base.collect_dates(bb_dir, "kline_1m.csv")
    )
    if not audited_days:
        raise SystemExit(f"No overlapping microstructure coverage found for {args.symbol}.")

    rows: list[MicroTrade] = []
    for day in audited_days:
        rows.extend(audit_day(args.symbol, day, args.taker_fee_bps_roundtrip))
    if not rows:
        raise SystemExit("No qualifying strategy triggers found in the audited microstructure days.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "crv_micro_trade_log.csv"
    md_path = OUT_DIR / "crv_micro_report.md"
    write_trade_log(csv_path, rows)
    md_path.write_text(build_report(rows, audited_days))
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
