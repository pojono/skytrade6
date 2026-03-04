#!/usr/bin/env python3
"""Enrich the recent replay-optimized trade window with sub-minute microstructure features."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "out"
DEFAULT_INPUT = OUT_DIR / "candidate_trades_v3_replayopt.csv"
DATALAKE = ROOT.parent / "datalake"


@dataclass(frozen=True)
class Candidate:
    row: dict[str, str]
    symbol: str
    day: str
    entry_ts_ms: int
    exit_ts_ms: int
    gross_pnl_bps: float
    entry_spread_bps: float
    entry_spread_abs_bps: float
    entry_spread_velocity_bps: float


def calc_net_bps(trade: Candidate) -> float:
    fee_bps_roundtrip = 6.0
    extra_slippage_bps = 1.0
    spread_slip_coeff = 0.10
    velocity_slip_coeff = 0.05
    size_slip_coeff = 1.5
    min_signal_bps = 10.0
    per_trade_allocation = 0.25
    base_alloc_ref = 0.10

    stretch = max(0.0, trade.entry_spread_abs_bps - min_signal_bps)
    size_multiple_above_base = max(0.0, per_trade_allocation / base_alloc_ref - 1.0)
    slip = (
        extra_slippage_bps
        + spread_slip_coeff * stretch
        + velocity_slip_coeff * trade.entry_spread_velocity_bps
        + size_slip_coeff * size_multiple_above_base
    )
    return trade.gross_pnl_bps - fee_bps_roundtrip - slip


def load_candidates(path: Path, start_day: str, end_day: str) -> list[Candidate]:
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            day = row["day"]
            if not (start_day <= day <= end_day):
                continue
            rows.append(
                Candidate(
                    row=row,
                    symbol=row["symbol"],
                    day=day,
                    entry_ts_ms=int(row["entry_ts_ms"]),
                    exit_ts_ms=int(row["exit_ts_ms"]),
                    gross_pnl_bps=float(row["gross_pnl_bps"]),
                    entry_spread_bps=float(row["entry_spread_bps"]),
                    entry_spread_abs_bps=float(row["entry_spread_abs_bps"]),
                    entry_spread_velocity_bps=float(row["entry_spread_velocity_bps"]),
                )
            )
    rows.sort(key=lambda row: (row.symbol, row.day, row.entry_ts_ms))
    return rows


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else math.nan


def median(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return math.nan
    vals = sorted(vals)
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def to_ms_from_binance_depth(ts_text: str) -> int:
    dt = datetime.strptime(ts_text, "%Y-%m-%d %H:%M:%S")
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def enrich_bybit_orderbook(
    file_path: Path,
    queries: list[Candidate],
) -> dict[int, dict[str, float]]:
    results: dict[int, dict[str, float]] = {}
    if not queries:
        return results

    bid_book: dict[float, float] = {}
    ask_book: dict[float, float] = {}
    qidx = 0
    current_ts = None

    def set_side(book: dict[float, float], updates: list[list[str]]) -> None:
        for price_s, size_s in updates:
            price = float(price_s)
            size = float(size_s)
            if size <= 0:
                book.pop(price, None)
            else:
                book[price] = size

    def snapshot_features(ts_ms: int) -> dict[str, float]:
        best_bid = max(bid_book) if bid_book else math.nan
        best_ask = min(ask_book) if ask_book else math.nan
        mid = (best_bid + best_ask) / 2.0 if bid_book and ask_book else math.nan
        top_bid_levels = sorted(bid_book.items(), reverse=True)[:5]
        top_ask_levels = sorted(ask_book.items())[:5]
        bid_notional_5 = sum(price * size for price, size in top_bid_levels)
        ask_notional_5 = sum(price * size for price, size in top_ask_levels)
        imbalance = (
            (bid_notional_5 - ask_notional_5) / (bid_notional_5 + ask_notional_5)
            if (bid_notional_5 + ask_notional_5) > 0
            else math.nan
        )
        spread_bps = (
            ((best_ask - best_bid) / mid) * 10000.0
            if bid_book and ask_book and mid > 0
            else math.nan
        )
        return {
            "bybit_book_lag_ms": max(0, queries[qidx].entry_ts_ms - ts_ms),
            "bybit_best_bid": best_bid,
            "bybit_best_ask": best_ask,
            "bybit_book_spread_bps": spread_bps,
            "bybit_top5_bid_notional": bid_notional_5,
            "bybit_top5_ask_notional": ask_notional_5,
            "bybit_top5_imbalance": imbalance,
        }

    with file_path.open() as handle:
        for line in handle:
            payload = json.loads(line)
            ts_ms = int(payload.get("cts") or payload["ts"])
            data = payload["data"]
            if payload["type"] == "snapshot":
                bid_book = {float(p): float(s) for p, s in data["b"] if float(s) > 0}
                ask_book = {float(p): float(s) for p, s in data["a"] if float(s) > 0}
            else:
                set_side(bid_book, data.get("b", []))
                set_side(ask_book, data.get("a", []))
            current_ts = ts_ms

            while qidx < len(queries) and queries[qidx].entry_ts_ms <= ts_ms:
                if current_ts is not None:
                    results[queries[qidx].entry_ts_ms] = snapshot_features(current_ts)
                qidx += 1
            if qidx >= len(queries):
                break

    if current_ts is not None:
        while qidx < len(queries):
            results[queries[qidx].entry_ts_ms] = snapshot_features(current_ts)
            qidx += 1
    return results


def enrich_bybit_trades(
    file_path: Path,
    queries: list[Candidate],
    lookback_ms: int = 5000,
) -> dict[int, dict[str, float]]:
    results: dict[int, dict[str, float]] = {}
    if not queries:
        return results

    window: deque[tuple[int, float, str]] = deque()
    qidx = 0
    with file_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ts_ms = int(float(row["timestamp"]) * 1000.0)
            price = float(row["price"])
            size = float(row["size"])
            notional = price * size
            side = row["side"]

            while qidx < len(queries) and queries[qidx].entry_ts_ms <= ts_ms:
                cutoff = queries[qidx].entry_ts_ms - lookback_ms
                while window and window[0][0] < cutoff:
                    window.popleft()
                buy_notional = sum(n for _, n, s in window if s == "Buy")
                sell_notional = sum(n for _, n, s in window if s == "Sell")
                total = buy_notional + sell_notional
                results[queries[qidx].entry_ts_ms] = {
                    "bybit_trade_count_5s": float(len(window)),
                    "bybit_buy_notional_5s": buy_notional,
                    "bybit_sell_notional_5s": sell_notional,
                    "bybit_trade_imbalance_5s": (
                        (buy_notional - sell_notional) / total if total > 0 else math.nan
                    ),
                }
                qidx += 1

            window.append((ts_ms, notional, side))

    while qidx < len(queries):
        cutoff = queries[qidx].entry_ts_ms - lookback_ms
        while window and window[0][0] < cutoff:
            window.popleft()
        buy_notional = sum(n for _, n, s in window if s == "Buy")
        sell_notional = sum(n for _, n, s in window if s == "Sell")
        total = buy_notional + sell_notional
        results[queries[qidx].entry_ts_ms] = {
            "bybit_trade_count_5s": float(len(window)),
            "bybit_buy_notional_5s": buy_notional,
            "bybit_sell_notional_5s": sell_notional,
            "bybit_trade_imbalance_5s": (
                (buy_notional - sell_notional) / total if total > 0 else math.nan
            ),
        }
        qidx += 1
    return results


def enrich_binance_trades(
    file_path: Path,
    queries: list[Candidate],
    lookback_ms: int = 5000,
) -> dict[int, dict[str, float]]:
    results: dict[int, dict[str, float]] = {}
    if not queries:
        return results

    window: deque[tuple[int, float, bool]] = deque()
    qidx = 0
    with file_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ts_ms = int(row["time"])
            notional = float(row["price"]) * float(row["qty"])
            is_buyer_maker = row["is_buyer_maker"].lower() == "true"

            while qidx < len(queries) and queries[qidx].entry_ts_ms <= ts_ms:
                cutoff = queries[qidx].entry_ts_ms - lookback_ms
                while window and window[0][0] < cutoff:
                    window.popleft()
                # if buyer is maker, aggressive side is sell; otherwise aggressive buy
                buy_notional = sum(n for _, n, buyer_maker in window if not buyer_maker)
                sell_notional = sum(n for _, n, buyer_maker in window if buyer_maker)
                total = buy_notional + sell_notional
                results[queries[qidx].entry_ts_ms] = {
                    "binance_trade_count_5s": float(len(window)),
                    "binance_buy_notional_5s": buy_notional,
                    "binance_sell_notional_5s": sell_notional,
                    "binance_trade_imbalance_5s": (
                        (buy_notional - sell_notional) / total if total > 0 else math.nan
                    ),
                }
                qidx += 1

            window.append((ts_ms, notional, is_buyer_maker))

    while qidx < len(queries):
        cutoff = queries[qidx].entry_ts_ms - lookback_ms
        while window and window[0][0] < cutoff:
            window.popleft()
        buy_notional = sum(n for _, n, buyer_maker in window if not buyer_maker)
        sell_notional = sum(n for _, n, buyer_maker in window if buyer_maker)
        total = buy_notional + sell_notional
        results[queries[qidx].entry_ts_ms] = {
            "binance_trade_count_5s": float(len(window)),
            "binance_buy_notional_5s": buy_notional,
            "binance_sell_notional_5s": sell_notional,
            "binance_trade_imbalance_5s": (
                (buy_notional - sell_notional) / total if total > 0 else math.nan
            ),
        }
        qidx += 1
    return results


def enrich_binance_bookdepth(
    file_path: Path,
    queries: list[Candidate],
) -> dict[int, dict[str, float]]:
    results: dict[int, dict[str, float]] = {}
    if not queries:
        return results

    snapshots: list[tuple[int, dict[float, tuple[float, float]]]] = []
    current_ts = None
    bucket: dict[float, tuple[float, float]] = {}

    with file_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ts_ms = to_ms_from_binance_depth(row["timestamp"])
            pct = float(row["percentage"])
            depth = float(row["depth"])
            notional = float(row["notional"])
            if current_ts is None:
                current_ts = ts_ms
            if ts_ms != current_ts:
                snapshots.append((current_ts, bucket))
                current_ts = ts_ms
                bucket = {}
            bucket[pct] = (depth, notional)
        if current_ts is not None:
            snapshots.append((current_ts, bucket))

    sidx = 0
    last_ts = None
    last_bucket: dict[float, tuple[float, float]] | None = None
    for query in queries:
        while sidx < len(snapshots) and snapshots[sidx][0] <= query.entry_ts_ms:
            last_ts, last_bucket = snapshots[sidx]
            sidx += 1
        if last_bucket is None or last_ts is None:
            continue
        bid_notional_1pct = sum(
            notional for pct, (_, notional) in last_bucket.items() if pct < 0 and abs(pct) <= 1.0
        )
        ask_notional_1pct = sum(
            notional for pct, (_, notional) in last_bucket.items() if pct > 0 and abs(pct) <= 1.0
        )
        total = bid_notional_1pct + ask_notional_1pct
        results[query.entry_ts_ms] = {
            "binance_depth_lag_ms": max(0, query.entry_ts_ms - last_ts),
            "binance_bid_notional_1pct": bid_notional_1pct,
            "binance_ask_notional_1pct": ask_notional_1pct,
            "binance_depth_imbalance_1pct": (
                (bid_notional_1pct - ask_notional_1pct) / total if total > 0 else math.nan
            ),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--start-day", default="2026-02-24")
    parser.add_argument("--end-day", default="2026-03-02")
    parser.add_argument("--output-csv", type=Path, default=OUT_DIR / "microstructure_window_analysis.csv")
    parser.add_argument("--output-md", type=Path, default=OUT_DIR / "microstructure_window_analysis.md")
    args = parser.parse_args()

    candidates = load_candidates(args.input, args.start_day, args.end_day)
    grouped: dict[tuple[str, str], list[Candidate]] = {}
    for trade in candidates:
        grouped.setdefault((trade.symbol, trade.day), []).append(trade)

    enriched_rows: list[dict[str, object]] = []
    missing_counts = Counter()

    for (symbol, day), trades in grouped.items():
        bybit_dir = DATALAKE / "bybit" / symbol
        binance_dir = DATALAKE / "binance" / symbol
        orderbook = bybit_dir / f"{day}_orderbook.jsonl"
        bybit_trades = bybit_dir / f"{day}_trades.csv"
        binance_depth = binance_dir / f"{day}_bookDepth.csv"
        binance_trades = binance_dir / f"{day}_trades.csv"

        orderbook_map = enrich_bybit_orderbook(orderbook, trades) if orderbook.exists() else {}
        bybit_trade_map = enrich_bybit_trades(bybit_trades, trades) if bybit_trades.exists() else {}
        binance_depth_map = enrich_binance_bookdepth(binance_depth, trades) if binance_depth.exists() else {}
        binance_trade_map = enrich_binance_trades(binance_trades, trades) if binance_trades.exists() else {}

        if not orderbook.exists():
            missing_counts["bybit_orderbook"] += len(trades)
        if not bybit_trades.exists():
            missing_counts["bybit_trades"] += len(trades)
        if not binance_depth.exists():
            missing_counts["binance_depth"] += len(trades)
        if not binance_trades.exists():
            missing_counts["binance_trades"] += len(trades)

        for trade in trades:
            net_bps = calc_net_bps(trade)
            row = dict(trade.row)
            row["net_pnl_bps_25pct_model"] = f"{net_bps:.6f}"
            row["is_winner_25pct_model"] = "1" if net_bps > 0 else "0"
            for mapping in (orderbook_map, bybit_trade_map, binance_depth_map, binance_trade_map):
                if trade.entry_ts_ms in mapping:
                    for key, value in mapping[trade.entry_ts_ms].items():
                        row[key] = f"{value:.6f}" if isinstance(value, float) else str(value)
            enriched_rows.append(row)

    fieldnames = list(enriched_rows[0].keys()) if enriched_rows else []
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched_rows)

    winners = [row for row in enriched_rows if row["is_winner_25pct_model"] == "1"]
    losers = [row for row in enriched_rows if row["is_winner_25pct_model"] == "0"]

    feature_names = [
        "bybit_book_spread_bps",
        "bybit_top5_imbalance",
        "bybit_trade_imbalance_5s",
        "bybit_trade_count_5s",
        "binance_trade_imbalance_5s",
        "binance_trade_count_5s",
        "binance_depth_imbalance_1pct",
        "bybit_book_lag_ms",
        "binance_depth_lag_ms",
    ]
    feature_lines = []
    for name in feature_names:
        wvals = [float(row[name]) for row in winners if name in row and row[name] != "" and math.isfinite(float(row[name]))]
        lvals = [float(row[name]) for row in losers if name in row and row[name] != "" and math.isfinite(float(row[name]))]
        if not wvals and not lvals:
            continue
        feature_lines.append(
            f"| {name} | {mean(wvals):.4f} | {mean(lvals):.4f} | {(mean(wvals) - mean(lvals)):.4f} |"
        )

    net_values = [float(row["net_pnl_bps_25pct_model"]) for row in enriched_rows]
    lines = [
        "# Microstructure Window Analysis",
        "",
        f"- Input trades: {args.input}",
        f"- Date window: {args.start_day} through {args.end_day}",
        f"- Trades analyzed: {len(enriched_rows)}",
        f"- Symbols: {', '.join(sorted({row['symbol'] for row in enriched_rows})) if enriched_rows else ''}",
        f"- Winners under frozen 25% model: {len(winners)}",
        f"- Losers under frozen 25% model: {len(losers)}",
        f"- Mean modeled net PnL: {mean(net_values):.4f} bps",
        f"- Median modeled net PnL: {median(net_values):.4f} bps",
        "",
        "## Feature Means (Winners vs Losers)",
        "",
        "| Feature | Winners | Losers | Delta |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(feature_lines or ["| n/a | n/a | n/a | n/a |"])
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            f"- Missing bybit orderbook enrichments: {missing_counts['bybit_orderbook']}",
            f"- Missing bybit trade enrichments: {missing_counts['bybit_trades']}",
            f"- Missing binance depth enrichments: {missing_counts['binance_depth']}",
            f"- Missing binance trade enrichments: {missing_counts['binance_trades']}",
            "",
            "## Notes",
            "",
            "- This is a first-pass microstructure overlay on the downloaded 7-day window only.",
            "- Bybit enrichments use true L2 order book updates from the ob200 feed.",
            "- Binance enrichments use trade flow plus aggregated `bookDepth` percentage buckets, not true top-of-book quotes.",
            "- Results here are descriptive. They help identify where the 1-minute edge may depend on intraminute conditions.",
        ]
    )
    args.output_md.write_text("\n".join(lines))

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
