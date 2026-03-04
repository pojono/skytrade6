#!/usr/bin/env python3
"""Strict fill replay for the accepted microstructure-gated 30d sleeve."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "out"
DATALAKE = ROOT.parent / "datalake"

DEFAULT_SOURCE = OUT_DIR / "microstructure_replay_train_gate_30d.csv"
DEFAULT_FILLS = OUT_DIR / "paper_fills_micro_train_gate_30d.csv"


@dataclass(frozen=True)
class ReplayTrade:
    symbol: str
    day: str
    entry_ts_ms: int
    exit_ts_ms: int
    entry_spread_bps: float
    gross_pnl_bps: float
    alloc_dollars: float
    modeled_net_bps: float
    modeled_pnl_dollars: float


def to_ms_from_binance_depth(ts_text: str) -> int:
    dt = datetime.strptime(ts_text, "%Y-%m-%d %H:%M:%S")
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def load_replay_trades(source_path: Path, fills_path: Path) -> list[ReplayTrade]:
    source_rows = {
        (row["symbol"], row["entry_ts_ms"]): row
        for row in csv.DictReader(source_path.open())
    }
    trades: list[ReplayTrade] = []
    for row in csv.DictReader(fills_path.open()):
        src = source_rows[(row["symbol"], row["entry_ts_ms"])]
        trades.append(
            ReplayTrade(
                symbol=row["symbol"],
                day=row["day"],
                entry_ts_ms=int(row["entry_ts_ms"]),
                exit_ts_ms=int(row["exit_ts_ms"]),
                entry_spread_bps=float(src["entry_spread_bps"]),
                gross_pnl_bps=float(row["gross_pnl_bps"]),
                alloc_dollars=float(row["alloc_dollars"]),
                modeled_net_bps=float(row["net_pnl_bps"]),
                modeled_pnl_dollars=float(row["pnl_dollars"]),
            )
        )
    trades.sort(key=lambda t: (t.symbol, t.day, t.entry_ts_ms))
    return trades


def simulate_bybit_fill(book: dict[str, dict[float, float]], side: str, leg_notional: float) -> tuple[bool, float, float]:
    bids = book["bids"]
    asks = book["asks"]
    if not bids or not asks:
        return False, math.nan, math.nan
    best_bid = max(bids)
    best_ask = min(asks)
    mid = (best_bid + best_ask) / 2.0
    if mid <= 0:
        return False, math.nan, math.nan
    target_base = leg_notional / mid
    remain = target_base
    filled_quote = 0.0
    levels = sorted(asks.items()) if side == "buy" else sorted(bids.items(), reverse=True)
    for price, size in levels:
        take = min(remain, size)
        filled_quote += take * price
        remain -= take
        if remain <= 1e-12:
            break
    if remain > 1e-9:
        return False, math.nan, math.nan
    avg_price = filled_quote / target_base
    if side == "buy":
        slippage_bps = ((avg_price - mid) / mid) * 10000.0
    else:
        slippage_bps = ((mid - avg_price) / mid) * 10000.0
    return True, slippage_bps, mid


def build_bybit_snapshot_map(file_path: Path, query_times: list[int]) -> dict[int, dict[str, dict[float, float]]]:
    result: dict[int, dict[str, dict[float, float]]] = {}
    if not query_times:
        return result

    query_times = sorted(query_times)
    qidx = 0
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}

    def set_side(book: dict[float, float], updates: list[list[str]]) -> None:
        for p_s, s_s in updates:
            p = float(p_s)
            s = float(s_s)
            if s <= 0:
                book.pop(p, None)
            else:
                book[p] = s

    with file_path.open() as handle:
        for line in handle:
            payload = json.loads(line)
            ts_ms = int(payload.get("cts") or payload["ts"])
            data = payload["data"]
            if payload["type"] == "snapshot":
                bids = {float(p): float(s) for p, s in data["b"] if float(s) > 0}
                asks = {float(p): float(s) for p, s in data["a"] if float(s) > 0}
            else:
                set_side(bids, data.get("b", []))
                set_side(asks, data.get("a", []))
            while qidx < len(query_times) and query_times[qidx] <= ts_ms:
                result[query_times[qidx]] = {"bids": dict(bids), "asks": dict(asks)}
                qidx += 1
            if qidx >= len(query_times):
                break
    if bids and asks:
        while qidx < len(query_times):
            result[query_times[qidx]] = {"bids": dict(bids), "asks": dict(asks)}
            qidx += 1
    return result


def build_binance_depth_map(file_path: Path, query_times: list[int]) -> dict[int, dict[float, float]]:
    result: dict[int, dict[float, float]] = {}
    if not query_times:
        return result

    snapshots: list[tuple[int, dict[float, float]]] = []
    current_ts = None
    bucket: dict[float, float] = {}
    with file_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ts_ms = to_ms_from_binance_depth(row["timestamp"])
            pct = float(row["percentage"])
            notional = float(row["notional"])
            if current_ts is None:
                current_ts = ts_ms
            if ts_ms != current_ts:
                snapshots.append((current_ts, bucket))
                current_ts = ts_ms
                bucket = {}
            bucket[pct] = notional
        if current_ts is not None:
            snapshots.append((current_ts, bucket))

    sidx = 0
    last_bucket: dict[float, float] | None = None
    for qt in sorted(query_times):
        while sidx < len(snapshots) and snapshots[sidx][0] <= qt:
            last_bucket = snapshots[sidx][1]
            sidx += 1
        if last_bucket is not None:
            result[qt] = dict(last_bucket)
    return result


def simulate_binance_fill(snapshot: dict[float, float], side: str, leg_notional: float) -> tuple[bool, float]:
    pct = 1.0 if side == "buy" else -1.0
    cap = snapshot.get(pct)
    if cap is None or cap <= 0:
        return False, math.nan
    frac = leg_notional / cap
    if frac > 1.0:
        return False, math.nan
    # Approximate average execution inside the 1% cumulative depth bucket.
    slippage_bps = 50.0 * frac
    return True, slippage_bps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--fills", type=Path, default=DEFAULT_FILLS)
    parser.add_argument("--output-csv", type=Path, default=OUT_DIR / "strict_fill_replay_train_gate_30d.csv")
    parser.add_argument("--output-md", type=Path, default=OUT_DIR / "strict_fill_replay_train_gate_30d.md")
    args = parser.parse_args()

    trades = load_replay_trades(args.source, args.fills)
    grouped: dict[tuple[str, str], list[ReplayTrade]] = defaultdict(list)
    for trade in trades:
        grouped[(trade.symbol, trade.day)].append(trade)

    output_rows: list[dict[str, str]] = []
    fee_bps_roundtrip = 6.0

    for (symbol, day), bucket in grouped.items():
        query_times = sorted({t.entry_ts_ms for t in bucket} | {t.exit_ts_ms for t in bucket})
        bybit_map = build_bybit_snapshot_map(DATALAKE / "bybit" / symbol / f"{day}_orderbook.jsonl", query_times)
        binance_map = build_binance_depth_map(DATALAKE / "binance" / symbol / f"{day}_bookDepth.csv", query_times)

        for trade in bucket:
            leg_notional = trade.alloc_dollars / 2.0
            spread_positive = trade.entry_spread_bps > 0
            # If Binance is rich, short Binance / long Bybit. Otherwise the reverse.
            bybit_entry_side = "buy" if spread_positive else "sell"
            bybit_exit_side = "sell" if spread_positive else "buy"
            binance_entry_side = "sell" if spread_positive else "buy"
            binance_exit_side = "buy" if spread_positive else "sell"

            entry_bybit_book = bybit_map.get(trade.entry_ts_ms)
            exit_bybit_book = bybit_map.get(trade.exit_ts_ms)
            entry_binance = binance_map.get(trade.entry_ts_ms)
            exit_binance = binance_map.get(trade.exit_ts_ms)

            bybit_entry_ok, bybit_entry_slip, _ = simulate_bybit_fill(entry_bybit_book, bybit_entry_side, leg_notional) if entry_bybit_book else (False, math.nan, math.nan)
            bybit_exit_ok, bybit_exit_slip, _ = simulate_bybit_fill(exit_bybit_book, bybit_exit_side, leg_notional) if exit_bybit_book else (False, math.nan, math.nan)
            binance_entry_ok, binance_entry_slip = simulate_binance_fill(entry_binance, binance_entry_side, leg_notional) if entry_binance else (False, math.nan)
            binance_exit_ok, binance_exit_slip = simulate_binance_fill(exit_binance, binance_exit_side, leg_notional) if exit_binance else (False, math.nan)

            fill_ok = bybit_entry_ok and bybit_exit_ok and binance_entry_ok and binance_exit_ok
            total_exec_slip = (
                bybit_entry_slip + bybit_exit_slip + binance_entry_slip + binance_exit_slip
                if fill_ok
                else math.nan
            )
            strict_net_bps = trade.gross_pnl_bps - fee_bps_roundtrip - total_exec_slip if fill_ok else math.nan
            strict_pnl_dollars = trade.alloc_dollars * (strict_net_bps / 10000.0) if fill_ok else math.nan
            output_rows.append(
                {
                    "symbol": trade.symbol,
                    "day": trade.day,
                    "entry_ts_ms": str(trade.entry_ts_ms),
                    "exit_ts_ms": str(trade.exit_ts_ms),
                    "alloc_dollars": f"{trade.alloc_dollars:.2f}",
                    "gross_pnl_bps": f"{trade.gross_pnl_bps:.6f}",
                    "modeled_net_bps": f"{trade.modeled_net_bps:.6f}",
                    "modeled_pnl_dollars": f"{trade.modeled_pnl_dollars:.6f}",
                    "fill_ok": "1" if fill_ok else "0",
                    "bybit_entry_slip_bps": f"{bybit_entry_slip:.6f}" if bybit_entry_ok else "",
                    "bybit_exit_slip_bps": f"{bybit_exit_slip:.6f}" if bybit_exit_ok else "",
                    "binance_entry_slip_bps": f"{binance_entry_slip:.6f}" if binance_entry_ok else "",
                    "binance_exit_slip_bps": f"{binance_exit_slip:.6f}" if binance_exit_ok else "",
                    "strict_exec_slip_bps": f"{total_exec_slip:.6f}" if fill_ok else "",
                    "strict_net_bps": f"{strict_net_bps:.6f}" if fill_ok else "",
                    "strict_pnl_dollars": f"{strict_pnl_dollars:.6f}" if fill_ok else "",
                }
            )

    fieldnames = list(output_rows[0].keys()) if output_rows else []
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    ok_rows = [row for row in output_rows if row["fill_ok"] == "1"]
    fill_rate = len(ok_rows) / len(output_rows) if output_rows else math.nan
    strict_nets = [float(row["strict_net_bps"]) for row in ok_rows]
    modeled_nets = [float(row["modeled_net_bps"]) for row in ok_rows]
    strict_pnls = [float(row["strict_pnl_dollars"]) for row in ok_rows]
    modeled_pnls = [float(row["modeled_pnl_dollars"]) for row in ok_rows]
    wins = sum(1 for row in ok_rows if float(row["strict_net_bps"]) > 0)

    lines = [
        "# Strict Fill Replay",
        "",
        "- Variant: current best microstructure-gated 30d sleeve (accepted fills only)",
        "- Bybit fills: exact simulation against archived L2 order book snapshots",
        "- Binance fills: approximate simulation against 1% cumulative `bookDepth` snapshots",
        "- Each leg uses half of the trade allocation as quote notional",
        "",
        "## Coverage",
        "",
        f"- Replayed fills: {len(output_rows)}",
        f"- Fully fillable under this strict replay: {len(ok_rows)}",
        f"- Fill success rate: {fill_rate:.2%}",
        "",
        "## PnL Comparison On Fillable Trades",
        "",
        f"- Modeled avg net: {(sum(modeled_nets) / len(modeled_nets)) if modeled_nets else math.nan:.4f} bps",
        f"- Strict-fill avg net: {(sum(strict_nets) / len(strict_nets)) if strict_nets else math.nan:.4f} bps",
        f"- Modeled total PnL: ${(sum(modeled_pnls)) if modeled_pnls else math.nan:.2f}",
        f"- Strict-fill total PnL: ${(sum(strict_pnls)) if strict_pnls else math.nan:.2f}",
        f"- Strict-fill win rate: {(wins / len(ok_rows)) if ok_rows else math.nan:.2%}",
        "",
        "## Strict Execution Cost",
        "",
        f"- Avg strict execution slippage: {(sum(float(r['strict_exec_slip_bps']) for r in ok_rows) / len(ok_rows)) if ok_rows else math.nan:.4f} bps",
        f"- Median strict execution slippage: {sorted(float(r['strict_exec_slip_bps']) for r in ok_rows)[len(ok_rows)//2] if ok_rows else math.nan:.4f} bps",
        "",
        "## Notes",
        "",
        "- This is stricter than the generic modeled slippage because it requires both legs to be fillable from archived depth snapshots.",
        "- Bybit is modeled more faithfully than Binance because only Bybit has true L2 snapshots here.",
        "- Binance remains an approximation using 1% cumulative depth buckets, so this is not yet a perfect venue-accurate simulator.",
    ]
    args.output_md.write_text("\n".join(lines))

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
