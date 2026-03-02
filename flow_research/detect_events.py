#!/usr/bin/env python3
"""
Stage-1 Forced Flow Event Detector.

Finds moments when the market transitions into disequilibrium:
one-sided taker flow overwhelming visible orderbook liquidity.

Input:  flow_research/data/{SYMBOL}/{YYYY-MM-DD}_trades.csv
        flow_research/data/{SYMBOL}/{YYYY-MM-DD}_orderbook.jsonl

Output: flow_research/output/{SYMBOL}/events_stage1.parquet
        flow_research/output/{SYMBOL}/daily_summary.csv
        flow_research/output/{SYMBOL}/sanity.json

Usage:
  # Single day test
  python detect_events.py DOGEUSDT --start 2025-09-01 --end 2025-09-01

  # Full month, all cores
  python detect_events.py DOGEUSDT --start 2025-09-01 --end 2025-09-30
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Constants / Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"

# Rolling window for metrics (seconds)
WINDOW_S = 15.0

# Evaluation interval (seconds) — time-based, NOT trade-based
EVAL_INTERVAL_S = 0.5  # 500ms

# Orderbook levels to use for depth
OB_LEVELS_K = 10

# Max book age before marking stale (seconds)
MAX_BOOK_AGE_S = 1.0

# Trigger thresholds (Section 6)
TH_IMPACT = 1.0       # FlowImpact >= 1.0
TH_IMB = 0.7          # Imbalance >= 0.7
TH_SAMESIDE = 0.75    # SameSideShare >= 0.75
TH_MIN_NOTIONAL = 50_000  # minimum AggTotal in USD
# TH_TRADES: adaptive — use q80 of last 1h (see below)
TH_TRADES_FALLBACK = 20  # absolute fallback if not enough history

# Cooldown after trigger (seconds)
COOLDOWN_S = 30.0

# Rolling baseline for DepthDrop / SpreadRatio (seconds)
BASELINE_WINDOW_S = 900.0  # 15 minutes

# Adaptive trade count: q80 over this window (seconds)
ADAPTIVE_TRADES_WINDOW_S = 3600.0  # 1 hour


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    ts: float          # unix seconds
    side: int          # +1 = Buy, -1 = Sell
    notional: float    # USD notional
    price: float


@dataclass
class BookSnapshot:
    ts: float          # unix seconds
    bid_prices: list   # top K bid prices
    bid_sizes_n: list  # top K bid sizes in notional
    ask_prices: list   # top K ask prices
    ask_sizes_n: list  # top K ask sizes in notional
    spread: float
    mid: float
    top_depth: float   # sum of top K bid+ask notional


@dataclass
class Event:
    event_id: str
    t0: float
    t0_iso: str
    symbol: str
    direction: str
    price_at_t0: float
    agg_buy: float
    agg_sell: float
    agg_total: float
    net_agg: float
    flow_impact: float
    imbalance: float
    same_side_share: float
    agg_trades_count: int
    top_depth: float
    spread: float
    mid: float
    book_ts: float
    book_age_ms: float
    book_stale: bool
    window_start: float
    window_end: float
    depth_drop: float
    spread_ratio: float
    trades_threshold_used: float


# ---------------------------------------------------------------------------
# Streaming data readers
# ---------------------------------------------------------------------------

def iter_trades(filepath: Path):
    """Yield TradeRecord from CSV, streaming line by line."""
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row["timestamp"])
            side = 1 if row["side"] == "Buy" else -1
            price = float(row["price"])
            notional = float(row["foreignNotional"])
            yield TradeRecord(ts=ts, side=side, notional=notional, price=price)


def iter_orderbook(filepath: Path, k: int = OB_LEVELS_K):
    """Yield BookSnapshot from JSONL, streaming line by line."""
    with open(filepath, "r") as f:
        for line in f:
            obj = json.loads(line)
            ts_ms = obj["ts"]
            ts = ts_ms / 1000.0
            data = obj["data"]
            bids = data["b"][:k]
            asks = data["a"][:k]

            bid_prices = [float(b[0]) for b in bids]
            ask_prices = [float(a[0]) for a in asks]

            # Convert sizes to notional (size * price)
            bid_sizes_n = [float(b[1]) * float(b[0]) for b in bids]
            ask_sizes_n = [float(a[1]) * float(a[0]) for a in asks]

            best_bid = bid_prices[0] if bid_prices else 0.0
            best_ask = ask_prices[0] if ask_prices else 0.0
            spread = best_ask - best_bid
            mid = (best_ask + best_bid) / 2.0 if best_bid > 0 else best_ask
            top_depth = sum(bid_sizes_n) + sum(ask_sizes_n)

            yield BookSnapshot(
                ts=ts,
                bid_prices=bid_prices,
                bid_sizes_n=bid_sizes_n,
                ask_prices=ask_prices,
                ask_sizes_n=ask_sizes_n,
                spread=spread,
                mid=mid,
                top_depth=top_depth,
            )


# ---------------------------------------------------------------------------
# Core: process one day
# ---------------------------------------------------------------------------

def process_day(symbol: str, date_str: str) -> dict:
    """Process one day of data. Returns dict with events list + daily stats."""
    trades_path = DATA_DIR / symbol / f"{date_str}_trades.csv"
    ob_path = DATA_DIR / symbol / f"{date_str}_orderbook.jsonl"

    if not trades_path.exists():
        return {"date": date_str, "error": "trades file missing", "events": []}
    if not ob_path.exists():
        return {"date": date_str, "error": "orderbook file missing", "events": []}

    t0_wall = time.monotonic()

    # --- Pre-load orderbook into sorted array for binary search ---
    # ob200 at 100ms = ~864K snapshots/day. Each snapshot we store compactly.
    # Memory: ~864K * ~100 bytes = ~86MB per day — acceptable.
    ob_ts_list = []
    ob_data_list = []

    ob_count = 0
    for snap in iter_orderbook(ob_path, k=OB_LEVELS_K):
        ob_ts_list.append(snap.ts)
        ob_data_list.append(snap)
        ob_count += 1
        if ob_count % 200000 == 0:
            print(f"  [{date_str}] loaded {ob_count:,} OB snapshots...", flush=True)

    ob_ts_arr = np.array(ob_ts_list, dtype=np.float64)
    print(f"  [{date_str}] OB loaded: {ob_count:,} snapshots", flush=True)

    def get_book(t: float) -> Optional[BookSnapshot]:
        """Get latest book snapshot with ts <= t."""
        idx = np.searchsorted(ob_ts_arr, t, side="right") - 1
        if idx < 0:
            return None
        return ob_data_list[idx]

    # --- Stream trades, evaluate on time grid ---
    # Rolling window state
    trade_window = deque()  # (ts, side, notional, price)
    # Baseline windows for DepthDrop/SpreadRatio
    depth_baseline = deque()  # (ts, top_depth)
    spread_baseline = deque()  # (ts, spread)
    # Adaptive trade count
    trades_count_history = deque()  # (eval_ts, trades_count_in_window)

    events = []
    event_counter = 0
    last_trigger_ts = -1e18
    last_eval_ts = -1e18
    eval_count = 0
    trades_processed = 0
    book_stale_count = 0
    book_ok_count = 0

    # Distributions for sanity
    book_ages = []
    flow_impacts = []
    imbalances = []
    events_per_hour = {}

    # We need the first trade ts to set up eval grid
    trades_iter = iter_trades(trades_path)
    first_trade = None

    for trade in trades_iter:
        first_trade = trade
        break

    if first_trade is None:
        return {"date": date_str, "error": "no trades", "events": []}

    # Push first trade
    trade_window.append((first_trade.ts, first_trade.side, first_trade.notional, first_trade.price))
    trades_processed = 1

    # Set up evaluation grid starting from first trade + WINDOW_S
    grid_start = first_trade.ts + WINDOW_S
    next_eval = grid_start

    last_price = first_trade.price

    for trade in trades_iter:
        trades_processed += 1

        # Add to rolling window
        trade_window.append((trade.ts, trade.side, trade.notional, trade.price))
        last_price = trade.price

        # Evaluate at each grid point that this trade has passed
        while trade.ts >= next_eval:
            t_eval = next_eval
            next_eval += EVAL_INTERVAL_S
            eval_count += 1

            # Evict old trades from window
            window_start = t_eval - WINDOW_S
            while trade_window and trade_window[0][0] < window_start:
                trade_window.popleft()

            # Compute trade aggregates in window [t_eval - W, t_eval)
            agg_buy = 0.0
            agg_sell = 0.0
            n_trades = 0
            for tw_ts, tw_side, tw_not, tw_price in trade_window:
                if tw_ts >= t_eval:
                    break
                n_trades += 1
                if tw_side == 1:
                    agg_buy += tw_not
                else:
                    agg_sell += tw_not

            agg_total = agg_buy + agg_sell
            net_agg = agg_buy - agg_sell
            eps = 1e-9
            same_side_share = max(agg_buy, agg_sell) / (agg_total + eps)
            imbalance = abs(net_agg) / (agg_total + eps)
            direction = "BUY" if net_agg > 0 else "SELL"

            # Get book
            book = get_book(t_eval)
            if book is None:
                continue

            book_age_s = t_eval - book.ts
            book_age_ms = book_age_s * 1000.0
            is_stale = book_age_s > MAX_BOOK_AGE_S

            top_depth = book.top_depth
            spread = max(book.spread, 0.0)
            mid = book.mid

            flow_impact = agg_total / (top_depth + eps)

            # Track sanity stats (sample every 10th eval to save memory)
            if eval_count % 10 == 0:
                book_ages.append(book_age_ms)
                flow_impacts.append(flow_impact)
                imbalances.append(imbalance)

            if is_stale:
                book_stale_count += 1
            else:
                book_ok_count += 1

            # Update baselines
            depth_baseline.append((t_eval, top_depth))
            spread_baseline.append((t_eval, spread))
            while depth_baseline and depth_baseline[0][0] < t_eval - BASELINE_WINDOW_S:
                depth_baseline.popleft()
            while spread_baseline and spread_baseline[0][0] < t_eval - BASELINE_WINDOW_S:
                spread_baseline.popleft()

            # Adaptive trades threshold
            trades_count_history.append((t_eval, n_trades))
            while trades_count_history and trades_count_history[0][0] < t_eval - ADAPTIVE_TRADES_WINDOW_S:
                trades_count_history.popleft()

            # Compute adaptive threshold: q80 of trade counts over last 1h
            if len(trades_count_history) >= 20:
                tc_vals = [x[1] for x in trades_count_history]
                th_trades = np.percentile(tc_vals, 80)
            else:
                th_trades = TH_TRADES_FALLBACK

            # --- Trigger check (Section 6) ---
            in_cooldown = (t_eval - last_trigger_ts) < COOLDOWN_S
            if in_cooldown:
                # Update peak FlowImpact and price extreme only (7.2)
                if events and events[-1]["t0"] == last_trigger_ts:
                    ev = events[-1]
                    if flow_impact > ev.get("flow_impact_peak", ev["flow_impact"]):
                        ev["flow_impact_peak"] = round(flow_impact, 4)
                    p = mid if mid > 0 else last_price
                    if ev["direction"] == "BUY":
                        ev["price_extreme"] = round(max(ev.get("price_extreme", p), p), 6)
                    else:
                        ev["price_extreme"] = round(min(ev.get("price_extreme", p), p), 6)
                continue

            triggered = (
                flow_impact >= TH_IMPACT
                and imbalance >= TH_IMB
                and same_side_share >= TH_SAMESIDE
                and n_trades >= th_trades
                and agg_total >= TH_MIN_NOTIONAL
                and not is_stale
            )

            if not triggered:
                continue

            # --- Compute DepthDrop, SpreadRatio ---
            depth_vals = [x[1] for x in depth_baseline]
            spread_vals = [x[1] for x in spread_baseline]
            median_depth = float(np.median(depth_vals)) if depth_vals else top_depth
            median_spread = float(np.median(spread_vals)) if spread_vals else spread

            depth_drop = top_depth / (median_depth + eps)
            spread_ratio = spread / (median_spread + eps) if median_spread > eps else 1.0

            # --- Create event ---
            event_counter += 1
            eid = f"{symbol}_{date_str}_{event_counter:04d}"
            t0_iso = datetime.utcfromtimestamp(t_eval).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            price_at_t0 = mid if mid > 0 else last_price

            ev = {
                "event_id": eid,
                "t0": t_eval,
                "t0_iso": t0_iso,
                "symbol": symbol,
                "direction": direction,
                "price_at_t0": round(price_at_t0, 6),
                "agg_buy": round(agg_buy, 2),
                "agg_sell": round(agg_sell, 2),
                "agg_total": round(agg_total, 2),
                "net_agg": round(net_agg, 2),
                "flow_impact": round(flow_impact, 4),
                "imbalance": round(imbalance, 4),
                "same_side_share": round(same_side_share, 4),
                "agg_trades_count": n_trades,
                "top_depth": round(top_depth, 2),
                "spread": round(spread, 6),
                "mid": round(mid, 6),
                "book_ts": round(book.ts, 4),
                "book_age_ms": round(book_age_ms, 1),
                "book_stale": is_stale,
                "window_start": round(t_eval - WINDOW_S, 4),
                "window_end": round(t_eval, 4),
                "depth_drop": round(depth_drop, 4),
                "spread_ratio": round(spread_ratio, 4),
                "trades_threshold_used": round(th_trades, 1),
            }
            events.append(ev)
            last_trigger_ts = t_eval

            # Track events per hour
            hour = int((t_eval % 86400) // 3600)
            events_per_hour[hour] = events_per_hour.get(hour, 0) + 1

        # Progress every 250K trades
        if trades_processed % 250000 == 0:
            elapsed = time.monotonic() - t0_wall
            print(
                f"  [{date_str}] {trades_processed:,} trades, "
                f"{eval_count:,} evals, {len(events)} events, "
                f"{elapsed:.0f}s",
                flush=True,
            )

    elapsed = time.monotonic() - t0_wall

    # Sanity stats
    sanity = {
        "book_age_ms_median": round(float(np.median(book_ages)), 1) if book_ages else None,
        "book_age_ms_p95": round(float(np.percentile(book_ages, 95)), 1) if book_ages else None,
        "book_stale_share": round(book_stale_count / max(book_stale_count + book_ok_count, 1), 4),
        "flow_impact_median": round(float(np.median(flow_impacts)), 4) if flow_impacts else None,
        "flow_impact_p95": round(float(np.percentile(flow_impacts, 95)), 4) if flow_impacts else None,
        "imbalance_median": round(float(np.median(imbalances)), 4) if imbalances else None,
        "imbalance_p95": round(float(np.percentile(imbalances, 95)), 4) if imbalances else None,
        "events_per_hour": events_per_hour,
    }

    print(
        f"  [{date_str}] DONE: {trades_processed:,} trades, "
        f"{eval_count:,} evals, {len(events)} events, "
        f"{elapsed:.1f}s",
        flush=True,
    )

    return {
        "date": date_str,
        "events": events,
        "trades_processed": trades_processed,
        "eval_count": eval_count,
        "ob_snapshots": ob_count,
        "elapsed_s": round(elapsed, 1),
        "sanity": sanity,
    }


# ---------------------------------------------------------------------------
# Multi-day orchestrator
# ---------------------------------------------------------------------------

def date_range(start: str, end: str):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    while s <= e:
        yield s.strftime("%Y-%m-%d")
        s += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(description="Stage-1 Forced Flow Event Detector")
    parser.add_argument("symbol", help="e.g. DOGEUSDT")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--workers", "-w", type=int, default=0,
                        help="Parallel workers (default: all CPUs)")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    dates = list(date_range(args.start, args.end))
    workers = args.workers or os.cpu_count() or 1

    print(f"Symbol:    {symbol}")
    print(f"Dates:     {args.start} -> {args.end} ({len(dates)} days)")
    print(f"Workers:   {workers}")
    print(f"Window:    {WINDOW_S}s, Eval interval: {EVAL_INTERVAL_S}s")
    print(f"Thresholds: Impact>={TH_IMPACT}, Imb>={TH_IMB}, SameSide>={TH_SAMESIDE}")
    print(f"Cooldown:  {COOLDOWN_S}s")
    print("=" * 60, flush=True)

    t0_total = time.monotonic()
    all_events = []
    daily_summaries = []
    all_sanity = {}

    # Process days in parallel
    with ProcessPoolExecutor(max_workers=min(workers, len(dates))) as executor:
        futures = {
            executor.submit(process_day, symbol, d): d
            for d in dates
        }

        for future in as_completed(futures):
            d = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"  [{d}] ERROR: {exc}", flush=True)
                continue

            if "error" in result:
                print(f"  [{d}] SKIP: {result['error']}", flush=True)
                continue

            evts = result["events"]
            all_events.extend(evts)

            san = result.get("sanity", {})
            all_sanity[d] = san

            n_depth_drop = sum(1 for e in evts if e.get("depth_drop", 1.0) < 0.7)
            daily_summaries.append({
                "date": d,
                "symbol": symbol,
                "events_count": len(evts),
                "trades_processed": result["trades_processed"],
                "eval_count": result["eval_count"],
                "ob_snapshots": result["ob_snapshots"],
                "median_flow_impact": round(
                    float(np.median([e["flow_impact"] for e in evts])), 4
                ) if evts else None,
                "p95_flow_impact": round(
                    float(np.percentile([e["flow_impact"] for e in evts], 95)), 4
                ) if evts else None,
                "share_book_stale": san.get("book_stale_share"),
                "share_depthdrop_lt_07": round(
                    n_depth_drop / max(len(evts), 1), 4
                ) if evts else 0.0,
                "book_age_ms_median": san.get("book_age_ms_median"),
                "elapsed_s": result["elapsed_s"],
            })

    total_elapsed = time.monotonic() - t0_total

    # Sort events by t0
    all_events.sort(key=lambda e: e["t0"])

    # --- Save outputs ---
    out_dir = OUTPUT_DIR / symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    # Events parquet
    if all_events:
        df_events = pd.DataFrame(all_events)
        events_path = out_dir / "events_stage1.parquet"
        df_events.to_parquet(events_path, index=False)
        print(f"\nEvents saved: {events_path} ({len(df_events):,} events)")

        # Also CSV for quick inspection
        csv_path = out_dir / "events_stage1.csv"
        df_events.to_csv(csv_path, index=False)
        print(f"Events CSV:   {csv_path}")
    else:
        print("\nNo events found!")

    # Daily summary
    if daily_summaries:
        df_daily = pd.DataFrame(daily_summaries).sort_values("date")
        daily_path = out_dir / "daily_summary.csv"
        df_daily.to_csv(daily_path, index=False)
        print(f"Daily summary: {daily_path}")

    # Sanity JSON
    sanity_path = out_dir / "sanity.json"
    with open(sanity_path, "w") as f:
        json.dump(all_sanity, f, indent=2, default=str)
    print(f"Sanity data:  {sanity_path}")

    # --- Print summary ---
    print(f"\n{'=' * 60}")
    print(f"TOTAL: {len(all_events):,} events across {len(dates)} days in {total_elapsed:.1f}s")
    if daily_summaries:
        evts_per_day = [s["events_count"] for s in daily_summaries]
        print(f"Events/day: min={min(evts_per_day)}, max={max(evts_per_day)}, "
              f"mean={np.mean(evts_per_day):.1f}, median={np.median(evts_per_day):.0f}")
    if all_events:
        directions = pd.Series([e["direction"] for e in all_events])
        print(f"Direction split: {dict(directions.value_counts())}")
        fi = [e["flow_impact"] for e in all_events]
        print(f"FlowImpact: median={np.median(fi):.3f}, p95={np.percentile(fi, 95):.3f}")
        dd = [e["depth_drop"] for e in all_events]
        print(f"DepthDrop: median={np.median(dd):.3f}, <0.7 share={sum(1 for d in dd if d<0.7)/len(dd):.1%}")


if __name__ == "__main__":
    main()
