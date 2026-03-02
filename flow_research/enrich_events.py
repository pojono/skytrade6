#!/usr/bin/env python3
"""
Stage-2: Enrich Stage-1 events with post-event dynamics.

For each event at t0, measure what happens in the market over [t0, t0+5s] and [t0, t0+15s]:
  - depth_min_5s:   min(TopDepth) in [t0, t0+5s] / TopDepth(t0)
  - spread_peak_5s: max(Spread) in [t0, t0+5s] / Spread(t0)
  - return_5s:      signed return (mid(t0+5s) - mid(t0)) / mid(t0)  * direction_sign
  - return_15s:     signed return at t0+15s
  - vwap_slippage:  VWAP of same-direction trades in [t0, t0+5s] vs mid(t0)
  - continuation:   share of same-direction notional in [t0, t0+5s]

Usage:
  python enrich_events.py DOGEUSDT
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"

# Post-event measurement horizons (seconds)
HORIZON_SHORT = 5.0
HORIZON_LONG = 15.0

# OB levels for depth measurement (match Stage-1)
OB_LEVELS_K = 10


# ---------------------------------------------------------------------------
# Data loading helpers (reused from detect_events.py)
# ---------------------------------------------------------------------------

def load_ob_arrays(filepath: Path, k: int = OB_LEVELS_K):
    """Load orderbook into parallel arrays for fast binary search.
    Returns (ts_array, depth_array, spread_array, mid_array).
    """
    ts_list = []
    depth_list = []
    spread_list = []
    mid_list = []

    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            ts = obj["ts"] / 1000.0
            data = obj["data"]
            bids = data["b"][:k]
            asks = data["a"][:k]

            bid_notional = sum(float(b[1]) * float(b[0]) for b in bids)
            ask_notional = sum(float(a[1]) * float(a[0]) for a in asks)
            top_depth = bid_notional + ask_notional

            best_bid = float(bids[0][0]) if bids else 0.0
            best_ask = float(asks[0][0]) if asks else 0.0
            spread = max(best_ask - best_bid, 0.0)
            mid = (best_ask + best_bid) / 2.0 if best_bid > 0 else best_ask

            ts_list.append(ts)
            depth_list.append(top_depth)
            spread_list.append(spread)
            mid_list.append(mid)

            if (i + 1) % 200000 == 0:
                print(f"    OB loaded {i+1:,}...", flush=True)

    return (
        np.array(ts_list, dtype=np.float64),
        np.array(depth_list, dtype=np.float64),
        np.array(spread_list, dtype=np.float64),
        np.array(mid_list, dtype=np.float64),
    )


def load_trades_arrays(filepath: Path):
    """Load trades into arrays: (ts, side, notional, price).
    side: +1=Buy, -1=Sell.
    """
    ts_list = []
    side_list = []
    not_list = []
    price_list = []

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            ts_list.append(float(row["timestamp"]))
            side_list.append(1 if row["side"] == "Buy" else -1)
            not_list.append(float(row["foreignNotional"]))
            price_list.append(float(row["price"]))

            if (i + 1) % 500000 == 0:
                print(f"    Trades loaded {i+1:,}...", flush=True)

    return (
        np.array(ts_list, dtype=np.float64),
        np.array(side_list, dtype=np.int8),
        np.array(not_list, dtype=np.float64),
        np.array(price_list, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# OB helpers using pre-loaded arrays
# ---------------------------------------------------------------------------

def ob_at(ob_ts, ob_depth, ob_spread, ob_mid, t):
    """Get OB metrics at time t (latest snapshot <= t)."""
    idx = np.searchsorted(ob_ts, t, side="right") - 1
    if idx < 0:
        return None, None, None
    return ob_depth[idx], ob_spread[idx], ob_mid[idx]


def ob_range_metrics(ob_ts, ob_depth, ob_spread, ob_mid, t_start, t_end):
    """Get min depth, max spread, and final mid in [t_start, t_end]."""
    i_start = np.searchsorted(ob_ts, t_start, side="left")
    i_end = np.searchsorted(ob_ts, t_end, side="right")

    if i_start >= i_end:
        # No snapshots in range — use boundary values
        d0, s0, m0 = ob_at(ob_ts, ob_depth, ob_spread, ob_mid, t_end)
        return d0, s0, m0

    sl = slice(i_start, i_end)
    min_depth = ob_depth[sl].min()
    max_spread = ob_spread[sl].max()

    # Final mid: last snapshot <= t_end
    final_idx = np.searchsorted(ob_ts, t_end, side="right") - 1
    final_mid = ob_mid[final_idx] if final_idx >= 0 else None

    return min_depth, max_spread, final_mid


# ---------------------------------------------------------------------------
# Trade helpers
# ---------------------------------------------------------------------------

def trades_in_range(tr_ts, tr_side, tr_not, tr_price, t_start, t_end):
    """Get trade arrays in [t_start, t_end)."""
    i_start = np.searchsorted(tr_ts, t_start, side="left")
    i_end = np.searchsorted(tr_ts, t_end, side="left")
    return tr_side[i_start:i_end], tr_not[i_start:i_end], tr_price[i_start:i_end]


# ---------------------------------------------------------------------------
# Process one day
# ---------------------------------------------------------------------------

def process_day(symbol: str, date_str: str, events_df: pd.DataFrame,
                next_date_str: str = None) -> list:
    """Enrich events for one day. Returns list of enrichment dicts."""
    t0_wall = time.monotonic()

    trades_path = DATA_DIR / symbol / f"{date_str}_trades.csv"
    ob_path = DATA_DIR / symbol / f"{date_str}_orderbook.jsonl"

    if not trades_path.exists() or not ob_path.exists():
        print(f"  [{date_str}] SKIP: missing data files", flush=True)
        return []

    print(f"  [{date_str}] Loading OB...", flush=True)
    ob_ts, ob_depth, ob_spread, ob_mid = load_ob_arrays(ob_path)

    print(f"  [{date_str}] Loading trades...", flush=True)
    tr_ts, tr_side, tr_not, tr_price = load_trades_arrays(trades_path)

    # If events near end of day need next-day data, load it too
    ob_ts_ext, ob_depth_ext, ob_spread_ext, ob_mid_ext = ob_ts, ob_depth, ob_spread, ob_mid
    tr_ts_ext, tr_side_ext, tr_not_ext, tr_price_ext = tr_ts, tr_side, tr_not, tr_price

    if next_date_str:
        next_trades = DATA_DIR / symbol / f"{next_date_str}_trades.csv"
        next_ob = DATA_DIR / symbol / f"{next_date_str}_orderbook.jsonl"
        if next_trades.exists() and next_ob.exists():
            print(f"  [{date_str}] Loading next-day boundary data...", flush=True)
            ob_ts_n, ob_depth_n, ob_spread_n, ob_mid_n = load_ob_arrays(next_ob)
            tr_ts_n, tr_side_n, tr_not_n, tr_price_n = load_trades_arrays(next_trades)

            # Concatenate
            ob_ts_ext = np.concatenate([ob_ts, ob_ts_n])
            ob_depth_ext = np.concatenate([ob_depth, ob_depth_n])
            ob_spread_ext = np.concatenate([ob_spread, ob_spread_n])
            ob_mid_ext = np.concatenate([ob_mid, ob_mid_n])
            tr_ts_ext = np.concatenate([tr_ts, tr_ts_n])
            tr_side_ext = np.concatenate([tr_side, tr_side_n])
            tr_not_ext = np.concatenate([tr_not, tr_not_n])
            tr_price_ext = np.concatenate([tr_price, tr_price_n])

    print(f"  [{date_str}] Enriching {len(events_df)} events...", flush=True)

    results = []
    for _, ev in events_df.iterrows():
        t0 = ev["t0"]
        direction_sign = 1 if ev["direction"] == "BUY" else -1

        # Depth and spread at t0
        d0, s0, m0 = ob_at(ob_ts_ext, ob_depth_ext, ob_spread_ext, ob_mid_ext, t0)
        if d0 is None or m0 is None or m0 == 0:
            results.append({"event_id": ev["event_id"], "enrich_ok": False})
            continue

        # --- 5s horizon ---
        min_depth_5, max_spread_5, mid_5 = ob_range_metrics(
            ob_ts_ext, ob_depth_ext, ob_spread_ext, ob_mid_ext, t0, t0 + HORIZON_SHORT
        )

        # --- 15s horizon ---
        _, _, mid_15 = ob_at(ob_ts_ext, ob_depth_ext, ob_spread_ext, ob_mid_ext,
                             t0 + HORIZON_LONG)

        # --- Trades in [t0, t0+5s] ---
        sides_5, nots_5, prices_5 = trades_in_range(
            tr_ts_ext, tr_side_ext, tr_not_ext, tr_price_ext, t0, t0 + HORIZON_SHORT
        )

        # VWAP of same-direction trades
        same_mask = sides_5 == direction_sign
        same_notional = nots_5[same_mask].sum()
        total_notional = nots_5.sum()

        if same_mask.any():
            vwap_same = np.average(prices_5[same_mask], weights=nots_5[same_mask])
        else:
            vwap_same = m0

        # Continuation: same-direction share
        continuation = same_notional / (total_notional + 1e-9)

        # Compute metrics
        eps = 1e-9
        depth_drop_5s = min_depth_5 / (d0 + eps) if min_depth_5 is not None else None
        spread_peak_5s = max_spread_5 / (s0 + eps) if (max_spread_5 is not None and s0 > eps) else 1.0

        return_5s = direction_sign * (mid_5 - m0) / (m0 + eps) if mid_5 is not None else None
        return_15s = direction_sign * (mid_15 - m0) / (m0 + eps) if mid_15 is not None else None

        vwap_slip = direction_sign * (vwap_same - m0) / (m0 + eps)

        # Trades in [t0, t0+15s] for extended metrics
        sides_15, nots_15, prices_15 = trades_in_range(
            tr_ts_ext, tr_side_ext, tr_not_ext, tr_price_ext, t0, t0 + HORIZON_LONG
        )
        total_notional_15 = nots_15.sum()
        same_mask_15 = sides_15 == direction_sign
        continuation_15 = nots_15[same_mask_15].sum() / (total_notional_15 + 1e-9)

        results.append({
            "event_id": ev["event_id"],
            "enrich_ok": True,
            "depth_at_t0": round(d0, 2),
            "spread_at_t0": round(s0, 8),
            "mid_at_t0": round(m0, 8),
            "depth_drop_5s": round(depth_drop_5s, 4) if depth_drop_5s is not None else None,
            "spread_peak_5s": round(spread_peak_5s, 4),
            "return_5s_bps": round(return_5s * 10000, 2) if return_5s is not None else None,
            "return_15s_bps": round(return_15s * 10000, 2) if return_15s is not None else None,
            "vwap_slip_bps": round(vwap_slip * 10000, 2),
            "continuation_5s": round(continuation, 4),
            "continuation_15s": round(continuation_15, 4),
            "same_notional_5s": round(same_notional, 2),
            "total_notional_5s": round(total_notional, 2),
            "n_trades_5s": len(sides_5),
            "n_trades_15s": len(sides_15),
        })

    elapsed = time.monotonic() - t0_wall
    ok_count = sum(1 for r in results if r.get("enrich_ok"))
    print(f"  [{date_str}] DONE: {ok_count}/{len(results)} enriched, {elapsed:.1f}s", flush=True)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def date_range(start: str, end: str):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    while s <= e:
        yield s.strftime("%Y-%m-%d")
        s += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(description="Stage-2: Enrich events with post-event dynamics")
    parser.add_argument("symbol", help="e.g. DOGEUSDT")
    parser.add_argument("--workers", "-w", type=int, default=0)
    args = parser.parse_args()

    symbol = args.symbol.upper()
    workers = args.workers or max(1, (os.cpu_count() or 1) // 2)  # conservative — 2 arrays per worker

    # Load events
    events_path = OUTPUT_DIR / symbol / "events_stage1.parquet"
    if not events_path.exists():
        print(f"ERROR: {events_path} not found. Run detect_events.py first.")
        sys.exit(1)

    df = pd.read_parquet(events_path)
    df["date"] = df["t0_iso"].str[:10]
    dates = sorted(df["date"].unique())

    print(f"Symbol:  {symbol}")
    print(f"Events:  {len(df):,} across {len(dates)} days")
    print(f"Workers: {workers}")
    print("=" * 60, flush=True)

    t_total = time.monotonic()
    all_results = []

    # Build date -> next_date map for boundary handling
    date_next = {}
    for i, d in enumerate(dates):
        if i + 1 < len(dates):
            # Check if next date is actually next calendar day
            d_dt = datetime.strptime(d, "%Y-%m-%d")
            n_dt = datetime.strptime(dates[i + 1], "%Y-%m-%d")
            if (n_dt - d_dt).days == 1:
                date_next[d] = dates[i + 1]

    # Process days — each day loads ~200MB, so limit parallelism
    with ProcessPoolExecutor(max_workers=min(workers, len(dates))) as executor:
        futures = {}
        for d in dates:
            ev_day = df[df["date"] == d].copy()
            # Only pass next-day if last event is within HORIZON_LONG of midnight
            need_next = False
            if len(ev_day) > 0:
                last_t0 = ev_day["t0"].max()
                # Check if last event's 15s window crosses into next day
                day_end_ts = (datetime.strptime(d, "%Y-%m-%d") + timedelta(days=1)).timestamp()
                if last_t0 + HORIZON_LONG > day_end_ts:
                    need_next = True

            next_d = date_next.get(d) if need_next else None
            futures[executor.submit(process_day, symbol, d, ev_day, next_d)] = d

        for future in as_completed(futures):
            d = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as exc:
                print(f"  [{d}] ERROR: {exc}", flush=True)

    total_elapsed = time.monotonic() - t_total

    # Merge enrichment into events
    enrich_df = pd.DataFrame(all_results)
    merged = df.merge(enrich_df, on="event_id", how="left")

    ok = merged["enrich_ok"].fillna(False).sum()
    print(f"\nEnriched: {ok:,} / {len(merged):,} events")

    # Save
    out_path = OUTPUT_DIR / symbol / "events_stage2.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")

    csv_path = OUTPUT_DIR / symbol / "events_stage2.csv"
    merged.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Quick summary
    ok_df = merged[merged["enrich_ok"] == True]
    if len(ok_df) > 0:
        print(f"\n{'=' * 60}")
        print(f"ENRICHMENT SUMMARY ({len(ok_df):,} events, {total_elapsed:.0f}s)")
        print(f"{'=' * 60}")

        for col in ["depth_drop_5s", "spread_peak_5s", "return_5s_bps", "return_15s_bps",
                     "vwap_slip_bps", "continuation_5s", "continuation_15s"]:
            vals = ok_df[col].dropna()
            if len(vals) > 0:
                print(f"  {col:20s}: median={vals.median():8.2f}  p10={vals.quantile(0.1):8.2f}  p90={vals.quantile(0.9):8.2f}")

        # VacuumScore: combine depth_drop_5s and continuation
        ok_df = ok_df.copy()
        ok_df["vacuum_score"] = (1 - ok_df["depth_drop_5s"]) * ok_df["continuation_5s"]
        vs = ok_df["vacuum_score"].dropna()
        print(f"\n  {'vacuum_score':20s}: median={vs.median():8.4f}  p10={vs.quantile(0.1):8.4f}  p90={vs.quantile(0.9):8.4f}")

        # VacuumScore quintiles → return
        ok_df["vs_quintile"] = pd.qcut(ok_df["vacuum_score"], 5, labels=False, duplicates="drop")
        print(f"\n  VacuumScore quintile → return_5s_bps:")
        for q in sorted(ok_df["vs_quintile"].dropna().unique()):
            sub = ok_df[ok_df["vs_quintile"] == q]
            r5 = sub["return_5s_bps"].dropna()
            r15 = sub["return_15s_bps"].dropna()
            print(f"    Q{int(q)}: n={len(sub):4d}  ret5s={r5.mean():+6.1f}bps (med {r5.median():+5.1f})  ret15s={r15.mean():+6.1f}bps (med {r15.median():+5.1f})")


if __name__ == "__main__":
    main()
