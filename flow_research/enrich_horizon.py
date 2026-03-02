#!/usr/bin/env python3
"""
Stage-2b: Extended horizon analysis + MAE/MFE curves.

For each Stage-2 event, compute:
  - Returns at 30s, 60s, 120s, 300s (direction-signed)
  - MAE (Max Adverse Excursion): worst drawdown against direction in [t0, t0+H]
  - MFE (Max Favorable Excursion): best move in direction in [t0, t0+H]
  - Retrace ratio: does price return to t0 level before continuing?
  - Time to MFE: when does the best move occur?

Uses OB mid-price sampled at ~100ms from raw orderbook data.

Usage:
  python enrich_horizon.py DOGEUSDT [--workers N]
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"

# Max forward horizon (seconds)
MAX_HORIZON_S = 300.0

# OB levels for depth (match Stage-1)
OB_LEVELS_K = 10

# Return horizons to compute
HORIZONS = [5, 15, 30, 60, 120, 300]

# MAE/MFE sampling: use raw OB timestamps (already ~100ms)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ob_arrays(filepath: Path, k: int = OB_LEVELS_K):
    """Load orderbook into arrays: (ts, depth, spread, mid)."""
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

            best_bid = float(bids[0][0]) if bids else 0.0
            best_ask = float(asks[0][0]) if asks else 0.0
            spread = max(best_ask - best_bid, 0.0)
            mid = (best_ask + best_bid) / 2.0 if best_bid > 0 else best_ask

            ts_list.append(ts)
            depth_list.append(bid_notional + ask_notional)
            spread_list.append(spread)
            mid_list.append(mid)

            if (i + 1) % 200000 == 0:
                print(f"    OB {i+1:,}...", flush=True)

    return (
        np.array(ts_list, dtype=np.float64),
        np.array(depth_list, dtype=np.float64),
        np.array(spread_list, dtype=np.float64),
        np.array(mid_list, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Core: compute path metrics for one event
# ---------------------------------------------------------------------------

def compute_event_metrics(ob_ts, ob_depth, ob_spread, ob_mid, t0, direction_sign):
    """
    For event at t0 with given direction, compute:
    - returns at each horizon
    - MAE/MFE at each horizon
    - retrace metrics
    - time_to_mfe
    """
    # Get mid at t0 — must have valid spread (not a blown book snapshot)
    idx0 = np.searchsorted(ob_ts, t0, side="right") - 1
    if idx0 < 0:
        return None
    mid0 = ob_mid[idx0]
    spread0 = ob_spread[idx0]
    if mid0 <= 0 or spread0 <= 0:
        return None

    # Extract OB path from t0 to t0 + MAX_HORIZON
    t_end = t0 + MAX_HORIZON_S
    i_start = np.searchsorted(ob_ts, t0, side="left")
    i_end = np.searchsorted(ob_ts, t_end, side="right")

    if i_start >= i_end:
        return None

    path_ts = ob_ts[i_start:i_end]
    path_mid = ob_mid[i_start:i_end]
    path_depth = ob_depth[i_start:i_end]
    path_spread = ob_spread[i_start:i_end]

    # Filter out blown-book snapshots (spread=0 → mid is half the real price)
    valid = path_spread > 0
    if valid.sum() < 10:
        return None
    path_ts = path_ts[valid]
    path_mid = path_mid[valid]
    path_depth = path_depth[valid]
    path_spread = path_spread[valid]

    # Direction-signed returns: positive = price moved in flow direction
    path_ret_bps = direction_sign * (path_mid - mid0) / mid0 * 10000
    path_elapsed = path_ts - t0

    result = {"mid_t0": mid0}

    # Returns at fixed horizons
    for h in HORIZONS:
        h_idx = np.searchsorted(path_elapsed, h, side="right") - 1
        if h_idx >= 0 and h_idx < len(path_ret_bps):
            result[f"ret_{h}s"] = round(float(path_ret_bps[h_idx]), 2)
        else:
            result[f"ret_{h}s"] = None

    # MAE / MFE at each horizon
    for h in HORIZONS:
        mask = path_elapsed <= h
        if not mask.any():
            result[f"mae_{h}s"] = None
            result[f"mfe_{h}s"] = None
            continue

        rets_h = path_ret_bps[mask]
        # MFE = max favorable = max of direction-signed returns (positive = good)
        mfe = float(rets_h.max())
        # MAE = max adverse = min of direction-signed returns (negative = bad)
        mae = float(rets_h.min())
        result[f"mfe_{h}s"] = round(mfe, 2)
        result[f"mae_{h}s"] = round(mae, 2)

    # Time to MFE (within 300s) — when does the best move occur?
    full_mask = path_elapsed <= MAX_HORIZON_S
    if full_mask.any():
        rets_full = path_ret_bps[full_mask]
        elapsed_full = path_elapsed[full_mask]
        mfe_idx = np.argmax(rets_full)
        result["time_to_mfe_300s"] = round(float(elapsed_full[mfe_idx]), 2)
        result["mfe_300s_total"] = round(float(rets_full[mfe_idx]), 2)
    else:
        result["time_to_mfe_300s"] = None
        result["mfe_300s_total"] = None

    # RETRACE ANALYSIS: does price return to t0 (or worse) before continuing?
    # For each horizon, check:
    # 1) retrace_depth: most negative excursion (MAE) before MFE
    # 2) retrace_to_t0: does ret go <= 0 between t0 and time_of_MFE?
    for h in [60, 300]:
        mask_h = path_elapsed <= h
        if not mask_h.any():
            result[f"retrace_to_t0_{h}s"] = None
            result[f"retrace_depth_{h}s"] = None
            result[f"mfe_after_retrace_{h}s"] = None
            continue

        rets_h = path_ret_bps[mask_h]
        elapsed_h = path_elapsed[mask_h]

        # Find MFE position
        mfe_pos = np.argmax(rets_h)
        mfe_val = float(rets_h[mfe_pos])

        # Check retrace: does return go <= 0 before MFE?
        if mfe_pos > 0:
            pre_mfe_rets = rets_h[:mfe_pos]
            retrace_occurred = bool((pre_mfe_rets <= 0).any())
            retrace_depth_val = float(pre_mfe_rets.min()) if len(pre_mfe_rets) > 0 else 0.0
        else:
            retrace_occurred = False
            retrace_depth_val = 0.0

        result[f"retrace_to_t0_{h}s"] = retrace_occurred
        result[f"retrace_depth_{h}s"] = round(retrace_depth_val, 2)
        result[f"mfe_after_retrace_{h}s"] = round(mfe_val, 2) if retrace_occurred else None

    # Depth path: min depth in various windows
    for h in [30, 60, 300]:
        mask_h = path_elapsed <= h
        if mask_h.any():
            depth0 = float(path_depth[0]) if len(path_depth) > 0 else 1.0
            min_depth = float(path_depth[mask_h].min())
            result[f"depth_drop_{h}s"] = round(min_depth / (depth0 + 1e-9), 4)
        else:
            result[f"depth_drop_{h}s"] = None

    return result


# ---------------------------------------------------------------------------
# Process one day
# ---------------------------------------------------------------------------

def process_day(symbol: str, date_str: str, events_df: pd.DataFrame,
                next_date_str: str = None) -> list:
    """Enrich events for one day with horizon + MAE/MFE data."""
    t0_wall = time.monotonic()

    ob_path = DATA_DIR / symbol / f"{date_str}_orderbook.jsonl"
    if not ob_path.exists():
        print(f"  [{date_str}] SKIP: missing OB", flush=True)
        return []

    print(f"  [{date_str}] Loading OB...", flush=True)
    ob_ts, ob_depth, ob_spread, ob_mid = load_ob_arrays(ob_path)

    # Load next-day OB if needed (events near end of day need 300s forward)
    if next_date_str:
        next_ob = DATA_DIR / symbol / f"{next_date_str}_orderbook.jsonl"
        if next_ob.exists():
            print(f"  [{date_str}] Loading next-day OB for boundary...", flush=True)
            ob_ts_n, ob_depth_n, ob_spread_n, ob_mid_n = load_ob_arrays(next_ob)
            ob_ts = np.concatenate([ob_ts, ob_ts_n])
            ob_depth = np.concatenate([ob_depth, ob_depth_n])
            ob_spread = np.concatenate([ob_spread, ob_spread_n])
            ob_mid = np.concatenate([ob_mid, ob_mid_n])

    print(f"  [{date_str}] Processing {len(events_df)} events...", flush=True)

    results = []
    for _, ev in events_df.iterrows():
        t0 = ev["t0"]
        direction_sign = 1 if ev["direction"] == "BUY" else -1

        metrics = compute_event_metrics(ob_ts, ob_depth, ob_spread, ob_mid, t0, direction_sign)
        if metrics is None:
            results.append({"event_id": ev["event_id"], "horizon_ok": False})
            continue

        metrics["event_id"] = ev["event_id"]
        metrics["horizon_ok"] = True
        results.append(metrics)

    elapsed = time.monotonic() - t0_wall
    ok_count = sum(1 for r in results if r.get("horizon_ok"))
    print(f"  [{date_str}] DONE: {ok_count}/{len(results)} enriched, {elapsed:.1f}s", flush=True)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage-2b: Horizon + MAE/MFE enrichment")
    parser.add_argument("symbol", help="e.g. DOGEUSDT")
    parser.add_argument("--workers", "-w", type=int, default=0)
    args = parser.parse_args()

    symbol = args.symbol.upper()
    workers = args.workers or max(1, (os.cpu_count() or 1) // 2)

    # Load Stage-2 events
    events_path = OUTPUT_DIR / symbol / "events_stage2.parquet"
    if not events_path.exists():
        print(f"ERROR: {events_path} not found. Run enrich_events.py first.")
        sys.exit(1)

    df = pd.read_parquet(events_path)
    # Only process events with valid enrichment
    df = df[df["enrich_ok"] == True].copy()
    df["date"] = df["t0_iso"].str[:10]
    dates = sorted(df["date"].unique())

    print(f"Symbol:  {symbol}")
    print(f"Events:  {len(df):,} across {len(dates)} days")
    print(f"Workers: {workers}")
    print(f"Horizons: {HORIZONS}")
    print("=" * 60, flush=True)

    t_total = time.monotonic()

    # Date -> next date map
    date_next = {}
    for i, d in enumerate(dates):
        if i + 1 < len(dates):
            d_dt = datetime.strptime(d, "%Y-%m-%d")
            n_dt = datetime.strptime(dates[i + 1], "%Y-%m-%d")
            if (n_dt - d_dt).days == 1:
                date_next[d] = dates[i + 1]

    all_results = []
    total_days = len(dates)
    with ProcessPoolExecutor(max_workers=min(workers, total_days)) as executor:
        futures = {}
        for d in dates:
            ev_day = df[df["date"] == d].copy()
            # Need next-day data if any event is within MAX_HORIZON of midnight
            need_next = False
            if len(ev_day) > 0:
                last_t0 = ev_day["t0"].max()
                day_end_ts = (datetime.strptime(d, "%Y-%m-%d") + timedelta(days=1)).timestamp()
                if last_t0 + MAX_HORIZON_S > day_end_ts:
                    need_next = True

            next_d = date_next.get(d) if need_next else None
            futures[executor.submit(process_day, symbol, d, ev_day, next_d)] = d

        done_count = 0
        for future in as_completed(futures):
            d = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as exc:
                print(f"  [{d}] ERROR: {exc}", flush=True)
            done_count += 1
            elapsed = time.monotonic() - t_total
            rate = elapsed / done_count
            eta = rate * (total_days - done_count)
            print(f"  >> Progress: {done_count}/{total_days} days  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    total_elapsed = time.monotonic() - t_total

    # Merge into events
    horizon_df = pd.DataFrame(all_results)
    merged = df.merge(horizon_df, on="event_id", how="left")

    ok = merged["horizon_ok"].fillna(False).sum()
    print(f"\nHorizon-enriched: {ok:,} / {len(merged):,} events in {total_elapsed:.0f}s")

    # Save
    out_path = OUTPUT_DIR / symbol / "events_stage2b.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")

    # --- Analysis ---
    ok_df = merged[merged["horizon_ok"] == True].copy()

    # Filter clean data (exclude blown-spread artifacts)
    clean = ok_df[ok_df["spread_at_t0"] > 0].copy()
    print(f"\nClean events (spread>0): {len(clean):,}")

    # VacuumScore
    clean["vs"] = np.clip(1 - clean["depth_drop_5s"], 0, 1) * clean["continuation_5s"]

    print(f"\n{'='*60}")
    print(f"HORIZON EXPANSION RESULTS")
    print(f"{'='*60}")

    # Returns at each horizon
    print(f"\n--- ALL EVENTS ({len(clean):,}) ---")
    for h in HORIZONS:
        col = f"ret_{h}s"
        vals = clean[col].dropna()
        print(f"  {col:12s}: med={vals.median():+6.1f}bp  mean={vals.mean():+6.1f}bp  "
              f"WR={(vals>0).mean():.1%}  p25={vals.quantile(0.25):+6.1f}  p75={vals.quantile(0.75):+6.1f}")

    # TopDecile VS
    p90 = clean["vs"].quantile(0.9)
    top_vs = clean[clean["vs"] >= p90]
    print(f"\n--- TOP DECILE VS (n={len(top_vs)}, VS>={p90:.3f}) ---")
    for h in HORIZONS:
        col = f"ret_{h}s"
        vals = top_vs[col].dropna()
        print(f"  {col:12s}: med={vals.median():+6.1f}bp  mean={vals.mean():+6.1f}bp  "
              f"WR={(vals>0).mean():.1%}  p25={vals.quantile(0.25):+6.1f}  p75={vals.quantile(0.75):+6.1f}")

    # HighVS + HighFI
    high_vs_fi = clean[(clean["vs"] >= 0.3) & (clean["flow_impact"] >= 3.0)]
    print(f"\n--- HIGH VS + HIGH FI (n={len(high_vs_fi)}) ---")
    for h in HORIZONS:
        col = f"ret_{h}s"
        vals = high_vs_fi[col].dropna()
        if len(vals) > 0:
            print(f"  {col:12s}: med={vals.median():+6.1f}bp  mean={vals.mean():+6.1f}bp  "
                  f"WR={(vals>0).mean():.1%}  p25={vals.quantile(0.25):+6.1f}  p75={vals.quantile(0.75):+6.1f}")

    # MAE / MFE analysis
    print(f"\n--- MAE/MFE ANALYSIS (Top Decile VS, n={len(top_vs)}) ---")
    for h in HORIZONS:
        mae_col = f"mae_{h}s"
        mfe_col = f"mfe_{h}s"
        mae = top_vs[mae_col].dropna()
        mfe = top_vs[mfe_col].dropna()
        if len(mae) > 0:
            print(f"  {h:4d}s: MAE med={mae.median():+6.1f}bp  MFE med={mfe.median():+6.1f}bp  "
                  f"MFE/|MAE|={mfe.median()/(-mae.median()+0.01):.2f}  "
                  f"MAE_p95={mae.quantile(0.05):+6.1f}bp")

    # RETRACE ANALYSIS — the key question
    print(f"\n--- RETRACE ANALYSIS (does price return to t0 before continuation?) ---")
    for h in [60, 300]:
        retrace_col = f"retrace_to_t0_{h}s"
        depth_col = f"retrace_depth_{h}s"
        mfe_after_col = f"mfe_after_retrace_{h}s"

        # Top decile VS
        sub = top_vs.dropna(subset=[retrace_col])
        if len(sub) == 0:
            continue
        retrace_pct = sub[retrace_col].mean()
        retracers = sub[sub[retrace_col] == True]
        non_retracers = sub[sub[retrace_col] == False]

        print(f"\n  Horizon {h}s (TopDecile VS, n={len(sub)}):")
        print(f"    Retrace to t0: {retrace_pct:.1%} of events")
        if len(retracers) > 0:
            rd = retracers[depth_col]
            mfe_ar = retracers[mfe_after_col].dropna()
            ret_h = retracers[f"ret_{h}s"].dropna()
            print(f"    Retracers ({len(retracers)}): "
                  f"retrace_depth={rd.median():+.1f}bp  mfe_after={mfe_ar.median():+.1f}bp  "
                  f"final_ret={ret_h.median():+.1f}bp  WR={(ret_h>0).mean():.1%}")
        if len(non_retracers) > 0:
            ret_h = non_retracers[f"ret_{h}s"].dropna()
            print(f"    Non-retracers ({len(non_retracers)}): "
                  f"final_ret={ret_h.median():+.1f}bp  WR={(ret_h>0).mean():.1%}")

    # Time to MFE distribution
    print(f"\n--- TIME TO MFE (300s window, TopDecile VS) ---")
    ttm = top_vs["time_to_mfe_300s"].dropna()
    if len(ttm) > 0:
        bins = [0, 5, 15, 30, 60, 120, 300]
        for i in range(len(bins)-1):
            mask = (ttm >= bins[i]) & (ttm < bins[i+1])
            print(f"  {bins[i]:3d}-{bins[i+1]:3d}s: {mask.sum():4d} ({mask.mean():.1%})")


if __name__ == "__main__":
    main()
