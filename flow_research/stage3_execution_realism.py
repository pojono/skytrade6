#!/usr/bin/env python3
"""
Stage-3: Execution Realism Analysis

Addresses three critical flaws in Stage-2b results:

1. LOOKAHEAD BIAS: VacuumScore uses 5s of future data (depth_drop_5s, continuation_5s).
   Fix: decision at t0+5s, entry/metrics measured from t0+5s (causal entry).

2. QUEUE REALISM: "price touched level" ≠ "limit order filled".
   Fix: measure executed volume through the entry level after the limit is placed.
   fill = True only if volume_through_level > visible_depth (queue cleared).

3. FAILURE PROFILING: characterize the ~40% of events where continuation fails.
   What distinguishes tail losses from winners?

Usage:
  python stage3_execution_realism.py DOGEUSDT [--workers 8]
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

OB_LEVELS_K = 10
DECISION_DELAY_S = 5.0      # Wait 5s after t0 to confirm VS, then decide
MAX_HORIZON_S = 300.0        # Track 300s after ENTRY (not t0)
HORIZONS = [5, 15, 30, 60, 120, 300]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ob_arrays(filepath: Path, k: int = OB_LEVELS_K):
    ts_list, depth_list, spread_list, mid_list = [], [], [], []
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            ts = obj["ts"] / 1000.0
            data = obj["data"]
            bids = data["b"][:k]
            asks = data["a"][:k]
            bid_n = sum(float(b[1]) * float(b[0]) for b in bids)
            ask_n = sum(float(a[1]) * float(a[0]) for a in asks)
            best_bid = float(bids[0][0]) if bids else 0.0
            best_ask = float(asks[0][0]) if asks else 0.0
            spread = max(best_ask - best_bid, 0.0)
            mid = (best_ask + best_bid) / 2.0 if best_bid > 0 else best_ask
            ts_list.append(ts)
            depth_list.append(bid_n + ask_n)
            spread_list.append(spread)
            mid_list.append(mid)
            if (i + 1) % 200000 == 0:
                print(f"    OB {i+1:,}...", flush=True)
    return (np.array(ts_list, dtype=np.float64),
            np.array(depth_list, dtype=np.float64),
            np.array(spread_list, dtype=np.float64),
            np.array(mid_list, dtype=np.float64))


def load_trades_arrays(filepath: Path):
    ts_l, side_l, not_l, price_l = [], [], [], []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            ts_l.append(float(row["timestamp"]))
            side_l.append(1 if row["side"] == "Buy" else -1)
            not_l.append(float(row["foreignNotional"]))
            price_l.append(float(row["price"]))
            if (i + 1) % 500000 == 0:
                print(f"    Trades {i+1:,}...", flush=True)
    return (np.array(ts_l, dtype=np.float64),
            np.array(side_l, dtype=np.int8),
            np.array(not_l, dtype=np.float64),
            np.array(price_l, dtype=np.float64))


# ---------------------------------------------------------------------------
# Core: compute causal metrics for one event
# ---------------------------------------------------------------------------

def compute_causal_metrics(
    ob_ts, ob_depth, ob_spread, ob_mid,
    tr_ts, tr_side, tr_not, tr_price,
    t0, direction_sign, vs_at_t0
):
    """
    Causal entry: decision at t_entry = t0 + DECISION_DELAY_S.
    All forward metrics measured from t_entry.
    """
    t_entry = t0 + DECISION_DELAY_S

    # Get OB state at t_entry
    idx_entry = np.searchsorted(ob_ts, t_entry, side="right") - 1
    if idx_entry < 0:
        return None
    mid_entry = ob_mid[idx_entry]
    spread_entry = ob_spread[idx_entry]
    depth_entry = ob_depth[idx_entry]
    if mid_entry <= 0 or spread_entry <= 0:
        return None

    # Get mid at t0 for comparison
    idx_t0 = np.searchsorted(ob_ts, t0, side="right") - 1
    mid_t0 = ob_mid[idx_t0] if idx_t0 >= 0 else mid_entry

    # How much has price already moved from t0 to t_entry? (direction-signed)
    slippage_from_t0 = direction_sign * (mid_entry - mid_t0) / (mid_t0 + 1e-15) * 10000

    # --- OB path from t_entry to t_entry + MAX_HORIZON ---
    t_end = t_entry + MAX_HORIZON_S
    i_start = np.searchsorted(ob_ts, t_entry, side="left")
    i_end = np.searchsorted(ob_ts, t_end, side="right")
    if i_start >= i_end:
        return None

    path_ts = ob_ts[i_start:i_end]
    path_mid = ob_mid[i_start:i_end]
    path_depth = ob_depth[i_start:i_end]
    path_spread = ob_spread[i_start:i_end]

    # Filter blown-spread
    valid = path_spread > 0
    if valid.sum() < 10:
        return None
    path_ts = path_ts[valid]
    path_mid = path_mid[valid]
    path_depth = path_depth[valid]
    path_spread = path_spread[valid]

    path_ret = direction_sign * (path_mid - mid_entry) / mid_entry * 10000
    path_elapsed = path_ts - t_entry

    result = {
        "mid_t0": round(mid_t0, 8),
        "mid_entry": round(mid_entry, 8),
        "spread_entry": round(spread_entry, 8),
        "depth_entry": round(depth_entry, 2),
        "slippage_from_t0_bps": round(slippage_from_t0, 2),
        "vs_at_t0": round(vs_at_t0, 4),
    }

    # Returns from entry at each horizon
    for h in HORIZONS:
        h_idx = np.searchsorted(path_elapsed, h, side="right") - 1
        if 0 <= h_idx < len(path_ret):
            result[f"ret_{h}s"] = round(float(path_ret[h_idx]), 2)
        else:
            result[f"ret_{h}s"] = None

    # MAE / MFE from entry
    for h in HORIZONS:
        mask = path_elapsed <= h
        if not mask.any():
            result[f"mae_{h}s"] = None
            result[f"mfe_{h}s"] = None
            continue
        rets_h = path_ret[mask]
        result[f"mfe_{h}s"] = round(float(rets_h.max()), 2)
        result[f"mae_{h}s"] = round(float(rets_h.min()), 2)

    # Retrace analysis from ENTRY point
    full_mask = path_elapsed <= MAX_HORIZON_S
    rets_full = path_ret[full_mask]
    elapsed_full = path_elapsed[full_mask]

    if len(rets_full) > 0:
        mfe_idx = np.argmax(rets_full)
        result["mfe_total"] = round(float(rets_full[mfe_idx]), 2)
        result["time_to_mfe"] = round(float(elapsed_full[mfe_idx]), 2)
        result["mae_total"] = round(float(rets_full.min()), 2)

        # Retrace: does price go below entry before MFE?
        if mfe_idx > 0:
            pre_mfe = rets_full[:mfe_idx]
            result["retrace_depth"] = round(float(pre_mfe.min()), 2)
            result["retrace_to_entry"] = bool((pre_mfe < 0).any())
        else:
            result["retrace_depth"] = 0.0
            result["retrace_to_entry"] = False
    else:
        result["mfe_total"] = None
        result["time_to_mfe"] = None
        result["mae_total"] = None
        result["retrace_depth"] = None
        result["retrace_to_entry"] = None

    # --- QUEUE REALISM ---
    # If we place a limit order at mid_entry (direction side):
    # For BUY: limit buy at best_bid-ish. Fill requires sells through that level.
    # Simplification: measure volume of OPPOSITE-direction trades at prices
    # at or below mid_entry (for BUY) in the retrace window [t_entry, t_entry+60s]
    queue_window = 60.0
    tr_start = np.searchsorted(tr_ts, t_entry, side="left")
    tr_end_q = np.searchsorted(tr_ts, t_entry + queue_window, side="left")

    if tr_start < tr_end_q:
        window_sides = tr_side[tr_start:tr_end_q]
        window_nots = tr_not[tr_start:tr_end_q]
        window_prices = tr_price[tr_start:tr_end_q]

        # For a BUY limit order at mid_entry: need SELL trades at <= mid_entry
        # For a SELL limit order at mid_entry: need BUY trades at >= mid_entry
        if direction_sign == 1:  # BUY
            fill_mask = (window_sides == -1) & (window_prices <= mid_entry)
        else:  # SELL
            fill_mask = (window_sides == 1) & (window_prices >= mid_entry)

        vol_through = float(window_nots[fill_mask].sum())
        n_trades_through = int(fill_mask.sum())

        # Also: total opposite-side volume (regardless of price)
        opp_mask = window_sides == -direction_sign
        total_opp_vol = float(window_nots[opp_mask].sum())

        # Visible depth at entry (one side only, rough: half of top_depth)
        one_side_depth = depth_entry / 2.0

        result["queue_vol_through"] = round(vol_through, 2)
        result["queue_trades_through"] = n_trades_through
        result["queue_opp_vol_total"] = round(total_opp_vol, 2)
        result["queue_depth_one_side"] = round(one_side_depth, 2)
        # Fill confidence: vol_through / depth_at_level
        result["queue_fill_ratio"] = round(vol_through / (one_side_depth + 1e-9), 4)
        # Conservative fill: would $10K notional fill?
        result["queue_fill_10k"] = vol_through > 10000
    else:
        result["queue_vol_through"] = 0
        result["queue_trades_through"] = 0
        result["queue_opp_vol_total"] = 0
        result["queue_depth_one_side"] = 0
        result["queue_fill_ratio"] = 0
        result["queue_fill_10k"] = False

    # --- Continuation / failure label ---
    ret_300 = result.get("ret_300s")
    if ret_300 is not None:
        result["continuation"] = ret_300 > 0
        result["strong_continuation"] = ret_300 > 10
        result["failure"] = ret_300 < -10
    else:
        result["continuation"] = None
        result["strong_continuation"] = None
        result["failure"] = None

    return result


# ---------------------------------------------------------------------------
# Process one day
# ---------------------------------------------------------------------------

def process_day(symbol: str, date_str: str, events_df: pd.DataFrame,
                next_date_str: str = None) -> list:
    t0_wall = time.monotonic()

    ob_path = DATA_DIR / symbol / f"{date_str}_orderbook.jsonl"
    tr_path = DATA_DIR / symbol / f"{date_str}_trades.csv"
    if not ob_path.exists() or not tr_path.exists():
        print(f"  [{date_str}] SKIP: missing data", flush=True)
        return []

    print(f"  [{date_str}] Loading...", flush=True)
    ob_ts, ob_depth, ob_spread, ob_mid = load_ob_arrays(ob_path)
    tr_ts, tr_side, tr_not, tr_price = load_trades_arrays(tr_path)

    # Append next-day data if needed
    if next_date_str:
        next_ob = DATA_DIR / symbol / f"{next_date_str}_orderbook.jsonl"
        next_tr = DATA_DIR / symbol / f"{next_date_str}_trades.csv"
        if next_ob.exists() and next_tr.exists():
            print(f"  [{date_str}] + next day boundary...", flush=True)
            ob_ts_n, ob_depth_n, ob_spread_n, ob_mid_n = load_ob_arrays(next_ob)
            tr_ts_n, tr_side_n, tr_not_n, tr_price_n = load_trades_arrays(next_tr)
            ob_ts = np.concatenate([ob_ts, ob_ts_n])
            ob_depth = np.concatenate([ob_depth, ob_depth_n])
            ob_spread = np.concatenate([ob_spread, ob_spread_n])
            ob_mid = np.concatenate([ob_mid, ob_mid_n])
            tr_ts = np.concatenate([tr_ts, tr_ts_n])
            tr_side = np.concatenate([tr_side, tr_side_n])
            tr_not = np.concatenate([tr_not, tr_not_n])
            tr_price = np.concatenate([tr_price, tr_price_n])

    print(f"  [{date_str}] Processing {len(events_df)} events...", flush=True)

    results = []
    for _, ev in events_df.iterrows():
        t0 = ev["t0"]
        direction_sign = 1 if ev["direction"] == "BUY" else -1
        vs = ev.get("vs", 0)

        metrics = compute_causal_metrics(
            ob_ts, ob_depth, ob_spread, ob_mid,
            tr_ts, tr_side, tr_not, tr_price,
            t0, direction_sign, vs
        )
        if metrics is None:
            results.append({"event_id": ev["event_id"], "causal_ok": False})
            continue

        metrics["event_id"] = ev["event_id"]
        metrics["causal_ok"] = True
        results.append(metrics)

    elapsed = time.monotonic() - t0_wall
    ok_count = sum(1 for r in results if r.get("causal_ok"))
    print(f"  [{date_str}] DONE: {ok_count}/{len(results)}, {elapsed:.1f}s", flush=True)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage-3: Execution Realism")
    parser.add_argument("symbol", help="e.g. DOGEUSDT")
    parser.add_argument("--workers", "-w", type=int, default=0)
    args = parser.parse_args()

    symbol = args.symbol.upper()
    workers = args.workers or max(1, (os.cpu_count() or 1) // 2)

    # Load Stage-2 events (with VS already computed)
    events_path = OUTPUT_DIR / symbol / "events_stage2.parquet"
    if not events_path.exists():
        print(f"ERROR: {events_path} not found.")
        sys.exit(1)

    df = pd.read_parquet(events_path)
    df = df[df["enrich_ok"] == True].copy()

    # Compute VS (same as before)
    df["vs"] = np.clip(1 - df["depth_drop_5s"], 0, 1) * df["continuation_5s"]
    df["date"] = df["t0_iso"].str[:10]

    # Only process TopDecile VS (the segment we care about)
    p90 = df["vs"].quantile(0.9)
    top = df[df["vs"] >= p90].copy()
    dates = sorted(top["date"].unique())

    print(f"Symbol:  {symbol}")
    print(f"Total events: {len(df):,}, TopDecile VS (>={p90:.3f}): {len(top):,}")
    print(f"Days: {len(dates)}, Workers: {workers}")
    print(f"Decision delay: {DECISION_DELAY_S}s (causal entry at t0+5s)")
    print("=" * 60, flush=True)

    t_total = time.monotonic()

    # Date -> next date
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
            ev_day = top[top["date"] == d].copy()
            need_next = False
            if len(ev_day) > 0:
                last_t0 = ev_day["t0"].max()
                day_end = (datetime.strptime(d, "%Y-%m-%d") + timedelta(days=1)).timestamp()
                if last_t0 + DECISION_DELAY_S + MAX_HORIZON_S > day_end:
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
            print(f"  >> {done_count}/{total_days} days  elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                  flush=True)

    total_elapsed = time.monotonic() - t_total

    # Merge
    causal_df = pd.DataFrame(all_results)
    merged = top.merge(causal_df, on="event_id", how="left")
    ok_df = merged[merged["causal_ok"] == True].copy()

    print(f"\nCausal-enriched: {len(ok_df):,} / {len(merged):,} in {total_elapsed:.0f}s")

    # Save
    out_path = OUTPUT_DIR / symbol / "events_stage3.parquet"
    ok_df.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")

    # =====================================================================
    # ANALYSIS
    # =====================================================================
    print(f"\n{'='*60}")
    print("STAGE-3: EXECUTION REALISM ANALYSIS")
    print(f"{'='*60}")

    # --- 1. SLIPPAGE FROM t0 to t_entry ---
    slip = ok_df["slippage_from_t0_bps"]
    print(f"\n--- 1. SLIPPAGE: t0 → t_entry (5s delay) ---")
    print(f"  Direction-signed (positive = price already moved in your direction):")
    print(f"  median={slip.median():+.1f}bp  mean={slip.mean():+.1f}bp  "
          f"p25={slip.quantile(0.25):+.1f}  p75={slip.quantile(0.75):+.1f}")
    print(f"  Interpretation: by the time you confirm VS and decide, price has already")
    print(f"  moved {slip.median():+.1f}bp in the flow direction — you enter WORSE.")

    # --- 2. RETURNS FROM CAUSAL ENTRY ---
    print(f"\n--- 2. RETURNS FROM CAUSAL ENTRY (t0+5s, n={len(ok_df)}) ---")
    for h in HORIZONS:
        col = f"ret_{h}s"
        r = ok_df[col].dropna()
        if len(r) == 0: continue
        print(f"  ret_{h:3d}s: med={r.median():+6.1f}bp  mean={r.mean():+6.1f}bp  "
              f"WR={(r>0).mean():.1%}  p25/p75={r.quantile(0.25):+5.1f}/{r.quantile(0.75):+5.1f}")

    # --- 3. MAE/MFE FROM CAUSAL ENTRY ---
    print(f"\n--- 3. MAE/MFE FROM CAUSAL ENTRY ---")
    for h in HORIZONS:
        mae = ok_df[f"mae_{h}s"].dropna()
        mfe = ok_df[f"mfe_{h}s"].dropna()
        if len(mae) == 0: continue
        print(f"  {h:4d}s: MAE med={mae.median():+6.1f}bp  MFE med={mfe.median():+6.1f}bp  "
              f"MFE/|MAE|={mfe.median()/(-mae.median()+0.01):.2f}")

    # --- 4. QUEUE REALISM ---
    print(f"\n--- 4. QUEUE REALISM (60s window after entry) ---")
    qfr = ok_df["queue_fill_ratio"]
    qvt = ok_df["queue_vol_through"]
    qf10 = ok_df["queue_fill_10k"]
    print(f"  Vol through entry level: med=${qvt.median():,.0f}  mean=${qvt.mean():,.0f}")
    print(f"  Fill ratio (vol/depth): med={qfr.median():.2f}  mean={qfr.mean():.2f}")
    print(f"  Fill ratio > 1 (queue cleared): {(qfr > 1).mean():.1%}")
    print(f"  Fill ratio > 2 (comfortable): {(qfr > 2).mean():.1%}")
    print(f"  $10K fill possible: {qf10.mean():.1%}")

    # Queue fill vs return
    print(f"\n  Queue fill vs continuation:")
    for label, mask in [("queue_cleared (FR>1)", qfr > 1),
                        ("queue_not_cleared (FR<1)", qfr <= 1)]:
        sub = ok_df[mask]
        r = sub["ret_300s"].dropna()
        if len(r) < 5: continue
        print(f"    {label}: n={len(sub):4d}  ret300={r.median():+.1f}bp  WR={(r>0).mean():.1%}")

    # --- 5. RETRACE FROM ENTRY ---
    print(f"\n--- 5. RETRACE FROM CAUSAL ENTRY ---")
    rt = ok_df["retrace_to_entry"]
    rd = ok_df["retrace_depth"]
    rt_valid = rt.dropna()
    print(f"  Retrace to entry level: {rt_valid.mean():.1%}")
    print(f"  Retrace depth: med={rd.median():+.1f}bp  p25={rd.quantile(0.25):+.1f}")

    # --- 6. FAILURE PROFILING ---
    print(f"\n--- 6. FAILURE PROFILING ---")
    cont = ok_df[ok_df["continuation"] == True]
    fail = ok_df[ok_df["failure"] == True]
    neutral = ok_df[(ok_df["continuation"] == False) & (ok_df["failure"] == False)]
    print(f"  Continuation (ret300>0): {len(cont)} ({len(cont)/len(ok_df):.1%})")
    print(f"  Failure (ret300<-10):    {len(fail)} ({len(fail)/len(ok_df):.1%})")
    print(f"  Neutral:                 {len(neutral)}")

    if len(fail) > 0 and len(cont) > 0:
        print(f"\n  Comparing winners vs failures:")
        for col in ["flow_impact", "vs_at_t0", "slippage_from_t0_bps",
                     "depth_entry", "spread_entry", "queue_fill_ratio",
                     "agg_total", "imbalance", "same_side_share"]:
            if col not in ok_df.columns:
                continue
            c_med = cont[col].median()
            f_med = fail[col].median()
            print(f"    {col:25s}: winners={c_med:+10.2f}  failures={f_med:+10.2f}  "
                  f"diff={c_med-f_med:+8.2f}")

    # --- 7. NET EDGE WITH REALISTIC ASSUMPTIONS ---
    print(f"\n--- 7. NET EDGE (REALISTIC) ---")
    maker_fee = 4  # bps RT
    # Only count events where queue cleared (realistic fill)
    fillable = ok_df[ok_df["queue_fill_ratio"] > 1].copy()
    print(f"  Fillable events (queue cleared): {len(fillable)} ({len(fillable)/len(ok_df):.1%})")
    if len(fillable) > 0:
        for h in [60, 120, 300]:
            r = fillable[f"ret_{h}s"].dropna()
            if len(r) == 0: continue
            gross = r.median()
            net = gross - maker_fee
            wr_net = (r > maker_fee).mean()
            events_day = len(fillable) / 30
            daily_net = events_day * net
            print(f"  ret_{h:3d}s: gross={gross:+5.1f}bp  net={net:+5.1f}bp  "
                  f"WR_net={wr_net:.1%}  n/day={events_day:.1f}  daily={daily_net:+.0f}bp")


if __name__ == "__main__":
    main()
