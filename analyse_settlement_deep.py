#!/usr/bin/env python3
"""
Deep Settlement Analysis — Full 60s trajectory, recovery, volume, OB, OI
=========================================================================
Answers:
  1. How does price behave over full 60s (not just first 5s)?
  2. What's the recovery/bounce pattern after initial drop?
  3. Can we predict post-settlement volume?
  4. Do orderbook changes predict post-settlement behavior?
  5. Does open interest matter?

Usage:
    python3 analyse_settlement_deep.py charts_settlement/*.jsonl
"""

import json
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd


def parse_jsonl(fp):
    """Parse JSONL file, return structured data streams."""
    trades = []      # (t_ms, price, qty, side, notional)
    ob1 = []         # (t_ms, bid_p, bid_q, ask_p, ask_q)
    ob200 = []       # (t_ms, bids_list, asks_list)
    tickers = []     # (t_ms, fr, oi, mark, index, vol24, turn24, price_change_24h)
    liquidations = []  # (t_ms, side, qty, price)

    with open(fp) as f:
        for line in f:
            try:
                msg = json.loads(line)
            except:
                continue

            t = msg.get("_t_ms", 0)
            topic = msg.get("topic", "")
            data = msg.get("data", {})

            if "publicTrade" in topic:
                for tr in (data if isinstance(data, list) else [data]):
                    p = float(tr.get("p", 0))
                    q = float(tr.get("v", 0))
                    s = tr.get("S", "")
                    trades.append((t, p, q, s, p * q))

            elif topic == "orderbook.1":
                b = data.get("b", [])
                a = data.get("a", [])
                if b and a:
                    ob1.append((t, float(b[0][0]), float(b[0][1]),
                                float(a[0][0]), float(a[0][1])))

            elif topic == "orderbook.200":
                bids = [(float(x[0]), float(x[1])) for x in data.get("b", [])]
                asks = [(float(x[0]), float(x[1])) for x in data.get("a", [])]
                if bids and asks:
                    ob200.append((t, bids, asks))

            elif "tickers" in topic:
                fr = float(data.get("fundingRate", 0))
                oi = float(data.get("openInterest", 0))
                mark = float(data.get("markPrice", 0))
                idx = float(data.get("indexPrice", 0))
                v24 = float(data.get("volume24h", 0))
                t24 = float(data.get("turnover24h", 0))
                pch = float(data.get("price24hPcnt", 0))
                tickers.append((t, fr, oi, mark, idx, v24, t24, pch))

            elif "Liquidation" in topic or "liquidation" in topic:
                ld = data if isinstance(data, dict) else {}
                side = ld.get("side", "")
                qty = float(ld.get("size", ld.get("qty", 0)))
                price = float(ld.get("price", 0))
                liquidations.append((t, side, qty, price))

    return trades, ob1, ob200, tickers, liquidations


def analyse_settlement(fp):
    """Full deep analysis of one settlement file."""
    trades, ob1, ob200, tickers, liquidations = parse_jsonl(fp)

    if not trades:
        return None

    stem = fp.stem
    parts = stem.split("_")
    symbol = parts[0]
    settle_label = "_".join(parts[1:])

    # Reference price = last trade before settlement
    pre_trades = [(t, p, q, s, n) for t, p, q, s, n in trades if t < 0]
    post_trades = [(t, p, q, s, n) for t, p, q, s, n in trades if t >= 0]

    if not pre_trades or not post_trades:
        return None

    ref_price = pre_trades[-1][1]
    bps = lambda p: (p / ref_price - 1) * 10000

    # FR
    pre_tickers = [tk for tk in tickers if tk[0] < 0]
    fr = pre_tickers[-1][1] * 10000 if pre_tickers else 0  # in bps

    feat = {
        "symbol": symbol,
        "settle_time": settle_label,
        "ref_price": ref_price,
        "fr_bps": fr,
        "fr_abs_bps": abs(fr),
    }

    # ════════════════════════════════════════════════════════════════
    # 1. FULL 60-SECOND PRICE TRAJECTORY
    # ════════════════════════════════════════════════════════════════
    time_windows = [100, 250, 500, 1000, 2000, 3000, 5000, 10000, 15000, 20000, 30000, 60000]

    for tw in time_windows:
        window_trades = [(t, p) for t, p, _, _, _ in post_trades if t <= tw]
        if window_trades:
            last_price = max(window_trades, key=lambda x: x[0])[1]
            min_price = min(p for _, p in window_trades)
            feat[f"price_{tw}ms_bps"] = bps(last_price)
            feat[f"worst_{tw}ms_bps"] = bps(min_price)
        else:
            feat[f"price_{tw}ms_bps"] = np.nan
            feat[f"worst_{tw}ms_bps"] = np.nan

    # Overall min
    all_post_bps = [bps(p) for _, p, _, _, _ in post_trades]
    feat["drop_min_bps"] = min(all_post_bps)
    feat["drop_final_bps"] = all_post_bps[-1]
    feat["max_post_ms"] = max(t for t, _, _, _, _ in post_trades)

    # Time to bottom
    min_bps = min(all_post_bps)
    min_idx = all_post_bps.index(min_bps)
    feat["time_to_bottom_ms"] = sorted(post_trades, key=lambda x: x[0])[min_idx][0]

    # ════════════════════════════════════════════════════════════════
    # 2. RECOVERY / BOUNCE ANALYSIS
    # ════════════════════════════════════════════════════════════════
    bottom_time = feat["time_to_bottom_ms"]
    bottom_price_bps = feat["drop_min_bps"]

    # Recovery after bottom
    recovery_trades = [(t, bps(p)) for t, p, _, _, _ in post_trades if t > bottom_time]
    if recovery_trades:
        # Recovery at various offsets from bottom
        for dt in [100, 250, 500, 1000, 2000, 5000, 10000, 30000]:
            target_time = bottom_time + dt
            candidates = [(t, bp) for t, bp in recovery_trades if t <= target_time]
            if candidates:
                feat[f"recovery_{dt}ms_bps"] = max(candidates, key=lambda x: x[0])[1] - bottom_price_bps
            else:
                feat[f"recovery_{dt}ms_bps"] = np.nan

        # Max recovery
        max_recovery_bps = max(bp for _, bp in recovery_trades) - bottom_price_bps
        feat["recovery_max_bps"] = max_recovery_bps
        feat["recovery_pct"] = max_recovery_bps / abs(bottom_price_bps) * 100 if bottom_price_bps != 0 else 0

        # Final recovery (last trade)
        feat["recovery_final_bps"] = recovery_trades[-1][1] - bottom_price_bps
        feat["recovery_final_pct"] = feat["recovery_final_bps"] / abs(bottom_price_bps) * 100 if bottom_price_bps != 0 else 0
    else:
        for dt in [100, 250, 500, 1000, 2000, 5000, 10000, 30000]:
            feat[f"recovery_{dt}ms_bps"] = np.nan
        feat["recovery_max_bps"] = feat["recovery_pct"] = np.nan
        feat["recovery_final_bps"] = feat["recovery_final_pct"] = np.nan

    # Does price fully recover? (return to ref_price)
    feat["full_recovery"] = 1 if feat.get("drop_final_bps", -999) > -5 else 0

    # ════════════════════════════════════════════════════════════════
    # 3. VOLUME ANALYSIS (by time window)
    # ════════════════════════════════════════════════════════════════
    pre_10s = [(t, p, q, s, n) for t, p, q, s, n in pre_trades if t >= -10000]

    feat["pre_vol_10s_usd"] = sum(n for _, _, _, _, n in pre_10s)

    for tw in [500, 1000, 2000, 5000, 10000, 30000, 60000]:
        window = [(t, p, q, s, n) for t, p, q, s, n in post_trades if t <= tw]
        total_vol = sum(n for _, _, _, _, n in window)
        sell_vol = sum(n for _, _, _, s, n in window if s == "Sell")
        buy_vol = sum(n for _, _, _, s, n in window if s == "Buy")
        n_trades = len(window)

        feat[f"vol_{tw}ms_usd"] = total_vol
        feat[f"sell_vol_{tw}ms_usd"] = sell_vol
        feat[f"buy_vol_{tw}ms_usd"] = buy_vol
        feat[f"sell_ratio_{tw}ms"] = sell_vol / total_vol if total_vol > 0 else 0.5
        feat[f"ntrades_{tw}ms"] = n_trades

    # Volume surge ratio (post/pre)
    pre_vol_rate = feat["pre_vol_10s_usd"] / 10.0 if feat["pre_vol_10s_usd"] > 0 else 1
    for tw in [1000, 5000, 10000]:
        post_vol_rate = feat[f"vol_{tw}ms_usd"] / (tw / 1000)
        feat[f"vol_surge_{tw}ms"] = post_vol_rate / pre_vol_rate

    # ════════════════════════════════════════════════════════════════
    # 4. ORDERBOOK CHANGES (pre vs post settlement)
    # ════════════════════════════════════════════════════════════════
    pre_ob200 = [o for o in ob200 if -10000 <= o[0] < 0]
    post_ob200 = [o for o in ob200 if 0 <= o[0] <= 60000]

    if pre_ob200:
        # Last pre-settlement OB snapshot
        _, pre_bids, pre_asks = pre_ob200[-1]
        pre_bid_depth = sum(p * q for p, q in pre_bids)
        pre_ask_depth = sum(p * q for p, q in pre_asks)
        feat["pre_bid_depth_usd"] = pre_bid_depth
        feat["pre_ask_depth_usd"] = pre_ask_depth
        feat["pre_total_depth_usd"] = pre_bid_depth + pre_ask_depth
        feat["pre_depth_imb"] = (pre_bid_depth - pre_ask_depth) / (pre_bid_depth + pre_ask_depth) if (pre_bid_depth + pre_ask_depth) > 0 else 0

        # Depth within X bps of mid
        mid = (pre_bids[0][0] + pre_asks[0][0]) / 2
        for bps_range in [20, 50, 100]:
            bid_in = sum(p * q for p, q in pre_bids if (mid - p) / mid * 10000 <= bps_range)
            ask_in = sum(p * q for p, q in pre_asks if (p - mid) / mid * 10000 <= bps_range)
            feat[f"pre_depth_{bps_range}bps_usd"] = bid_in + ask_in

        # Track OB changes over time after settlement
        for post_snap in post_ob200:
            t_snap, p_bids, p_asks = post_snap
            if t_snap > 60000:
                break

            snap_bid = sum(p * q for p, q in p_bids)
            snap_ask = sum(p * q for p, q in p_asks)

            # Store at key time points
            for tw in [1000, 5000, 10000, 30000]:
                if abs(t_snap - tw) < 2000 and f"post_bid_depth_{tw}ms" not in feat:
                    feat[f"post_bid_depth_{tw}ms"] = snap_bid
                    feat[f"post_ask_depth_{tw}ms"] = snap_ask
                    feat[f"post_depth_change_{tw}ms"] = (snap_bid + snap_ask) - (pre_bid_depth + pre_ask_depth)
                    feat[f"post_depth_change_pct_{tw}ms"] = feat[f"post_depth_change_{tw}ms"] / feat["pre_total_depth_usd"] * 100 if feat["pre_total_depth_usd"] > 0 else 0

        # OB.200 snapshots over time (depth trajectory)
        ob_trajectory = []
        for t_snap, p_bids, p_asks in sorted(ob200, key=lambda x: x[0]):
            if -10000 <= t_snap <= 60000:
                total = sum(p*q for p,q in p_bids) + sum(p*q for p,q in p_asks)
                imb = ((sum(p*q for p,q in p_bids) - sum(p*q for p,q in p_asks)) / total) if total > 0 else 0
                ob_trajectory.append((t_snap, total, imb))

        if len(ob_trajectory) > 5:
            # Did depth drain before settlement?
            pre_depths = [d for t, d, _ in ob_trajectory if -5000 <= t < 0]
            post_depths = [d for t, d, _ in ob_trajectory if 0 <= t <= 5000]
            if pre_depths and post_depths:
                feat["depth_drain_pre_pct"] = (pre_depths[-1] - pre_depths[0]) / pre_depths[0] * 100 if pre_depths[0] > 0 else 0
                feat["depth_drain_post_pct"] = (post_depths[-1] - post_depths[0]) / post_depths[0] * 100 if post_depths[0] > 0 else 0
    else:
        for k in ["pre_bid_depth_usd", "pre_ask_depth_usd", "pre_total_depth_usd", "pre_depth_imb",
                   "pre_depth_20bps_usd", "pre_depth_50bps_usd", "pre_depth_100bps_usd"]:
            feat[k] = np.nan

    # ════════════════════════════════════════════════════════════════
    # 5. OPEN INTEREST ANALYSIS
    # ════════════════════════════════════════════════════════════════
    pre_tks = [(t, fr_v, oi, mk, idx, v24, t24, pch) for t, fr_v, oi, mk, idx, v24, t24, pch in tickers if -60000 <= t < 0]
    post_tks = [(t, fr_v, oi, mk, idx, v24, t24, pch) for t, fr_v, oi, mk, idx, v24, t24, pch in tickers if 0 <= t <= 60000]

    if pre_tks:
        oi_values_pre = [oi for _, _, oi, _, _, _, _, _ in pre_tks if oi > 0]
        if len(oi_values_pre) >= 2:
            feat["oi_pre_first"] = oi_values_pre[0]
            feat["oi_pre_last"] = oi_values_pre[-1]
            feat["oi_pre_change_pct"] = (oi_values_pre[-1] - oi_values_pre[0]) / oi_values_pre[0] * 100

            # OI change in last 10s vs last 60s
            oi_last_10 = [oi for t, _, oi, _, _, _, _, _ in pre_tks if t >= -10000 and oi > 0]
            if len(oi_last_10) >= 2:
                feat["oi_change_10s_pct"] = (oi_last_10[-1] - oi_last_10[0]) / oi_last_10[0] * 100
        else:
            feat["oi_pre_first"] = feat["oi_pre_last"] = np.nan
            feat["oi_pre_change_pct"] = feat["oi_change_10s_pct"] = np.nan

        # Basis
        basis_vals = [(mk - idx) / idx * 10000 for _, _, _, mk, idx, _, _, _ in pre_tks if idx > 0]
        if basis_vals:
            feat["basis_pre_bps"] = basis_vals[-1]
            feat["basis_pre_trend"] = basis_vals[-1] - basis_vals[0] if len(basis_vals) > 1 else 0
    else:
        feat["oi_pre_first"] = feat["oi_pre_last"] = np.nan
        feat["oi_pre_change_pct"] = feat["oi_change_10s_pct"] = np.nan
        feat["basis_pre_bps"] = feat["basis_pre_trend"] = np.nan

    # Post-settlement OI
    if post_tks:
        oi_values_post = [(t, oi) for t, _, oi, _, _, _, _, _ in post_tks if oi > 0]
        if oi_values_post and feat.get("oi_pre_last", 0) > 0:
            for tw in [5000, 10000, 30000, 60000]:
                oi_at = [oi for t, oi in oi_values_post if t <= tw]
                if oi_at:
                    feat[f"oi_change_{tw}ms_pct"] = (oi_at[-1] - feat["oi_pre_last"]) / feat["oi_pre_last"] * 100

    # ════════════════════════════════════════════════════════════════
    # 6. LIQUIDATION ANALYSIS (post-settlement)
    # ════════════════════════════════════════════════════════════════
    post_liqs = [(t, s, q, p) for t, s, q, p in liquidations if 0 <= t <= 60000]
    feat["post_liq_count"] = len(post_liqs)
    feat["post_liq_vol_usd"] = sum(q * p for _, _, q, p in post_liqs)

    # Pre-settlement liquidations (signal of stress)
    pre_liqs = [(t, s, q, p) for t, s, q, p in liquidations if -10000 <= t < 0]
    feat["pre_liq_count"] = len(pre_liqs)
    feat["pre_liq_vol_usd"] = sum(q * p for _, _, q, p in pre_liqs)

    return feat


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyse_settlement_deep.py charts_settlement/*.jsonl")
        sys.exit(1)

    files = sorted(Path(f) for f in sys.argv[1:])
    print(f"Deep analysis of {len(files)} files...\n")
    t0 = _time.time()

    results = []
    for i, fp in enumerate(files, 1):
        if not fp.exists():
            continue
        r = analyse_settlement(fp)
        if r:
            results.append(r)
        if i % 20 == 0:
            print(f"  [{i}/{len(files)}] processed, {len(results)} valid, {_time.time()-t0:.1f}s elapsed")

    if not results:
        print("No valid results")
        return

    df = pd.DataFrame(results)
    out = Path("settlement_deep_analysis.csv")
    df.to_csv(out, index=False)

    elapsed = _time.time() - t0
    print(f"\n{'='*70}")
    print(f"DEEP ANALYSIS COMPLETE: {len(df)} settlements, {len(df.columns)} features, {elapsed:.1f}s")
    print(f"Saved to: {out}")
    print(f"{'='*70}")

    # ── Quick analysis ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ANALYSIS 1: PRICE TRAJECTORY OVER 60 SECONDS")
    print(f"{'='*70}")

    time_cols = [c for c in df.columns if c.startswith("price_") and c.endswith("_bps")]
    worst_cols = [c for c in df.columns if c.startswith("worst_") and c.endswith("_bps")]

    print(f"\n  Time window → avg LAST price (bps from ref) | avg WORST price")
    for tc, wc in zip(sorted(time_cols), sorted(worst_cols)):
        tw = tc.replace("price_", "").replace("_bps", "")
        mean_price = df[tc].mean()
        mean_worst = df[wc].mean()
        print(f"  T+{tw:>8s}: last={mean_price:+7.1f} bps | worst={mean_worst:+7.1f} bps")

    print(f"\n  Avg time to bottom: {df['time_to_bottom_ms'].mean():.0f} ms (median: {df['time_to_bottom_ms'].median():.0f} ms)")
    print(f"  Max recording window: {df['max_post_ms'].mean()/1000:.1f}s avg ({df['max_post_ms'].min()/1000:.1f}s min)")

    print(f"\n{'='*70}")
    print("ANALYSIS 2: RECOVERY / BOUNCE AFTER DROP")
    print(f"{'='*70}")

    rec_cols = [c for c in df.columns if c.startswith("recovery_") and c.endswith("_bps") and "final" not in c and "max" not in c]
    print(f"\n  Recovery from bottom (avg bps bounced back):")
    for rc in sorted(rec_cols, key=lambda x: int(x.replace("recovery_", "").replace("ms_bps", ""))):
        tw = rc.replace("recovery_", "").replace("ms_bps", "")
        val = df[rc].dropna()
        if len(val) > 0:
            print(f"  Bottom + {tw:>6s}ms: +{val.mean():5.1f} bps recovery (N={len(val)})")

    print(f"\n  Max recovery: {df['recovery_max_bps'].mean():+.1f} bps ({df['recovery_pct'].mean():.0f}% of drop)")
    print(f"  Final recovery: {df['recovery_final_bps'].mean():+.1f} bps ({df['recovery_final_pct'].mean():.0f}% of drop)")
    print(f"  Full recovery (price returns to ref): {df['full_recovery'].sum()}/{len(df)} ({df['full_recovery'].mean()*100:.0f}%)")

    # Recovery by FR magnitude
    print(f"\n  Recovery by FR magnitude:")
    for lo, hi, label in [(0, 30, "|FR|<30"), (30, 60, "|FR| 30-60"), (60, 100, "|FR| 60-100"), (100, 999, "|FR|>100")]:
        mask = (df['fr_abs_bps'] >= lo) & (df['fr_abs_bps'] < hi)
        sub = df[mask]
        if len(sub) > 0:
            print(f"  {label:12s}: N={len(sub):3d} | drop={sub['drop_min_bps'].mean():+7.1f} | recovery={sub['recovery_pct'].mean():5.1f}% | bottom@{sub['time_to_bottom_ms'].mean():6.0f}ms")

    print(f"\n{'='*70}")
    print("ANALYSIS 3: VOLUME PATTERNS")
    print(f"{'='*70}")

    vol_cols = [c for c in df.columns if c.startswith("vol_") and c.endswith("_usd") and "surge" not in c and "pre" not in c and "sell" not in c and "buy" not in c]
    print(f"\n  Post-settlement volume by time window:")
    for vc in sorted(vol_cols, key=lambda x: int(x.replace("vol_", "").replace("ms_usd", ""))):
        tw = vc.replace("vol_", "").replace("ms_usd", "")
        val = df[vc].dropna()
        sell_col = f"sell_ratio_{tw}ms"
        sr = df[sell_col].mean() if sell_col in df.columns else np.nan
        print(f"  T+{tw:>6s}ms: ${val.mean():>10,.0f} avg | sell ratio: {sr:.1%}")

    surge_cols = [c for c in df.columns if c.startswith("vol_surge_")]
    if surge_cols:
        print(f"\n  Volume surge ratio (post rate / pre rate):")
        for sc in sorted(surge_cols):
            tw = sc.replace("vol_surge_", "").replace("ms", "")
            print(f"  First {tw}ms: {df[sc].mean():.1f}x vs pre-settlement rate")

    # Volume predictability
    print(f"\n  FR vs post-settlement volume correlations:")
    for tw in [1000, 5000, 10000, 30000]:
        vc = f"vol_{tw}ms_usd"
        if vc in df.columns:
            r = df['fr_abs_bps'].corr(df[vc])
            print(f"  |FR| vs vol_{tw}ms: r={r:+.3f}")

    print(f"\n{'='*70}")
    print("ANALYSIS 4: ORDERBOOK CHANGES")
    print(f"{'='*70}")

    if "pre_total_depth_usd" in df.columns and df["pre_total_depth_usd"].notna().sum() > 5:
        print(f"\n  Pre-settlement depth: ${df['pre_total_depth_usd'].mean():,.0f} avg")
        print(f"  Pre depth imbalance: {df['pre_depth_imb'].mean():+.3f} (>0 = more bids)")

        for tw in [1000, 5000, 10000, 30000]:
            col = f"post_depth_change_pct_{tw}ms"
            if col in df.columns:
                val = df[col].dropna()
                if len(val) > 0:
                    print(f"  Depth change at T+{tw}ms: {val.mean():+.1f}% (N={len(val)})")

        # Depth drain as predictor
        if "depth_drain_pre_pct" in df.columns:
            print(f"\n  Pre-settlement depth drain (last 5s): {df['depth_drain_pre_pct'].mean():+.1f}%")
            r = df['depth_drain_pre_pct'].corr(df['drop_min_bps'])
            print(f"  Depth drain vs drop: r={r:+.3f}")

        # Pre-depth features vs drop
        print(f"\n  Orderbook features vs drop correlation:")
        ob_feats = ["pre_total_depth_usd", "pre_depth_imb", "pre_depth_20bps_usd",
                     "pre_depth_50bps_usd", "pre_depth_100bps_usd"]
        for f in ob_feats:
            if f in df.columns:
                r = df[f].corr(df['drop_min_bps'])
                print(f"  {f:30s}: r={r:+.3f}")
    else:
        print("  (No OB.200 data available)")

    print(f"\n{'='*70}")
    print("ANALYSIS 5: OPEN INTEREST")
    print(f"{'='*70}")

    if "oi_pre_change_pct" in df.columns:
        print(f"\n  OI change in 60s before settlement: {df['oi_pre_change_pct'].mean():+.2f}%")
        if "oi_change_10s_pct" in df.columns:
            print(f"  OI change in last 10s: {df['oi_change_10s_pct'].mean():+.2f}%")

        print(f"\n  Post-settlement OI changes:")
        for tw in [5000, 10000, 30000, 60000]:
            col = f"oi_change_{tw}ms_pct"
            if col in df.columns:
                val = df[col].dropna()
                if len(val) > 0:
                    print(f"  T+{tw/1000:.0f}s: {val.mean():+.2f}% (N={len(val)})")

        # OI vs drop/recovery
        print(f"\n  OI correlations:")
        oi_feats = ["oi_pre_change_pct", "oi_change_10s_pct"]
        for f in oi_feats:
            if f in df.columns:
                r_drop = df[f].corr(df['drop_min_bps'])
                r_rec = df[f].corr(df['recovery_pct']) if 'recovery_pct' in df.columns else np.nan
                print(f"  {f:25s}: vs drop r={r_drop:+.3f} | vs recovery r={r_rec:+.3f}")

    # Liquidation summary
    print(f"\n  Liquidations:")
    print(f"  Pre-settlement: {df['pre_liq_count'].mean():.1f} avg, ${df['pre_liq_vol_usd'].mean():,.0f} avg vol")
    print(f"  Post-settlement: {df['post_liq_count'].mean():.1f} avg, ${df['post_liq_vol_usd'].mean():,.0f} avg vol")

    # ── Correlations with targets ────────────────────────────────────
    print(f"\n{'='*70}")
    print("TOP CORRELATIONS WITH TARGETS")
    print(f"{'='*70}")

    # Exclude target/meta columns
    exclude_prefixes = ["price_", "worst_", "drop_", "recovery_", "vol_", "sell_vol_", "buy_vol_",
                        "sell_ratio_", "ntrades_", "vol_surge_", "post_", "time_to_", "max_post",
                        "full_recovery", "oi_change_5", "oi_change_1", "oi_change_3", "oi_change_6"]
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if not any(c.startswith(p) for p in exclude_prefixes)
                    and c not in ("ref_price",)]

    for target, label in [("drop_min_bps", "Max drop"), ("recovery_pct", "Recovery %"), ("time_to_bottom_ms", "Time to bottom")]:
        if target in df.columns:
            print(f"\n  Top features predicting {label}:")
            corrs = df[feature_cols].corrwith(df[target]).abs().sort_values(ascending=False).head(10)
            for col, val in corrs.items():
                sign = "+" if df[col].corr(df[target]) > 0 else "-"
                print(f"    {col:35s}: {sign}{val:.3f}")


if __name__ == "__main__":
    main()
