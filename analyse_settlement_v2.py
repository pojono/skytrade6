#!/usr/bin/env python3
"""
Settlement Feature Extractor V2 — Deep Feature Engineering
============================================================
Extracts 75+ features from JSONL market data files for ML prediction
of post-settlement sell waves.

Feature groups:
  1. Funding rate (2)
  2. Spread dynamics + trends (12)
  3. Orderbook imbalance + trends (12)
  4. Orderbook shape + depth (10)
  5. Trade microstructure (16)
  6. Ticker-derived: OI, basis, context (10)
  7. Liquidation (3)
  8. Interaction features (6)
  9. Targets: regression + classification (10)

Usage:
    python3 analyse_settlement_v2.py charts_settlement/*.jsonl
"""

import json
import sys
import time as _time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats as sp_stats


def _slope(values):
    """Linear regression slope over index."""
    if len(values) < 3:
        return 0.0
    x = np.arange(len(values))
    try:
        slope, _ = np.polyfit(x, values, 1)
        return slope
    except:
        return 0.0


def _safe_div(a, b, default=0.0):
    return a / b if b != 0 else default


def extract_features(filepath: Path) -> dict | None:
    """Extract deep features from a JSONL settlement recording."""

    symbol = filepath.stem.split("_")[0]
    settle_label = "_".join(filepath.stem.split("_")[1:])

    try:
        settle_dt = datetime.strptime(settle_label, "%Y%m%d_%H%M%S")
    except Exception:
        print(f"  ⚠ {symbol}: bad filename {settle_label}")
        return None

    # ── parse all messages ──────────────────────────────────────────────
    trades = []
    ob1 = []
    ob50 = []
    ob200 = []
    tickers = []
    liquidations = []

    with open(filepath) as f:
        for line in f:
            m = json.loads(line)
            t_ms = m["_t_ms"]
            topic = m.get("topic", "")

            if topic.startswith("publicTrade"):
                for tr in m.get("data", []):
                    p = float(tr["p"])
                    q = float(tr["v"])
                    trades.append((t_ms, p, q, tr["S"], p * q))

            elif topic.startswith("orderbook.1."):
                d = m.get("data", {})
                b = d.get("b", [])
                a = d.get("a", [])
                if b and a:
                    ob1.append((
                        t_ms,
                        float(b[0][0]), float(b[0][1]),
                        float(a[0][0]), float(a[0][1]),
                    ))

            elif topic.startswith("orderbook.50."):
                d = m.get("data", {})
                bids = [(float(p), float(q)) for p, q in d.get("b", [])]
                asks = [(float(p), float(q)) for p, q in d.get("a", [])]
                if bids and asks:
                    ob50.append((t_ms, bids, asks))

            elif topic.startswith("orderbook.200."):
                d = m.get("data", {})
                bids = [(float(p), float(q)) for p, q in d.get("b", [])]
                asks = [(float(p), float(q)) for p, q in d.get("a", [])]
                if bids and asks:
                    ob200.append((t_ms, bids, asks))

            elif topic.startswith("tickers"):
                d = m.get("data", {})
                tickers.append((
                    t_ms,
                    float(d["fundingRate"]) if d.get("fundingRate") else None,
                    float(d.get("openInterestValue", 0) or 0),
                    float(d["markPrice"]) if d.get("markPrice") else None,
                    float(d["indexPrice"]) if d.get("indexPrice") else None,
                    float(d.get("volume24h", 0) or 0),
                    float(d.get("turnover24h", 0) or 0),
                    float(d["prevPrice1h"]) if d.get("prevPrice1h") else None,
                    float(d.get("price24hPcnt", 0) or 0),
                ))

            elif topic.startswith("allLiquidation"):
                for liq in m.get("data", []):
                    liquidations.append((
                        t_ms,
                        liq.get("S", "?"),
                        float(liq.get("v", 0)),
                        float(liq.get("p", 0)),
                    ))

    if not trades:
        print(f"  ⚠ {symbol}: no trades")
        return None

    # ref price = last trade before T=0
    pre_trades = [t for t in trades if t[0] < 0]
    if not pre_trades:
        print(f"  ⚠ {symbol}: no pre-settlement trades")
        return None
    ref_price = pre_trades[-1][1]

    def bps(p):
        return (p - ref_price) / ref_price * 10000

    feat = {"symbol": symbol, "settle_time": settle_dt.strftime("%Y-%m-%d %H:%M:%S"), "ref_price": ref_price}

    # ════════════════════════════════════════════════════════════════════
    # GROUP 1 — FUNDING RATE
    # ════════════════════════════════════════════════════════════════════
    pre_tickers = [t for t in tickers if t[0] < 0]
    fr_vals = [t[1] for t in pre_tickers if t[1] is not None]
    if fr_vals:
        feat["fr_bps"] = fr_vals[-1] * 10000
        feat["fr_abs_bps"] = abs(feat["fr_bps"])
    else:
        feat["fr_bps"] = feat["fr_abs_bps"] = np.nan

    # ════════════════════════════════════════════════════════════════════
    # GROUP 2 — SPREAD DYNAMICS + TRENDS  (orderbook.1)
    # ════════════════════════════════════════════════════════════════════
    pre_ob1_10 = [(t, bp, bq, ap, aq) for t, bp, bq, ap, aq in ob1 if -10000 <= t < 0]
    pre_ob1_5  = [(t, bp, bq, ap, aq) for t, bp, bq, ap, aq in ob1 if -5000 <= t < 0]
    pre_ob1_2  = [(t, bp, bq, ap, aq) for t, bp, bq, ap, aq in ob1 if -2000 <= t < 0]
    pre_ob1_1  = [(t, bp, bq, ap, aq) for t, bp, bq, ap, aq in ob1 if -1000 <= t < 0]

    if pre_ob1_10:
        spreads_10 = [bps(ap) - bps(bp) for _, bp, _, ap, _ in pre_ob1_10]
        feat["spread_mean_bps"]  = np.mean(spreads_10)
        feat["spread_std_bps"]   = np.std(spreads_10)
        feat["spread_max_bps"]   = np.max(spreads_10)
        feat["spread_min_bps"]   = np.min(spreads_10)
        feat["spread_trend_10s"] = _slope(spreads_10)

        if pre_ob1_5:
            spreads_5 = [bps(ap) - bps(bp) for _, bp, _, ap, _ in pre_ob1_5]
            feat["spread_trend_5s"] = _slope(spreads_5)
            feat["spread_last_vs_mean"] = spreads_5[-1] - np.mean(spreads_10)
        else:
            feat["spread_trend_5s"] = feat["spread_last_vs_mean"] = np.nan

        if pre_ob1_2:
            spreads_2 = [bps(ap) - bps(bp) for _, bp, _, ap, _ in pre_ob1_2]
            feat["spread_trend_2s"] = _slope(spreads_2)
        else:
            feat["spread_trend_2s"] = np.nan

        # Qty imbalance at BBO
        imbs_10 = [_safe_div(bq - aq, bq + aq) for _, _, bq, _, aq in pre_ob1_10]
        feat["qty_imb_mean"]     = np.mean(imbs_10)
        feat["qty_imb_std"]      = np.std(imbs_10)
        feat["qty_imb_trend_10s"] = _slope(imbs_10)

        if pre_ob1_1:
            imbs_1 = [_safe_div(bq - aq, bq + aq) for _, _, bq, _, aq in pre_ob1_1]
            feat["qty_imb_last_1s"] = np.mean(imbs_1)
            feat["qty_imb_surge"]   = np.mean(imbs_1) - np.mean(imbs_10)
        else:
            feat["qty_imb_last_1s"] = feat["qty_imb_surge"] = np.nan
    else:
        for k in ["spread_mean_bps", "spread_std_bps", "spread_max_bps", "spread_min_bps",
                   "spread_trend_10s", "spread_trend_5s", "spread_trend_2s", "spread_last_vs_mean",
                   "qty_imb_mean", "qty_imb_std", "qty_imb_trend_10s",
                   "qty_imb_last_1s", "qty_imb_surge"]:
            feat[k] = np.nan

    # ════════════════════════════════════════════════════════════════════
    # GROUP 3 — DEPTH IMBALANCE + TRENDS  (orderbook.50)
    # ════════════════════════════════════════════════════════════════════
    pre_ob50_10 = [(t, b, a) for t, b, a in ob50 if -10000 <= t < 0]
    pre_ob50_5  = [(t, b, a) for t, b, a in ob50 if -5000 <= t < 0]

    if pre_ob50_10:
        bid10_vals = [sum(p * q for p, q in b[:10]) for _, b, _ in pre_ob50_10]
        ask10_vals = [sum(p * q for p, q in a[:10]) for _, _, a in pre_ob50_10]
        depth_imbs = [_safe_div(b - a, b + a) for b, a in zip(bid10_vals, ask10_vals)]

        feat["bid10_mean_usd"]     = np.mean(bid10_vals)
        feat["ask10_mean_usd"]     = np.mean(ask10_vals)
        feat["depth_imb_mean"]     = np.mean(depth_imbs)
        feat["depth_imb_std"]      = np.std(depth_imbs)
        feat["depth_imb_trend_10s"] = _slope(depth_imbs)
        feat["bid_depth_trend"]    = _slope(bid10_vals)
        feat["ask_depth_trend"]    = _slope(ask10_vals)

        if pre_ob50_5:
            d5 = [_safe_div(
                sum(p * q for p, q in b[:10]) - sum(p * q for p, q in a[:10]),
                sum(p * q for p, q in b[:10]) + sum(p * q for p, q in a[:10])
            ) for _, b, a in pre_ob50_5]
            feat["depth_imb_trend_5s"] = _slope(d5)
        else:
            feat["depth_imb_trend_5s"] = np.nan
    else:
        for k in ["bid10_mean_usd", "ask10_mean_usd", "depth_imb_mean", "depth_imb_std",
                   "depth_imb_trend_10s", "depth_imb_trend_5s", "bid_depth_trend", "ask_depth_trend"]:
            feat[k] = np.nan

    # ════════════════════════════════════════════════════════════════════
    # GROUP 4 — ORDERBOOK SHAPE + TOTAL DEPTH  (orderbook.200)
    # ════════════════════════════════════════════════════════════════════
    pre_ob200_10 = [(t, b, a) for t, b, a in ob200 if -10000 <= t < 0]

    if pre_ob200_10:
        total_bids = [sum(p * q for p, q in b) for _, b, _ in pre_ob200_10]
        total_asks = [sum(p * q for p, q in a) for _, _, a in pre_ob200_10]
        total_deps = [tb + ta for tb, ta in zip(total_bids, total_asks)]

        feat["total_bid_mean_usd"]   = np.mean(total_bids)
        feat["total_ask_mean_usd"]   = np.mean(total_asks)
        feat["total_depth_usd"]      = np.mean(total_deps)
        feat["total_depth_imb_mean"] = np.mean([_safe_div(b - a, b + a) for b, a in zip(total_bids, total_asks)])
        feat["total_depth_trend"]    = _slope(total_deps)

        # Concentration: top-10 vs full depth
        last_ob200 = pre_ob200_10[-1]
        _, lb, la = last_ob200
        bid10_last = sum(p * q for p, q in lb[:10])
        ask10_last = sum(p * q for p, q in la[:10])
        total_bid_last = sum(p * q for p, q in lb)
        total_ask_last = sum(p * q for p, q in la)
        feat["bid_concentration"]  = _safe_div(bid10_last, total_bid_last)
        feat["ask_concentration"]  = _safe_div(ask10_last, total_ask_last)

        # Depth within price bands
        mid = (lb[0][0] + la[0][0]) / 2 if lb and la else ref_price
        feat["depth_within_50bps"] = (
            sum(p * q for p, q in lb if abs(p - mid) / mid * 10000 < 50) +
            sum(p * q for p, q in la if abs(p - mid) / mid * 10000 < 50)
        )
        feat["thin_side_depth"]    = min(total_bid_last, total_ask_last)
    else:
        for k in ["total_bid_mean_usd", "total_ask_mean_usd", "total_depth_usd",
                   "total_depth_imb_mean", "total_depth_trend",
                   "bid_concentration", "ask_concentration",
                   "depth_within_50bps", "thin_side_depth"]:
            feat[k] = np.nan

    # ════════════════════════════════════════════════════════════════════
    # GROUP 5 — TRADE MICROSTRUCTURE
    # ════════════════════════════════════════════════════════════════════
    pre_t10 = [(t, p, q, s, n) for t, p, q, s, n in trades if -10000 <= t < 0]
    pre_t5  = [(t, p, q, s, n) for t, p, q, s, n in trades if -5000 <= t < 0]
    pre_t2  = [(t, p, q, s, n) for t, p, q, s, n in trades if -2000 <= t < 0]
    pre_t1  = [(t, p, q, s, n) for t, p, q, s, n in trades if -1000 <= t < 0]

    if pre_t10:
        notionals = [n for _, _, _, _, n in pre_t10]
        buy_vol  = sum(n for _, _, _, s, n in pre_t10 if s == "Buy")
        sell_vol = sum(n for _, _, _, s, n in pre_t10 if s == "Sell")
        total_vol = buy_vol + sell_vol

        feat["pre_trade_count"]        = len(pre_t10)
        feat["pre_total_vol_usd"]      = total_vol
        feat["trade_flow_imb"]         = _safe_div(buy_vol - sell_vol, total_vol)
        feat["pre_avg_trade_size_usd"] = np.mean(notionals)

        # Size distribution
        feat["trade_size_median"]  = np.median(notionals)
        feat["trade_size_p90"]     = np.percentile(notionals, 90)
        feat["trade_size_p99"]     = np.percentile(notionals, 99)
        feat["trade_size_max"]     = np.max(notionals)
        feat["trade_size_skew"]    = float(sp_stats.skew(notionals)) if len(notionals) > 2 else 0.0

        large = [n for n in notionals if n > 500]
        feat["large_trade_count"]  = len(large)
        feat["large_trade_pct"]    = len(large) / len(notionals)

        # Large trade imbalance
        large_buy  = sum(n for _, _, _, s, n in pre_t10 if s == "Buy" and n > 500)
        large_sell = sum(n for _, _, _, s, n in pre_t10 if s == "Sell" and n > 500)
        feat["large_trade_imb"]    = _safe_div(large_buy - large_sell, large_buy + large_sell)

        # Price volatility
        prices = [p for _, p, _, _, _ in pre_t10]
        pch = [bps(prices[i]) - bps(prices[i - 1]) for i in range(1, len(prices))]
        feat["pre_price_vol_bps"]  = np.std(pch) if pch else 0.0

        # Trade rate dynamics
        feat["trade_rate_10s"] = len(pre_t10) / 10.0
        feat["trade_rate_2s"]  = len(pre_t2) / 2.0 if pre_t2 else 0.0
        feat["trade_rate_accel"] = (feat["trade_rate_2s"] - feat["trade_rate_10s"]) / max(feat["trade_rate_10s"], 1)

        # Volume acceleration
        vol_10 = total_vol / 10.0
        vol_2  = sum(n for _, _, _, _, n in pre_t2) / 2.0 if pre_t2 else 0.0
        feat["vol_rate_accel"] = _safe_div(vol_2 - vol_10, vol_10)

        # Buy pressure last 1s vs 10s
        if pre_t1:
            buy1  = sum(n for _, _, _, s, n in pre_t1 if s == "Buy")
            sell1 = sum(n for _, _, _, s, n in pre_t1 if s == "Sell")
            feat["buy_imb_last_1s"]    = _safe_div(buy1 - sell1, buy1 + sell1)
            feat["buy_pressure_surge"] = feat["buy_imb_last_1s"] - feat["trade_flow_imb"]
        else:
            feat["buy_imb_last_1s"] = feat["buy_pressure_surge"] = np.nan

        # VWAP vs mid
        vwap = sum(p * n for _, p, _, _, n in pre_t10) / total_vol if total_vol > 0 else ref_price
        if pre_ob1_10:
            mid = (pre_ob1_10[-1][1] + pre_ob1_10[-1][3]) / 2
        else:
            mid = ref_price
        feat["vwap_vs_mid_bps"] = bps(vwap) - bps(mid)
    else:
        for k in ["pre_trade_count", "pre_total_vol_usd", "trade_flow_imb",
                   "pre_avg_trade_size_usd", "trade_size_median", "trade_size_p90",
                   "trade_size_p99", "trade_size_max", "trade_size_skew",
                   "large_trade_count", "large_trade_pct", "large_trade_imb",
                   "pre_price_vol_bps", "trade_rate_10s", "trade_rate_2s",
                   "trade_rate_accel", "vol_rate_accel",
                   "buy_imb_last_1s", "buy_pressure_surge", "vwap_vs_mid_bps"]:
            feat[k] = np.nan

    # ════════════════════════════════════════════════════════════════════
    # GROUP 6 — TICKER-DERIVED: OI, BASIS, CONTEXT
    # ════════════════════════════════════════════════════════════════════
    if pre_tickers:
        # Open interest
        oi_vals = [t[2] for t in pre_tickers if t[2] > 0]
        if len(oi_vals) >= 2:
            feat["oi_value_usd"]      = oi_vals[-1]
            feat["oi_change_60s"]     = oi_vals[-1] - oi_vals[0]
            feat["oi_change_pct_60s"] = _safe_div(feat["oi_change_60s"], oi_vals[0]) * 100
        else:
            feat["oi_value_usd"] = oi_vals[-1] if oi_vals else np.nan
            feat["oi_change_60s"] = feat["oi_change_pct_60s"] = np.nan

        # Basis (mark - index)
        marks   = [(t[0], t[3]) for t in pre_tickers if t[3] is not None]
        indices = [(t[0], t[4]) for t in pre_tickers if t[4] is not None]
        if marks and indices:
            m_last = marks[-1][1]
            i_last = indices[-1][1]
            feat["basis_bps"]   = (m_last - i_last) / i_last * 10000 if i_last else np.nan
            feat["basis_abs_bps"] = abs(feat["basis_bps"]) if not np.isnan(feat["basis_bps"]) else np.nan
            # Basis trend
            if len(marks) >= 5 and len(indices) >= 5:
                basis_series = [(m - i) / i * 10000 for (_, m), (_, i) in zip(marks[-20:], indices[-20:])]
                feat["basis_trend"] = _slope(basis_series)
            else:
                feat["basis_trend"] = np.nan
        else:
            feat["basis_bps"] = feat["basis_abs_bps"] = feat["basis_trend"] = np.nan

        # Market context
        feat["volume_24h"]         = pre_tickers[-1][5]
        feat["turnover_24h_usd"]   = pre_tickers[-1][6]
        prev1h = pre_tickers[-1][7]
        if prev1h and prev1h > 0:
            feat["price_change_1h_bps"] = (ref_price - prev1h) / prev1h * 10000
        else:
            feat["price_change_1h_bps"] = np.nan
        feat["price_change_24h_pct"] = pre_tickers[-1][8] * 100
    else:
        for k in ["oi_value_usd", "oi_change_60s", "oi_change_pct_60s",
                   "basis_bps", "basis_abs_bps", "basis_trend",
                   "volume_24h", "turnover_24h_usd",
                   "price_change_1h_bps", "price_change_24h_pct"]:
            feat[k] = np.nan

    # ════════════════════════════════════════════════════════════════════
    # GROUP 7 — LIQUIDATION
    # ════════════════════════════════════════════════════════════════════
    pre_liqs = [l for l in liquidations if l[0] < 0]
    feat["liq_count_pre"]   = len(pre_liqs)
    feat["liq_volume_usd"]  = sum(l[2] * l[3] for l in pre_liqs)
    long_liqs  = sum(1 for l in pre_liqs if l[1] == "Sell")   # long liq = forced sell
    short_liqs = sum(1 for l in pre_liqs if l[1] == "Buy")    # short liq = forced buy
    feat["liq_direction"]   = _safe_div(long_liqs - short_liqs, long_liqs + short_liqs)

    # ════════════════════════════════════════════════════════════════════
    # GROUP 8 — INTERACTION FEATURES
    # ════════════════════════════════════════════════════════════════════
    fr = feat.get("fr_bps", 0) if not np.isnan(feat.get("fr_bps", np.nan)) else 0
    depth = feat.get("total_depth_usd", 0) if not np.isnan(feat.get("total_depth_usd", np.nan)) else 0
    spread = feat.get("spread_mean_bps", 0) if not np.isnan(feat.get("spread_mean_bps", np.nan)) else 0
    imb = feat.get("qty_imb_mean", 0) if not np.isnan(feat.get("qty_imb_mean", np.nan)) else 0
    vol = feat.get("pre_total_vol_usd", 0) if not np.isnan(feat.get("pre_total_vol_usd", np.nan)) else 0

    feat["fr_x_depth"]     = fr * np.log1p(depth) if depth > 0 else 0
    feat["fr_x_spread"]    = fr * spread
    feat["fr_x_imb"]       = fr * imb
    feat["imb_x_vol"]      = imb * np.log1p(vol) if vol > 0 else 0
    feat["spread_x_depth"] = spread * np.log1p(depth) if depth > 0 else 0
    feat["fr_squared"]     = fr * fr

    # ════════════════════════════════════════════════════════════════════
    # TARGETS (regression + classification)
    # ════════════════════════════════════════════════════════════════════
    post = [(t, p, q, s, n) for t, p, q, s, n in trades if 0 <= t <= 5000]
    if not post:
        print(f"  ⚠ {symbol}: no post-settlement trades")
        return None

    post_prices_bps = [bps(p) for _, p, _, _, _ in sorted(post, key=lambda x: x[0])]

    # Regression targets
    feat["drop_min_bps"]   = min(post_prices_bps)
    feat["drop_final_bps"] = post_prices_bps[-1]
    feat["recovery_bps"]   = post_prices_bps[-1] - min(post_prices_bps)

    min_idx = post_prices_bps.index(min(post_prices_bps))
    feat["time_to_bottom_ms"] = sorted(post, key=lambda x: x[0])[min_idx][0]

    # Horizon targets: price at CLOSEST trade to each horizon
    for horizon, label in [(100, "100ms"), (500, "500ms"), (1000, "1s"), (5000, "5s")]:
        h_trades = [(t, bps(p)) for t, p, _, _, _ in post if t <= horizon]
        if h_trades:
            # Last trade price before/at horizon = price at that point in time
            last_trade = max(h_trades, key=lambda x: x[0])
            feat[f"price_{label}_bps"] = last_trade[1]
            # Also keep min drop within window (worst price reached so far)
            feat[f"worst_{label}_bps"] = min(bp for _, bp in h_trades)
        else:
            feat[f"price_{label}_bps"] = np.nan
            feat[f"worst_{label}_bps"] = np.nan

    # Sell wave
    post_buy  = sum(n for _, _, _, s, n in post if s == "Buy")
    post_sell = sum(n for _, _, _, s, n in post if s == "Sell")
    feat["post_sell_vol_usd"] = post_sell
    feat["post_sell_ratio"]   = _safe_div(post_sell, post_buy + post_sell)

    # Classification targets
    feat["target_profitable"]  = 1 if feat["drop_min_bps"] < -40 else 0
    feat["target_drop_class"]  = (
        0 if feat["drop_min_bps"] > -40 else
        1 if feat["drop_min_bps"] > -80 else
        2 if feat["drop_min_bps"] > -120 else 3
    )
    feat["target_fast_drop"]   = 1 if feat["time_to_bottom_ms"] < 500 else 0

    # Data quality
    feat["has_ob1"]  = len(pre_ob1_10) > 0
    feat["has_ob50"] = len(pre_ob50_10) > 0

    fr_str = f"{feat['fr_bps']:+.1f}" if not np.isnan(feat["fr_bps"]) else "N/A"
    print(f"  ✓ {symbol} {settle_label}: FR={fr_str}bps drop={feat['drop_min_bps']:.1f}bps class={feat['target_drop_class']}")

    return feat


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyse_settlement_v2.py charts_settlement/*.jsonl")
        sys.exit(1)

    files = sorted(Path(f) for f in sys.argv[1:])
    print(f"Extracting V2 features from {len(files)} files...\n")
    t0 = _time.time()

    results = []
    for i, fp in enumerate(files, 1):
        if not fp.exists():
            continue
        r = extract_features(fp)
        if r:
            results.append(r)
        if i % 10 == 0:
            print(f"  [{i}/{len(files)}] processed, {len(results)} valid, {_time.time()-t0:.1f}s elapsed")

    if not results:
        print("\n⚠ No valid results")
        return

    df = pd.DataFrame(results)
    out = Path("settlement_features_v2.csv")
    df.to_csv(out, index=False)

    elapsed = _time.time() - t0
    print(f"\n{'='*80}")
    print(f"V2 FEATURE EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Settlements: {len(df)} | Features: {len(df.columns)} | Time: {elapsed:.1f}s")
    print(f"Saved to: {out}")

    # Quick stats
    print(f"\n--- Feature coverage ---")
    nan_pct = df.isna().mean().sort_values(ascending=False)
    has_nan = nan_pct[nan_pct > 0]
    if len(has_nan) > 0:
        for col, pct in has_nan.head(15).items():
            print(f"  {col:35s}: {pct*100:5.1f}% missing")

    print(f"\n--- Target distribution ---")
    print(f"  Profitable (drop > 40bps):  {df['target_profitable'].sum()}/{len(df)} ({df['target_profitable'].mean()*100:.0f}%)")
    print(f"  Drop class 0 (SKIP):        {(df['target_drop_class']==0).sum()}")
    print(f"  Drop class 1 (MARGINAL):    {(df['target_drop_class']==1).sum()}")
    print(f"  Drop class 2 (GOOD):        {(df['target_drop_class']==2).sum()}")
    print(f"  Drop class 3 (EXCELLENT):   {(df['target_drop_class']==3).sum()}")
    print(f"  Fast drop (<500ms):         {df['target_fast_drop'].sum()}/{len(df)} ({df['target_fast_drop'].mean()*100:.0f}%)")

    print(f"\n--- FR vs drop correlation ---")
    valid = df.dropna(subset=["fr_bps", "drop_min_bps"])
    if len(valid) > 2:
        corr = valid[["fr_bps", "drop_min_bps"]].corr().iloc[0, 1]
        print(f"  Pearson r = {corr:+.3f}  (N={len(valid)})")

    # Top correlations with drop_min_bps
    print(f"\n--- Top 15 correlations with drop_min_bps ---")
    numeric = df.select_dtypes(include=[np.number])
    corrs = numeric.corr()["drop_min_bps"].drop(
        [c for c in numeric.columns if c.startswith("target_") or c.startswith("drop_") or
         c.startswith("post_") or c in ("time_to_bottom_ms", "recovery_bps", "ref_price")],
        errors="ignore"
    ).dropna().abs().sort_values(ascending=False).head(15)
    for col, val in corrs.items():
        sign = "+" if numeric.corr()["drop_min_bps"][col] > 0 else "-"
        print(f"  {col:35s}: {sign}{val:.3f}")


if __name__ == "__main__":
    main()
