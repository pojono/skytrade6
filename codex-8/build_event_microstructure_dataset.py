#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATALAKE = ROOT / "datalake"
OUT_DIR = Path(__file__).resolve().parent / "out"
CACHE_DIR = OUT_DIR / "cache"

DEFAULT_PANEL = OUT_DIR / "dislocation_panel_survivors.csv.gz"
DEFAULT_SYMBOLS = ["CRVUSDT", "GALAUSDT", "SEIUSDT"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build event-level dislocation + microstructure dataset.")
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--start-date", default="2026-02-01")
    parser.add_argument("--end-date", default="2026-03-04")
    parser.add_argument("--min-gap-bps", type=float, default=10.0)
    parser.add_argument("--cooldown-minutes", type=int, default=60)
    parser.add_argument("--pair-fee-bps-roundtrip", type=float, default=8.0)
    parser.add_argument("--output", default=str(OUT_DIR / "event_microstructure_dataset.csv"))
    parser.add_argument("--required-symbol-days")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def parse_symbols(raw: str) -> list[str]:
    symbols = [part.strip().upper() for part in raw.split(",") if part.strip()]
    return symbols if symbols else list(DEFAULT_SYMBOLS)


def load_required_symbol_days(path: str | None) -> set[tuple[str, str]] | None:
    if not path:
        return None
    df = pd.read_csv(path, usecols=["symbol", "date"])
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["date"] = df["date"].astype(str)
    return set(zip(df["symbol"], df["date"]))


def rolling_z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(window // 4, 20)).mean()
    std = series.rolling(window, min_periods=max(window // 4, 20)).std()
    return (series - mean) / std.replace(0, np.nan)


def prepare_panel(panel_path: Path, symbols: list[str], start_date: str, end_date: str, pair_fee_bps: float) -> pd.DataFrame:
    df = pd.read_csv(panel_path, parse_dates=["ts"])
    df = df[df["symbol"].isin(symbols)].copy()
    df = df[(df["ts"] >= pd.Timestamp(start_date, tz="UTC")) & (df["ts"] <= pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1))].copy()
    if df.empty:
        raise SystemExit("panel filter returned no rows")
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    group = df.groupby("symbol", sort=False)

    df["gap_bps"] = 10000.0 * (pd.to_numeric(df["bn_close"], errors="coerce") / pd.to_numeric(df["bb_close"], errors="coerce") - 1.0)
    df["bn_premium"] = pd.to_numeric(df["bn_premium"], errors="coerce")
    df["bb_premium"] = pd.to_numeric(df["bb_premium"], errors="coerce")
    df["premium_gap_bps"] = 10000.0 * (df["bn_premium"] - df["bb_premium"])
    df["bn_ls_ratio"] = pd.to_numeric(df["bn_ls_ratio"], errors="coerce")
    df["bb_buy_ratio"] = pd.to_numeric(df["bb_buy_ratio"], errors="coerce")
    df["bb_sell_ratio"] = pd.to_numeric(df["bb_sell_ratio"], errors="coerce")
    df["bn_oi_value"] = pd.to_numeric(df["bn_oi_value"], errors="coerce")
    df["bb_oi_value"] = pd.to_numeric(df["bb_open_interest"], errors="coerce") * pd.to_numeric(df["bb_close"], errors="coerce")
    df["bn_quote_volume"] = pd.to_numeric(df["bn_quote_volume"], errors="coerce")
    df["bn_taker_buy_quote_volume"] = pd.to_numeric(df["bn_taker_buy_quote_volume"], errors="coerce")
    df["bb_turnover"] = pd.to_numeric(df["bb_turnover"], errors="coerce")

    df["bn_crowding"] = df["bn_ls_ratio"] - 1.0
    df["bb_crowding"] = 2.0 * (df["bb_buy_ratio"] - 0.5)
    df["crowding_gap"] = df["bn_crowding"] - df["bb_crowding"]
    df["bn_taker_imbalance"] = 2.0 * (df["bn_taker_buy_quote_volume"] / df["bn_quote_volume"].replace(0, np.nan)) - 1.0

    for lag in [1, 5, 15, 30]:
        df[f"bb_ret_{lag}m_bps"] = 10000.0 * group["bb_close"].pct_change(lag)
        df[f"bn_ret_{lag}m_bps"] = 10000.0 * group["bn_close"].pct_change(lag)
    df["rel_ret_5m_bps"] = df["bn_ret_5m_bps"] - df["bb_ret_5m_bps"]
    df["rel_ret_15m_bps"] = df["bn_ret_15m_bps"] - df["bb_ret_15m_bps"]
    df["bb_realized_vol_15m_bps"] = group["bb_close"].transform(lambda s: 10000.0 * s.pct_change().rolling(15, min_periods=10).std())
    df["bn_quote_volume_z_60"] = group["bn_quote_volume"].transform(lambda s: rolling_z(s, 60))
    df["bb_turnover_z_60"] = group["bb_turnover"].transform(lambda s: rolling_z(s, 60))
    df["gap_z_60"] = group["gap_bps"].transform(lambda s: rolling_z(s, 60))
    df["gap_z_240"] = group["gap_bps"].transform(lambda s: rolling_z(s, 240))
    df["premium_gap_z_240"] = group["premium_gap_bps"].transform(lambda s: rolling_z(s, 240))
    df["crowding_gap_z_240"] = group["crowding_gap"].transform(lambda s: rolling_z(s, 240))
    df["bn_oi_chg_30m"] = group["bn_oi_value"].pct_change(30, fill_method=None)
    df["bb_oi_chg_30m"] = group["bb_oi_value"].pct_change(30, fill_method=None)
    df["oi_gap_30m"] = df["bn_oi_chg_30m"] - df["bb_oi_chg_30m"]
    df["oi_gap_30m_z_240"] = group["oi_gap_30m"].transform(lambda s: rolling_z(s, 240))

    for horizon in [5, 15, 30]:
        df[f"future_gap_bps_{horizon}m"] = group["gap_bps"].shift(-horizon)
        df[f"gap_close_{horizon}m_bps"] = df["gap_bps"] - df[f"future_gap_bps_{horizon}m"]

    # Best convergence achieved within 15m / 30m using future minimum gap.
    for horizon in [15, 30]:
        future_min = group["gap_bps"].transform(lambda s, h=horizon: s.shift(-1).rolling(h, min_periods=1).min().shift(-(h - 1)))
        df[f"max_gap_close_{horizon}m_bps"] = df["gap_bps"] - future_min

    df["pair_net_15m_bps"] = df["gap_close_15m_bps"] - pair_fee_bps
    df["pair_net_30m_bps"] = df["gap_close_30m_bps"] - pair_fee_bps
    df["pair_net_max_15m_bps"] = df["max_gap_close_15m_bps"] - pair_fee_bps
    df["pair_win_15m"] = (df["pair_net_15m_bps"] > 0).astype(int)
    df["pair_win_max_15m"] = (df["pair_net_max_15m_bps"] > 0).astype(int)

    minute_of_day = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df["tod_sin"] = np.sin(2.0 * np.pi * minute_of_day / 1440.0)
    df["tod_cos"] = np.cos(2.0 * np.pi * minute_of_day / 1440.0)
    return df.replace([np.inf, -np.inf], np.nan)


def select_events(df: pd.DataFrame, min_gap_bps: float, cooldown_minutes: int) -> pd.DataFrame:
    rows: list[pd.Series] = []
    cooldown = pd.Timedelta(minutes=cooldown_minutes)
    for _, part in df.groupby("symbol", sort=False):
        last_event_ts: pd.Timestamp | None = None
        for row in part.itertuples(index=False):
            ts = row.ts
            gap_bps = row.gap_bps
            if not np.isfinite(gap_bps) or gap_bps < min_gap_bps:
                continue
            if not np.isfinite(row.pair_net_15m_bps):
                continue
            if last_event_ts is None or ts - last_event_ts >= cooldown:
                rows.append(pd.Series(row._asdict()))
                last_event_ts = ts
    if not rows:
        raise SystemExit("no events survived the gap/cooldown filters")
    events = pd.DataFrame(rows).reset_index(drop=True)
    events["event_sec"] = (events["ts"].astype("int64") // 10**9).astype("int64")
    events["date"] = events["ts"].dt.strftime("%Y-%m-%d")
    return events


def aggregate_binance_trades(path: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, compression="gzip", chunksize=1_000_000):
        chunk["sec"] = (pd.to_numeric(chunk["time"], errors="coerce") // 1000).astype("int64")
        chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
        chunk["qty"] = pd.to_numeric(chunk["qty"], errors="coerce")
        chunk["notional"] = chunk["price"] * chunk["qty"]
        chunk["sign"] = np.where(chunk["is_buyer_maker"].astype(str).str.lower() == "true", -1.0, 1.0)
        chunk["signed_notional"] = chunk["notional"] * chunk["sign"]
        parts.append(
            chunk.groupby("sec").agg(
                bn_signed_notional=("signed_notional", "sum"),
                bn_notional=("notional", "sum"),
                bn_trade_count=("id", "count"),
                bn_last_price=("price", "last"),
            )
        )
    out = pd.concat(parts)
    out = out.groupby(level=0).agg(
        {
            "bn_signed_notional": "sum",
            "bn_notional": "sum",
            "bn_trade_count": "sum",
            "bn_last_price": "last",
        }
    )
    return out.sort_index()


def aggregate_bybit_trades(path: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, compression="gzip", chunksize=1_000_000):
        chunk["sec"] = pd.to_numeric(chunk["timestamp"], errors="coerce").astype(float).astype("int64")
        chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
        chunk["size"] = pd.to_numeric(chunk["size"], errors="coerce")
        chunk["notional"] = chunk["price"] * chunk["size"]
        chunk["sign"] = np.where(chunk["side"].astype(str).str.lower() == "buy", 1.0, -1.0)
        chunk["signed_notional"] = chunk["notional"] * chunk["sign"]
        parts.append(
            chunk.groupby("sec").agg(
                bb_signed_notional=("signed_notional", "sum"),
                bb_notional=("notional", "sum"),
                bb_trade_count=("side", "count"),
                bb_last_price=("price", "last"),
            )
        )
    out = pd.concat(parts)
    out = out.groupby(level=0).agg(
        {
            "bb_signed_notional": "sum",
            "bb_notional": "sum",
            "bb_trade_count": "sum",
            "bb_last_price": "last",
        }
    )
    return out.sort_index()


def load_binance_depth(path: Path) -> pd.DataFrame:
    depth = pd.read_csv(path, compression="gzip")
    depth["ts"] = pd.to_datetime(depth["timestamp"], utc=True)
    depth["sec"] = (depth["ts"].astype("int64") // 10**9).astype("int64")
    pivot = depth.pivot_table(index="sec", columns="percentage", values="notional", aggfunc="last").sort_index()
    pivot.columns = [f"bn_depth_{val:g}" for val in pivot.columns]
    pivot = pivot.reset_index().sort_values("sec")
    for pct in ["0.2", "1", "5"]:
        bid_col = f"bn_depth_-{pct}"
        ask_col = f"bn_depth_{pct}"
        if bid_col not in pivot.columns:
            pivot[bid_col] = np.nan
        if ask_col not in pivot.columns:
            pivot[ask_col] = np.nan
        bid = pivot[bid_col]
        ask = pivot[ask_col]
        pivot[f"bn_depth_imbalance_{pct}"] = (bid - ask) / (bid + ask)
    pivot["bn_depth_pressure"] = pivot["bn_depth_imbalance_0.2"] - pivot["bn_depth_imbalance_5"]
    return pivot


def apply_updates(side: dict[float, float], updates: list[list[str]]) -> None:
    for price_str, size_str in updates:
        price = float(price_str)
        size = float(size_str)
        if size == 0.0:
            side.pop(price, None)
        else:
            side[price] = size


def summarize_book(sec: int, bids: dict[float, float], asks: dict[float, float]) -> dict[str, float]:
    if not bids or not asks:
        return {"sec": sec}
    top_bids = sorted(bids.items(), key=lambda x: x[0], reverse=True)
    top_asks = sorted(asks.items(), key=lambda x: x[0])
    best_bid_px, best_bid_sz = top_bids[0]
    best_ask_px, best_ask_sz = top_asks[0]
    mid = (best_bid_px + best_ask_px) / 2.0
    bid5 = sum(px * sz for px, sz in top_bids[:5])
    ask5 = sum(px * sz for px, sz in top_asks[:5])
    bid20 = sum(px * sz for px, sz in top_bids[:20])
    ask20 = sum(px * sz for px, sz in top_asks[:20])
    return {
        "sec": sec,
        "bb_best_bid_px": best_bid_px,
        "bb_best_bid_sz": best_bid_sz,
        "bb_best_ask_px": best_ask_px,
        "bb_best_ask_sz": best_ask_sz,
        "bb_mid_px_ob": mid,
        "bb_spread_bps_ob": (best_ask_px - best_bid_px) / mid * 10000.0,
        "bb_top5_bid_notional": bid5,
        "bb_top5_ask_notional": ask5,
        "bb_top20_bid_notional": bid20,
        "bb_top20_ask_notional": ask20,
        "bb_top5_imbalance": (bid5 - ask5) / (bid5 + ask5) if (bid5 + ask5) else np.nan,
        "bb_top20_imbalance": (bid20 - ask20) / (bid20 + ask20) if (bid20 + ask20) else np.nan,
    }


def load_bybit_orderbook(path: Path) -> pd.DataFrame:
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    current_sec: int | None = None
    rows: list[dict[str, float]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            msg = json.loads(line)
            sec = int(msg["ts"]) // 1000
            data = msg["data"]
            if msg.get("type") == "snapshot":
                bids = {float(px): float(sz) for px, sz in data["b"]}
                asks = {float(px): float(sz) for px, sz in data["a"]}
            else:
                apply_updates(bids, data["b"])
                apply_updates(asks, data["a"])
            if current_sec is None:
                current_sec = sec
            if sec != current_sec:
                rows.append(summarize_book(current_sec, bids, asks))
                current_sec = sec
        if current_sec is not None:
            rows.append(summarize_book(current_sec, bids, asks))
    out = pd.DataFrame(rows).sort_values("sec").reset_index(drop=True)
    out["bb_top5_bid_delta"] = out["bb_top5_bid_notional"].diff()
    out["bb_top5_ask_delta"] = out["bb_top5_ask_notional"].diff()
    out["bb_top5_pull_pressure"] = out["bb_top5_bid_delta"] - out["bb_top5_ask_delta"]
    out["bb_top5_pull_pressure_5s"] = out["bb_top5_pull_pressure"].rolling(5).sum()
    out["bb_top5_pull_pressure_15s"] = out["bb_top5_pull_pressure"].rolling(15).sum()
    out["bb_top20_imbalance_chg_5s"] = out["bb_top20_imbalance"].diff(5)
    out["bb_best_sz_imbalance"] = (
        (out["bb_best_bid_sz"] - out["bb_best_ask_sz"]) / (out["bb_best_bid_sz"] + out["bb_best_ask_sz"])
    )
    return out


def build_day_microstructure(symbol: str, day: str, force_rebuild: bool) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{symbol}_{day}_event_micro_sec.csv.gz"
    if cache_path.exists() and not force_rebuild:
        try:
            return pd.read_csv(cache_path)
        except Exception:
            cache_path.unlink(missing_ok=True)

    bn_trade_path = DATALAKE / "binance" / symbol / f"{day}_trades.csv.gz"
    bb_trade_path = DATALAKE / "bybit" / symbol / f"{day}_trades.csv.gz"
    bn_depth_path = DATALAKE / "binance" / symbol / f"{day}_bookDepth.csv.gz"
    bb_book_path = DATALAKE / "bybit" / symbol / f"{day}_orderbook.jsonl.gz"
    if not (bn_trade_path.exists() and bb_trade_path.exists() and bn_depth_path.exists() and bb_book_path.exists()):
        raise FileNotFoundError(f"missing microstructure files for {symbol} {day}")

    bn_trades = aggregate_binance_trades(bn_trade_path)
    bb_trades = aggregate_bybit_trades(bb_trade_path)
    bn_depth = load_binance_depth(bn_depth_path)
    bb_book = load_bybit_orderbook(bb_book_path)

    start = max(int(bn_trades.index.min()), int(bb_trades.index.min()), int(bn_depth["sec"].min()), int(bb_book["sec"].min()))
    end = min(int(bn_trades.index.max()), int(bb_trades.index.max()), int(bn_depth["sec"].max()), int(bb_book["sec"].max()))
    sec = pd.DataFrame(index=pd.Index(range(start, end + 1), name="sec"))
    sec = sec.join(bn_trades, how="left").join(bb_trades, how="left").fillna(0.0)
    for col in ["bn_last_price", "bb_last_price"]:
        sec[col] = sec[col].replace(0, np.nan).ffill().bfill()
    sec = sec.reset_index().sort_values("sec")
    sec = pd.merge_asof(sec, bn_depth, on="sec", direction="backward")
    sec = pd.merge_asof(
        sec,
        bb_book[
            [
                "sec",
                "bb_top5_imbalance",
                "bb_top20_imbalance",
                "bb_best_sz_imbalance",
                "bb_spread_bps_ob",
                "bb_top5_pull_pressure",
                "bb_top5_pull_pressure_5s",
                "bb_top5_pull_pressure_15s",
                "bb_top20_imbalance_chg_5s",
                "bb_mid_px_ob",
            ]
        ],
        on="sec",
        direction="backward",
    )

    sec["bn_flow_ratio"] = sec["bn_signed_notional"] / sec["bn_notional"].replace(0, np.nan)
    sec["bb_flow_ratio"] = sec["bb_signed_notional"] / sec["bb_notional"].replace(0, np.nan)
    sec["flow_divergence"] = sec["bn_flow_ratio"] - sec["bb_flow_ratio"]
    sec["sec_gap_bps"] = 10000.0 * (sec["bn_last_price"] / sec["bb_last_price"] - 1.0)
    sec["ob_gap_bps"] = 10000.0 * (sec["bn_last_price"] / sec["bb_mid_px_ob"] - 1.0)

    for window in [5, 15, 30, 60, 180]:
        sec[f"bn_signed_notional_{window}s"] = sec["bn_signed_notional"].rolling(window, min_periods=max(2, window // 3)).sum()
        sec[f"bb_signed_notional_{window}s"] = sec["bb_signed_notional"].rolling(window, min_periods=max(2, window // 3)).sum()
        sec[f"combo_signed_notional_{window}s"] = sec[f"bn_signed_notional_{window}s"] + sec[f"bb_signed_notional_{window}s"]
        sec[f"bn_flow_ratio_{window}s"] = sec["bn_flow_ratio"].rolling(window, min_periods=max(2, window // 3)).mean()
        sec[f"bb_flow_ratio_{window}s"] = sec["bb_flow_ratio"].rolling(window, min_periods=max(2, window // 3)).mean()
        sec[f"flow_divergence_{window}s"] = sec["flow_divergence"].rolling(window, min_periods=max(2, window // 3)).mean()
        sec[f"sec_gap_change_{window}s"] = sec["sec_gap_bps"] - sec["sec_gap_bps"].shift(window)
        sec[f"ob_gap_change_{window}s"] = sec["ob_gap_bps"] - sec["ob_gap_bps"].shift(window)

    sec["bb_mid_gap_vs_trades_bps"] = 10000.0 * (sec["bb_mid_px_ob"] / sec["bb_last_price"] - 1.0)
    sec["symbol"] = symbol
    sec["date"] = day
    sec = sec.replace([np.inf, -np.inf], np.nan)
    sec.to_csv(cache_path, index=False, compression="gzip")
    return sec


def asof_lookup(query: pd.DataFrame, source: pd.DataFrame, cols: list[str], offset: int = 0) -> pd.DataFrame:
    lookup = query[["sec"]].copy()
    lookup["lookup_sec"] = lookup["sec"] + offset
    src = source[["sec"] + cols].copy().rename(columns={"sec": "source_sec"}).sort_values("source_sec")
    merged = pd.merge_asof(
        lookup.sort_values("lookup_sec"),
        src,
        left_on="lookup_sec",
        right_on="source_sec",
        direction="backward",
    )
    return merged.drop(columns=["lookup_sec", "source_sec"]).sort_values("sec").reset_index(drop=True)


def cumulative_frame(source: pd.DataFrame, col: str) -> pd.DataFrame:
    frame = source[["sec", col]].copy().sort_values("sec").reset_index(drop=True)
    value = pd.to_numeric(frame[col], errors="coerce")
    frame[f"{col}_sum_cum"] = value.fillna(0.0).cumsum()
    frame[f"{col}_count_cum"] = value.notna().astype("int64").cumsum()
    return frame[["sec", f"{col}_sum_cum", f"{col}_count_cum"]]


def rolling_sum_at_events(query: pd.DataFrame, source: pd.DataFrame, col: str, window: int) -> pd.Series:
    cum = cumulative_frame(source, col)
    sum_col = f"{col}_sum_cum"
    curr = asof_lookup(query, cum, [sum_col])
    prev = asof_lookup(query, cum, [sum_col], offset=-window)
    return curr[sum_col].fillna(0.0) - prev[sum_col].fillna(0.0)


def rolling_mean_at_events(query: pd.DataFrame, source: pd.DataFrame, col: str, window: int) -> pd.Series:
    cum = cumulative_frame(source, col)
    sum_col = f"{col}_sum_cum"
    count_col = f"{col}_count_cum"
    curr = asof_lookup(query, cum, [sum_col, count_col])
    prev = asof_lookup(query, cum, [sum_col, count_col], offset=-window)
    sum_diff = curr[sum_col].fillna(0.0) - prev[sum_col].fillna(0.0)
    count_diff = curr[count_col].fillna(0.0) - prev[count_col].fillna(0.0)
    return sum_diff / count_diff.replace(0.0, np.nan)


def build_day_microstructure_for_events(symbol: str, day: str, event_secs: pd.Series, force_rebuild: bool) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    full_cache_path = CACHE_DIR / f"{symbol}_{day}_event_micro_sec.csv.gz"
    point_cache_path = CACHE_DIR / f"{symbol}_{day}_event_micro_points.csv.gz"
    query = pd.DataFrame({"sec": sorted(pd.Series(event_secs, dtype="int64").dropna().astype("int64").unique())})

    if full_cache_path.exists() and not force_rebuild:
        try:
            cached = pd.read_csv(full_cache_path, usecols=["sec"] + MICRO_COLS)
            return query.merge(cached, on="sec", how="left")
        except Exception:
            full_cache_path.unlink(missing_ok=True)

    if point_cache_path.exists() and not force_rebuild:
        try:
            cached = pd.read_csv(point_cache_path)
            if set(query["sec"]).issubset(set(cached["sec"])):
                return query.merge(cached, on="sec", how="left")
        except Exception:
            point_cache_path.unlink(missing_ok=True)

    bn_trade_path = DATALAKE / "binance" / symbol / f"{day}_trades.csv.gz"
    bb_trade_path = DATALAKE / "bybit" / symbol / f"{day}_trades.csv.gz"
    bn_depth_path = DATALAKE / "binance" / symbol / f"{day}_bookDepth.csv.gz"
    bb_book_path = DATALAKE / "bybit" / symbol / f"{day}_orderbook.jsonl.gz"
    if not (bn_trade_path.exists() and bb_trade_path.exists() and bn_depth_path.exists() and bb_book_path.exists()):
        raise FileNotFoundError(f"missing microstructure files for {symbol} {day}")

    bn_trades = aggregate_binance_trades(bn_trade_path).reset_index().sort_values("sec").reset_index(drop=True)
    bb_trades = aggregate_bybit_trades(bb_trade_path).reset_index().sort_values("sec").reset_index(drop=True)
    bn_depth = load_binance_depth(bn_depth_path).sort_values("sec").reset_index(drop=True)
    bb_book = load_bybit_orderbook(bb_book_path).sort_values("sec").reset_index(drop=True)

    start = max(
        int(bn_trades["sec"].min()),
        int(bb_trades["sec"].min()),
        int(bn_depth["sec"].min()),
        int(bb_book["sec"].min()),
    )
    end = min(
        int(bn_trades["sec"].max()),
        int(bb_trades["sec"].max()),
        int(bn_depth["sec"].max()),
        int(bb_book["sec"].max()),
    )
    needed_secs: set[int] = set()
    for sec in query["sec"].tolist():
        lo = max(start, int(sec) - 60)
        hi = min(end, int(sec))
        needed_secs.update(range(lo, hi + 1))
    sec = pd.DataFrame(index=pd.Index(sorted(needed_secs), name="sec"))
    sec = sec.join(bn_trades.set_index("sec"), how="left").join(bb_trades.set_index("sec"), how="left").fillna(0.0)
    for col in ["bn_last_price", "bb_last_price"]:
        sec[col] = sec[col].replace(0.0, np.nan).ffill().bfill()
    sec = sec.reset_index().sort_values("sec")
    sec = pd.merge_asof(sec, bn_depth, on="sec", direction="backward")
    sec = pd.merge_asof(sec, bb_book, on="sec", direction="backward")

    sec["bn_flow_ratio"] = sec["bn_signed_notional"] / sec["bn_notional"].replace(0.0, np.nan)
    sec["bb_flow_ratio"] = sec["bb_signed_notional"] / sec["bb_notional"].replace(0.0, np.nan)
    sec["flow_divergence"] = sec["bn_flow_ratio"] - sec["bb_flow_ratio"]
    sec["sec_gap_bps"] = 10000.0 * (sec["bn_last_price"] / sec["bb_last_price"] - 1.0)
    sec["ob_gap_bps"] = 10000.0 * (sec["bn_last_price"] / sec["bb_mid_px_ob"] - 1.0)

    for window in [5, 15, 60]:
        sec[f"bn_signed_notional_{window}s"] = sec["bn_signed_notional"].rolling(window, min_periods=max(2, window // 3)).sum()
        sec[f"bb_signed_notional_{window}s"] = sec["bb_signed_notional"].rolling(window, min_periods=max(2, window // 3)).sum()
        sec[f"combo_signed_notional_{window}s"] = sec[f"bn_signed_notional_{window}s"] + sec[f"bb_signed_notional_{window}s"]
        sec[f"bn_flow_ratio_{window}s"] = sec["bn_flow_ratio"].rolling(window, min_periods=max(2, window // 3)).mean()
        sec[f"bb_flow_ratio_{window}s"] = sec["bb_flow_ratio"].rolling(window, min_periods=max(2, window // 3)).mean()
        sec[f"flow_divergence_{window}s"] = sec["flow_divergence"].rolling(window, min_periods=max(2, window // 3)).mean()
        if window in [5, 15]:
            sec[f"sec_gap_change_{window}s"] = sec["sec_gap_bps"] - sec["sec_gap_bps"].shift(window)
            sec[f"ob_gap_change_{window}s"] = sec["ob_gap_bps"] - sec["ob_gap_bps"].shift(window)

    sec["bb_mid_gap_vs_trades_bps"] = 10000.0 * (sec["bb_mid_px_ob"] / sec["bb_last_price"] - 1.0)
    out = query.merge(sec[["sec"] + MICRO_COLS], on="sec", how="left").replace([np.inf, -np.inf], np.nan)
    out.to_csv(point_cache_path, index=False, compression="gzip")
    return out


MICRO_COLS = [
    "bn_depth_imbalance_0.2",
    "bn_depth_imbalance_1",
    "bn_depth_imbalance_5",
    "bn_depth_pressure",
    "bb_top5_imbalance",
    "bb_top20_imbalance",
    "bb_best_sz_imbalance",
    "bb_spread_bps_ob",
    "bb_top5_pull_pressure",
    "bb_top5_pull_pressure_5s",
    "bb_top5_pull_pressure_15s",
    "bb_top20_imbalance_chg_5s",
    "bb_mid_gap_vs_trades_bps",
    "ob_gap_bps",
    "ob_gap_change_5s",
    "ob_gap_change_15s",
    "sec_gap_bps",
    "sec_gap_change_5s",
    "sec_gap_change_15s",
    "bn_signed_notional_15s",
    "bn_signed_notional_60s",
    "bb_signed_notional_15s",
    "bb_signed_notional_60s",
    "combo_signed_notional_15s",
    "combo_signed_notional_60s",
    "bn_flow_ratio_15s",
    "bn_flow_ratio_60s",
    "bb_flow_ratio_15s",
    "bb_flow_ratio_60s",
    "flow_divergence_15s",
    "flow_divergence_60s",
]

PANEL_COLS = [
    "symbol",
    "ts",
    "date",
    "event_sec",
    "gap_bps",
    "gap_z_60",
    "gap_z_240",
    "premium_gap_bps",
    "premium_gap_z_240",
    "crowding_gap",
    "crowding_gap_z_240",
    "bn_taker_imbalance",
    "rel_ret_5m_bps",
    "rel_ret_15m_bps",
    "bb_realized_vol_15m_bps",
    "bn_quote_volume_z_60",
    "bb_turnover_z_60",
    "oi_gap_30m",
    "oi_gap_30m_z_240",
    "gap_close_5m_bps",
    "gap_close_15m_bps",
    "gap_close_30m_bps",
    "max_gap_close_15m_bps",
    "max_gap_close_30m_bps",
    "pair_net_15m_bps",
    "pair_net_30m_bps",
    "pair_net_max_15m_bps",
    "pair_win_15m",
    "pair_win_max_15m",
    "tod_sin",
    "tod_cos",
]


def enrich_event_group(
    subset: pd.DataFrame,
    force_rebuild: bool,
) -> pd.DataFrame:
    symbol = str(subset["symbol"].iloc[0])
    day = str(subset["date"].iloc[0])
    micro = build_day_microstructure_for_events(symbol, day, subset["event_sec"], force_rebuild=force_rebuild).sort_values("sec")
    joined = pd.merge_asof(
        subset.sort_values("event_sec")[PANEL_COLS],
        micro[["sec"] + MICRO_COLS].sort_values("sec"),
        left_on="event_sec",
        right_on="sec",
        direction="backward",
        tolerance=5,
    )
    return joined


def enrich_events(events: pd.DataFrame, force_rebuild: bool, workers: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    groups = [subset.copy() for _, subset in events.groupby(["symbol", "date"], sort=True)]
    if workers <= 1:
        frames = [enrich_event_group(subset, force_rebuild) for subset in groups]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(enrich_event_group, subset, force_rebuild) for subset in groups]
            for future in futures:
                frames.append(future.result())
    out = pd.concat(frames, ignore_index=True)
    out = out.drop(columns=["sec"])
    return out.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = parse_symbols(args.symbols)
    required_symbol_days = load_required_symbol_days(args.required_symbol_days)
    panel = prepare_panel(Path(args.panel), symbols, args.start_date, args.end_date, args.pair_fee_bps_roundtrip)
    events = select_events(panel, args.min_gap_bps, args.cooldown_minutes)
    if required_symbol_days is not None:
        keep_mask = [(symbol, date) in required_symbol_days for symbol, date in zip(events["symbol"], events["date"])]
        events = events.loc[keep_mask].reset_index(drop=True)
        if events.empty:
            raise SystemExit("required symbol-day filter removed all events")
    dataset = enrich_events(events, force_rebuild=args.force_rebuild_cache, workers=args.workers)
    output_path = Path(args.output)
    dataset.to_csv(output_path, index=False)
    summary = {
        "panel": args.panel,
        "symbols": symbols,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "min_gap_bps": args.min_gap_bps,
        "cooldown_minutes": args.cooldown_minutes,
        "pair_fee_bps_roundtrip": args.pair_fee_bps_roundtrip,
        "rows": int(len(dataset)),
        "symbols_with_events": int(dataset["symbol"].nunique()),
        "pair_net_15m_bps_mean": float(dataset["pair_net_15m_bps"].mean()),
        "pair_win_15m_rate": float(dataset["pair_win_15m"].mean()),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(f"wrote {output_path}")
    print(f"events={len(dataset):,} symbols={dataset['symbol'].nunique()} mean_pair_net_15m={dataset['pair_net_15m_bps'].mean():+.2f}bps")


if __name__ == "__main__":
    main()
