#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATALAKE = ROOT / "datalake"
OUT_DIR = Path(__file__).resolve().parent / "out"
CACHE_DIR = OUT_DIR / "cache"


def aggregate_binance_trades(path: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=1_000_000):
        chunk["sec"] = (chunk["time"] // 1000).astype("int64")
        chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
        chunk["qty"] = pd.to_numeric(chunk["qty"], errors="coerce")
        chunk["notional"] = chunk["price"] * chunk["qty"]
        chunk["sign"] = np.where(chunk["is_buyer_maker"].astype(str).str.lower() == "true", -1.0, 1.0)
        chunk["signed_notional"] = chunk["notional"] * chunk["sign"]
        parts.append(
            chunk.groupby("sec").agg(
                bn_signed_notional=("signed_notional", "sum"),
                bn_notional=("notional", "sum"),
                bn_last_price=("price", "last"),
            )
        )
    out = pd.concat(parts)
    return out.groupby(level=0).agg(
        {"bn_signed_notional": "sum", "bn_notional": "sum", "bn_last_price": "last"}
    )


def aggregate_bybit_trades(path: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=1_000_000):
        chunk["sec"] = chunk["timestamp"].astype(float).astype("int64")
        chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
        chunk["size"] = pd.to_numeric(chunk["size"], errors="coerce")
        chunk["notional"] = chunk["price"] * chunk["size"]
        chunk["sign"] = np.where(chunk["side"].astype(str).str.lower() == "buy", 1.0, -1.0)
        chunk["signed_notional"] = chunk["notional"] * chunk["sign"]
        parts.append(
            chunk.groupby("sec").agg(
                bb_signed_notional=("signed_notional", "sum"),
                bb_notional=("notional", "sum"),
                bb_last_price=("price", "last"),
            )
        )
    out = pd.concat(parts)
    return out.groupby(level=0).agg(
        {"bb_signed_notional": "sum", "bb_notional": "sum", "bb_last_price": "last"}
    )


def _apply_updates(side: dict[float, float], updates: list[list[str]]) -> None:
    for price_str, size_str in updates:
        price = float(price_str)
        size = float(size_str)
        if size == 0.0:
            side.pop(price, None)
        else:
            side[price] = size


def _summarize_book(sec: int, bids: dict[float, float], asks: dict[float, float]) -> dict[str, float]:
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
        "bb_spread_bps": (best_ask_px - best_bid_px) / mid * 10000.0,
        "bb_top5_bid_notional": bid5,
        "bb_top5_ask_notional": ask5,
        "bb_top20_bid_notional": bid20,
        "bb_top20_ask_notional": ask20,
        "bb_top5_imbalance": (bid5 - ask5) / (bid5 + ask5) if (bid5 + ask5) else np.nan,
        "bb_top20_imbalance": (bid20 - ask20) / (bid20 + ask20) if (bid20 + ask20) else np.nan,
    }


def build_bybit_book_sec(symbol: str, day: str, force_rebuild: bool = False) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{symbol}_{day}_bybit_orderbook_sec.csv"
    if cache_path.exists() and not force_rebuild:
        return pd.read_csv(cache_path)

    path = DATALAKE / "bybit" / symbol / f"{day}_orderbook.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    current_sec: int | None = None
    rows: list[dict[str, float]] = []

    with path.open() as handle:
        for line in handle:
            msg = json.loads(line)
            sec = int(msg["ts"]) // 1000
            data = msg["data"]
            if msg.get("type") == "snapshot":
                bids = {float(px): float(sz) for px, sz in data["b"]}
                asks = {float(px): float(sz) for px, sz in data["a"]}
            else:
                _apply_updates(bids, data["b"])
                _apply_updates(asks, data["a"])

            if current_sec is None:
                current_sec = sec
            if sec != current_sec:
                rows.append(_summarize_book(current_sec, bids, asks))
                current_sec = sec
        if current_sec is not None:
            rows.append(_summarize_book(current_sec, bids, asks))

    out = pd.DataFrame(rows).sort_values("sec").reset_index(drop=True)
    out["bb_top5_bid_delta"] = out["bb_top5_bid_notional"].diff()
    out["bb_top5_ask_delta"] = out["bb_top5_ask_notional"].diff()
    out["bb_top5_pull_pressure"] = out["bb_top5_bid_delta"] - out["bb_top5_ask_delta"]
    out["bb_best_sz_imbalance"] = (
        (out["bb_best_bid_sz"] - out["bb_best_ask_sz"]) / (out["bb_best_bid_sz"] + out["bb_best_ask_sz"])
    )
    out.to_csv(cache_path, index=False)
    return out


def build_joined_sec(symbol: str, day: str, force_rebuild: bool = False) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{symbol}_{day}_joined_microstructure.csv"
    if cache_path.exists() and not force_rebuild:
        return pd.read_csv(cache_path)

    bn_trade_path = DATALAKE / "binance" / symbol / f"{day}_trades.csv"
    bb_trade_path = DATALAKE / "bybit" / symbol / f"{day}_trades.csv"
    if not bn_trade_path.exists() or not bb_trade_path.exists():
        raise FileNotFoundError(f"Missing trade files for {symbol} {day}")

    bn = aggregate_binance_trades(bn_trade_path)
    bb = aggregate_bybit_trades(bb_trade_path)
    ob = build_bybit_book_sec(symbol, day, force_rebuild=force_rebuild)

    start = max(int(bn.index.min()), int(bb.index.min()), int(ob["sec"].min()))
    end = min(int(bn.index.max()), int(bb.index.max()), int(ob["sec"].max()))
    sec = pd.DataFrame(index=pd.Index(range(start, end + 1), name="sec"))
    sec = sec.join(bn, how="left").join(bb, how="left").fillna(0.0)
    for col in ["bn_last_price", "bb_last_price"]:
        sec[col] = sec[col].replace(0, np.nan).ffill().bfill()
    sec["mid_px"] = (sec["bn_last_price"] + sec["bb_last_price"]) / 2.0
    sec["combo_signed_notional"] = sec["bn_signed_notional"] + sec["bb_signed_notional"]
    sec["combo_flow_60s"] = sec["combo_signed_notional"].rolling(60).sum()
    sec["combo_flow_30s"] = sec["combo_signed_notional"].rolling(30).sum()
    sec = pd.merge_asof(sec.reset_index().sort_values("sec"), ob.sort_values("sec"), on="sec", direction="backward")

    sec["bb_mid_gap_bps"] = (sec["bb_mid_px_ob"] - sec["mid_px"]) / sec["mid_px"] * 10000.0
    sec["bb_top5_pull_pressure_5s"] = sec["bb_top5_pull_pressure"].rolling(5).sum()
    sec["bb_top5_pull_pressure_15s"] = sec["bb_top5_pull_pressure"].rolling(15).sum()
    sec["bb_top20_imbalance_chg_5s"] = sec["bb_top20_imbalance"].diff(5)
    for horizon in [30, 60, 120]:
        sec[f"future_ret_{horizon}s"] = sec["mid_px"].shift(-horizon) / sec["mid_px"] - 1.0

    sec = sec.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    sec.to_csv(cache_path, index=False)
    return sec


def eval_rule(df: pd.DataFrame, feature: str, threshold: float, tail: str, signal: int, horizon: int, fee_bps: float) -> dict[str, float] | None:
    mask = df[feature] >= threshold if tail == "high" else df[feature] <= threshold
    sample = df.loc[mask]
    if len(sample) < 50:
        return None
    signed = sample[f"future_ret_{horizon}s"] * signal
    gross = signed.mean() * 10000.0
    return {
        "count": float(len(sample)),
        "gross_bps": float(gross),
        "net_bps": float(gross - fee_bps),
        "win_rate": float((signed > 0).mean()),
    }


def scan_symbol(symbol: str, train_day: str, test_day: str, fee_bps: float, force_rebuild: bool) -> pd.DataFrame:
    train = build_joined_sec(symbol, train_day, force_rebuild=force_rebuild)
    test = build_joined_sec(symbol, test_day, force_rebuild=force_rebuild)

    features = [
        "bb_spread_bps",
        "bb_top5_imbalance",
        "bb_top20_imbalance",
        "bb_best_sz_imbalance",
        "bb_top5_pull_pressure",
        "bb_top5_pull_pressure_5s",
        "bb_top5_pull_pressure_15s",
        "bb_top20_imbalance_chg_5s",
        "bb_mid_gap_bps",
        "combo_flow_30s",
        "combo_flow_60s",
    ]

    rows: list[dict[str, float | str]] = []
    for feature in features:
        series = train[feature].dropna()
        for q in [0.95, 0.98, 0.99]:
            for tail, qv in [("high", q), ("low", 1.0 - q)]:
                threshold = float(series.quantile(qv))
                probe = eval_rule(train, feature, threshold, tail, 1, 120, fee_bps)
                if probe is None:
                    continue
                signal = 1 if probe["gross_bps"] >= 0 else -1
                for horizon in [60, 120]:
                    tr = eval_rule(train, feature, threshold, tail, signal, horizon, fee_bps)
                    te = eval_rule(test, feature, threshold, tail, signal, horizon, fee_bps)
                    if tr is None or te is None:
                        continue
                    rows.append(
                        {
                            "symbol": symbol,
                            "feature": feature,
                            "tail": tail,
                            "quantile": q,
                            "threshold": threshold,
                            "signal": "long" if signal > 0 else "short",
                            "horizon_s": horizon,
                            "train_count": tr["count"],
                            "train_net_bps": tr["net_bps"],
                            "train_win_rate": tr["win_rate"],
                            "test_count": te["count"],
                            "test_net_bps": te["net_bps"],
                            "test_win_rate": te["win_rate"],
                        }
                    )

    # Conjunction: book pull pressure + trade flow in same direction.
    for q_book, q_flow in [(0.98, 0.95), (0.99, 0.98)]:
        hi_book = float(train["bb_top5_pull_pressure_5s"].quantile(q_book))
        lo_book = float(train["bb_top5_pull_pressure_5s"].quantile(1.0 - q_book))
        hi_flow = float(train["combo_flow_60s"].quantile(q_flow))
        lo_flow = float(train["combo_flow_60s"].quantile(1.0 - q_flow))
        for tail, b_thr, f_thr, signal in [
            ("high", hi_book, hi_flow, 1),
            ("low", lo_book, lo_flow, -1),
        ]:
            for horizon in [60, 120]:
                def stats(frame: pd.DataFrame) -> dict[str, float] | None:
                    if tail == "high":
                        mask = (frame["bb_top5_pull_pressure_5s"] >= b_thr) & (frame["combo_flow_60s"] >= f_thr)
                    else:
                        mask = (frame["bb_top5_pull_pressure_5s"] <= b_thr) & (frame["combo_flow_60s"] <= f_thr)
                    sample = frame.loc[mask]
                    if len(sample) < 50:
                        return None
                    signed = sample[f"future_ret_{horizon}s"] * signal
                    gross = signed.mean() * 10000.0
                    return {
                        "count": float(len(sample)),
                        "net_bps": float(gross - fee_bps),
                        "win_rate": float((signed > 0).mean()),
                    }

                tr = stats(train)
                te = stats(test)
                if tr is None or te is None:
                    continue
                rows.append(
                    {
                        "symbol": symbol,
                        "feature": "bb_top5_pull_pressure_5s+combo_flow_60s",
                        "tail": tail,
                        "quantile": q_book,
                        "threshold": b_thr,
                        "signal": "long" if signal > 0 else "short",
                        "horizon_s": horizon,
                        "train_count": tr["count"],
                        "train_net_bps": tr["net_bps"],
                        "train_win_rate": tr["win_rate"],
                        "test_count": te["count"],
                        "test_net_bps": te["net_bps"],
                        "test_win_rate": te["win_rate"],
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan Bybit orderbook pressure edges.")
    parser.add_argument("--symbols", nargs="*", default=["BTCUSDT", "SOLUSDT"])
    parser.add_argument("--train-day", default="2026-03-02")
    parser.add_argument("--test-day", default="2026-03-03")
    parser.add_argument("--fee-bps-roundtrip", type=float, default=8.0)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    frames = []
    for symbol in args.symbols:
        try:
            res = scan_symbol(symbol, args.train_day, args.test_day, args.fee_bps_roundtrip, args.force_rebuild)
        except FileNotFoundError as exc:
            print(f"skip {symbol}: {exc}")
            continue
        if res.empty:
            print(f"skip {symbol}: no rules")
            continue
        frames.append(res)
        top = res.sort_values(["test_net_bps", "train_net_bps"], ascending=[False, False]).iloc[0]
        print(
            f"{symbol}: best={top['feature']} {top['tail']} {top['signal']} "
            f"h={int(top['horizon_s'])}s test_net={top['test_net_bps']:.2f}bps "
            f"train_net={top['train_net_bps']:.2f}bps"
        )

    if not frames:
        raise SystemExit("No results.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard = pd.concat(frames, ignore_index=True)
    leaderboard = leaderboard.sort_values(
        ["test_net_bps", "train_net_bps", "test_win_rate", "test_count"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    leaderboard.to_csv(OUT_DIR / "bybit_orderbook_rule_leaderboard.csv", index=False)
    robust = leaderboard[
        (leaderboard["train_net_bps"] > 0)
        & (leaderboard["test_net_bps"] > 0)
        & (leaderboard["train_count"] >= 100)
        & (leaderboard["test_count"] >= 100)
    ]
    (robust if not robust.empty else leaderboard.head(10)).to_csv(
        OUT_DIR / "bybit_orderbook_top_candidates.csv", index=False
    )
    print(f"wrote {OUT_DIR / 'bybit_orderbook_rule_leaderboard.csv'}")
    print(f"wrote {OUT_DIR / 'bybit_orderbook_top_candidates.csv'}")


if __name__ == "__main__":
    main()
