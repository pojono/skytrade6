#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
        # buyer is maker => seller is taker/aggressor.
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
    return out.groupby(level=0).agg(
        {
            "bn_signed_notional": "sum",
            "bn_notional": "sum",
            "bn_trade_count": "sum",
            "bn_last_price": "last",
        }
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
                bb_trade_count=("side", "count"),
                bb_last_price=("price", "last"),
            )
        )
    out = pd.concat(parts)
    return out.groupby(level=0).agg(
        {
            "bb_signed_notional": "sum",
            "bb_notional": "sum",
            "bb_trade_count": "sum",
            "bb_last_price": "last",
        }
    )


def load_book_depth(path: Path) -> pd.DataFrame:
    depth = pd.read_csv(path)
    depth["ts"] = pd.to_datetime(depth["timestamp"], utc=True)
    depth["sec"] = (depth["ts"].astype("int64") // 10**9).astype("int64")
    piv = depth.pivot_table(index="sec", columns="percentage", values="notional", aggfunc="last").sort_index()
    piv.columns = [f"depth_{val:g}" for val in piv.columns]
    return piv.reset_index().sort_values("sec")


def build_symbol_day(symbol: str, day: str, force_rebuild: bool = False) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{symbol}_{day}_sec_features.csv"
    if cache_path.exists() and not force_rebuild:
        return pd.read_csv(cache_path)

    bn_trade_path = DATALAKE / "binance" / symbol / f"{day}_trades.csv"
    bb_trade_path = DATALAKE / "bybit" / symbol / f"{day}_trades.csv"
    bd_path = DATALAKE / "binance" / symbol / f"{day}_bookDepth.csv"
    if not (bn_trade_path.exists() and bb_trade_path.exists() and bd_path.exists()):
        raise FileNotFoundError(f"Missing required files for {symbol} {day}")

    bn = aggregate_binance_trades(bn_trade_path)
    bb = aggregate_bybit_trades(bb_trade_path)
    start = max(int(bn.index.min()), int(bb.index.min()))
    end = min(int(bn.index.max()), int(bb.index.max()))
    sec = pd.DataFrame(index=pd.Index(range(start, end + 1), name="sec"))
    sec = sec.join(bn, how="left").join(bb, how="left").fillna(0.0)
    for col in ["bn_last_price", "bb_last_price"]:
        sec[col] = sec[col].replace(0, np.nan).ffill().bfill()
    sec["mid_px"] = (sec["bn_last_price"] + sec["bb_last_price"]) / 2.0

    sec["combo_signed_notional"] = sec["bn_signed_notional"] + sec["bb_signed_notional"]
    sec["flow_ratio"] = sec["combo_signed_notional"] / (
        sec["bn_notional"] + sec["bb_notional"]
    ).replace(0, np.nan)
    for window in [5, 15, 30, 60]:
        sec[f"combo_flow_{window}s"] = sec["combo_signed_notional"].rolling(window).sum()
        sec[f"bn_flow_{window}s"] = sec["bn_signed_notional"].rolling(window).sum()
        sec[f"bb_flow_{window}s"] = sec["bb_signed_notional"].rolling(window).sum()
    sec["flow_ratio_60s"] = sec["flow_ratio"].rolling(60).mean()

    depth = load_book_depth(bd_path)
    sec = pd.merge_asof(sec.reset_index().sort_values("sec"), depth, on="sec", direction="backward")
    for pct in ["0.2", "1", "5"]:
        bid = sec[f"depth_-{pct}"]
        ask = sec[f"depth_{pct}"]
        sec[f"depth_imbalance_{pct}"] = (bid - ask) / (bid + ask)
    sec["depth_pressure"] = sec["depth_imbalance_0.2"] - sec["depth_imbalance_5"]

    for horizon in [30, 60, 120]:
        sec[f"future_ret_{horizon}s"] = sec["mid_px"].shift(-horizon) / sec["mid_px"] - 1.0

    sec = sec.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    sec.to_csv(cache_path, index=False)
    return sec


def score_threshold_rule(
    df: pd.DataFrame,
    feature: str,
    threshold: float,
    tail: str,
    signal: int,
    horizon: int,
    fee_bps_roundtrip: float,
) -> dict[str, float] | None:
    if tail == "high":
        mask = df[feature] >= threshold
    else:
        mask = df[feature] <= threshold
    sample = df.loc[mask]
    if len(sample) < 50:
        return None
    signed = sample[f"future_ret_{horizon}s"] * signal
    gross_bps = signed.mean() * 10000.0
    return {
        "count": float(len(sample)),
        "gross_bps": float(gross_bps),
        "net_bps": float(gross_bps - fee_bps_roundtrip),
        "win_rate": float((signed > 0).mean()),
        "coverage": float(len(sample) / len(df)),
    }


def scan_symbol(symbol: str, train_day: str, test_day: str, fee_bps_roundtrip: float, force_rebuild: bool) -> pd.DataFrame:
    train = build_symbol_day(symbol, train_day, force_rebuild=force_rebuild)
    test = build_symbol_day(symbol, test_day, force_rebuild=force_rebuild)

    rules: list[dict[str, float | str]] = []
    feature_horizons = {
        "combo_flow_30s": [60, 120],
        "combo_flow_60s": [60, 120],
        "bn_flow_60s": [60, 120],
        "bb_flow_60s": [60, 120],
        "flow_ratio_60s": [60, 120],
        "depth_imbalance_0.2": [60, 120],
        "depth_imbalance_1": [60, 120],
        "depth_imbalance_5": [60, 120],
        "depth_pressure": [60, 120],
    }
    quantiles = [0.95, 0.98, 0.99]

    for feature, horizons in feature_horizons.items():
        series = train[feature].dropna()
        for q in quantiles:
            for tail, quantile in [("high", q), ("low", 1.0 - q)]:
                threshold = float(series.quantile(quantile))
                probe = score_threshold_rule(train, feature, threshold, tail, 1, 120, fee_bps_roundtrip)
                if probe is None:
                    continue
                signal = 1 if probe["gross_bps"] >= 0 else -1
                for horizon in horizons:
                    train_stats = score_threshold_rule(train, feature, threshold, tail, signal, horizon, fee_bps_roundtrip)
                    test_stats = score_threshold_rule(test, feature, threshold, tail, signal, horizon, fee_bps_roundtrip)
                    if train_stats is None or test_stats is None:
                        continue
                    rules.append(
                        {
                            "symbol": symbol,
                            "feature": feature,
                            "tail": tail,
                            "quantile": q,
                            "threshold": threshold,
                            "signal": "long" if signal > 0 else "short",
                            "horizon_s": horizon,
                            "train_count": train_stats["count"],
                            "train_gross_bps": train_stats["gross_bps"],
                            "train_net_bps": train_stats["net_bps"],
                            "train_win_rate": train_stats["win_rate"],
                            "test_count": test_stats["count"],
                            "test_gross_bps": test_stats["gross_bps"],
                            "test_net_bps": test_stats["net_bps"],
                            "test_win_rate": test_stats["win_rate"],
                        }
                    )

    # Simple conjunctions: strong depth + strong trade flow in same direction.
    for depth_feature in ["depth_imbalance_0.2", "depth_imbalance_1", "depth_imbalance_5"]:
        for q_depth, q_flow in [(0.98, 0.95), (0.99, 0.98)]:
            hi_depth = float(train[depth_feature].quantile(q_depth))
            lo_depth = float(train[depth_feature].quantile(1.0 - q_depth))
            hi_flow = float(train["combo_flow_60s"].quantile(q_flow))
            lo_flow = float(train["combo_flow_60s"].quantile(1.0 - q_flow))
            for tail, d_thr, f_thr, signal in [
                ("high", hi_depth, hi_flow, 1),
                ("low", lo_depth, lo_flow, -1),
            ]:
                for horizon in [60, 120]:
                    def eval_combo(frame: pd.DataFrame) -> dict[str, float] | None:
                        if tail == "high":
                            mask = (frame[depth_feature] >= d_thr) & (frame["combo_flow_60s"] >= f_thr)
                        else:
                            mask = (frame[depth_feature] <= d_thr) & (frame["combo_flow_60s"] <= f_thr)
                        sample = frame.loc[mask]
                        if len(sample) < 50:
                            return None
                        signed = sample[f"future_ret_{horizon}s"] * signal
                        gross_bps = signed.mean() * 10000.0
                        return {
                            "count": float(len(sample)),
                            "gross_bps": float(gross_bps),
                            "net_bps": float(gross_bps - fee_bps_roundtrip),
                            "win_rate": float((signed > 0).mean()),
                        }

                    train_stats = eval_combo(train)
                    test_stats = eval_combo(test)
                    if train_stats is None or test_stats is None:
                        continue
                    rules.append(
                        {
                            "symbol": symbol,
                            "feature": f"{depth_feature}+combo_flow_60s",
                            "tail": tail,
                            "quantile": q_depth,
                            "threshold": d_thr,
                            "signal": "long" if signal > 0 else "short",
                            "horizon_s": horizon,
                            "train_count": train_stats["count"],
                            "train_gross_bps": train_stats["gross_bps"],
                            "train_net_bps": train_stats["net_bps"],
                            "train_win_rate": train_stats["win_rate"],
                            "test_count": test_stats["count"],
                            "test_gross_bps": test_stats["gross_bps"],
                            "test_net_bps": test_stats["net_bps"],
                            "test_win_rate": test_stats["win_rate"],
                        }
                    )

    return pd.DataFrame(rules)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan microstructure pre-move edges from trades + depth.")
    parser.add_argument("--symbols", nargs="*", default=["BTCUSDT", "SOLUSDT"])
    parser.add_argument("--train-day", default="2026-03-02")
    parser.add_argument("--test-day", default="2026-03-03")
    parser.add_argument("--fee-bps-roundtrip", type=float, default=8.0)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    all_rules = []
    for symbol in args.symbols:
        try:
            rules = scan_symbol(
                symbol,
                train_day=args.train_day,
                test_day=args.test_day,
                fee_bps_roundtrip=args.fee_bps_roundtrip,
                force_rebuild=args.force_rebuild,
            )
        except FileNotFoundError as exc:
            print(f"skip {symbol}: {exc}")
            continue
        if rules.empty:
            print(f"skip {symbol}: no rules met minimum sample counts")
            continue
        all_rules.append(rules)
        top = rules.sort_values(["test_net_bps", "train_net_bps"], ascending=[False, False]).iloc[0]
        print(
            f"{symbol}: best={top['feature']} {top['tail']} {top['signal']} "
            f"h={int(top['horizon_s'])}s test_net={top['test_net_bps']:.2f}bps "
            f"train_net={top['train_net_bps']:.2f}bps"
        )

    if not all_rules:
        raise SystemExit("No usable symbol results.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard = pd.concat(all_rules, ignore_index=True)
    leaderboard = leaderboard.sort_values(
        ["test_net_bps", "train_net_bps", "test_win_rate", "test_count"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    leaderboard.to_csv(OUT_DIR / "microstructure_rule_leaderboard.csv", index=False)

    robust = leaderboard[
        (leaderboard["train_net_bps"] > 0)
        & (leaderboard["test_net_bps"] > 0)
        & (leaderboard["train_count"] >= 100)
        & (leaderboard["test_count"] >= 100)
    ]
    summary_source = robust if not robust.empty else leaderboard.head(10)
    summary_source.to_csv(OUT_DIR / "microstructure_top_candidates.csv", index=False)
    print(f"wrote {OUT_DIR / 'microstructure_rule_leaderboard.csv'}")
    print(f"wrote {OUT_DIR / 'microstructure_top_candidates.csv'}")


if __name__ == "__main__":
    main()
