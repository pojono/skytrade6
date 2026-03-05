#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATALAKE = ROOT / "datalake"
OUT_DIR = Path(__file__).resolve().parent / "out"


@dataclass(frozen=True)
class Rule:
    feature: str
    tail: str
    quantile: float
    signal: int
    threshold: float


def list_symbol_days(exchange: str, symbol: str, suffix: str) -> list[str]:
    base = DATALAKE / exchange / symbol
    if not base.exists():
        return []
    days = []
    for path in base.glob(f"*{suffix}"):
        day = path.name[:10]
        if len(day) == 10:
            days.append(day)
    return sorted(set(days))


def load_csvs(paths: list[Path], usecols: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frames.append(pd.read_csv(path, usecols=usecols))
        except ValueError:
            frames.append(pd.read_csv(path))
            frames[-1] = frames[-1][usecols]
    if not frames:
        return pd.DataFrame(columns=usecols)
    return pd.concat(frames, ignore_index=True)


def load_binance(symbol: str, days: list[str]) -> pd.DataFrame:
    base = DATALAKE / "binance" / symbol
    kline_paths = [base / f"{day}_kline_1m.csv" for day in days if (base / f"{day}_kline_1m.csv").exists()]
    metric_paths = [base / f"{day}_metrics.csv" for day in days if (base / f"{day}_metrics.csv").exists()]
    if not kline_paths:
        return pd.DataFrame()

    k = load_csvs(
        kline_paths,
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "taker_buy_volume",
            "taker_buy_quote_volume",
        ],
    )
    k["ts"] = pd.to_datetime(k["open_time"], unit="ms", utc=True)
    k = k.rename(
        columns={
            "open": "bn_open",
            "high": "bn_high",
            "low": "bn_low",
            "close": "bn_close",
            "volume": "bn_volume",
            "taker_buy_volume": "bn_taker_buy_volume",
            "taker_buy_quote_volume": "bn_taker_buy_quote_volume",
        }
    )
    for col in ["bn_open", "bn_high", "bn_low", "bn_close", "bn_volume", "bn_taker_buy_volume", "bn_taker_buy_quote_volume"]:
        k[col] = pd.to_numeric(k[col], errors="coerce")
    k = k.sort_values("ts").drop_duplicates("ts")

    m = load_csvs(
        metric_paths,
        [
            "create_time",
            "sum_open_interest",
            "sum_open_interest_value",
            "count_long_short_ratio",
            "sum_taker_long_short_vol_ratio",
        ],
    )
    if not m.empty:
        m["ts"] = pd.to_datetime(m["create_time"], utc=True)
        m = m.rename(
            columns={
                "sum_open_interest": "bn_oi",
                "sum_open_interest_value": "bn_oi_value",
                "count_long_short_ratio": "bn_ls_ratio",
                "sum_taker_long_short_vol_ratio": "bn_taker_ls_ratio",
            }
        )
        for col in ["bn_oi", "bn_oi_value", "bn_ls_ratio", "bn_taker_ls_ratio"]:
            m[col] = pd.to_numeric(m[col], errors="coerce")
        m = m.sort_values("ts").drop_duplicates("ts")
        k = pd.merge_asof(k, m[["ts", "bn_oi", "bn_oi_value", "bn_ls_ratio", "bn_taker_ls_ratio"]], on="ts", direction="backward", tolerance=pd.Timedelta("5min"))
    return k


def load_bybit(symbol: str, days: list[str]) -> pd.DataFrame:
    base = DATALAKE / "bybit" / symbol
    kline_paths = [base / f"{day}_kline_1m.csv" for day in days if (base / f"{day}_kline_1m.csv").exists()]
    oi_paths = [base / f"{day}_open_interest_5min.csv" for day in days if (base / f"{day}_open_interest_5min.csv").exists()]
    ls_paths = [base / f"{day}_long_short_ratio_5min.csv" for day in days if (base / f"{day}_long_short_ratio_5min.csv").exists()]
    if not kline_paths:
        return pd.DataFrame()

    k = load_csvs(kline_paths, ["startTime", "open", "high", "low", "close", "volume", "turnover"])
    k["ts"] = pd.to_datetime(k["startTime"], unit="ms", utc=True)
    k = k.rename(
        columns={
            "open": "bb_open",
            "high": "bb_high",
            "low": "bb_low",
            "close": "bb_close",
            "volume": "bb_volume",
            "turnover": "bb_turnover",
        }
    )
    for col in ["bb_open", "bb_high", "bb_low", "bb_close", "bb_volume", "bb_turnover"]:
        k[col] = pd.to_numeric(k[col], errors="coerce")
    k = k.sort_values("ts").drop_duplicates("ts")

    oi = load_csvs(oi_paths, ["timestamp", "openInterest"])
    if not oi.empty:
        oi["ts"] = pd.to_datetime(oi["timestamp"], unit="ms", utc=True)
        oi = oi.rename(columns={"openInterest": "bb_oi"})
        oi["bb_oi"] = pd.to_numeric(oi["bb_oi"], errors="coerce")
        oi = oi.sort_values("ts").drop_duplicates("ts")
        k = pd.merge_asof(k, oi[["ts", "bb_oi"]], on="ts", direction="backward", tolerance=pd.Timedelta("5min"))

    ls = load_csvs(ls_paths, ["timestamp", "buyRatio", "sellRatio"])
    if not ls.empty:
        ls["ts"] = pd.to_datetime(ls["timestamp"], unit="ms", utc=True)
        ls = ls.rename(columns={"buyRatio": "bb_buy_ratio", "sellRatio": "bb_sell_ratio"})
        for col in ["bb_buy_ratio", "bb_sell_ratio"]:
            ls[col] = pd.to_numeric(ls[col], errors="coerce")
        ls = ls.sort_values("ts").drop_duplicates("ts")
        k = pd.merge_asof(
            k,
            ls[["ts", "bb_buy_ratio", "bb_sell_ratio"]],
            on="ts",
            direction="backward",
            tolerance=pd.Timedelta("5min"),
        )
    return k


def build_symbol_frame(symbol: str, max_days: int) -> pd.DataFrame:
    bn_days = set(list_symbol_days("binance", symbol, "_kline_1m.csv"))
    bb_days = set(list_symbol_days("bybit", symbol, "_kline_1m.csv"))
    common_days = sorted(bn_days & bb_days)
    if len(common_days) < 10:
        return pd.DataFrame()
    days = common_days[-max_days:]
    bn = load_binance(symbol, days)
    bb = load_bybit(symbol, days)
    if bn.empty or bb.empty:
        return pd.DataFrame()

    df = pd.merge(bn, bb, on="ts", how="inner")
    if df.empty:
        return pd.DataFrame()
    df["symbol"] = symbol
    return df.sort_values("ts").reset_index(drop=True)


def add_features(df: pd.DataFrame, move_horizon: int, move_threshold_bps: float) -> pd.DataFrame:
    df = df.copy()
    px = (df["bn_close"] + df["bb_close"]) / 2.0
    df["mid_close"] = px

    for window in [1, 3, 5, 15]:
        df[f"bn_ret_{window}m"] = df["bn_close"].pct_change(window)
        df[f"bb_ret_{window}m"] = df["bb_close"].pct_change(window)
        df[f"mid_ret_{window}m"] = px.pct_change(window)

    df["basis"] = (df["bn_close"] - df["bb_close"]) / px
    df["basis_z_60"] = (df["basis"] - df["basis"].rolling(60).mean()) / df["basis"].rolling(60).std()
    df["bn_volume_z_60"] = (df["bn_volume"] - df["bn_volume"].rolling(60).mean()) / df["bn_volume"].rolling(60).std()
    df["bb_volume_z_60"] = (df["bb_volume"] - df["bb_volume"].rolling(60).mean()) / df["bb_volume"].rolling(60).std()
    df["cross_volume_ratio"] = np.log1p(df["bn_volume"]) - np.log1p(df["bb_volume"])
    df["cross_volume_ratio_z_60"] = (
        (df["cross_volume_ratio"] - df["cross_volume_ratio"].rolling(60).mean())
        / df["cross_volume_ratio"].rolling(60).std()
    )
    df["bn_taker_buy_share"] = df["bn_taker_buy_volume"] / df["bn_volume"].replace(0, np.nan)
    df["bn_taker_buy_share_z_60"] = (
        (df["bn_taker_buy_share"] - df["bn_taker_buy_share"].rolling(60).mean())
        / df["bn_taker_buy_share"].rolling(60).std()
    )
    df["bn_oi_chg_5m"] = df["bn_oi"].pct_change(5)
    df["bb_oi_chg_5m"] = df["bb_oi"].pct_change(5)
    df["oi_spread_chg_5m"] = df["bn_oi_chg_5m"] - df["bb_oi_chg_5m"]
    df["bn_ls_centered"] = df["bn_ls_ratio"] - 1.0
    df["bn_taker_ls_centered"] = df["bn_taker_ls_ratio"] - 1.0
    df["bb_long_bias"] = df["bb_buy_ratio"] - df["bb_sell_ratio"]
    df["bb_long_bias_chg_5m"] = df["bb_long_bias"].diff(5)
    df["bn_ls_centered_chg_5m"] = df["bn_ls_centered"].diff(5)
    df["realized_vol_15m"] = df["mid_ret_1m"].rolling(15).std()

    df["future_ret"] = df["mid_close"].shift(-move_horizon) / df["mid_close"] - 1.0
    threshold = move_threshold_bps / 10000.0
    df["big_up"] = (df["future_ret"] >= threshold).astype(int)
    df["big_down"] = (df["future_ret"] <= -threshold).astype(int)
    df["big_move"] = ((df["future_ret"].abs() >= threshold)).astype(int)
    return df


def split_train_test(df: pd.DataFrame, frac: float = 0.6) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []
    for _, part in df.groupby("symbol", sort=False):
        cut = max(int(len(part) * frac), 1)
        train_parts.append(part.iloc[:cut])
        test_parts.append(part.iloc[cut:])
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def evaluate_rule(df: pd.DataFrame, rule: Rule, fee_bps_roundtrip: float) -> dict[str, float] | None:
    feature = df[rule.feature]
    if rule.tail == "high":
        mask = feature >= rule.threshold
    else:
        mask = feature <= rule.threshold
    trades = df.loc[mask].copy()
    if len(trades) < 25:
        return None

    signed_ret = trades["future_ret"] * rule.signal
    big_dir_col = "big_up" if rule.signal > 0 else "big_down"
    gross_bps = signed_ret.mean() * 10000.0
    net_bps = gross_bps - fee_bps_roundtrip
    return {
        "trades": float(len(trades)),
        "gross_bps_per_trade": float(gross_bps),
        "net_bps_per_trade": float(net_bps),
        "win_rate": float((signed_ret > 0).mean()),
        "hit_big_move_rate": float(trades[big_dir_col].mean()),
        "base_big_move_rate": float(df[big_dir_col].mean()),
        "coverage": float(len(trades) / len(df)),
        "total_net_bps": float(net_bps * len(trades)),
    }


def fit_best_rule(train: pd.DataFrame, features: list[str], fee_bps_roundtrip: float) -> tuple[Rule | None, pd.DataFrame]:
    candidates: list[dict[str, float | str]] = []
    for feature in features:
        series = train[feature].replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) < 500:
            continue
        for tail, quantile in [("high", 0.90), ("high", 0.95), ("high", 0.98), ("low", 0.10), ("low", 0.05), ("low", 0.02)]:
            threshold = float(series.quantile(quantile))
            probe = Rule(feature=feature, tail=tail, quantile=quantile, signal=1, threshold=threshold)
            tmp = evaluate_rule(train, probe, fee_bps_roundtrip)
            if tmp is None:
                continue
            signal = 1 if tmp["gross_bps_per_trade"] >= 0 else -1
            rule = Rule(feature=feature, tail=tail, quantile=quantile, signal=signal, threshold=threshold)
            stats = evaluate_rule(train, rule, fee_bps_roundtrip)
            if stats is None:
                continue
            candidates.append(
                {
                    "feature": feature,
                    "tail": tail,
                    "quantile": quantile,
                    "signal": signal,
                    "threshold": threshold,
                    **stats,
                }
            )
    if not candidates:
        return None, pd.DataFrame()

    scored = pd.DataFrame(candidates)
    scored = scored.sort_values(
        ["net_bps_per_trade", "hit_big_move_rate", "coverage", "trades"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    top = scored.iloc[0]
    rule = Rule(
        feature=str(top["feature"]),
        tail=str(top["tail"]),
        quantile=float(top["quantile"]),
        signal=int(top["signal"]),
        threshold=float(top["threshold"]),
    )
    return rule, scored


def summarize_feature_conditional(df: pd.DataFrame, feature: str, quantile: float = 0.95) -> list[dict[str, float | str]]:
    series = df[feature].replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 500:
        return []
    out = []
    for tail, q in [("high", quantile), ("low", 1.0 - quantile)]:
        threshold = float(series.quantile(q))
        if tail == "high":
            sample = df.loc[df[feature] >= threshold]
        else:
            sample = df.loc[df[feature] <= threshold]
        if len(sample) < 25:
            continue
        out.append(
            {
                "feature": feature,
                "tail": tail,
                "threshold": threshold,
                "trades": float(len(sample)),
                "mean_future_bps": float(sample["future_ret"].mean() * 10000.0),
                "up_move_rate": float(sample["big_up"].mean()),
                "down_move_rate": float(sample["big_down"].mean()),
                "any_big_move_rate": float(sample["big_move"].mean()),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Search for pre-move signals across Binance and Bybit futures data.")
    parser.add_argument("--symbols", nargs="*", default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"])
    parser.add_argument("--max-days", type=int, default=45)
    parser.add_argument("--move-horizon", type=int, default=15)
    parser.add_argument("--move-threshold-bps", type=float, default=35.0)
    parser.add_argument("--fee-bps-roundtrip", type=float, default=8.0)
    args = parser.parse_args()

    frames = []
    for symbol in args.symbols:
        frame = build_symbol_frame(symbol, args.max_days)
        if frame.empty:
            print(f"skip {symbol}: insufficient overlapping Binance/Bybit data")
            continue
        frame = add_features(frame, args.move_horizon, args.move_threshold_bps)
        frames.append(frame)
        print(f"loaded {symbol}: {len(frame):,} merged rows")

    if not frames:
        raise SystemExit("No symbols with usable overlap found.")

    data = pd.concat(frames, ignore_index=True)
    feature_cols = [
        "bn_ret_1m",
        "bb_ret_1m",
        "mid_ret_1m",
        "bn_ret_3m",
        "bb_ret_3m",
        "mid_ret_3m",
        "bn_ret_5m",
        "bb_ret_5m",
        "mid_ret_5m",
        "basis",
        "basis_z_60",
        "bn_volume_z_60",
        "bb_volume_z_60",
        "cross_volume_ratio_z_60",
        "bn_taker_buy_share_z_60",
        "bn_oi_chg_5m",
        "bb_oi_chg_5m",
        "oi_spread_chg_5m",
        "bn_ls_centered",
        "bn_taker_ls_centered",
        "bb_long_bias",
        "bb_long_bias_chg_5m",
        "bn_ls_centered_chg_5m",
        "realized_vol_15m",
    ]
    required = ["future_ret", "big_up", "big_down", "big_move", *feature_cols]
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=required).reset_index(drop=True)

    train, test = split_train_test(data, frac=0.6)
    train_rule, leaderboard = fit_best_rule(train, feature_cols, args.fee_bps_roundtrip)
    if train_rule is None:
        raise SystemExit("No candidate rule met minimum sample requirements.")

    enriched_rows = []
    for row in leaderboard.itertuples(index=False):
        candidate = Rule(
            feature=str(row.feature),
            tail=str(row.tail),
            quantile=float(row.quantile),
            signal=int(row.signal),
            threshold=float(row.threshold),
        )
        test_stats = evaluate_rule(test, candidate, args.fee_bps_roundtrip)
        if test_stats is None:
            continue
        enriched_rows.append(
            {
                "feature": candidate.feature,
                "tail": candidate.tail,
                "quantile": candidate.quantile,
                "signal": candidate.signal,
                "threshold": candidate.threshold,
                "train_trades": row.trades,
                "train_gross_bps_per_trade": row.gross_bps_per_trade,
                "train_net_bps_per_trade": row.net_bps_per_trade,
                "train_win_rate": row.win_rate,
                "train_hit_big_move_rate": row.hit_big_move_rate,
                "train_base_big_move_rate": row.base_big_move_rate,
                "train_coverage": row.coverage,
                "train_total_net_bps": row.total_net_bps,
                "test_trades": test_stats["trades"],
                "test_gross_bps_per_trade": test_stats["gross_bps_per_trade"],
                "test_net_bps_per_trade": test_stats["net_bps_per_trade"],
                "test_win_rate": test_stats["win_rate"],
                "test_hit_big_move_rate": test_stats["hit_big_move_rate"],
                "test_base_big_move_rate": test_stats["base_big_move_rate"],
                "test_coverage": test_stats["coverage"],
                "test_total_net_bps": test_stats["total_net_bps"],
            }
        )
    if not enriched_rows:
        raise SystemExit("No candidate rule survived minimum trade count on the test split.")

    leaderboard = pd.DataFrame(enriched_rows).sort_values(
        ["test_net_bps_per_trade", "train_net_bps_per_trade", "test_hit_big_move_rate", "test_trades"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    robust = leaderboard[
        (leaderboard["train_net_bps_per_trade"] > 0)
        & (leaderboard["test_net_bps_per_trade"] > 0)
        & (leaderboard["train_trades"] >= 100)
        & (leaderboard["test_trades"] >= 100)
    ]
    chosen = robust.iloc[0] if not robust.empty else leaderboard.iloc[0]
    rule = Rule(
        feature=str(chosen["feature"]),
        tail=str(chosen["tail"]),
        quantile=float(chosen["quantile"]),
        signal=int(chosen["signal"]),
        threshold=float(chosen["threshold"]),
    )
    train_stats = {
        "trades": float(chosen["train_trades"]),
        "gross_bps_per_trade": float(chosen["train_gross_bps_per_trade"]),
        "net_bps_per_trade": float(chosen["train_net_bps_per_trade"]),
        "win_rate": float(chosen["train_win_rate"]),
        "hit_big_move_rate": float(chosen["train_hit_big_move_rate"]),
        "base_big_move_rate": float(chosen["train_base_big_move_rate"]),
        "coverage": float(chosen["train_coverage"]),
        "total_net_bps": float(chosen["train_total_net_bps"]),
    }
    test_stats = {
        "trades": float(chosen["test_trades"]),
        "gross_bps_per_trade": float(chosen["test_gross_bps_per_trade"]),
        "net_bps_per_trade": float(chosen["test_net_bps_per_trade"]),
        "win_rate": float(chosen["test_win_rate"]),
        "hit_big_move_rate": float(chosen["test_hit_big_move_rate"]),
        "base_big_move_rate": float(chosen["test_base_big_move_rate"]),
        "coverage": float(chosen["test_coverage"]),
        "total_net_bps": float(chosen["test_total_net_bps"]),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(OUT_DIR / "rule_leaderboard.csv", index=False)

    conditionals = []
    for feature in feature_cols:
        conditionals.extend(summarize_feature_conditional(train, feature))
    pd.DataFrame(conditionals).to_csv(OUT_DIR / "feature_conditionals.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "symbols": ",".join(args.symbols),
                "max_days": args.max_days,
                "move_horizon_min": args.move_horizon,
                "move_threshold_bps": args.move_threshold_bps,
                "fee_bps_roundtrip": args.fee_bps_roundtrip,
                "best_feature": rule.feature,
                "best_tail": rule.tail,
                "best_quantile": rule.quantile,
                "best_signal": "long" if rule.signal > 0 else "short",
                "threshold": rule.threshold,
                "train_trades": train_stats["trades"],
                "train_net_bps_per_trade": train_stats["net_bps_per_trade"],
                "train_hit_big_move_rate": train_stats["hit_big_move_rate"],
                "train_base_big_move_rate": train_stats["base_big_move_rate"],
                "test_trades": test_stats["trades"],
                "test_net_bps_per_trade": test_stats["net_bps_per_trade"],
                "test_hit_big_move_rate": test_stats["hit_big_move_rate"],
                "test_base_big_move_rate": test_stats["base_big_move_rate"],
            }
        ]
    )
    summary.to_csv(OUT_DIR / "best_rule_summary.csv", index=False)

    print("\nBest rule")
    print(
        f"feature={rule.feature} tail={rule.tail} quantile={rule.quantile:.2f} "
        f"signal={'long' if rule.signal > 0 else 'short'} threshold={rule.threshold:.6f}"
    )
    print(
        f"train: trades={train_stats['trades']:.0f}, net_bps/trade={train_stats['net_bps_per_trade']:.2f}, "
        f"big-move-hit={train_stats['hit_big_move_rate']:.3f} vs base={train_stats['base_big_move_rate']:.3f}"
    )
    print(
        f"test: trades={test_stats['trades']:.0f}, net_bps/trade={test_stats['net_bps_per_trade']:.2f}, "
        f"big-move-hit={test_stats['hit_big_move_rate']:.3f} vs base={test_stats['base_big_move_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
